"""robosuite glue shared by the dataset converter and the eval env.

Everything here mirrors stage 0 / eval of the object-centric pipeline
(`rerender_robomimic.py`, `eval_closed_loop.py` in the main repo): same env
construction from the HDF5 `env_args`, same per-demo model-XML reload, same
depth->metres and intrinsics, same image flip. The point cloud is built from
robosuite's per-geom ("element") GT segmentation restricted to the task's
`gt_bodies` — the oracle analogue of the SAM3 masks.
"""

import json

import numpy as np

from .camera import (
    T_cam_world_from_sim,
    depth_zbuffer_to_meters,
    intrinsics_from_fovy,
    resolve_camera_id,
)
from .tasks import TASK_SPECS, register_mimicgen_envs

# Panda gripper: finger1 in [0, 0.04], finger2 in [-0.04, 0] -> open width ~0.08 m.
PANDA_MAX_OPEN_WIDTH = 0.08


# ─────────────────────────── env construction ──────────────────────────────

def load_env_meta(hdf5_path: str) -> dict:
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        return json.loads(f["data"].attrs["env_args"])


def build_env(hdf5_path: str, resolution: int, camera: str, task_name: str):
    """Raw robosuite env from the HDF5's env_args, with RGB + depth + per-geom
    segmentation on `camera`. Returns (env, env_name)."""
    import robosuite
    env_meta = load_env_meta(hdf5_path)
    env_name = env_meta["env_name"]
    expected = TASK_SPECS[task_name]["env_name"]
    if env_name != expected:
        raise SystemExit(
            f"{hdf5_path} was collected on env '{env_name}', but task "
            f"'{task_name}' expects '{expected}' (see tasks.py)")
    register_mimicgen_envs(env_name)
    kwargs = dict(env_meta["env_kwargs"])
    kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=[camera],
        camera_heights=resolution,
        camera_widths=resolution,
        camera_depths=True,
        camera_segmentations="element",
        ignore_done=True,
    )
    env = robosuite.make(env_name, **kwargs)
    return env, env_name


def load_init_states(hdf5_path: str, n_demos: int | None = None):
    """Per-demo (demo_key, model_xml, first flattened state), in demo order.
    A robosuite demo replay needs the demo's own model XML — geometry (and
    for Square_D1 the peg pose) is re-sampled per hard reset."""
    import h5py
    out = []
    with h5py.File(hdf5_path, "r") as f:
        keys = sorted((k for k in f["data"].keys() if k.startswith("demo_")),
                      key=lambda k: int(k.split("_")[1]))
        if n_demos is not None:
            keys = keys[:n_demos]
        for k in keys:
            xml = f[f"data/{k}"].attrs["model_file"]
            if isinstance(xml, bytes):
                xml = xml.decode("utf-8")
            out.append((k, xml, f[f"data/{k}/states"][0]))
    return out


def edit_model_xml(env, xml: str) -> str:
    """Rewrite the collector-machine asset paths in a stored model XML to the
    local robosuite install."""
    if hasattr(env, "edit_model_xml"):
        return env.edit_model_xml(xml)
    from robosuite.utils.mjcf_utils import postprocess_model_xml
    return postprocess_model_xml(xml)


def set_init_state(env, xml: str, state: np.ndarray) -> dict:
    """Load a demo's model XML + first state (mirrors robomimic's reset_to;
    call after env.reset()). Returns the refreshed obs dict."""
    env.reset_from_xml_string(edit_model_xml(env, xml))
    env.sim.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    return env._get_observations(force_update=True)


def camera_params(sim, camera: str, resolution: int):
    """(K, znear_m, zfar_m) from the live sim. Re-read after any model
    reload — `stat.extent` (and hence the clip planes) is model-derived."""
    extent = float(sim.model.stat.extent)
    znear_m = float(sim.model.vis.map.znear) * extent
    zfar_m = float(sim.model.vis.map.zfar) * extent
    cam_id = resolve_camera_id(sim, camera)
    fovy = float(sim.model.cam_fovy[cam_id])
    K = intrinsics_from_fovy(resolution, resolution, fovy)
    return K, znear_m, zfar_m


# ─────────────────────────── segmentation → point cloud ────────────────────

def object_geom_ids(sim, body_names) -> np.ndarray:
    """Geom ids of every geom attached to `body_names` or any body in their
    subtrees (composite objects such as the bins have child bodies)."""
    roots = set()
    for name in body_names:
        try:
            roots.add(int(sim.model.body_name2id(name)))
        except Exception:
            raise SystemExit(
                f"GT body '{name}' not found in the model. Available bodies:\n"
                f"{sorted(sim.model.body_names)}\nFix TASK_SPECS in tasks.py.")
    parent = np.asarray(sim.model.body_parentid)
    keep = set()
    for b in range(len(parent)):
        p = b
        while True:
            if p in roots:
                keep.add(b)
                break
            if p == 0:
                break
            p = int(parent[p])
    geom_body = np.asarray(sim.model.geom_bodyid)
    ids = np.nonzero(np.isin(geom_body, sorted(keep)))[0]
    if len(ids) == 0:
        raise SystemExit(f"bodies {list(body_names)} carry no geoms")
    return ids.astype(np.int32)


def render_frames(obs: dict, camera: str, znear_m: float, zfar_m: float):
    """(rgb uint8 (H,W,3), depth_m float32 (H,W), seg int32 (H,W) geom ids,
    -1 = background). All three flipped vertically like stage 0 does, so a
    standard OpenCV K (y down) applies."""
    rgb = obs[f"{camera}_image"][::-1].copy()
    z = obs[f"{camera}_depth"].squeeze()[::-1].copy()
    depth_m = depth_zbuffer_to_meters(z, znear_m, zfar_m)
    depth_m[depth_m >= zfar_m * 0.999] = 0.0
    seg = obs[f"{camera}_segmentation_element"].squeeze()[::-1].copy()
    return rgb.astype(np.uint8), depth_m, seg.astype(np.int32)


def object_point_cloud(depth_m, seg, geom_ids, K, T_world_cam,
                       max_points: int | None = None, rng=None) -> np.ndarray:
    """Backproject the pixels of `geom_ids` into a WORLD-frame (M,3) float32
    cloud. OpenCV backprojection with K, then T_world_cam (the inverse of
    `T_cam_world_from_sim(..., gl_to_cv=True)`)."""
    mask = np.isin(seg, geom_ids) & (depth_m > 0)
    vs, us = np.nonzero(mask)
    if len(vs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth_m[vs, us].astype(np.float64)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    x = (us.astype(np.float64) - cx) / fx * z
    y = (vs.astype(np.float64) - cy) / fy * z
    pts_cam = np.stack([x, y, z], axis=-1)
    pts = pts_cam @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
    if max_points is not None and len(pts) > max_points:
        rng = np.random if rng is None else rng
        pts = pts[rng.choice(len(pts), size=max_points, replace=False)]
    return pts.astype(np.float32)


# ─────────────────────────── state / action layout ─────────────────────────

def make_state(obs: dict) -> np.ndarray:
    """EquiBot 13-dim eef state (dof=7 layout, see base_env._get_obs):
    [eef_pos(3), R[:,0](3), R[:,2](3), gravity_dir(3), gripper(1)].
    gripper = open width / 0.08 in [0,1] (EquiBot's own envs use a 0/1
    attached flag; a normalized width is the same scalar slot, more
    informative). Scalars are NOT normalized by the agent, so keep it O(1)."""
    from robosuite.utils.transform_utils import quat2mat
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    R = quat2mat(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))  # xyzw
    q = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    width = float(q[0] - q[1])
    grip = np.clip(width / PANDA_MAX_OPEN_WIDTH, 0.0, 1.0)
    return np.concatenate([
        pos, R[:, 0], R[:, 2], np.array([0.0, 0.0, -1.0]), [grip],
    ]).astype(np.float32)


def osc_to_equibot_action(a7: np.ndarray) -> np.ndarray:
    """robosuite OSC_POSE [dx,dy,dz, ax,ay,az, grip] -> EquiBot dof=7
    [grip, dx,dy,dz, ax,ay,az]. Both are world-frame deltas in [-1,1]; EquiBot
    treats index 0 as the scalar channel and 1:4 / 4:7 as two type-1 vectors
    (rotated by the equivariant layers, position additionally scaled)."""
    a7 = np.asarray(a7, dtype=np.float32).reshape(7)
    return np.concatenate([a7[6:7], a7[0:6]]).astype(np.float32)


def equibot_to_osc_action(a7: np.ndarray) -> np.ndarray:
    a7 = np.asarray(a7, dtype=np.float64).reshape(7)
    out = np.concatenate([a7[1:7], a7[0:1]])
    return np.clip(out, -1.0, 1.0)
