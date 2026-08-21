"""Camera helpers for the robosuite viewpoint-shift protocol.

These are VERBATIM ports of `src_pipeline2/libero_camera.py` (and the two
depth/intrinsics helpers of `rerender_libero_object.py`) from the
view-agnostic-imitation-learning repo, so that the EquiBot baseline sees
exactly the same camera orbit as the object-centric policy: same pivot rule,
same rotation, same RNG consumption order. Do not "improve" them here — any
change must be mirrored in the main repo or the two evals stop being
comparable.

Conventions: MuJoCo cameras are OpenGL (+X right, +Y up, -Z forward); the
point-cloud code in this package is OpenCV (+X right, +Y down, +Z forward),
hence the diag(1,-1,-1) flip in `T_cam_world_from_sim`.
"""

import numpy as np


def quat_wxyz_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),         2 * (x * z + y * w)],
        [2 * (x * y + z * w),         1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [2 * (x * z - y * w),         2 * (y * z + x * w),         1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def rotation_matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> WXYZ unit quaternion (Shepperd's method)."""
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def rotvec_to_matrix(rv: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: 3-vec axis-angle -> 3x3 rotation matrix."""
    rv = np.asarray(rv, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    k = rv / theta
    kx, ky, kz = k
    K = np.array([
        [0.0, -kz, ky],
        [kz, 0.0, -kx],
        [-ky, kx, 0.0],
    ], dtype=np.float64)
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * K
        + (1.0 - np.cos(theta)) * (K @ K)
    )


def resolve_camera_id(sim, name: str) -> int:
    try:
        return sim.model.camera_name2id(name)
    except Exception:
        names = list(sim.model.camera_names)
        return names.index(name)


def T_cam_world_from_sim(sim, camera_name: str, gl_to_cv: bool = True) -> np.ndarray:
    """Build T_cam_world from the pose MuJoCo is *actually rendering with*
    (`sim.data.cam_xpos/cam_xmat`), immune to XML-vs-runtime camera edits."""
    cam_id = resolve_camera_id(sim, camera_name)
    T_world_cam = np.eye(4, dtype=np.float64)
    T_world_cam[:3, :3] = np.asarray(
        sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    T_world_cam[:3, 3] = np.asarray(sim.data.cam_xpos[cam_id], dtype=np.float64)
    T_cam_world = np.linalg.inv(T_world_cam)
    if gl_to_cv:
        T_cam_world = np.diag([1.0, -1.0, -1.0, 1.0]) @ T_cam_world
    return T_cam_world


def orbit_shift_camera(
    sim,
    camera_name: str,
    table_z: float,
    shift_deg: float,
    radius_scale: float = 1.0,
    min_elev_deg: float = 5.0,
    max_tries: int = 100,
    axis_mode: str = "sphere",
    elev_ratio: float = 0.3,
    elev_cap_deg: float = 15.0,
    max_elev_deg: float = 85.0,
):
    """Move a fixed camera `shift_deg` of arc along the sphere centred on the
    pivot (optical axis ∩ the z=table_z plane) in a random tangent direction,
    then scale the pivot distance by `radius_scale`.

    `axis_mode="azimuth"` instead rotates about the WORLD VERTICAL through the
    pivot, in a random per-episode direction (sign). The camera's height and
    its elevation above the table are then preserved exactly.

    `axis_mode="azimuth_elev"` is that same orbit plus a random up/down tilt
    about the horizontal axis perpendicular to the vertical plane through the
    camera (moves elevation, leaves azimuth untouched). Its bound GROWS with
    the shift:

        elev_deg ~ U(-b, +b),   b = min(elev_ratio * |shift_deg|, elev_cap_deg)

    clamped so the result stays within [min_elev_deg, max_elev_deg] above the
    table. b = 0 at shift_deg = 0, so the identity check is still exact.

    Implemented as a rigid rotation of the whole camera pose about the pivot:
    the camera stays aimed at the pivot, roll is preserved, and shift_deg=0
    is exactly the identity. The direction is drawn from np.random — seed
    upstream for reproducibility. Writes sim.model.cam_pos/cam_quat and calls
    sim.forward(); re-read the extrinsic afterwards (T_cam_world_from_sim).
    Returns (pivot (3,), direction_deg, elev_deg).
    """
    try:
        cam_id = sim.model.camera_name2id(camera_name)
    except Exception:
        cam_id = list(sim.model.camera_names).index(camera_name)
    p = np.asarray(sim.data.cam_xpos[cam_id], dtype=np.float64).copy()
    R_gl = np.asarray(
        sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3).copy()
    d = -R_gl[:, 2]                       # GL camera looks along its -Z
    if d[2] >= -1e-6:
        raise ValueError(
            f"camera {camera_name!r} does not look down at the table plane")
    pivot = p + ((table_z - p[2]) / d[2]) * d
    r = p - pivot
    radius = float(np.linalg.norm(r))

    if axis_mode in ("azimuth", "azimuth_elev"):
        sign = 1.0 if np.random.rand() < 0.5 else -1.0
        R_shift = rotvec_to_matrix(
            np.array([0.0, 0.0, 1.0]) * np.deg2rad(sign * shift_deg))
        r_new = R_shift @ r
        elev_deg = 0.0
        if axis_mode == "azimuth_elev":
            axis = np.cross(r_new, np.array([0.0, 0.0, 1.0]))
            n_axis = float(np.linalg.norm(axis))
            if n_axis < 1e-8:
                raise ValueError(
                    f"camera {camera_name!r} is directly above the pivot — "
                    f"elevation tilt is undefined there (use axis_mode="
                    f"'azimuth')")
            axis /= n_axis
            elev0 = np.degrees(
                np.arcsin(np.clip(r_new[2] / radius, -1.0, 1.0)))
            bound = min(elev_ratio * abs(shift_deg), elev_cap_deg)
            lo = max(-bound, min_elev_deg - elev0)
            hi = min(bound, max_elev_deg - elev0)
            elev_deg = float(np.random.uniform(lo, hi)) if hi > lo else 0.0
            R_elev = rotvec_to_matrix(axis * np.deg2rad(elev_deg))
            R_shift = R_elev @ R_shift
            r_new = R_elev @ r_new
        p_new = pivot + radius_scale * r_new
        sim.model.cam_pos[cam_id] = p_new
        sim.model.cam_quat[cam_id] = rotation_matrix_to_quat_wxyz(R_shift @ R_gl)
        sim.forward()
        if not np.allclose(sim.data.cam_xpos[cam_id], p_new, atol=1e-6):
            raise RuntimeError(
                f"camera {camera_name!r} did not land at the requested pose — "
                f"it is not attached to the worldbody (model cam_pos is "
                f"parent-relative)")
        return pivot, float(sign * shift_deg), elev_deg
    if axis_mode != "sphere":
        raise ValueError(
            f"unknown axis_mode {axis_mode!r} (sphere|azimuth|azimuth_elev)")

    e1 = np.cross(d, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(e1) < 1e-8:         # optical axis (near-)vertical
        e1 = np.cross(d, np.array([1.0, 0.0, 0.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)

    min_z = np.sin(np.deg2rad(min_elev_deg)) * radius
    for _ in range(max_tries):
        phi = float(np.random.uniform(0.0, 2.0 * np.pi))
        axis = np.cross(d, np.cos(phi) * e1 + np.sin(phi) * e2)
        R_shift = rotvec_to_matrix(axis / np.linalg.norm(axis)
                                   * np.deg2rad(shift_deg))
        r_new = R_shift @ r
        if r_new[2] >= min_z:
            break
    else:
        raise RuntimeError(
            f"no orbit direction keeps the camera >= {min_elev_deg} deg above "
            f"the table plane (shift_deg={shift_deg})")

    p_new = pivot + radius_scale * r_new
    sim.model.cam_pos[cam_id] = p_new
    sim.model.cam_quat[cam_id] = rotation_matrix_to_quat_wxyz(R_shift @ R_gl)
    sim.forward()
    if not np.allclose(sim.data.cam_xpos[cam_id], p_new, atol=1e-6):
        raise RuntimeError(
            f"camera {camera_name!r} did not land at the requested pose — it "
            f"is not attached to the worldbody (model cam_pos is "
            f"parent-relative)")
    return pivot, float(np.degrees(phi)), 0.0


def table_z_for_env(inner_env) -> float:
    """Table-top height — the orbit pivot is the optical axis ∩ this plane."""
    arena = getattr(inner_env.model, "mujoco_arena", None)
    offset = getattr(arena, "table_offset", None)
    if offset is not None:          # TableArena-based (Lift, NutAssembly, Stack)
        return float(offset[2])
    bin1 = getattr(inner_env, "bin1_pos", None)   # PickPlace uses BinsArena:
    if bin1 is not None:                          # no table_offset, bin top ~z
        return float(bin1[2])
    floor = getattr(arena, "floor_body", None)
    if floor is not None:
        pos = floor.get("pos")
        return float(pos.split()[2]) if pos else 0.0
    raise RuntimeError(
        "camera orbit found neither mujoco_arena.table_offset, env.bin1_pos, "
        "nor mujoco_arena.floor_body to place the pivot")


def depth_zbuffer_to_meters(z: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    """OpenGL z-buffer in [0,1] -> linear depth in metres."""
    z = z.astype(np.float64)
    return (near_m / (1.0 - z * (1.0 - near_m / far_m))).astype(np.float32)


def intrinsics_from_fovy(height: int, width: int, fovy_deg: float) -> np.ndarray:
    fovy_rad = np.deg2rad(fovy_deg)
    fy = 0.5 * height / np.tan(0.5 * fovy_rad)
    fx = fy  # robosuite uses square pixels
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
