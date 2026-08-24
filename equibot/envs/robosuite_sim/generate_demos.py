"""Convert a robomimic / MimicGen demo HDF5 into EquiBot's per-step dataset.

Replays every demo in robosuite from the TRAINING camera (`agentview`, no
orbit), renders depth + per-geom segmentation, and writes one file per step:

    <out_dir>/pcs/<task>_ep<EEE>_t<TTTT>.npz
        pc       (M,3) float32   world-frame cloud of the task objects (<= --max_points)
        eef_pos  (1,13) float32  EquiBot eef state (sim_utils.make_state)
        action   (7,)  float32   [grip, dx,dy,dz, ax,ay,az]  (sim_utils.osc_to_equibot_action)
    <out_dir>/meta.json          camera K / extrinsic / clip planes, bodies, replay success
    <out_dir>/preview_ep<EEE>.png   first frame with the object mask (every --preview_every)

`data.dataset.path` for training is `<out_dir>/pcs`.

    python -m equibot.envs.robosuite_sim.generate_demos \
        --dataset /data/demos/stack_d1.hdf5 --task stack_d1 \
        --out_dir /data/equibot/stack_d1 --num_demos 200
"""

import argparse
import json
import os
import time

import numpy as np
from tqdm import tqdm

from .camera import T_cam_world_from_sim
from .sim_utils import (
    build_env,
    fallback_pc,
    camera_params,
    load_init_states,
    make_state,
    object_geom_ids,
    object_point_cloud,
    osc_to_equibot_action,
    render_frames,
    set_init_state,
)
from .tasks import TASK_SPECS


def _save_preview(path, rgb, seg, geom_ids):
    from PIL import Image
    img = rgb.copy()
    m = np.isin(seg, geom_ids)
    img[m] = (0.4 * img[m] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
    Image.fromarray(img).save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="demo HDF5")
    p.add_argument("--task", required=True, choices=sorted(TASK_SPECS))
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_demos", type=int, default=200)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--camera", default="agentview")
    p.add_argument("--max_points", type=int, default=4096,
                   help="cap on stored points per frame (EquiBot stores 4096; "
                        "the loader samples num_points=1024 from them)")
    p.add_argument("--preview_every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    spec = TASK_SPECS[args.task]
    pcs_dir = os.path.join(args.out_dir, "pcs")
    os.makedirs(pcs_dir, exist_ok=True)

    import h5py
    env, env_name = build_env(args.dataset, args.resolution, args.camera, args.task)
    init_states = load_init_states(args.dataset, args.num_demos)
    print(f"[generate_demos] {env_name}: {len(init_states)} demos -> {pcs_dir}")

    n_success = 0
    n_empty = 0
    ep_lengths = {}
    n_points_stats = []
    meta_cam = None
    t0 = time.time()
    with h5py.File(args.dataset, "r") as f:
        for ep_i, (demo_key, xml, state0) in enumerate(tqdm(init_states, desc=args.task)):
            states = f[f"data/{demo_key}/states"][()]
            actions = f[f"data/{demo_key}/actions"][()].astype(np.float32)
            assert len(states) == len(actions), demo_key
            assert actions.shape[1] == 7, f"{demo_key}: expected 7-dim OSC actions, got {actions.shape}"

            env.reset()
            set_init_state(env, xml, state0)
            sim = env.sim
            K, znear, zfar = camera_params(sim, args.camera, args.resolution)
            T_cw = T_cam_world_from_sim(sim, args.camera)
            T_wc = np.linalg.inv(T_cw)
            geom_ids = object_geom_ids(sim, spec["gt_bodies"])
            if meta_cam is None:
                meta_cam = {"K": K.tolist(), "T_cam_world": T_cw.tolist(),
                            "znear_m": znear, "zfar_m": zfar}
            elif not np.allclose(np.array(meta_cam["T_cam_world"]), T_cw, atol=1e-6):
                print(f"[WARN] {demo_key}: training camera differs from demo 0's")

            ep_idx = int(demo_key.split("_")[1])
            prev_pc = None
            for t, (state, action) in enumerate(zip(states, actions)):
                sim.set_state_from_flattened(state)
                sim.forward()
                obs = env._get_observations(force_update=True)
                rgb, depth_m, seg = render_frames(obs, args.camera, znear, zfar)
                pc = object_point_cloud(depth_m, seg, geom_ids, K, T_wc,
                                        max_points=args.max_points, rng=rng)
                n_points_stats.append(len(pc))
                if len(pc) == 0:
                    # Full occlusion (e.g. the gripper covering the nut
                    # mid-grasp): carry the last visible cloud forward — an
                    # empty pc would crash BaseDataset's np.random.choice.
                    # No previous cloud yet (empty from the very first frame)
                    # -> a fixed sentinel blob; see sim_utils.fallback_pc.
                    n_empty += 1
                    print(f"[WARN] {demo_key} t={t}: no object pixels — using "
                          + ("previous frame's cloud" if prev_pc is not None
                             else "fallback sentinel cloud"))
                    pc = prev_pc if prev_pc is not None else fallback_pc()
                else:
                    prev_pc = pc
                np.savez(
                    os.path.join(pcs_dir, f"{args.task}_ep{ep_idx:03d}_t{t:04d}.npz"),
                    pc=pc,
                    eef_pos=make_state(obs)[None],
                    action=osc_to_equibot_action(action),
                )
                if t == 0 and ep_i % args.preview_every == 0:
                    _save_preview(os.path.join(args.out_dir, f"preview_ep{ep_idx:03d}.png"),
                                  rgb, seg, geom_ids)
            ep_lengths[demo_key] = int(len(states))
            if env._check_success():
                n_success += 1

    rate = n_success / max(1, len(init_states))
    print(f"[generate_demos] replay success rate {rate:.2f} "
          f"({n_success}/{len(init_states)}); ~1.0 expected for robomimic ph, "
          f"~0.7 is NORMAL for MimicGen (terminal state not stored).")
    print(f"[generate_demos] points/frame: min {np.min(n_points_stats)}, "
          f"median {np.median(n_points_stats):.0f}; "
          f"{n_empty} fully-occluded frames carried forward; "
          f"{time.time() - t0:.0f}s")
    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump({
            "task": args.task, "env_name": env_name, "dataset": os.path.abspath(args.dataset),
            "camera": args.camera, "resolution": args.resolution,
            "gt_bodies": spec["gt_bodies"], "max_points": args.max_points,
            "num_demos": len(init_states), "episode_lengths": ep_lengths,
            "replay_success_rate": rate, "min_points": int(np.min(n_points_stats)),
            "empty_frames_carried": n_empty,
            **meta_cam,
        }, fh, indent=2)
    env.close()


if __name__ == "__main__":
    main()
