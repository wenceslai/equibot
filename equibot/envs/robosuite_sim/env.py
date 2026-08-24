"""EquiBot-interface wrapper around a robosuite / MimicGen task, with the
per-episode camera orbit of the viewpoint-shift protocol.

Interface consumed by `equibot.policies.eval.run_eval` (non-vectorized):
    reset() -> state (num_eef=1, 13)
    render() -> {"pc": (M,3) world-frame object cloud, "images": [rgb]}
    step(action, dummy_reward) -> (state, rew, done, info)
    compute_reward() -> 1.0 once the task succeeded, else 0.0
    args.max_episode_length

Per reset (same order as the main repo's eval loop, so the per-episode camera
is a pure function of (seed, episode)):
    np.random.seed(seed + ep) -> env.reset() -> [demo XML + first state] ->
    orbit_shift_camera(cam_shift_deg) -> refresh obs, K/clip planes, extrinsic,
    object geom ids.
"""

import numpy as np

from .camera import T_cam_world_from_sim, orbit_shift_camera, table_z_for_env
from .sim_utils import (
    build_env,
    camera_params,
    fallback_pc,
    equibot_to_osc_action,
    load_init_states,
    make_state,
    object_geom_ids,
    object_point_cloud,
    render_frames,
    set_init_state,
)
from .tasks import TASK_SPECS


class RobosuiteEnv:
    def __init__(self, args):
        self.args = args
        self.seed = int(args.seed)
        self.task_name = str(args.task_name)
        self.spec = TASK_SPECS[self.task_name]
        self.camera = str(args.camera)
        self.resolution = int(args.resolution)
        self.max_episode_length = int(args.max_episode_length)
        self.max_points = int(args.get("max_points", 4096))
        self.video_resolution = int(args.get("video_resolution", 128))

        init_mode = str(args.get("init_states", "demo"))
        if init_mode not in ("demo", "random"):
            raise ValueError(f"init_states must be demo|random, got {init_mode}")
        self.init_states = (load_init_states(args.dataset_path)
                            if init_mode == "demo" else None)

        cam_shift_deg = args.get("cam_shift_deg", None)
        self.cam_shift_deg = None if cam_shift_deg is None else float(cam_shift_deg)
        self.cam_shift_mode = str(args.get("cam_shift_mode", "azimuth_elev"))
        self.cam_elev_ratio = float(args.get("cam_elev_ratio", 0.3))
        self.cam_elev_cap_deg = float(args.get("cam_elev_cap_deg", 15.0))
        self.cam_radius_scale = float(args.get("cam_radius_scale", 1.0))

        self.env, self.env_name = build_env(
            args.dataset_path, self.resolution, self.camera, self.task_name)
        self.dof = 7
        self.num_eef = 1

        self._ep = 0
        self._t = 0
        self._success = False
        self._obs = None
        self.episode_log = []

    # ── camera bookkeeping ────────────────────────────────────────────────
    def _refresh_camera(self):
        sim = self.env.sim
        self.K, self.znear, self.zfar = camera_params(sim, self.camera, self.resolution)
        self.T_cw = T_cam_world_from_sim(sim, self.camera)
        self.T_wc = np.linalg.inv(self.T_cw)
        # Geom ids can change with the per-demo model reload.
        self.geom_ids = object_geom_ids(sim, self.spec["gt_bodies"])

    # ── EquiBot env interface ─────────────────────────────────────────────
    def reset(self, dummy_obs=False):
        ep = self._ep
        # Same seeding as eval_closed_loop.py: fixes the random layout AND
        # the orbit direction/tilt for this episode index.
        np.random.seed(self.seed + ep)
        obs = self.env.reset()
        demo_key = None
        if self.init_states is not None:
            demo_key, xml, state = self.init_states[ep % len(self.init_states)]
            obs = set_init_state(self.env, xml, state)

        cam = {"shift_deg": self.cam_shift_deg}
        if self.cam_shift_deg is not None:
            # Re-applied every episode: robosuite hard resets (and the demo
            # XML reload) recompile the model and wipe cam_pos/cam_quat edits.
            pivot, dir_deg, elev_deg = orbit_shift_camera(
                self.env.sim, self.camera, table_z_for_env(self.env),
                self.cam_shift_deg, self.cam_radius_scale,
                axis_mode=self.cam_shift_mode,
                elev_ratio=self.cam_elev_ratio,
                elev_cap_deg=self.cam_elev_cap_deg,
            )
            obs = self.env._get_observations(force_update=True)
            cam.update(mode=self.cam_shift_mode, direction_deg=dir_deg,
                       elev_deg=elev_deg, radius_scale=self.cam_radius_scale,
                       pivot=[float(v) for v in pivot])
            print(f"[robosuite_env] ep {ep}: camera orbit {self.cam_shift_mode} "
                  f"{self.cam_shift_deg:.1f} deg (dir {dir_deg:+.0f}, elev "
                  f"{elev_deg:+.1f}), pivot {np.round(pivot, 3)}", flush=True)
        self._refresh_camera()
        cam["T_cam_world"] = self.T_cw.tolist()

        self._obs = obs
        self._t = 0
        self._success = False
        self._last_pc = None
        self._empty_frames = 0
        self._ep += 1
        self.episode_log.append({"episode": ep, "demo_key": demo_key,
                                 "seed": self.seed + ep, "camera": cam})
        return make_state(obs)[None]

    def render(self, **kwargs):
        rgb, depth_m, seg = render_frames(self._obs, self.camera, self.znear, self.zfar)
        pc = object_point_cloud(depth_m, seg, self.geom_ids, self.K, self.T_wc,
                                max_points=self.max_points)
        if len(pc) == 0:
            # Full occlusion — same treatment as the dataset converter: carry
            # the last visible cloud forward, or the sentinel blob if nothing
            # was ever visible. (An empty cloud would otherwise end the
            # episode inside run_eval; a degenerate one would NaN the encoder.)
            self._empty_frames += 1
            pc = self._last_pc if self._last_pc is not None else fallback_pc()
        else:
            self._last_pc = pc
        s = max(1, rgb.shape[0] // self.video_resolution)
        return {"pc": pc, "images": [rgb[::s, ::s]]}

    def step(self, action, dummy_reward=False, dummy_obs=False):
        a = equibot_to_osc_action(np.asarray(action).reshape(-1)[:7])
        obs, _, _, _ = self.env.step(a)
        self._obs = obs
        self._t += 1
        if self.env._check_success():
            self._success = True
        done = self._success or self._t >= self.max_episode_length
        if done:
            self.episode_log[-1].update(success=bool(self._success), steps=self._t,
                                        occluded_frames=self._empty_frames)
        rew = 0.0 if dummy_reward else float(self._success)
        return make_state(obs)[None], rew, done, {}

    def compute_reward(self):
        return float(self._success)

    def close(self):
        self.env.close()
