# EquiBot as a viewpoint-robustness baseline on robosuite / MimicGen

This fork adds a robosuite back-end to EquiBot so it can be run as the
baseline for the *train-on-one-view, evaluate-under-camera-orbit* experiment
of the view-agnostic imitation-learning project. The policy code is untouched;
everything new lives in `equibot/envs/robosuite_sim/`, two configs, and
`scripts/`.

**TL;DR — run it**

```bash
# 1. clone
git clone git@github.com:wenceslai/equibot.git && cd equibot

# 2. one conda env (EquiBot stack + robosuite 1.4.1 + mimicgen), ~15 min
bash scripts/install.sh
conda activate equibot

# 3. scratch dirs (demos ~5 GB, point clouds ~6 GB, logs ~2 GB) — not NFS home
export DATA_ROOT=/scratch/$USER/equibot_data
export LOG_ROOT=/scratch/$USER/equibot_logs

# 4. run everything (needs a GPU node; set MUJOCO_GL=osmesa if you get EGL errors)
bash scripts/run_viewpoint_experiment.sh
#    or one SLURM job per task (edit partition/account in the sbatch first;
#    DATA_ROOT/LOG_ROOT are inherited by sbatch):
#      for t in can square_d1 stack_d1 stack_three_d1; do TASKS=$t sbatch scripts/slurm_example.sbatch; done
#      STAGES=collect bash scripts/run_viewpoint_experiment.sh      # after all four finished

# 5. result
cat $LOG_ROOT/results_equibot.md        # tasks x angles success rate
#   + $LOG_ROOT/eval/*/eval_results.json (per-episode camera direction/tilt) — send both back
```

The script downloads the demos, converts them, trains one policy per task on
the training camera, evaluates every policy with the camera orbited by
0/5/15/30/45/60/90°, and writes a tasks × angles success-rate table. Every
stage is skipped if its output exists, so re-submitting after a time-out
resumes where it stopped. If something breaks, the log line names the stage;
`SMOKE=1 TASKS=can bash scripts/run_viewpoint_experiment.sh` re-runs the
whole chain at toy scale in ~10 min for debugging.

## What is being measured

| | object-centric pipeline (main repo) | EquiBot baseline (this fork) |
|---|---|---|
| environment | robosuite 1.4 env rebuilt from the HDF5 `env_args`, per-demo model XML reload | identical (`sim_utils.build_env`, `set_init_state`) |
| tasks | can (robomimic ph), Square_D1 / Stack_D1 / StackThree_D1 (MimicGen) | same HDF5s, same demo order |
| training camera | `agentview`, unshifted | same |
| object input | SAM3 masks → SAM-3D shape + FoundationPose pose, per object | GT per-geom segmentation of the same `gt_bodies` → one world-frame point cloud (1024 pts sampled from ≤4096) |
| proprio | eef pos + axis-angle + gripper | eef pos + two rotation columns + gravity + gripper width (EquiBot's 13-dim layout) |
| action | OSC_POSE world-frame delta, gripper ±1 | same numbers, EquiBot ordering `[grip, dxyz, d-axis-angle]` |
| eval layouts | `np.random.seed(seed+ep)` before each reset; `--init_states random` (fresh layouts) or `demo` | same (`INIT_STATES`, default `random`) |
| viewpoint shift | `orbit_shift_camera` (pivot = optical axis ∩ table plane; `azimuth_elev`: ring orbit + bounded tilt), re-applied after every reset, live extrinsic re-read | **verbatim copy** of that function (`robosuite_sim/camera.py`) called in the same order → same camera per `(seed, episode)` |
| extrinsics at eval | known (live sim) | known (live sim) — the cloud is unprojected with the shifted camera's `T_world_cam` |
| success | `_check_success()` after every step, ≤600 steps | same (`RobosuiteEnv.step`) |

So the two methods see the *same scene from the same moved camera*; what
differs is only how the observation is encoded. For EquiBot the camera move
changes which surface points are visible (self-occlusion, the far side of the
objects) and nothing else — that is exactly the partial-view sensitivity the
baseline is meant to quantify. With the DP baseline (`AGENT=dp`) the encoder
is a plain PointNet, so the same data also gives the non-equivariant
point-cloud reference.

Two protocol decisions worth knowing:

* **`can` gets the target bin (`bin2`) in its cloud.** EquiBot centres the
  cloud on its centroid and expresses the end-effector relative to it, so
  absolute world position is not observable; without the bin the policy has
  no way to know where to carry the can. Square/stack already contain their
  target (peg / lower cube). The object-centric pipeline gets this from its
  anchor token instead. Edit `gt_bodies` in `tasks.py` to change.
* **Gripper scalar = open width / 0.08** (EquiBot's own envs use a 0/1
  "attached" flag). It sits in the un-normalised scalar channel, so it is
  kept O(1).

## Install

`scripts/install.sh` does the following; run it line by line if your cluster
needs different CUDA wheels.

1. EquiBot's own stack (README above): python 3.10, torch 2.1 + cu118,
   pytorch3d (only `knn_points` is used — the prebuilt wheel index
   `py310_cu118_pyt210` avoids a source build), `pip install -e .`.
2. `robosuite==1.4.1` (pinned in `setup.py`; the demo XMLs and MimicGen do
   not work with 1.5) and `h5py`.
3. MimicGen envs: `git clone https://github.com/NVlabs/mimicgen && pip install -e mimicgen --no-deps`.
   `--no-deps` is deliberate — mimicgen would otherwise move the robosuite
   pin. Do **not** run from inside a directory containing the `mimicgen`
   clone (the outer dir shadows the installed package).
4. Offscreen rendering: `MUJOCO_GL=egl` needs a GPU visible to the job; if
   you see `EGL`/`/dev/dri` errors use `MUJOCO_GL=osmesa` (CPU, ~5× slower).

Verify: `python -c "import robosuite, mimicgen; from pytorch3d.ops.knn import knn_points"`.
The experiment script runs the same preflight and stops with the missing
piece named.

## Running

All knobs are environment variables (defaults in the script header):

| var | default | meaning |
|---|---|---|
| `TASKS` | `can square_d1 stack_d1 stack_three_d1` | any subset of `tasks.TASK_SPECS` (also `square` = robomimic square ph) |
| `ANGLES` | `0 5 15 30 45 60 90` | camera orbit degrees; 0 = identity sanity check |
| `AGENT` | `equibot` | `equibot` or `dp` |
| `STAGES` | `download data train eval collect` | any subset |
| `NUM_DEMOS` | 200 | demos used for training (can ph has 200; MimicGen 1000) |
| `EPOCHS` | 200 | ≈1k optimizer steps per epoch at 200 demos |
| `N_EPISODES` | 20 | eval episodes per (task, angle) |
| `CAM_SHIFT_MODE` | `azimuth_elev` | `sphere` / `azimuth` / `azimuth_elev` (+ `CAM_ELEV_RATIO`, `CAM_ELEV_CAP_DEG`) |
| `INIT_STATES` | `random` | `random` = fresh layouts from the env sampler (held-out by construction); `demo` = the HDF5 layouts in demo order (the first ones are training layouts) |
| `MAX_STEPS` | 600 | env steps per eval episode |
| `RESOLUTION` | 256 | depth/segmentation render size |
| `DATA_ROOT`, `LOG_ROOT` | `./data`, `./logs` | put them on fast scratch, not NFS home |
| `MUJOCO_GL` | `egl` | `osmesa` if EGL is unavailable |
| `USE_WANDB` | `false` | `true` needs `wandb.entity/project` in `configs/base.yaml` |
| `FORCE` | 0 | 1 = redo stages whose output exists |
| `SMOKE` | 0 | 1 = 3 demos / 2 epochs / 2 episodes / angles `0 30` |

Individual steps, if you prefer to drive them by hand:

```bash
python -m equibot.envs.robosuite_sim.generate_demos --dataset data/demos/stack_d1.hdf5 \
    --task stack_d1 --out_dir data/equibot/stack_d1 --num_demos 200

python -m equibot.policies.train --config-name robosuite_equibot prefix=train_stack_d1_equibot \
    use_wandb=false data.dataset.path=$PWD/data/equibot/stack_d1/pcs \
    env.args.dataset_path=$PWD/data/demos/stack_d1.hdf5 env.args.task_name=stack_d1

python -m equibot.policies.eval --config-name robosuite_equibot mode=eval prefix=eval_stack_d1_equibot_cam30 \
    use_wandb=false training.ckpt=$PWD/logs/train/train_stack_d1_equibot/ckpt00199.pth \
    training.num_eval_episodes=20 env.args.dataset_path=$PWD/data/demos/stack_d1.hdf5 \
    env.args.task_name=stack_d1 env.args.cam_shift_deg=30
```

Use absolute paths — Hydra changes the working directory to the run dir.
`env.vectorize` must stay `false` (the robosuite env is single-process).

## Outputs

```
data/demos/<task>.hdf5                      downloaded demos
data/equibot/<task>/pcs/<task>_ep###_t####.npz   pc (M,3) | eef_pos (1,13) | action (7,)
data/equibot/<task>/meta.json               K, T_cam_world, clip planes, replay success rate, points/frame
data/equibot/<task>/preview_ep###.png       first frame with the object mask in red — LOOK AT ONE
logs/train/train_<task>_<agent>/ckpt*.pth
logs/eval/eval_<task>_<agent>_cam<deg>/eval_results.json   success_rate + per-episode camera (direction, tilt, T_cam_world), videos, info.npz
logs/results_<agent>.md / .csv             the table
```

`eval_results.json` → `episodes[i].camera` records the orbit direction and
elevation tilt actually used, so a run can be cross-checked against the main
repo's per-episode log.

## Sanity checks the script prints — read them

* **Replay success rate** after the data stage: ≈1.0 for robomimic `can`;
  **≈0.7 is normal for MimicGen** (its HDF5s omit the terminal state). A low
  value on `can` means the local robosuite diverges from the collection
  version — stop there.
* **points/frame min** in `meta.json`: should be in the hundreds at 256².
  A `no object pixels visible` warning means a `gt_bodies` name is wrong or
  the object left the view.
* **Fully-occluded frames** (`ValueError: a must be greater than 0` from
  `np.random.choice` during training): a stored frame had an empty cloud —
  the gripper briefly covered every object pixel. Datasets generated before
  2026-08-24 can be fixed in place with
  `python scripts/repair_empty_pcs.py $DATA_ROOT/equibot/*/pcs`;
  the converter and the eval env now carry the last visible cloud forward
  through such frames — or a fixed sentinel blob if nothing was visible yet —
  (`empty_frames_carried` in `meta.json`, `occluded_frames` per episode in
  `eval_results.json`). Many such frames still means something is wrong with
  the segmentation — look at a `preview_ep*.png`.
* **Angle 0 must match the training-view number** (the orbit is an exact
  identity at 0°). If it doesn't, something in the env, not the camera, is
  off.
* Rollouts are seeded (`np.random` and `torch`), but 20 episodes is still
  noisy: differences of ±0.1 between adjacent angles are within noise.

## Compute

At 200 demos (≈30k frames/task) and 256² OSMesa rendering, the data stage is
~20–40 min per task (EGL much faster). Training at `EPOCHS=200` is ≈200k
iterations ≈ 4–6 h per task on one GPU (EquiBot's default for its own tasks
is ~150k). The eval sweep is 7 angles × 20 episodes × ≤600 steps with a
100-step DDPM per 8 executed actions: ≈1–2 h per task. Everything is
embarrassingly parallel across tasks and across angles (see the SLURM
example).

## Files added / changed

* `equibot/envs/robosuite_sim/camera.py` — verbatim ports of the camera
  orbit, extrinsic, depth and intrinsics helpers from the main repo
  (`src_pipeline2/libero_camera.py`, `rerender_libero_object.py`). Keep in
  sync; the experiment's comparability rests on these being identical.
* `equibot/envs/robosuite_sim/tasks.py` — task table (env name, `gt_bodies`,
  download URL), MimicGen registration.
* `equibot/envs/robosuite_sim/sim_utils.py` — env build, demo init states,
  GT-segmentation point cloud, 13-dim state, action reordering.
* `equibot/envs/robosuite_sim/env.py` — `RobosuiteEnv`, EquiBot's env
  interface with the per-reset camera orbit.
* `equibot/envs/robosuite_sim/generate_demos.py` — HDF5 → per-step npz.
* `equibot/policies/configs/robosuite_{dp,equibot}.yaml`.
* `equibot/policies/utils/misc.py` — `env_class: robosuite`.
* `equibot/policies/eval.py` — writes `eval_results.json`; seeds torch;
  fixes a stock-repo crash in the non-vectorised path (`rew_values` was
  never set).
* `setup.py` — `robosuite==1.4.1`, `h5py`.
* `scripts/` — `run_viewpoint_experiment.sh`, `collect_results.py`,
  `install.sh`, `slurm_example.sbatch`.
