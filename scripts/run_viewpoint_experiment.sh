#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# EquiBot viewpoint-robustness experiment, end to end:
#   download demos -> convert to point-cloud dataset -> train one policy per
#   task on the TRAINING camera -> evaluate it with the camera orbited by
#   0/5/15/30/45/60/90 deg -> results table.
#
# Everything is an env var with a default; nothing needs editing.
#
#   bash scripts/run_viewpoint_experiment.sh                      # everything
#   SMOKE=1 bash scripts/run_viewpoint_experiment.sh              # 10-min sanity run
#   STAGES="eval collect" TASKS=can ANGLES="0 30" bash scripts/run_viewpoint_experiment.sh
#   AGENT=dp bash scripts/run_viewpoint_experiment.sh             # point-cloud DP baseline
#
# Stages are idempotent: a stage whose output already exists is skipped
# (FORCE=1 re-runs it). Safe to resubmit after a time-out.
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ─── Knobs ────────────────────────────────────────────────────────────────
TASKS=${TASKS:-"can square_d1 stack_d1 stack_three_d1"}   # keys of tasks.TASK_SPECS
ANGLES=${ANGLES:-"0 5 15 30 45"}                           # camera orbit, degrees
AGENT=${AGENT:-equibot}                                    # equibot | dp
STAGES=${STAGES:-"download data train eval collect"}
NUM_DEMOS=${NUM_DEMOS:-}             # empty = per-task default from tasks.py
                                     # (can/square 200, MimicGen tasks 800);
                                     # set to force one count for all tasks
EPOCHS=${EPOCHS:-}                   # empty = computed per task from the dataset
                                     # size so training runs ~STEPS_TARGET
                                     # optimizer steps; set to force a count
STEPS_TARGET=${STEPS_TARGET:-1500000}  # ~2x the paper's robomimic budget
                                     # (2000 epochs on 100 demos ~ 750k steps);
                                     # at can's 200 demos this reproduces
                                     # ~2000 epochs
BATCH_SIZE=32                        # must match training.batch_size (base.yaml)
N_EPISODES=${N_EPISODES:-20}         # eval episodes per (task, angle)
SEED=${SEED:-0}
RESOLUTION=${RESOLUTION:-256}        # render size for depth/segmentation
MAX_STEPS=${MAX_STEPS:-600}          # env steps per eval episode
INIT_STATES=${INIT_STATES:-random}   # random (fresh layouts from the env sampler) | demo (HDF5 layouts, in demo order)
CAM_SHIFT_MODE=${CAM_SHIFT_MODE:-azimuth_elev}   # sphere | azimuth | azimuth_elev
CAM_ELEV_RATIO=${CAM_ELEV_RATIO:-0.3}
CAM_ELEV_CAP_DEG=${CAM_ELEV_CAP_DEG:-15}
DATA_ROOT=${DATA_ROOT:-$ROOT/data}
LOG_ROOT=${LOG_ROOT:-$ROOT/logs}
USE_WANDB=${USE_WANDB:-false}        # true needs wandb.entity/project in configs/base.yaml
FORCE=${FORCE:-0}
SMOKE=${SMOKE:-0}
export MUJOCO_GL=${MUJOCO_GL:-egl}   # egl (GPU) | osmesa (CPU, slow but always works)
[ "$MUJOCO_GL" = osmesa ] && export PYOPENGL_PLATFORM=osmesa

if [ "$SMOKE" = 1 ]; then
    NUM_DEMOS=3; EPOCHS=2; N_EPISODES=2; MAX_STEPS=40
    export NUM_DEMOS EPOCHS
    [ "${ANGLES}" = "0 5 15 30 45" ] && ANGLES="0 30"
    # Own roots: a smoke run must never leave 3-demo datasets / 2-epoch
    # checkpoints / 2-episode eval results where the real run would pick
    # them up via the skip-if-exists logic. (Demos are still shared.)
    DEMO_DIR="$DATA_ROOT/demos"
    DATA_ROOT="$DATA_ROOT/smoke"; LOG_ROOT="$LOG_ROOT/smoke"
    mkdir -p "$DATA_ROOT/demos"
    for f in "$DEMO_DIR"/*.hdf5; do [ -e "$f" ] && ln -sf "$f" "$DATA_ROOT/demos/"; done
    echo "[smoke] NUM_DEMOS=$NUM_DEMOS EPOCHS=$EPOCHS N_EPISODES=$N_EPISODES ANGLES='$ANGLES' MAX_STEPS=$MAX_STEPS"
    echo "[smoke] outputs isolated under $DATA_ROOT and $LOG_ROOT"
fi

echo "════ EquiBot viewpoint experiment ════"
echo "AGENT=$AGENT TASKS='$TASKS' ANGLES='$ANGLES' STAGES='$STAGES'"
echo "NUM_DEMOS=${NUM_DEMOS:-per-task} EPOCHS=${EPOCHS:-auto (~$STEPS_TARGET steps)} N_EPISODES=$N_EPISODES SEED=$SEED"
echo "CAM_SHIFT_MODE=$CAM_SHIFT_MODE (elev ratio $CAM_ELEV_RATIO, cap $CAM_ELEV_CAP_DEG) INIT_STATES=$INIT_STATES"
echo "DATA_ROOT=$DATA_ROOT LOG_ROOT=$LOG_ROOT MUJOCO_GL=$MUJOCO_GL"
mkdir -p "$DATA_ROOT/demos" "$LOG_ROOT"

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }
hdf5_of()   { echo "$DATA_ROOT/demos/$1.hdf5"; }
pcs_of()    { echo "$DATA_ROOT/equibot/$1/pcs"; }
train_prefix() { echo "train_${1}_${AGENT}"; }
demos_of()  { if [ -n "$NUM_DEMOS" ]; then echo "$NUM_DEMOS"; else python -m equibot.envs.robosuite_sim.tasks train_demos "$1"; fi }
# Epochs from the dataset actually on disk, so the optimizer-step budget is
# held constant across tasks with very different demo counts/lengths.
epochs_of() {
    if [ -n "$EPOCHS" ]; then echo "$EPOCHS"; return; fi
    local n
    n=$(find "$(pcs_of "$1")" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" -gt 0 ] || { echo "ERROR: no dataset at $(pcs_of "$1") — run the data stage first (epochs are computed from it)" >&2; return 1; }
    python -c "import math,sys; n,b,t=map(int,sys.argv[1:]); print(max(1, math.ceil(t/max(1, n//b))))" "$n" "$BATCH_SIZE" "$STEPS_TARGET"
}
ckpt_of()   { local ep; ep=$(epochs_of "$1") || return 1; printf "%s/train/%s/ckpt%05d.pth" "$LOG_ROOT" "$(train_prefix "$1")" $((ep - 1)); }

# ─── Preflight: imports ───────────────────────────────────────────────────
python - "$TASKS" <<'PY'
import importlib, sys
missing = []
for m in ["torch", "robosuite", "h5py", "hydra", "diffusers", "equibot"]:
    try: importlib.import_module(m)
    except Exception as e: missing.append(f"{m}: {e}")
try: from pytorch3d.ops.knn import knn_points  # noqa
except Exception as e: missing.append(f"pytorch3d: {e}")
from equibot.envs.robosuite_sim.tasks import TASK_SPECS, MIMICGEN_ENVS
tasks = sys.argv[1].split()
bad = [t for t in tasks if t not in TASK_SPECS]
if bad: missing.append(f"unknown TASKS {bad}; known: {sorted(TASK_SPECS)}")
if any(TASK_SPECS[t]["env_name"] in MIMICGEN_ENVS for t in tasks if t in TASK_SPECS):
    try: import mimicgen  # noqa
    except Exception as e: missing.append(f"mimicgen (needed for MimicGen tasks): {e}")
import robosuite
if not robosuite.__version__.startswith("1.4"):
    missing.append(f"robosuite {robosuite.__version__}: need 1.4.x (mimicgen + the demo XMLs)")
if missing:
    print("PREFLIGHT FAILED:\n  " + "\n  ".join(missing)); sys.exit(1)
print("preflight ok: torch, robosuite", robosuite.__version__)
PY

# ─── Stage: download ──────────────────────────────────────────────────────
if has_stage download; then
    for task in $TASKS; do
        f=$(hdf5_of "$task")
        if [ -f "$f" ] && [ "$FORCE" != 1 ]; then echo "[download] $f exists"; continue; fi
        url=$(python -m equibot.envs.robosuite_sim.tasks url "$task")
        echo "[download] $task <- $url"
        wget -c -O "$f.part" "$url" && mv "$f.part" "$f"
    done
fi

# A dataset counts as done only if it was built with the CURRENT NUM_DEMOS —
# a leftover from a smaller run (e.g. an old smoke run) must be rebuilt.
data_ok() {  # $1 = task
    python -c "import json,sys; m=json.load(open(sys.argv[1])); sys.exit(0 if m['num_demos']==int(sys.argv[2]) else 1)" \
        "$DATA_ROOT/equibot/$1/meta.json" "$(demos_of "$1")" 2>/dev/null
}
# A checkpoint counts as done only if the dataset it trained on still matches:
# same NUM_DEMOS and the data's meta.json is OLDER than the checkpoint.
train_ok() {  # $1 = task
    data_ok "$1" || return 1
    python -c "import os,sys; c,m=sys.argv[1:3]; sys.exit(0 if os.path.exists(c) and os.path.getmtime(m)<os.path.getmtime(c) else 1)" \
        "$(ckpt_of "$1")" "$DATA_ROOT/equibot/$1/meta.json" 2>/dev/null
}
# An eval counts as done only if it scored >= the current N_EPISODES and ran
# AFTER the checkpoint it evaluates was written.
eval_ok() {  # $1 = eval_results.json  $2 = ckpt
    python -c "import json,os,sys; r,c=sys.argv[1:3]; d=json.load(open(r)); sys.exit(0 if len(d.get('rew_values',[]))>=int(sys.argv[3]) and os.path.getmtime(c)<os.path.getmtime(r) else 1)" \
        "$1" "$2" "$N_EPISODES" 2>/dev/null
}

# ─── Stage: data (replay demos -> point-cloud npz) ────────────────────────
if has_stage data; then
    for task in $TASKS; do
        out="$DATA_ROOT/equibot/$task"
        if data_ok "$task" && [ "$FORCE" != 1 ]; then echo "[data] $out exists (num_demos=$(demos_of "$task"))"; continue; fi
        if [ -d "$out" ]; then
            echo "[data] $out is stale (different num_demos) — rebuilding"
            rm -rf "$out"
        fi
        python -m equibot.envs.robosuite_sim.generate_demos \
            --dataset "$(hdf5_of "$task")" --task "$task" --out_dir "$out" \
            --num_demos "$(demos_of "$task")" --resolution "$RESOLUTION" --seed "$SEED"
    done
fi

# ─── Stage: train ─────────────────────────────────────────────────────────
if has_stage train; then
    for task in $TASKS; do
        ckpt=$(ckpt_of "$task")
        if train_ok "$task" && [ "$FORCE" != 1 ]; then echo "[train] $ckpt exists and matches the dataset"; continue; fi
        prefix=$(train_prefix "$task")
        np=$(python -m equibot.envs.robosuite_sim.tasks num_points "$task")
        ep=$(epochs_of "$task") || exit 1
        echo "[train] $task: $(demos_of "$task") demos, $ep epochs (~$STEPS_TARGET steps), $np points"
        python -m equibot.policies.train --config-name "robosuite_${AGENT}" \
            mode=train prefix="$prefix" hydra.run.dir="$LOG_ROOT/train/$prefix" \
            use_wandb="$USE_WANDB" seed="$SEED" \
            training.num_epochs="$ep" data.dataset.num_points="$np" \
            data.dataset.path="$(pcs_of "$task")" \
            env.args.dataset_path="$(hdf5_of "$task")" env.args.task_name="$task" \
            env.args.resolution="$RESOLUTION"
        [ -f "$ckpt" ] || { echo "[train] expected $ckpt after training" >&2; exit 1; }
    done
fi

# ─── Stage: eval sweep ────────────────────────────────────────────────────
if has_stage eval; then
    for task in $TASKS; do
        ckpt=$(ckpt_of "$task")
        [ -f "$ckpt" ] || { echo "[eval] missing checkpoint $ckpt (run the train stage)" >&2; exit 1; }
        np=$(python -m equibot.envs.robosuite_sim.tasks num_points "$task")
        for deg in $ANGLES; do
            prefix="eval_${task}_${AGENT}_cam${deg}"
            dir="$LOG_ROOT/eval/$prefix"
            if eval_ok "$dir/eval_results.json" "$ckpt" && [ "$FORCE" != 1 ]; then echo "[eval] $prefix done"; continue; fi
            python -m equibot.policies.eval --config-name "robosuite_${AGENT}" \
                mode=eval prefix="$prefix" hydra.run.dir="$dir" \
                use_wandb="$USE_WANDB" seed="$SEED" \
                training.ckpt="$ckpt" training.num_eval_episodes="$N_EPISODES" \
                data.dataset.num_points="$np" \
                data.dataset.path="$(pcs_of "$task")" \
                env.args.dataset_path="$(hdf5_of "$task")" env.args.task_name="$task" \
                env.args.resolution="$RESOLUTION" env.args.max_episode_length="$MAX_STEPS" \
                env.args.init_states="$INIT_STATES" \
                env.args.cam_shift_deg="$deg" env.args.cam_shift_mode="$CAM_SHIFT_MODE" \
                env.args.cam_elev_ratio="$CAM_ELEV_RATIO" env.args.cam_elev_cap_deg="$CAM_ELEV_CAP_DEG"
        done
    done
fi

# ─── Stage: collect ───────────────────────────────────────────────────────
if has_stage collect; then
    python scripts/collect_results.py --log_root "$LOG_ROOT" --agent "$AGENT" \
        --tasks $TASKS --angles $ANGLES --out "$LOG_ROOT/results_${AGENT}"
fi
echo "════ done ════"
