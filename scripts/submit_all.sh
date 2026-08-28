#!/usr/bin/env bash
# Submit the whole experiment as PARALLEL SLURM jobs with dependencies:
#
#   per task:   [download + data + train]          (4 jobs, run concurrently)
#   per (task, angle): [eval]  after that task's train   (28 jobs)
#   finally:    [collect]      after every eval          (1 job)
#
# Wall time ~ one training + one eval instead of the ~10 h sequential run.
# Every job runs scripts/run_viewpoint_experiment.sh with the stage skips, so
# resubmitting after failures only redoes what's missing or stale.
#
#   bash scripts/submit_all.sh
#   TASKS="square_d1" ANGLES="0 30" bash scripts/submit_all.sh   # subset
#
# DATA_ROOT/LOG_ROOT/EPOCHS/... are inherited by the jobs (sbatch exports the
# environment) — export them before submitting, same as for the plain script.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TASKS=${TASKS:-"can square_d1 stack_d1 stack_three_d1"}
ANGLES=${ANGLES:-"0 5 15 30 45 60 90"}
EVAL_TIME=${EVAL_TIME:-4:00:00}     # per-eval-job time limit (overrides the sbatch header)
SB=scripts/slurm_example.sbatch
mkdir -p logs

eval_deps=""
for t in $TASKS; do
    tid=$(TASKS=$t STAGES="download data train" sbatch --parsable \
          --job-name="eqv-train-$t" "$SB")
    echo "train[$t] -> job $tid"
    for a in $ANGLES; do
        eid=$(TASKS=$t ANGLES=$a STAGES=eval sbatch --parsable \
              --job-name="eqv-eval-$t-$a" --time="$EVAL_TIME" \
              --dependency=afterok:$tid "$SB")
        echo "  eval[$t @ ${a}deg] -> job $eid (after $tid)"
        eval_deps+=":$eid"
    done
done
cid=$(TASKS="$TASKS" ANGLES="$ANGLES" STAGES=collect sbatch --parsable \
      --job-name=eqv-collect --time=0:15:00 --dependency=afterok${eval_deps} "$SB")
echo "collect -> job $cid  (writes \$LOG_ROOT/results_\${AGENT:-equibot}.md)"
echo
echo "watch:  squeue -u \$USER | grep eqv-"
echo "NOTE: afterok means a failed upstream job leaves dependents pending"
echo "      forever — scancel them and resubmit after fixing (skips make it cheap)."
