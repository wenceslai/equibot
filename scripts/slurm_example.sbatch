#!/bin/bash
#SBATCH --job-name=equibot-view
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%x-%j.out
# Adapt partition/account to your cluster. One job = one task, all stages.
# For the fully parallel fan-out (per-task train jobs -> per-angle eval jobs
# -> collect), use scripts/submit_all.sh instead of submitting this by hand.
#
#   mkdir -p logs
#   for t in can square_d1 stack_d1 stack_three_d1; do
#       TASKS=$t sbatch scripts/slurm_example.sbatch
#   done
#
# To spread the eval sweep over jobs instead (after training finished):
#   STAGES="eval" TASKS=can ANGLES="0 5 15" sbatch scripts/slurm_example.sbatch
#   STAGES="eval" TASKS=can ANGLES="30 45 60 90" sbatch scripts/slurm_example.sbatch
# and finally  STAGES=collect bash scripts/run_viewpoint_experiment.sh

# Environment: sbatch inherits the submitting shell's environment, so if you
# submitted from an activated venv/conda env, python is already the right one.
# Only fall back to the conda env from install.sh when equibot isn't importable.
if ! python -c "import equibot" 2>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate equibot
fi
export MUJOCO_GL=${MUJOCO_GL:-egl}
bash scripts/run_viewpoint_experiment.sh
