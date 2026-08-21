#!/bin/bash
#SBATCH --job-name=equibot-view
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
# Adapt partition/account to your cluster. One job = one task, all stages.
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

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate equibot
export MUJOCO_GL=${MUJOCO_GL:-egl}
bash scripts/run_viewpoint_experiment.sh
