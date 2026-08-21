#!/usr/bin/env bash
# Conda env for the robosuite viewpoint experiment (Linux, CUDA 11.8 wheels).
# Run line by line if anything fails; see README_VIEWPOINT.md "Install".
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME=${ENV_NAME:-equibot}

conda create -n "$ENV_NAME" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

conda install -y fvcore iopath ffmpeg -c iopath -c fvcore
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# pytorch3d: only knn_points is used. Prebuilt wheel first, source build as fallback.
pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html \
  || pip install "git+https://github.com/facebookresearch/pytorch3d.git"

pip install robosuite==1.4.1 h5py
pip install -e "$ROOT"

# MimicGen envs (Square_D1 / Stack_D1 / StackThree_D1). --no-deps on purpose:
# mimicgen does not support robosuite 1.5 and would move the pin.
if [ ! -d "$ROOT/../mimicgen" ]; then
    git clone https://github.com/NVlabs/mimicgen "$ROOT/../mimicgen"
fi
pip install -e "$ROOT/../mimicgen" --no-deps

python -c "import robosuite, mimicgen, h5py, torch; from pytorch3d.ops.knn import knn_points; print('ok', robosuite.__version__)"
