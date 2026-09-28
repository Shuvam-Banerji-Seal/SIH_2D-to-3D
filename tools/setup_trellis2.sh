#!/usr/bin/env bash
# Usage: git clone --recursive https://github.com/microsoft/TRELLIS.2 third_party/TRELLIS.2 && tools/setup_trellis2.sh
# TRELLIS.2 in its own venv (the drone3d venv is Python 3.14 / torch 2.14): torch 2.7.1+cu128, xformers, extensions for sm_80.
set -euo pipefail
cd "${TRELLIS2_DIR:-$(dirname "$0")/../third_party/TRELLIS.2}"
export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=8
unset OMP_PROC_BIND OMP_PLACES
[ -d .venv ] || uv venv --python 3.11 .venv
. .venv/bin/activate
uv pip install torch==2.7.1 torchvision==0.22.1 xformers==0.0.31 --index-url https://download.pytorch.org/whl/cu128
uv pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers pandas lpips zstandard kornia timm pillow setuptools wheel
uv pip install "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
mkdir -p ext
[ -d ext/nvdiffrast ] || git clone -q -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git ext/nvdiffrast
[ -d ext/nvdiffrec ] || git clone -q -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git ext/nvdiffrec
[ -d ext/CuMesh ] || git clone -q --recursive https://github.com/JeffreyXiang/CuMesh.git ext/CuMesh
[ -d ext/FlexGEMM ] || git clone -q --recursive https://github.com/JeffreyXiang/FlexGEMM.git ext/FlexGEMM
for pkg in ext/nvdiffrast ext/nvdiffrec ext/CuMesh ext/FlexGEMM o-voxel; do
  echo "=== building $pkg"; uv pip install --no-build-isolation "./$pkg"
done
python -c "import torch, xformers, nvdiffrast, cumesh, flex_gemm, o_voxel; print('ok', torch.__version__, torch.cuda.is_available())"
