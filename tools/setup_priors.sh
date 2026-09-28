#!/usr/bin/env bash
# Usage: tools/setup_priors.sh   -> third_party/priors/{da3,moge}/.venv, for experiments/prior_bench.py
# The depth priors compared against the fast profile's Depth Anything V2 (the drone3d venv is Python 3.14 /
# torch 2.14, too new for them). Two environments, because Depth Anything 3 pins numpy<2 and MoGe-3 needs
# numpy>=2 and Triton >= 3.4 (torch 2.8): da3/ holds DA3 and DA-V2, moge/ holds MoGe.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)/third_party/priors"
unset OMP_PROC_BIND OMP_PLACES
export UV_HTTP_TIMEOUT=600
unset HF_HUB_ENABLE_HF_TRANSFER  # hf_transfer is not installed in these environments
mkdir -p "$root/da3" "$root/moge"

cd "$root/da3"
[ -d .venv ] || uv venv --python 3.11 .venv
VIRTUAL_ENV="$PWD/.venv" uv pip install torch==2.7.1 torchvision==0.22.1 xformers==0.0.31 --index-url https://download.pytorch.org/whl/cu128
VIRTUAL_ENV="$PWD/.venv" uv pip install "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git" "transformers<5" "huggingface_hub<1.0" "numpy<2" addict pillow scipy
.venv/bin/python -c "import torch, depth_anything_3, transformers; print('da3 ok', torch.__version__, torch.cuda.is_available())"

cd "$root/moge"
[ -d .venv ] || uv venv --python 3.11 .venv
VIRTUAL_ENV="$PWD/.venv" uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
VIRTUAL_ENV="$PWD/.venv" uv pip install "git+https://github.com/microsoft/MoGe.git" pillow scipy
.venv/bin/python -c "import torch, triton, moge; print('moge ok', torch.__version__, triton.__version__, torch.cuda.is_available())"
