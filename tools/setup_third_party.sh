#!/usr/bin/env bash
# Usage: tools/setup_third_party.sh   (after `uv sync --all-extras`)
# Source checkouts the fast profile imports but PyPI does not carry, installed editable into the project venv
# at the commits tested: RoMa v2 (MIT) merges an edit's passes into one model; MoGe (MIT) is the optional fill
# prior (dense.mono_model: Ruicheng/moge-3-vitl). MoGe's demo dependencies (gradio, ...) are left out.
set -euo pipefail
cd "$(dirname "$0")/.."
unset OMP_PROC_BIND OMP_PLACES
clone() { [ -d "third_party/$1" ] || git clone -q "$2" "third_party/$1"; git -C "third_party/$1" checkout -q "$3"; }
clone RoMaV2 https://github.com/Parskatt/RoMaV2.git 95c9968
clone MoGe https://github.com/microsoft/MoGe.git 74fbce0
uv pip install --no-deps -e third_party/RoMaV2 einops rich  # RoMa runs without its fused-local-corr CUDA extension
uv pip install --no-deps -e third_party/MoGe \
  "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" \
  "pipeline @ git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4" \
  "flex-gemm @ git+https://github.com/JeffreyXiang/FlexGEMM.git@b2fadb29d41846c7981ade6801ffc689fae119cf"
uv run python -c "import romav2, moge; print('ok')"
