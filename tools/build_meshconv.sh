#!/usr/bin/env bash
# Build .tools/bin/meshconv (FBX export for drone3d) against conda-forge assimp:
#   /store/miniforge3/bin/mamba create -y -p .tools/assimp -c conda-forge assimp && tools/build_meshconv.sh
set -euo pipefail
cd "$(dirname "$0")/.."
A=.tools/assimp
mkdir -p .tools/bin
gcc -O2 -o .tools/bin/meshconv tools/meshconv.c -I"$A/include" -L"$A/lib" -lassimp -Wl,-rpath,"$(pwd)/$A/lib"
echo "built .tools/bin/meshconv"
