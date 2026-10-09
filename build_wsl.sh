#!/usr/bin/env bash
# Builds inside WSL using the local CUDA 11.7 toolchain (micromamba env at ~/tools/cuda117).
# On a machine with a normal CUDA install (lab PC, Google Colab) just run:  make ARCH=sm_XX
set -e
P="$HOME/tools/cuda117"
export PATH="$P/bin:$PATH"
cd "$(dirname "$0")"
make CCBIN=x86_64-conda-linux-gnu-g++ EXTRA_LDFLAGS="-Xlinker -rpath,$P/lib" "$@"
