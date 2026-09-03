#!/usr/bin/env bash
# prefill-moe lane runtime env (host-native, 1 GPU, no NCCL/CXI needed).
set -euo pipefail
export PROJECTDIR=/projects/s6p
export VENV="$PROJECTDIR/$USER/envs/sglang-0.5.14"
source "$HOME/projects/isambard-catalog/env.sh"
export PYTHONPATH="$SCRATCH/src/sglang-prefill-moe-0902/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
cd "$SCRATCH/grace-1m/lanes/prefill-moe"
exec "$@"
