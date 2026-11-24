#!/usr/bin/env bash
# Source this file to activate this standalone ManiSkill installation.
export MANISKILL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$MANISKILL_ROOT/.venv/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export MS_ASSET_DIR="$MANISKILL_ROOT/assets"
export XDG_CACHE_HOME="$MANISKILL_ROOT/cache"
export PIP_CACHE_DIR="$MANISKILL_ROOT/cache/pip"
export UV_CACHE_DIR="$MANISKILL_ROOT/cache/uv"
export TORCH_HOME="$MANISKILL_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$MANISKILL_ROOT/cache/torch_extensions"
export HF_HOME="$MANISKILL_ROOT/cache/huggingface"
export CUDA_CACHE_PATH="$MANISKILL_ROOT/cache/cuda"
export SAPIEN_CACHE_DIR="$MANISKILL_ROOT/cache/sapien"
export MPLCONFIGDIR="$MANISKILL_ROOT/cache/matplotlib"
export TMPDIR="$MANISKILL_ROOT/tmp"
if [[ -n "${MANISKILL_GPU_UUID:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$MANISKILL_GPU_UUID"
fi
