#!/usr/bin/env bash
# Source explicitly; this does not change any legacy ARX environment file.
export VIRTUAL_ENV=/home/zhaoyechong/projects/ARX-ManiSkill/modules/curobo_adapter/.venv
export CUDA_HOME=/home/zhaoyechong/projects/ARX-ManiSkill/environments/runtime_bundle/cuda
export CUDA_PATH="$CUDA_HOME"
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUROBO_KERNEL_BACKEND=cuda_core
export XDG_CACHE_HOME=/home/zhaoyechong/projects/ARX-ManiSkill/modules/curobo_adapter/cache
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export TMPDIR=/home/zhaoyechong/projects/ARX-ManiSkill/modules/curobo_adapter/tmp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=2
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST=8.9
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export GRASP_TRAJOPT_POLICY=interactive
unset PYTHONPATH
mkdir -p "$XDG_CACHE_HOME" "$WARP_CACHE_PATH" "$CUDA_CACHE_PATH" "$TORCH_EXTENSIONS_DIR" "$TMPDIR"
