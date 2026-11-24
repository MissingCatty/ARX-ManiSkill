#!/usr/bin/env bash
# Source this file. Dependencies may live elsewhere; no helper under /tmp is sourced.
ARX_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export ARX_PROJECT_ROOT="$(cd -- "${ARX_REPO_ROOT}/../.." && pwd -P)"
export ARX_INFERENCE_ENV="${ARX_INFERENCE_ENV:-${ARX_PROJECT_ROOT}/environments/runtime_bundle/env}"
export ARX_NEURAL_PYTHON="${ARX_NEURAL_PYTHON:-${ARX_INFERENCE_ENV}/bin/python}"
export CUROBO_PYTHON="${CUROBO_PYTHON:-${ARX_PROJECT_ROOT}/environments/curobo/bin/python}"
export CUDA_HOME="${CUDA_HOME:-${ARX_PROJECT_ROOT}/environments/runtime_bundle/cuda}"
export CUDA_PATH="$CUDA_HOME"
export PATH="$(dirname -- "$ARX_NEURAL_PYTHON"):$CUDA_HOME/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ARX_GPU_ASSIGNMENT="$(python3 "$ARX_REPO_ROOT/scripts/curobo_arx/gpu_policy.py" shell "${LILO_GPU:-2}" "${ARX_NEURAL_GPU:-3}" "${ARX_PLANNER_GPU:-3}" "${ARX_SIM_GPU:-3}")" || { return 1 2>/dev/null || exit 1; }
read -r LILO_GPU ARX_NEURAL_GPU ARX_PLANNER_GPU ARX_SIM_GPU <<< "$ARX_GPU_ASSIGNMENT"
export LILO_GPU ARX_NEURAL_GPU ARX_PLANNER_GPU ARX_SIM_GPU
export CUROBO_KERNEL_BACKEND=cuda_core
export CUROBO_SWEPT_MESH_BOUNDS="${CUROBO_SWEPT_MESH_BOUNDS:-1}"
export GRASP_TRAJOPT_POLICY="${GRASP_TRAJOPT_POLICY:-interactive}"
export ARX_RESEARCH_AUDIT="${ARX_RESEARCH_AUDIT:-0}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ARX_PROJECT_ROOT}/tmp/curobo/arx-curobo/cache}"
export TORCH_EXTENSIONS_DIR="${ARX_PROJECT_ROOT}/tmp/curobo/torch_extensions_sm89"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS="${MAX_JOBS:-2}"
export CC="${CC:-/usr/bin/gcc}" CXX="${CXX:-/usr/bin/g++}" CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++}"
export OCNN_DISABLE_TRITON=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export COOLAS_PYTHON="$ARX_NEURAL_PYTHON" ZEROGRASP_PYTHON="$ARX_NEURAL_PYTHON"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
if [[ -x "$ARX_NEURAL_PYTHON" ]]; then
  ARX_NVIDIA_INCLUDES="$("$ARX_NEURAL_PYTHON" -c 'import sysconfig; from pathlib import Path; print(":".join(str(p) for p in (Path(sysconfig.get_path("purelib"))/"nvidia").glob("*/include")))')"
  export CPLUS_INCLUDE_PATH="${ARX_NVIDIA_INCLUDES}${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
fi
