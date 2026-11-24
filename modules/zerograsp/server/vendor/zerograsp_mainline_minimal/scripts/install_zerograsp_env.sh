#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-zerograsp}"
CUDA_FLAVOR="${CUDA_FLAVOR:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.0}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
PIP_RETRIES="${PIP_RETRIES:-10}"
PIP_TIMEOUT="${PIP_TIMEOUT:-120}"

CONDA_BIN="${CONDA_BIN:-/home/arx/doc/miniconda3/bin/conda}"
CONDA_ROOT="${CONDA_ROOT:-/home/arx/doc/miniconda3}"

CUDA_TOOLKIT_HOME="${CUDA_TOOLKIT_HOME:-/usr/lib/nvidia-cuda-toolkit}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${CONDA_ROOT}/envs/${ENV_NAME}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"

echo "========================================"
echo "ZeroGrasp 环境安装"
echo "========================================"
echo "  ENV_NAME=${ENV_NAME}"
echo "  CONDA_BIN=${CONDA_BIN}"
echo "  CONDA_ROOT=${CONDA_ROOT}"
echo "  CUDA_TOOLKIT_HOME=${CUDA_TOOLKIT_HOME}"
echo "  ROOT_DIR=${ROOT_DIR}"
echo "  ENV_PREFIX=${ENV_PREFIX}"
echo "========================================"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "ERROR: conda not found at $CONDA_BIN"
  echo "Set CONDA_BIN=/path/to/conda if your conda lives elsewhere."
  exit 1
fi

CURRENT_ACTIVE=$("$CONDA_BIN" info --envs 2>/dev/null | awk '/\*/ {print $1}' || echo "")

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo ""
  echo "[1/3] 创建 conda 环境 ${ENV_NAME} ..."

  if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -n "$CURRENT_ACTIVE" ]]; then
    echo "  注意: 已激活环境 '${CURRENT_ACTIVE}', 将在后台创建新环境."
  fi

  "$CONDA_BIN" create -y -n "$ENV_NAME" python=3.11 pip setuptools wheel
  echo "  基础环境创建完成."
else
  echo ""
  echo "[1/3] 环境 ${ENV_NAME} 已存在, 跳过创建."
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: 环境创建失败, python 未找到: $PYTHON_BIN"
  exit 1
fi

echo ""
echo "[2/3] 检查 CUDA 工具链 ..."

export TORCH_CUDA_ARCH_LIST
if [[ -d "$CUDA_TOOLKIT_HOME" ]]; then
  export CUDA_HOME="$CUDA_TOOLKIT_HOME"
  export CUDA_PATH="$CUDA_TOOLKIT_HOME"
  export PATH="$CUDA_TOOLKIT_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_TOOLKIT_HOME/lib64:${LD_LIBRARY_PATH:-}"
  echo "  CUDA_HOME=${CUDA_HOME}"
  echo "  nvcc: $("$CUDA_TOOLKIT_HOME/bin/nvcc" --version 2>/dev/null | grep release || echo "detection failed")"
else
  echo "WARNING: CUDA toolkit not found at $CUDA_TOOLKIT_HOME"
  echo "C++ extensions (dwconv, ocnn, octree_feature_extractor) may fail to compile."
  echo "Set CUDA_TOOLKIT_HOME=/path/to/cuda-X.Y if needed."
fi

echo ""
echo "[3/3] 安装 ZeroGrasp 依赖 ..."

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
PIP_INSTALL=("$PYTHON_BIN" -m pip install --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT")

echo "  → 升级 pip / setuptools / wheel"
"${PIP_INSTALL[@]}" --upgrade pip setuptools wheel

echo "  → 安装环境基础包 (numpy<2 避免与 OpenCV/PyTorch C++ 模块不兼容)"
"${PIP_INSTALL[@]}" "numpy<2" pyyaml websockets==16.0 msgpack==1.1.2 fastapi uvicorn "opencv-python<4.11"

if [[ "$CUDA_FLAVOR" == "cu128" ]]; then
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
  PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu128.html"
else
  echo "ERROR: Unsupported CUDA_FLAVOR=${CUDA_FLAVOR}"
  exit 1
fi

echo "  → 安装 PyTorch ${TORCH_VERSION}+${CUDA_FLAVOR}"
"${PIP_INSTALL[@]}" \
  --index-url "$TORCH_INDEX_URL" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}"

echo "  → 安装 torch-scatter / torch-cluster"
"${PIP_INSTALL[@]}" torch-scatter -f "$PYG_WHEEL_URL"
"${PIP_INSTALL[@]}" torch-cluster -f "$PYG_WHEEL_URL"

echo "  → 安装 dwconv / ocnn (CUDA extensions)"
if [[ -n "${CUDA_HOME:-}" ]]; then
  "${PIP_INSTALL[@]}" --no-build-isolation \
    "dwconv @ git+https://github.com/octree-nn/dwconv.git@ae53057eaf36dab01aa2727fcc93a749fd995af5#egg=dwconv" \
    "ocnn @ git+https://github.com/octree-nn/ocnn-pytorch.git"
else
  echo "  WARNING: 跳过 dwconv/ocnn (无 CUDA toolkit)"
fi

echo "  → 安装 ZeroGrasp 上游依赖"
grep -Ev '^(dwconv|ocnn) @ ' requirements.upstream.txt > /tmp/requirements.zerograsp.filtered.txt
"${PIP_INSTALL[@]}" -r /tmp/requirements.zerograsp.filtered.txt || {
  echo "  WARNING: 部分上游依赖安装失败, 尝试逐个安装 ..."
  "${PIP_INSTALL[@]}" --no-build-isolation -r /tmp/requirements.zerograsp.filtered.txt || true
}

echo "  → 安装 graspnetAPI / transforms3d / autolab_core / cvxopt / grasp_nms"
"${PIP_INSTALL[@]}" graspnetAPI --no-deps
"${PIP_INSTALL[@]}" transforms3d==0.4.2 autolab_core cvxopt grasp_nms

echo "  → 编译 octree_feature_extractor (ofe)"
cd "$ROOT_DIR/third_party/octree_feature_extractor"
rm -rf build dist ofe_pytorch.egg-info
if [[ -f setup.py ]]; then
  if [[ -n "${CUDA_HOME:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}"
    echo "    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  fi
  "$PYTHON_BIN" setup.py build_ext --inplace && {
    OFE_SO=$(find ofe/cuda -name "ofe.cpython-*.so" 2>/dev/null | head -1)
    if [[ -n "$OFE_SO" ]]; then
      cp "$OFE_SO" "$ROOT_DIR/ofe/cuda/"
      echo "    ofe 编译完成, 已复制到 $ROOT_DIR/ofe/cuda/"
    fi
  } || {
    echo "  WARNING: octree_feature_extractor 编译失败"
  }
elif [[ -f pyproject.toml ]]; then
  "${PIP_INSTALL[@]}" -e .
else
  echo "  WARNING: octree_feature_extractor 无 setup.py 或 pyproject.toml, 跳过."
fi

cd "$ROOT_DIR"

echo ""
echo "========================================"
echo "ZeroGrasp 环境安装完成"
echo "  Python: ${PYTHON_BIN}"
echo "  使用方式:"
echo "    ${PYTHON_BIN} run_mainline.py --img_path ... --depth_path ... --mask_path ... --camera_info_path ... --output_dir ..."
echo "  或在总控脚本中:"
echo "    python3 scripts/run_zerograsp_arx_once.py --zg-python ${PYTHON_BIN}"
echo "========================================"
