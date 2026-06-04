#!/bin/bash
set -ex

LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(basename "$0" .sh)_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ============================================================
# 03_megatron.sh  安装 Megatron-LM
# ============================================================
# 前置条件: 02_kernels.sh 已跑完
# 用法:     bash 03_megatron.sh
# ============================================================

source /jizhicfs/johnnyslin/anaconda3/etc/profile.d/conda.sh
ENV_PREFIX="/tmp/linguangming/slime-workspace/slime_env"
conda activate "$ENV_PREFIX"

export CUDA_HOME=/tmp/linguangming/slime-workspace/slime_env
export PATH=$CUDA_HOME/bin:$PATH
export PATH=/tmp/linguangming/slime-workspace/protoc/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

BASE_DIR=/tmp/linguangming/slime-workspace
MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"

cd "$BASE_DIR"

# 4.1 克隆 Megatron-LM（含 submodules）
if [ ! -d "$BASE_DIR/Megatron-LM" ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi

# 4.2 预安装编译依赖
pip install -v "setuptools<80.0.0" pybind11 "packaging>=24.2"

# 4.3 切到目标 commit 并安装
cd "$BASE_DIR/Megatron-LM"
git checkout "${MEGATRON_COMMIT}"
pip install -v -e . --no-build-isolation

echo "==== 03_megatron.sh 完成，日志: $LOG_FILE ===="
