#!/bin/bash
set -ex

LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(basename "$0" .sh)_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ============================================================
# 04_slime.sh  安装 slime + int4_qat kernel + 打补丁
# ============================================================
# 前置条件: 03_megatron.sh 已跑完
# 用法:     bash 04_slime.sh
# ============================================================

source /jizhicfs/johnnyslin/anaconda3/etc/profile.d/conda.sh
ENV_PREFIX="/tmp/linguangming/slime-workspace/slime_env"
conda activate "$ENV_PREFIX"

export CUDA_HOME=/tmp/linguangming/slime-workspace/slime_env
export PATH=$CUDA_HOME/bin:$PATH
export PATH=/tmp/linguangming/slime-workspace/protoc/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

BASE_DIR=/tmp/linguangming/slime-workspace
PATCH_VERSION="latest"

export SLIME_DIR="$BASE_DIR/slime"

# 5.1 安装 slime 的 Python 依赖
cd "$SLIME_DIR"
pip install -v -r requirements.txt
pip install -v -e . --no-deps

# 5.2 安装 int4_qat kernel
cd "$SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat"
pip install -v . --no-build-isolation

# 5.3 修复和收尾
pip install -v nvidia-cudnn-cu12==9.16.0.29
pip install -v "numpy<2"
pip install -v "kernels<0.15.0"

# 5.4 打 sglang 补丁
cd "$BASE_DIR/sglang"
if git apply --check "$SLIME_DIR/docker/patch/${PATCH_VERSION}/sglang.patch" 2>/dev/null; then
  git update-index --refresh || true
  git apply "$SLIME_DIR/docker/patch/${PATCH_VERSION}/sglang.patch" --3way
  if grep -R -n '^<<<<<<< ' .; then
    echo "sglang patch failed to apply cleanly. Please resolve conflicts." >&2
    exit 1
  fi
else
  echo "sglang patch already applied or not applicable, skipping"
fi

# 5.5 打 megatron 补丁
cd "$BASE_DIR/Megatron-LM"
if git apply --check "$SLIME_DIR/docker/patch/${PATCH_VERSION}/megatron.patch" 2>/dev/null; then
  git update-index --refresh || true
  git apply "$SLIME_DIR/docker/patch/${PATCH_VERSION}/megatron.patch" --3way
  if grep -R -n '^<<<<<<< ' .; then
    echo "megatron patch failed to apply cleanly. Please resolve conflicts." >&2
    exit 1
  fi
else
  echo "megatron patch already applied or not applicable, skipping"
fi

echo "==== 04_slime.sh 完成，日志: $LOG_FILE ===="
