#!/bin/bash
set -ex

LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(basename "$0" .sh)_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ============================================================
# 02_kernels.sh  安装 flash-attn / APEX / Transformer Engine /
#                torch_memory_saver 等 kernel 包
# ============================================================
# 前置条件: 01_sglang.sh 已跑完
# 用法:     bash 02_kernels.sh
# ============================================================

source /jizhicfs/johnnyslin/anaconda3/etc/profile.d/conda.sh
ENV_PREFIX="/tmp/linguangming/slime-workspace/slime_env"
conda activate "$ENV_PREFIX"

export CUDA_HOME=/tmp/linguangming/slime-workspace/slime_env
export PATH=$CUDA_HOME/bin:$PATH
export PATH=/tmp/linguangming/slime-workspace/protoc/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

BASE_DIR=/tmp/linguangming/slime-workspace

cd "$BASE_DIR"

# 3.1 编译工具
pip install -v cmake ninja

# 3.2 flash-attn v2（Megatron 支持的最新版本）
MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation

# 3.3 各种 kernel 包
pip install -v git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install -v flash-linear-attention==0.4.1
pip install -v git+https://github.com/QwenLM/FlashQLA.git --no-build-isolation
pip install -v tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

# 3.4 Transformer Engine
MAX_JOBS=4 pip install -v --no-build-isolation "transformer_engine[pytorch]==2.10.0"

# 3.5 APEX（编译较慢，可根据核数调整 --threads 和 MAX_JOBS）
NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

# 3.6 torch_memory_saver
TMS_CUDA_MAJOR=$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')
export TMS_CUDA_MAJOR
pip install -v git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926 \
  --no-cache-dir --force-reinstall --no-build-isolation

# 3.7 Megatron-Bridge + modelopt + sglang-router
pip install -v git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation
pip install -v nvidia-modelopt[torch]>=0.37.0 --no-build-isolation
pip install -v https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

echo "==== 02_kernels.sh 完成，日志: $LOG_FILE ===="
