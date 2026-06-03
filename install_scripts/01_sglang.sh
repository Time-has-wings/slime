#!/bin/bash
set -ex

LOG_FILE="$(basename "$0" .sh)_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ============================================================
# 01_sglang.sh  克隆 sglang + 安装 + torch cu129 版本修复
# ============================================================
# 前置条件: 00_conda_env.sh 已跑完，conda activate slime 已执行
# 用法:     bash 01_sglang.sh
# ============================================================

# 增加一系列环境变量
export CUDA_HOME=/jizhicfs/johnnyslin/anaconda3/envs/slime
export PATH=$CUDA_HOME/bin:$PATH
export PATH=/apdcephfs_zwfy2_303541817/share_303541817/pkuhetu/guangming/NewWorkspace/slime-workspace/software/protoc/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

BASE_DIR=/apdcephfs_zwfy2_303541817/share_303541817/pkuhetu/guangming/NewWorkspace/slime-workspace
SGLANG_COMMIT="5a15cde858ea09b77116212a39356f2fc51b8584"

cd "$BASE_DIR"

# 2.1 克隆 sglang 并切到目标 commit
if [ ! -d "$BASE_DIR/sglang" ]; then
  git clone https://github.com/sgl-project/sglang.git
fi
cd "$BASE_DIR/sglang"
git checkout "${SGLANG_COMMIT}"

# 2.2 安装 sglang（editable 模式 + 指定 cu129 torch index）
pip install -v -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu129

# 2.3 强制重装 torch → cu129 版本
pip install -v --force-reinstall --no-deps \
  torch==2.11.0 torchvision torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu129

# 2.4 强制重装 sglang-kernel / sgl-deep-gemm → cu129
pip install -v --force-reinstall --no-deps \
  sglang-kernel==0.4.2.post2 sgl-deep-gemm==0.1.0 \
  --index-url https://docs.sglang.ai/whl/cu129/

# 2.5 卸载 cu13 的 nvidia 包（可能有些不存在，|| true 忽略错误）
pip uninstall -y \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc \
  nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cufft \
  nvidia-cufile nvidia-curand nvidia-cusolver \
  nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 \
  nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx \
  nvidia-cutlass-dsl-libs-cu13 \
  || true

# 2.6 重新安装 cu12 版本的 nvidia 包
pip install -v --force-reinstall --no-deps \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 \
  nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 \
  nvidia-cudnn-cu12==9.16.0.29 nvidia-cufft-cu12 \
  nvidia-cufile-cu12 nvidia-curand-cu12 \
  nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
  nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 \
  nvidia-nvtx-cu12 \
  --index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://pypi.org/simple

# 验证 torch 能正常导入
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"

echo "==== 01_sglang.sh 完成 ===="
