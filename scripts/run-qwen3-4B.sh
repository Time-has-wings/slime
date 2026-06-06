#!/bin/bash

# GRPO training script for Qwen3-4B on slime.
#
# Prerequisites:
#   - models/Qwen3-4B/                  (HF checkpoint, under WORKSPACE_DIR)
#   - models/Qwen3-4B_torch_dist/       (from tools/convert_hf_to_torch_dist.py)
#   - datasets/dapo-math-17k/dapo-math-17k.jsonl
#   - datasets/aime-2024/aime-2024.jsonl  (for evaluation)

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$(dirname "$WORKSPACE_DIR")/slime_env"
RAY_TMP_DIR="/tmp/linguangming/ray_logs"
mkdir -p "$RAY_TMP_DIR"

mkdir -p "$WORKSPACE_DIR/logs"
LOG_FILE="$WORKSPACE_DIR/logs/run-qwen3-4B_$(date +'%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] Logging to: $LOG_FILE"

# clean any leftover ray/sglang from this venv only
pkill -9 -f "$VENV_DIR/.*sglang" 2>/dev/null || true
ray stop --temp-dir "$RAY_TMP_DIR" --force 2>/dev/null || true
pkill -9 -f "$VENV_DIR/.*(ray|python)" 2>/dev/null || true
sleep 2

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    DETECTED_GPUS=0
fi
NUM_GPUS=${NUM_GPUS:-${DETECTED_GPUS}}
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -le 0 ]; then
    NUM_GPUS=8
fi
echo "NUM_GPUS: $NUM_GPUS"

source "${SCRIPT_DIR}/models/qwen3-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint "$WORKSPACE_DIR/models/Qwen3-4B/"
   --ref-load "$WORKSPACE_DIR/models/Qwen3-4B_torch_dist/"
   # --load "$WORKSPACE_DIR/models/Qwen3-4B_slime/"
   # --save "$WORKSPACE_DIR/models/Qwen3-4B_slime/"
   # --save-interval 20
   # --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data "$WORKSPACE_DIR/datasets/dapo-math-17k/dapo-math-17k.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 20
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 128
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data aime "$WORKSPACE_DIR/datasets/aime-2024/aime-2024.jsonl@[0:10]"
   --eval-input-key text
   --eval-label-key label
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-4B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir="$RAY_TMP_DIR"
rm -rf "$WORKSPACE_DIR/ray_logs"
ln -sf "$RAY_TMP_DIR" "$WORKSPACE_DIR/ray_logs"
echo "Ray logs linked at: $WORKSPACE_DIR/ray_logs -> $RAY_TMP_DIR"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "'"$(dirname "$WORKSPACE_DIR")/Megatron-LM"'",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "LD_LIBRARY_PATH": "'"$(dirname "$WORKSPACE_DIR")/slime_env/lib64"':/lib64:/usr/lib64",
        "NCCL_NVLS_ENABLE": "'"${HAS_NVLINK}"'"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
