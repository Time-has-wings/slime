SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
MEGATRON_DIR="$(dirname "$BASE_DIR")/Megatron-LM"

LOGS_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGS_DIR"
LOG_FILE="$LOGS_DIR/convert_qwen3-4B_$(date +'%Y%m%d_%H%M%S').log"

source "$BASE_DIR/scripts/models/qwen3-4B.sh"

PYTHONPATH="$MEGATRON_DIR" python "$BASE_DIR/tools/convert_hf_to_torch_dist.py" \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint "$BASE_DIR/models/Qwen3-4B/" \
    --save "$BASE_DIR/models/Qwen3-4B_torch_dist/" 2>&1 | tee "$LOG_FILE"
