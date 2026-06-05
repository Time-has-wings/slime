base_dir="$(cd "$(dirname "$0")/.." && pwd)"
models_dir="${base_dir}/models"
logs_dir="${base_dir}/download_scripts/logs"
mkdir -p "${models_dir}" "${logs_dir}"

log_file="${logs_dir}/download_qwen2.5-0.5B-Instruct_$(date +'%Y%m%d_%H%M%S').log"

export HF_DEBUG=1
unset HF_ENDPOINT

hf download \
    --repo-type model Qwen/Qwen2.5-0.5B-Instruct \
    --local-dir "${models_dir}/Qwen2.5-0.5B-Instruct" 2>&1 | tee "${log_file}"
