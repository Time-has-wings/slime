base_dir="$(cd "$(dirname "$0")/.." && pwd)"
datasets_dir="${base_dir}/datasets"
logs_dir="${base_dir}/download_scripts/logs"
mkdir -p "${datasets_dir}" "${logs_dir}"

log_file="${logs_dir}/download_dapo_math_17k_$(date +'%Y%m%d_%H%M%S').log"

export HF_DEBUG=1
unset HF_ENDPOINT

hf download \
    --repo-type dataset zhuzilin/dapo-math-17k \
    --local-dir "${datasets_dir}/dapo-math-17k" 2>&1 | tee "${log_file}"