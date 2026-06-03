base_dir="/apdcephfs_zwfy2_303541817/share_303541817/pkuhetu/guangming/NewWorkspace/slime-workspace/slime"
datasets_dir="${base_dir}/datasets"
logs_dir="${base_dir}/download_scripts/logs"
mkdir -p "${datasets_dir}" "${logs_dir}"

log_file="${logs_dir}/download_gsm8k_$(date +'%Y%m%d_%H%M%S').log"

export HF_DEBUG=1
unset HF_ENDPOINT

hf download \
    --repo-type dataset openai/gsm8k \
    --local-dir "${datasets_dir}/gsm8k" 2>&1 | tee "${log_file}"
