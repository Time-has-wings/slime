source scripts/models/qwen2.5-0.5B.sh
PYTHONPATH=/apdcephfs_zwfy2_303541817/share_303541817/pkuhetu/guangming/NewWorkspace/slime-workspace/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint models/Qwen2.5-0.5B-Instruct/ \
    --save models/Qwen2.5-0.5B-Instruct_torch_dist/
