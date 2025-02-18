set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

MODEL_PATH=$YOUR_MODEL_PATH
DPO_VR_RESULTS_PATH=$YOUR_VR_JSON_PATH
SAVE_PATH=$NEW_MODEL_SAVE_PATH

training_commands="openrlhf.cli.train_dpo \
    --save_path ${SAVE_PATH} \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --train_batch_size 64 \
    --micro_train_batch_size 1 \
    --pretrain ${MODEL_PATH} \
    --bf16 \
    --max_epochs 1 \
    --max_len 2048 \
    --zero_stage 3 \
    --learning_rate 5e-7 \
    --beta 0.1 \
    --dataset ${DPO_VR_RESULTS_PATH} \
    --apply_chat_template \
    --prompt_key query \
    --chosen_key chosen_response \
    --rejected_key reject_response \
    --flash_attn \
    --load_checkpoint \
    --gradient_checkpointing"

if [[ -z "${1}" || "${1}" != "slurm" ]]; then
    deepspeed --module $training_commands
fi