set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

PROMPT_TYPE="qwen25-math-cot"

MODEL_NAME=$YOUR_MODEL_NAME

MODEL_PATH=$YOUR_MODEL_PATH

OUTPUT_DIR=$OUTPUT_DIR

MAX_K=8
DATA_NAME="math8k"
SPLIT="train"

NUM_TEST_SAMPLE=-1
SEED=0
TEMPERATURE=0.7
START=0
END=-1
MAX_TOKENS=2048

TOKENIZERS_PARALLELISM=false \
python -u "evaluation/math_eval.py" \
    --model_name_or_path ${MODEL_PATH} \
    --max_tokens_per_call ${MAX_TOKENS} \
    --data_name ${DATA_NAME} \
    --data_dir "evaluation/data" \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed ${SEED} \
    --temperature ${TEMPERATURE} \
    --top_p 1 \
    --n_sampling ${MAX_K} \
    --start ${START} \
    --end ${END} \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --gpu_memory_utilization 0.9