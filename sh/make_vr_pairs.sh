RESULTS_PATH=$YOUR_COLLECTED_DATA_PATH


python -u "evaluation/VR_score.py" \
      --model_name ${MODEL_PATH} \
      --results_path ${RESULTS_PATH} \

VR_RESULTS_PATH="${RESULTS_PATH}_VRs.jsonl"
python -u "evaluation/divide_dataset.py" \
      --results_path ${VR_RESULTS_PATH} \