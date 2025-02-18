from detect_hacking import brief_info,Solution
import json
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass
import random

def get_dpo_data_vr(results: list[dict], max_tokens: int = 2048):

    dpo_data = []
    
    for item in results:
        responses = item.get("response", [])
        vr_scores = item.get("vr_score", [])
        context_lengths = item.get("token_length", [])
        query = item.get("query", "")
        
        # 筛选正样本、负样本
        positive_samples = [(resp, length) for resp, score, length in zip(responses, vr_scores, context_lengths) if score == 1]
        negative_samples = [(resp, length) for resp, score, length in zip(responses, vr_scores, context_lengths) if score == -1]
        zero_samples = [(resp, length) for resp, score, length in zip(responses, vr_scores, context_lengths) if score == 0]

        # 选择正样本
        if positive_samples:
            positive_samples = sorted(positive_samples, key=lambda x: -x[1])  # 按照长度降序排列
            chosen_response, chosen_length = next(
                ((resp, length) for resp, length in positive_samples if length < max_tokens), 
                positive_samples[0]
            )
            chosen_reward = 1
        elif zero_samples:
            zero_samples = sorted(zero_samples, key=lambda x: -x[1])  # 按照长度降序排列
            chosen_response = zero_samples[0][0]  # 选择最长的 `zero_sample`
            chosen_reward = 0
        else:
            continue  # 如果没有正样本和零样本，跳过

        # 选择负样本
        if negative_samples: #如果有负样本，就直接随便选一个
            reject_response, reject_length = random.choice(negative_samples)
            reject_reward = -1
        elif (not negative_samples) and positive_samples and zero_samples: #如果在有正样本的情况下，没有负样本，就零样本随机选
            reject_response, reject_length = random.choice(zero_samples)
            reject_reward = 0
        elif zero_samples and (not positive_samples) and len(zero_samples) > 1:  #如果没有正样本，就随机选择剩下的 `zero_samples` 作为负样本
            reject_response, reject_length = random.choice(zero_samples[1:])
            reject_reward = 0
        else:
            continue  # 没有负样本可选，跳过

        dpo_data.append({
            "query": query,
            "chosen_response": chosen_response,
            "chosen_reward": chosen_reward,
            "chosen_length": chosen_length,
            "reject_response": reject_response,
            "reject_reward": reject_reward,
            "reject_length": reject_length
        })
    
    return dpo_data


if __name__=="__main__":
    #raise TypeError('undone')
    parser = ArgumentParser()
    parser.add_argument('--results_path', type=str,
                        default="/home/tsj/OpenRLHF/outputs/Qwen2.5-Math-7B/math8k/train_qwen25-math-cot_-1_seed0_t0.7_s0_e-1_VRs.jsonl"
                        )
    args = parser.parse_args()
    results_path = Path(args.results_path)
    output_path = results_path.parent/f'{results_path.stem}_dpo.jsonl'
    
    with open(results_path, 'r') as f:
        results = [json.loads(line) for line in f]
    
    print(brief_info(results))

    outputs = get_dpo_data_vr(results)


    print(brief_info(outputs))

    with open(output_path, 'w') as f:
        f.writelines([json.dumps(output)+'\n' for output in outputs])