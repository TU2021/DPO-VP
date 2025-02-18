import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, Qwen2Tokenizer, Qwen2Model
import torch.nn.functional as F
import json
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import re
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

SEP = '\n\n'
SYSTEM = 'Please reason step by step, and put your final answer within \\boxed{{}}.'
PRM_STEP_SEP = '<extra_0>'
# PRM_STEP_SEP = '\n\n'

def brief_info(data):
    info: dict[str, int|list] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            info[k] = brief_info(v)
        return info
    elif isinstance(data, list):
        return (len(data), brief_info(data[0])) if len(data) > 0 else (len(data), )
    else:
        return str(type(data))
    
def preprocess(results_path: str|Path, tokenizer):
    results_path = results_path if isinstance(results_path, Path) else Path(results_path)
    with open(results_path, 'r') as f:
        results = [json.loads(line) for line in f]
    print(brief_info(results))

    for result in tqdm(results, desc="Preprocessing"):
        if 'question' in result:
            result['query'] = result.pop('question')
        if 'code' in result:
            result['response'] = result.pop('code')
        result['system'] = SYSTEM
        result['token_length'] = [len(tokenizer.encode(output)) for output in result['response']] 
        
    print(brief_info(results))
    return results

def score(results: list[dict]):
    for item in results:
        vr_score = []
        for sc, resp in zip(item.get("score", []), item.get("response", [])):
            boxed_matches = re.findall(r"\\boxed{(.*?)}", resp)
            if boxed_matches:
                contains_value = any(match.strip() for match in boxed_matches)  # 确保 `\boxed{}` 内有数值
                vr_score.append(1 if sc and contains_value else 0)
            else:
                vr_score.append(-1)

        item["vr_score"] = vr_score
        # item["len"] = [len(resp) for resp in item.get("response", [])]
    
    return results
            
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="/home/tsj/OpenRLHF/hf_models/Qwen2.5-Math-7B"
                        )
    parser.add_argument('--results_path', type=str,
                        default="/home/tsj/OpenRLHF/outputs/Qwen2.5-Math-7B/math8k/train_qwen25-math-cot_-1_seed0_t0.7_s0_e-1.jsonl"
                        )
    args = parser.parse_args()
    results_path = Path(args.results_path)
    output_path = results_path.parent/f'{results_path.stem}_VRs.jsonl'

    tokenizer = AutoTokenizer.from_pretrained(
                args.model_name, trust_remote_code=True
            )


    results = preprocess(results_path, tokenizer)
    
    results = score(results)

    with open(output_path, 'w') as f:
        f.writelines([json.dumps(result)+'\n' for result in results])