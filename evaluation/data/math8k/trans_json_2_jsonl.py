import json

# 读取原始 JSON 文件
with open('/home/tsj/OpenRLHF/evaluation/data/math8k/train_json.json', 'r', encoding='utf-8') as f:
    data = json.load(f)[:100]

# 需要去除的键
keys_to_remove = {"input", "gt_answer", "ground_truth_answer", "target"}

# 处理数据并写入 JSONL 文件
with open('/home/tsj/OpenRLHF/evaluation/data/math8k/debug.jsonl', 'w', encoding='utf-8') as f:
    for item in data:
        filtered_item = {k: v for k, v in item.items() if k not in keys_to_remove}
        f.write(json.dumps(filtered_item, ensure_ascii=False) + '\n')

print("JSONL 文件已成功生成: output.jsonl")