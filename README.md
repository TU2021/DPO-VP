<div align="center">

# Improving Math reasoning through Self-improvement and Direct Preference Optimization with Verifiable Pairs



</div>

**TL;DR:** 我们仅通过 Verifiable Reward 筛选 + DPO 的自提升训练范式以提高 LLM 的数学推理能力。最后在 Qwen-2.5-Math-7B 模型上展现出与当前基于 RL 的方法相近的数学推理能力。整个框架无需模型并行，因此可以只在1张A800上复现。

## 🎉 News:

- [2025/02/18] 

## 📖 Introduction

2025年以来，通过基于可验证奖励 (Verifiable Reward, VR) 的强化学习微调 (Reinforcement learning Fine-Tuning, ReFT) 被证明可以使能力足够强的基础模型涌现出卓越的数学推理能力。如[DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1), [SimpleRL-Zero](https://github.com/hkust-nlp/simpleRL-reason) 和 [PURE](https://github.com/CJReinforce/PURE) 等。这些工作基于 PPO 或 GRPO 等RL算法，对计算资源的需求仍然较为苛刻，少则需要8块A100，多则需要几十块。我们的目的是在更加有限的计算资源下，尝试从同样的基础模型开始，不借助外部蒸馏提升 LLM 的数学推理能力。

本质上，以直接偏好优化 (Direct Preference Optimization, [DPO](https://arxiv.org/abs/2305.18290)) 为代表的工作与 RL 的优化目标相同，目的在于优化 LLM 的生成分布，使其接近给定数据集中的接受答案，远离拒绝答案。在 [DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1) 中，我们观察到 Verifiable Reward 是一个 0/1 的离散分布，代表模型的输出是否符合预定义的正确性标准。 这与 DPO 的设计初衷类似：符合预定义的正确样本归类为正类，不符合的归类为负类，进而构造 Verifiable Pairs 优化模型。

由于 DPO 完全是离线优化算法，无法满足模型自提升的需求。我们参考 [Iterative DPO](https://arxiv.org/abs/2305.18290) 的思想，基于 Qwen2.5-Math-7B 基础模型，在 8K 的困难 MATH 数据集 (与 [SimpleRL-Zero](https://github.com/hkust-nlp/simpleRL-reason) 和 [PURE](https://github.com/CJReinforce/PURE) 相同) 的 prompt 上进行采样-筛选-构造偏好数据集自提升的流程，具体地：

- 每轮采样中对每个 prompt 采 8 个样本，并从8个样本中筛选出一对样本构建当前的偏好数据集：
  - 给每个回答打分：参考 VR 打分方式，若回答正确且符合格式，则打 1 分；若回答错误但符合格式，打 0 分；若回答不符合格式，则不论对错，打 -1 分；
  - 正样本：若回答候选中有得分 1 的样本，则选其中 token 长度最长的样本为正样本；若没有得分 1 的样本，但有得分 0 的样本，则选取长度最长的 0 分样本为正样本；否则跳过该问题；
  - 负样本：若回答候选有得分为 -1 的样本，则随机选取一个作为负样本；若没有得分 -1 的样本，但有得分 0 的样本，则随机选取一个为负样本；否则跳过该问题；
- 构建完数据集后进行 1 个 epoch 的 DPO 迭代，训练出新的模型，再基于新的模型重复以上过程。

经过调试，我们一共进行 6 轮 DPO 迭代，并借鉴逐步升温采样的思想提高数据的多样性：其中，前 3 轮，我们设置温度系数 Tempurature = 0.7 进行采样；4 - 5 轮，设置Tempurature = 1.0；最后一轮，设置 Tempurature = 1.2。 

我们在 4 张 A800 上进行采样与训练；DPO 只做了数据并行操作，理论上可以在 1 张 80G 显卡甚至更低的条件下进行训练。在我们的 4 卡实验中，每轮采样约 2 -2.5 小时，每轮训练约 1 小时。 因此，最终模型大约需要 80 小时 A800 机时，在单卡情况下需要约 3 天可复现。

最终的结果在五个数学推理的 benchmark 中取得 48.2 的均分，与 Qwen2.5-Math-7B-Instruct 和其他在同等数据条件下使用 RL 的方法性能相当。


***All results are in pass@1 accuracy***

|*pass@1 acc*| MATH500 | Minerva Math | Olymapaidbench   | AMC23 | AIME24 | Avg.   |
| -------------------------- | --------- | -------- | -------- | ------------ | ------------- | -------- |
| [Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B) *         | 64.8      | 15.4     | 25.6     | 37.5         | 16.7          | 32.0     |
| [Qwen2.5-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct) *| 83.2      | 33.5     | 38.4     | 62.5         | 20.0          | 47.5     |
| [rStar-Math-7B](https://arxiv.org/abs/2501.04519) ^       | 78.4      | -     | 47.1     | 47.5         | 26.7          | -     |
| [Eurus-2-7B-PRIME](https://huggingface.co/PRIME-RL/Eurus-2-7B-PRIME) *        | 74.0      | 39.7     | 35.6     | 57.5         | 23.3          | 46.0     |
| **[Qwen2.5-7B-Simple-RL-Zero](https://github.com/hkust-nlp/simpleRL-reason)** ^   | 77.2      | 33.5     | 37.6     | 62.5         | 33.3          | 48.8     |
| **[Qwen2.5-7B-Simple-RL-Zero](https://huggingface.co/Bradley/Qwen-2.5-7B-Simple-RL)** *   | 75.6      | 34.2     | 39.0     | 52.5         | 26.7          | 45.6     |
| **[Qwen2.5-7B-PURE-VR](https://huggingface.co/jinachris/PURE-VR)** *    | 79.8      | 36.8     | 41.9     | 60.0         | 20.0          | 47.7     |
| **Qwen2.5-7B-DPO-VP**    |   74.8   | 35.3 | 36.9 | 67.5 | 26.7 | 48.2|

表格中，所有的模型都基于 Qwen2.5-Math-7B 的基模微调，加粗的模型代表使用完全相同的 prompts 作自提升的方法调整的模型；* 尾标的结果为我自己评估的结果；^ 尾标的结果由对应模型的技术报告中记载得到。其中 Qwen2.5-7B-Simple-RL-Zero 并未开源其训练好的模型，我们从 Huggingface 上找了一个他人复现的结果作为评估。另外，我们注意到由于 Qwen 官方的评估代码对模型进行切片，在不同数量的卡上进行评估的结果会有微小差异。我们的模型和我们复现的结果均采用 4 张 A800 进行评估。

***Data and GPUs comparison of different approaches***


|                | Qwen2.5-Math-7B-Instruct        | rStar-Math-7B                  | Eurus-2-7B-PRIME         | Qwen2.5-7B-SimpleRL-Zero | Qwen2.5-7B-PURE-VR         | Qwen2.5-7B-DPO-VP         |
| -------------- | ------------------------------- | ------------------------------ | ------------------------ | ------------------------ | ----------------------- | ----------------------- |
| **Base Model** | Qwen2.5-Math-7B                 | Qwen2.5-Math-7B                | Qwen2.5-Math-7B          | Qwen2.5-Math-7B          | Qwen2.5-Math-7B         | Qwen2.5-Math-7B         |
| **SFT Data**   | 2.5M (open-source and in-house) | ~7.3M (MATH, NuminaMath, etc.) | 230K                     | 0                        | 0                       | 0 |
| **RM Data**    | 618K (in-house)                 | ~7k (in-house)                 | 0                        | 0                        | 0      | 0|
| **RM**         | Qwen2.5-Math-RM (72B)           | None                           | Eurus-2-7B-SFT           | None                     | None  | None |
| **Self-improve Method**    | RL + ORM     | MCTS + PPM       | RL + PRM|  RL + VR   | RL + VR   | DPO + VR
| **Self-improve Data**    | 66K     | ~3.647M                    | 150K| 8K   | 8K   | 8K
| **GPUs**       | -                               | 80 H100 at most                | 8 A100                   | 40 H100                  | 8 A100                  | 1 A800 or even less

## 🔧 Quick Start

### Installation

Our code is implemented based on OpenRLHF. Please follow [OpenRLHF's guidance](https://github.com/OpenRLHF/OpenRLHF/tree/main?tab=readme-ov-file#installation) to configure required environments. Then run `pip install -r requirements.txt`

### 


## 📝 TODO:



## 🎈 Citation


## 🌻 Acknowledgement
