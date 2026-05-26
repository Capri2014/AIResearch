# Shuffle-R1: Efficient RL Framework for Multimodal LLMs via Data-centric Dynamic Shuffle

## TL;DR

**Shuffle-R1** (ICLR 2026) improves RL fine-tuning efficiency by dynamically restructuring trajectory sampling and batch composition. It matches GRPO performance with only **50% of the training steps**.

**Two key innovations:**
1. **Pairwise Trajectory Sampling (PTS)** — Pair high/low advantage responses to amplify learning signals
2. **Advantage-based Batch Shuffle (ABS)** — Dynamically rebalance batches to keep non-zero gradients flowing

---

## 1. Why Standard RL Fails

### Problem 1: Advantage Collapsing

In standard GRPO, advantages cluster near zero:

```python
# Standard GRPO: advantages are small and clustered
advantages = (rewards - mean_rewards) / std  # Most values ≈ 0
```

This drowns out meaningful learning signals.

### Problem 2: Rollout Silencing

By mid-training, 70-80% of samples produce **zero gradients** — computational waste.

---

## 2. Method: Shuffle-R1 Architecture

### 2.1 High-Level Pipeline

```
Query q ──► Generate 2N responses ──► Compute Rewards
                                      │
         ┌────────────────────────────┴───┐
         ▼                                    ▼
  Pairwise Trajectory          Advantage-based
  Sampling (PTS)            Batch Shuffle (ABS)
         │                                    │
         └────────────────────────────┬──────┘
                                      ▼
                              Filter & Shuffle
                                      │
                                      ▼
                              Strong Batched Loss
                                      │
                                      ▼
                              Update Policy
```

### 2.2 Pairwise Trajectory Sampling (PTS)

From `shuffle_r1/custom_algos.py`:

```python
import torch
from collections import defaultdict
from typing import Tuple
import heapq

# Core PTS algorithm - pair highest-advantage with lowest-advantage responses
@torch.no_grad()
def compute_pairwise_purning_grpo_advantage(
    token_level_rewards: torch.Tensor, 
    eos_mask: torch.Tensor, 
    index: torch.Tensor, 
    eps: float = 1e-6,  
    purning: bool = False, 
    purning_ratio: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Key idea: Pair highest-advantage with lowest-advantage responses.
    This amplifies the learning signal compared to individual advantages.
    """
    if isinstance(index, list):
        index = index[0]
        
    Purning = purning
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)  # [batch_size]
    
    # Group by prompt index
    id2score = defaultdict(list)
    id2mean, id2std, id2gps = {}, {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append({'score': scores[i], 'index': i})
    
    # Compute group-level statistics
    for idx in id2score:
        id2gps[idx] = len(id2score[idx])
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor([item['score'] for item in id2score[idx]]))
            id2std[idx] = torch.std(torch.tensor([item['score'] for item in id2score[idx]]))

    # Pairing logic
    if Purning:
        pairwise_list = []
        for idx in id2score:
            # Calculate advantages for each response in group
            group_scores = [
                {"adv": (item['score'] - id2mean[idx]) / (id2std[idx] + eps), "index": item['index']} 
                for item in id2score[idx]
            ]
            group_scores.sort(key=lambda x: x["adv"], reverse=True)
            
            # Pair: best with worst (opposite ends)
            pairs = []
            assert len(group_scores) % 2 == 0, "Number of responses in a group should be even"
            left, right = 0, len(group_scores) - 1
            while left < right:
                adv1, index1 = group_scores[left]["adv"], group_scores[left]["index"]
                adv2, index2 = group_scores[right]["adv"], group_scores[right]["index"]
                pairs.append({
                    'pair': [group_scores[left], group_scores[right]], 
                    'adv_sum': abs(adv1) + abs(adv2),
                    'adv_max': max(abs(adv1), abs(adv2)),
                    'adv_mean': (abs(adv1) + abs(adv2)) / 2,
                    'pos_adv': abs(torch.max(adv1, adv2))
                })
                left += 1
                right -= 1
            
            # Prune bottom 50% of pairs
            bucket_size = int(len(pairs) * (1.0 - purning_ratio))
            pairs_to_keep = pairs[:bucket_size]
            
            for pair in pairs_to_keep:
                pairwise_list.append({
                    "pair_index": [item["index"] for item in pair['pair']], 
                    "adv_sum": pair['adv_sum'].item(),
                    "adv_max": pair['adv_max'].item(),
                    "adv_mean": pair['adv_mean'].item(),
                    "pos_adv": pair['pos_adv'].item()
                })
    else:
        pairwise_list = None

    # Normalize advantages to token level
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return scores, scores, pairwise_list
```

### 2.3 Advantage-based Batch Shuffle (ABS)

Dynamically re-balance batch composition:

```python
import numpy as np

def normalize_advantage(
    advantages: np.ndarray, 
    current_step: int, 
    total_training_step: int, 
    method: str = 'power', 
    p: int = 1, 
    t_max: int = 2, 
    t_min: int = 1
):
    """
    ABS: Adjust advantage magnitude dynamically.
    
    method='power':  adv^p * (1 - step/total) for emphasis decay
    method='softmax': temperature increases with training for softer distribution
    method='log_softmax': log-based smoothing
    """
    assert method in ['power', 'softmax', 'log_softmax', 'inverse', 'std', 'buffer']
    
    progress = current_step / total_training_step
    
    if method in ['power', 'buffer']:
        advantages = advantages ** p
    elif method == 'softmax':
        # Bigger temperature = smaller probability difference
        temperature = (t_max - t_min) * np.cos(0.5 * np.pi * progress) + t_min
        advantages = np.exp(advantages / temperature)
    elif method == 'log_softmax':
        # Bigger temperature = bigger probability difference  
        temperature = (t_max - t_min) * np.sin(0.5 * np.pi * progress) + t_min
        advantages = np.log(1 + advantages / temperature)
    elif method == 'inverse':
        advantages = 1 / (advantages + 1e-6)
    
    probabilities = advantages / advantages.sum()
    return probabilities
```

### 2.4 Training Script

```bash
# examples/qwen2_5_vl_3b.sh
python -m shuffle_r1.main \
    --config examples/config.yaml \
    algorithm.adv_estimator=shuffle_r1 \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-3B-Instruct \
    trainer.total_episodes=15 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8
```

Key config (`examples/config.yaml`):

```yaml
algorithm:
  adv_estimator: grpo
  disable_kl: false
  use_kl_loss: true
  kl_penalty: low_var_kl
  kl_coef: 1.0e-2

worker:
  actor:
    global_batch_size: 128
    micro_batch_size_per_device_for_update: 4
    micro_batch_size_per_device_for_experience: 16
    model:
      model_path: Qwen/Qwen2.5-VL-3B-Instruct
      
  rollout:
    temperature: 1.0
    n: 5                      # Generate N responses per query
    
  reward:
    reward_type: function
    compute_score: math
```

### 2.5 Inference

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model_path = "XenoZLH/Shuffle-R1-Qwen-7B"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_path)

# IMPORTANT: Add thinking prompt
system_prompt = """You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <reasoning> </reasoning> tags. The final answer MUST BE put in \\boxed{}."""

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/your/image"},
            {"type": "text", "text": system_prompt + "YOUR QUESTION"},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
output_text = processor.batch_decode(
    generated_ids[:, len(inputs.input_ids[0]):], 
    skip_special_tokens=True
)
print(output_text)
```

Or with vLLM:

```bash
python inference/infer_vllm.py \
    --model XenoZLH/Shuffle-R1-Qwen-7B \
    --output-dir ./outputs \
    --input-file ./data.jsonl \
    --tensor-parallel-size 1 \
    --min-pixels 262144 \
    --max-pixels 4194304 \
    --max-model-len 8192 \
    --temperature 0.5
```

---

## 3. Results

| Model | MathVerse | MathVision | MathVista | WeMath | HallusionBench | ChartQA | **Avg** |
|-------|-----------|------------|----------|--------|--------------|---------|---------|
| Qwen2.5-VL-3B | 34.8 | 21.9 | 58.4 | 51.7 | 59.8 | 73.1 | 49.9 |
| Qwen2.5-VL-7B | 42.6 | 25.8 | 67.4 | 63.5 | 65.2 | 79.8 | 57.4 |
| **Shuffle-R1-3B** | 44.2 | 26.8 | 70.4 | 66.5 | 69.2 | 79.9 | **59.5** |
| **Shuffle-R1-7B** | 53.9 | 30.0 | 77.0 | 72.3 | 71.0 | 84.1 | **64.7** |

**Key finding:** Shuffle-R1-7B achieves 64.7 avg (vs Qwen2.5-VL-7B's 57.4) with only **50% of training steps**.

---

## 4. Quick Reference Decision Table

| Scenario | Recommendation |
|----------|-------------|
| Limited compute (≤8 GPUs) | Shuffle-R1 with smaller n (n=3) |
| Many prompts per group | Higher pruning_ratio (0.3-0.5) |
| Sparse rewards | PTS essential |
| Dense rewards | Standard GRPO often sufficient |
| Mid-training silencing | Use ABS with 'power' method |

---

## 5. Key Files in Repo

| File | Purpose |
|------|---------|
| `shuffle_r1/custom_algos.py` | **Core PTS + ABS algorithms** (189 lines) |
| `shuffle_r1/main.py` | Training entry point |
| `shuffle_r1/config.py` | Config dataclasses |
| `examples/config.yaml` | Training config |
| `inference/infer_vllm.py` | vLLM inference |

---

## References

- **Paper:** [Shuffle-R1 (ICLR 2026)](https://arxiv.org/abs/2508.05612)
- **Code:** [xiaomi-research/shuffle-r1](https://github.com/xiaomi-research/shuffle-r1)
- **Models:** [Shuffle-R1-Qwen-3B](https://huggingface.co/XenoZLH/Shuffle-R1-Qwen-3B), [Shuffle-R1-Qwen-7B](https://huggingface.co/XenoZLH/Shuffle-R1-Qwen-7B)
- **Dataset:** [MMRL30k](https://huggingface.co/datasets/XenoZLH/MMRL30k)