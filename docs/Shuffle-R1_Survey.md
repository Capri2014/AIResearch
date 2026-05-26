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

By mid-training, 70-80% of samples produce **zero gradients** — computational waste:

```python
# trad_grpo.py (standard approach)
for batch in dataloader:
    advantages = compute_advantages(rewards, responses)  # Most ≈ 0!
    loss = policy_loss(advantages)  # Weak signal
    optimizer.step()  # Barely learns
```

---

## 2. Method: Shuffle-R1 Architecture

### 2.1 High-Level Pipeline

```
Query q ──► Generate 2N responses ──► Compute Rewards
                                      │
         ┌────────────────────────────┴───┐
         ▼                                    ▼
  Pairwise Trajectory          Advantage-based
  Sampling (PTS)              Batch Shuffle (ABS)
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

Pair the best response with the worst from each group:

```python
# shuffle_r1/custom_algos.py
def compute_pairwise_purning_grpo_advantage(
    token_level_rewards, eos_mask, index, 
    purning_ratio=0.5
):
    """
    Key idea: Pair highest-advantage with lowest-advantage responses.
    This amplifies the learning signal compared to individual advantages.
    """
    scores = token_level_rewards.sum(dim=-1)  # [batch_size]
    
    # Group by prompt index
    id2score = defaultdict(list)
    for i in range(batch_size):
        id2score[index[i]].append({'score': scores[i], 'index': i})
    
    # Compute group-level normalized advantages
    for idx in id2score:
        mean = torch.mean([item['score'] for item in id2score[idx]])
        std = torch.std([item['score'] for item in id2score[idx]])
        group_scores = [
            {"adv": (item['score'] - mean) / (std + eps), "index": item['index']}
            for item in id2score[idx]
        ]
        # Sort by advantage descending
        group_scores.sort(key=lambda x: x["adv"], reverse=True)
        
        # Pair: best with worst (opposite ends)
        pairs = []
        left, right = 0, len(group_scores) - 1
        while left < right:
            pairs.append({
                'pair_index': [group_scores[left]["index"], group_scores[right]["index"]],
                'pos_adv': max(abs(group_scores[left]["adv"]), abs(group_scores[right]["adv"]))
            })
            left += 1
            right -= 1
        
        # Prune bottom 50% of pairs (keep most informative)
        bucket_size = int(len(pairs) * (1.0 - purning_ratio))
        pairs_to_keep = pairs[:bucket_size]
    
    # Extend advantages to token level
    scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return scores, scores, pairwise_list
```

**Why it works:** Pairing extremes amplifies the contrast — the model learns "this is better than that" directly.

### 2.3 Advantage-based Batch Shuffle (ABS)

Dynamically re-balance batches so low-advantage samples don't drown out the signal:

```python
# shuffle_r1/custom_algos.py
def normalize_advantage(
    advantages, current_step, total_steps,
    method='power', p=1, t_max=2, t_min=1
):
    """
    ABS: Adjust advantage magnitude dynamically.
    
    method='power':  adv * (1 - step/total)^p for emphasis decay
    method='softmax': temperature increases with training
    """
    progress = current_step / total_steps
    
    if method == 'power':
        # Early: strong signal → Late: reduced emphasis
        return advantages * ((1 - progress) ** p)
    elif method == 'softmax':
        # Higher temperature later = softer distribution
        temperature = t_min + (t_max - t_min) * progress
        return torch.softmax(advantages / temperature, dim=-1)
```

Combine with PTS for maximum efficiency:

```python
# Full Shuffle-R1 algorithm
def shuffle_r1_advantage(
    rewards, eos_mask, index, 
    step, total_steps,
    purning_ratio=0.5
):
    # Step 1: Pairwise sampling (PTS)
    adv, _, pairwise_list = compute_pairwise_purning_grpo_advantage(
        rewards, eos_mask, index, purning_ratio=True, 
        purning_ratio=purning_ratio
    )
    
    # Step 2: Dynamic normalization (ABS)
    adv = normalize_advantage(adv, step, total_steps, method='power')
    
    return adv, pairwise_list
```

### 2.4 Training Script

Run with the provided config:

```bash
# examples/qwen2_5_vl_3b.sh
#!/bin/bash
# Shuffle-R1 training launcher

python -m shuffle_r1.main \
    --config examples/config.yaml \
    algorithm.adv_estimator=shuffle_r1 \
    algorithm.purning_ratio=0.5 \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-3B-Instruct \
    trainer.total_episodes=15 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8
```

Key config parameters:

```yaml
# examples/config.yaml
algorithm:
  adv_estimator: grpo          # → change to shuffle_r1 in practice
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
    compute_score: math        # Match function to your task
```

### 2.5 Inference

```python
# From README.md - inference example
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_path = "XenoZLH/Shuffle-R1-Qwen-7B"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_path)

# IMPORTANT: Add thinking prompt
system_prompt = """
You FIRST think about the reasoning process as an internal monologue 
and then provide the final answer. The reasoning process MUST BE 
enclosed within <reasoning> </reasoning> tags. The final answer 
MUST BE put in \\boxed{}.
"""

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

Or with vLLM for batch inference:

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
| `shuffle_r1/custom_algos.py` | **Core PTS + ABS algorithms** |
| `shuffle_r1/main.py` | Training entry point |
| `shuffle_r1/config.py` | Config dataclasses |
| `shuffle_r1/fsdp.py` | FSDP worker setup |
| `examples/config.yaml` | Training config |
| `inference/infer_vllm.py` | vLLM inference |

---

## References

- **Paper:** [Shuffle-R1 (ICLR 2026)](https://arxiv.org/abs/2508.05612)
- **Code:** [xiaomi-research/shuffle-r1](https://github.com/xiaomi-research/shuffle-r1)
- **Models:** [Shuffle-R1-Qwen-3B](https://huggingface.co/XenoZLH/Shuffle-R1-Qwen-3B), [Shuffle-R1-Qwen-7B](https://huggingface.co/XenoZLH/Shuffle-R1-Qwen-7B)
- **Dataset:** [MMRL30k](https://huggingface.co/datasets/XenoZLH/MMRL30k)