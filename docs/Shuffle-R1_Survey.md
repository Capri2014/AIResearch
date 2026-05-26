# Shuffle-R1 Survey

**Date:** 2026-05-25  
**Surveyed by:** Agent (pipeline)  
**Source:** Zhu et al., ICLR 2026 - https://arxiv.org/abs/2508.05612, https://github.com/xiaomi-research/shuffle-r1

---

## TL;DR

**Shuffle-R1** is an ICLR 2026 paper that improves RL fine-tuning efficiency for Multimodal Large Language Models (MLLMs) by dynamically restructuring trajectory sampling and batch composition.

**Key findings:**
- Achieves superior performance against GRPO while using only **50% of training steps**
- Introduces two key modules: **Pairwise Trajectory Sampling (PTS)** and **Advantage-based Batch Shuffle (ABS)**
- Shuffle-R1-7B achieves **64.7% average** across benchmarks (+7.3% over base Qwen2.5-VL-7B)

---

## Key Insights

### 1. Two Critical Inefficiencies in Standard RL Pipelines

**Problem 1: Advantage Collapsing**
- In standard GRPO, most computed advantages cluster tightly around zero
- Weak gradient updates that don't meaningfully improve the model
- The learning signal gets drowned out by high-variance normalization

**Problem 2: Rollout Silencing**
- By mid-training, 70-80% of rollouts produce **zero gradient updates**
- Massive computational waste despite generating these rollouts
- Progressive decline in active learning samples

### 2. Root Cause: Static Sampling Paradigm

Both issues stem from uniformly processing all trajectories regardless of evolving learning signal quality during training.

> **Core insight:** What data the model updates on matters as much as how it updates.

### 3. Solution: Pairwise Trajectory Sampling (PTS)

Instead of treating each response individually, pair the highest-advantage response with the lowest-advantage response in each group:

```
Group responses: [r1, r2, r3, r4, r5, r6, r7, r8] (sorted by advantage)
Pairs: (r1,r8), (r2,r7), (r3,r6), (r4,r5)
```

**Why it works:**
- Amplifies contrast: model learns "r1 is better than r8" directly
- Reduces advantage variance within pairs
- Keeps high-advantage samples active longer

### 4. Solution: Advantage-based Batch Shuffle (ABS)

Dynamically adjust advantage magnitudes during training to prevent silencing:

- **Early training:** Strong signal (high power exponent)
- **Late training:** Reduced emphasis to prevent collapse
- Temperature scheduling for softmax methods

---

## Performance

| Model | MathVerse | MathVision | MathVista | WeMath | HallusionBench | ChartQA | **Avg** |
|-------|-----------|------------|----------|--------|--------------|---------|---------|
| Qwen2.5-VL-3B | 34.8 | 21.9 | 58.4 | 51.7 | 59.8 | 73.1 | 49.9 |
| Qwen2.5-VL-7B | 42.6 | 25.8 | 67.4 | 63.5 | 65.2 | 79.8 | 57.4 |
| **Shuffle-R1-3B** | 44.2 | 26.8 | 70.4 | 66.5 | 69.2 | 79.9 | **59.5** |
| **Shuffle-R1-7B** | 53.9 | 30.0 | 77.0 | 72.3 | 71.0 | 84.1 | **64.7** |

All models evaluated under CoT prompt. Shuffle-R1-7B achieves **+7.3%** improvement with only **50% of training steps** vs GRPO.

---

## Technical Details

### Architecture Overview

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
                              Update Policy
```

### Core Algorithm: Pairwise Trajectory Sampling

From `shuffle_r1/custom_algos.py` (189 lines):

```python
# Line 58-110: compute_pairwise_purning_grpo_advantage
@torch.no_grad()
def compute_pairwise_purning_grpo_advantage(
    token_level_rewards: torch.Tensor, 
    eos_mask: torch.Tensor, 
    index: torch.Tensor, 
    eps: float = 1e-6,  
    purning: bool = False, 
    purning_ratio: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pair highest-advantage with lowest-advantage responses."""
    
    # Step 1: Sum rewards along token dimension
    scores = token_level_rewards.sum(dim=-1)  # [batch_size]
    
    # Step 2: Group responses by prompt index
    id2score = defaultdict(list)
    for i in range(bsz):
        id2score[index[i]].append({'score': scores[i], 'index': i})
    
    # Step 3: Compute group-level normalized advantages
    for idx in id2score:
        id2gps[idx] = len(id2score[idx])
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor([item['score'] for item in id2score[idx]]))
            id2std[idx] = torch.std(torch.tensor([item['score'] for item in id2score[idx]]))
    
    # Step 4: Pair highest with lowest advantage
    if Purning:
        pairwise_list = []
        for idx in id2score:
            group_scores = [
                {"adv": (item['score'] - id2mean[idx]) / (id2std[idx] + eps), "index": item['index']} 
                for item in id2score[idx]]
            group_scores.sort(key=lambda x: x["adv"], reverse=True)
            
            # Pair: best with worst (opposite ends)
            pairs = []
            left, right = 0, len(group_scores) - 1
            while left < right:
                pairs.append({
                    'pair': [group_scores[left], group_scores[right]], 
                    'adv_sum': abs(adv1) + abs(adv2),
                    'adv_max': max(abs(adv1), abs(adv2)),
                })
                left += 1
                right -= 1
            
            # Prune bottom 50% of pairs
            bucket_size = int(len(pairs) * (1.0 - purning_ratio))
            pairs_to_keep = pairs[:bucket_size]
    
    # Step 5: Extend to token-level advantages
    scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return scores, scores, pairwise_list
```

### Core Algorithm: Advantage Normalization (ABS)

```python
# Line 112-147: normalize_advantage
def normalize_advantage(
    advantages: np.ndarray, 
    current_step: int, 
    total_training_step: int, 
    method: str = 'power', 
    p: int = 1, 
    t_max: int = 2, 
    t_min: int = 1
):
    """Dynamically adjust advantage magnitude."""
    assert method in EXPERIENCE_SAMPLING_METHODS
    
    progress = current_step / total_training_step
    
    if method in ['power', 'buffer']:
        # Power law decay: early = strong, late = weak
        advantages = advantages ** p
    elif method == 'softmax':
        # Cosine temperature schedule
        temperature = (t_max - t_min) * np.cos(0.5 * np.pi * progress) + t_min
        advantages = np.exp(advantages / temperature)
    elif method == 'log_softmax':
        # Sinusoidal temperature
        temperature = (t_max - t_min) * np.sin(0.5 * np.pi * progress) + t_min
        advantages = np.log(1 + advantages / temperature)
    elif method == 'inverse':
        advantages = 1 / (advantages + 1e-6)
    
    probabilities = advantages / advantages.sum()
    return probabilities
```

### Training Configuration

```yaml
# examples/config.yaml (key sections)
data:
  train_files: hiyouga/math12k@train
  prompt_key: problem
  answer_key: answer
  max_prompt_length: 2048
  max_response_length: 2048
  rollout_batch_size: 512

algorithm:
  adv_estimator: grpo
  use_kl_loss: true
  kl_penalty: low_var_kl
  kl_coef: 1.0e-2

worker:
  actor:
    global_batch_size: 128
    micro_batch_size_per_device_for_update: 4
    micro_batch_size_per_device_for_experience: 16
    model:
      model_path: Qwen/Qwen2.5-VL-7B-Instruct
  rollout:
    temperature: 1.0
    n: 5  # Generate N=5 responses per query
    gpu_memory_utilization: 0.6
  reward:
    reward_type: function
    compute_score: math

trainer:
  total_episodes: 15
  n_gpus_per_node: 8
  nnodes: 1
  val_freq: 5
  save_freq: 5
```

### Training Execution

```bash
# From repo: examples/qwen2_5_vl_3b.sh
#!/bin/bash
python -m shuffle_r1.main \
    --config examples/config.yaml \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-3B-Instruct \
    trainer.total_episodes=15 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8

# Or 7B model:
bash examples/qwen2_5_vl_7b.sh
```

**Hardware:** All training conducted on 8x H800-80G GPUs.

### Inference

#### Using Transformers

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

# IMPORTANT: Add thinking prompt (required for best results)
system_prompt = """You FIRST think about the reasoning process as an internal monologue 
and then provide the final answer. The reasoning process MUST BE 
enclosed within <reasoning> </reasoning> tags. The final answer 
MUST BE put in \\boxed{}. """

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
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output_text)
```

#### Using vLLM (Batch Inference)

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

**Input format** (`data.jsonl`):
```json
{"image_path": "path/to/image/1", "question": "question 1"}
{"image_path": "path/to/image/2", "question": "question 2"}
```

---

## Open Source Implementations

| Repository | Type | Notes |
|------------|------|-------|
| https://github.com/xiaomi-research/shuffle-r1 | Code | Official, 189 lines in `custom_algos.py` |
| https://huggingface.co/XenoZLH/Shuffle-R1-Qwen-3B | Model | 3B checkpoint |
| https://huggingface.co/XenoZLH/Shuffle-R1-Qwen-7B | Model | 7B checkpoint |
| https://huggingface.co/datasets/XenoZLH/MMRL30k | Dataset | 29.1k training samples |

**Dependencies:**
- Based on **EasyR1** (https://github.com/hiyouga/EasyR1)
- Uses **VERL** library (https://verl.readthedocs.io)
- Qwen2.5-VL as base model

---

## Relevance to Autonomous Driving

### Potential Applications

1. **VLA Fine-tuning** — Efficient RL fine-tuning for Vision-Language-Action models in autonomous driving
2. **Preference Learning** — Pairwise comparison for driving behavior optimization (comfort vs efficiency)
3. **Reward Shaping** — Better advantage estimation for safety-critical driving policies

### Why Shuffle-R1 Matters for Driving

| Aspect | Standard GRPO | Shuffle-R1 | Driving Impact |
|--------|--------------|-----------|----------------|
| Training Steps | Full (100%) | 50% | Faster iteration cycles |
| Sample Efficiency | Low | High | Less driving data needed |
| Gradient Signal | Collapsed | Preserved | Better policy learning |
| Implementation | Standard | Custom + EasyR1 | Integration effort |

### For Our Pipeline

```
VLA Training Flow:
  Waymo Episodes → SSL Pretrain → SFT → Shuffle-R1 RL → Driving Policy
                                     ↑
                           (50% fewer steps than GRPO)
```

**Implementation path:**
1. Use Shuffle-R1-7B as initial checkpoint for driving VLA
2. Fine-tune on Waymo/CARLA with Shuffle-R1 for behavior cloning
3. Apply ABS scheduling for late-training stabilization

---

## References

- Paper: https://arxiv.org/abs/2508.05612
- Code: https://github.com/xiaomi-research/shuffle-r1
- Models: https://huggingface.co/XenoZLH/Shuffle-R1-Qwen-7B
- Dataset: https://huggingface.co/datasets/XenoZLH/MMRL30k
- Based on: https://github.com/hiyouga/EasyR1