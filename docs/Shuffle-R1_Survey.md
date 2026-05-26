# Shuffle-R1 Survey

**Date:** 2026-05-25  
**Surveyed by:** Agent (pipeline)  
**Source:** Zhu et al., ICLR 2026 - https://arxiv.org/abs/2508.05612, https://github.com/xiaomi-research/shuffle-r1

## TL;DR

**Shuffle-R1** improves RL fine-tuning efficiency for Multimodal LLMs by dynamically restructuring trajectory sampling and batch composition. Matches GRPO performance with only **50% of training steps**.

**Two key innovations:**
1. **Pairwise Trajectory Sampling (PTS)** — Pair high/low advantage responses to amplify learning signals
2. **Advantage-based Batch Shuffle (ABS)** — Dynamically rebalance batches to prevent gradient silencing

---

## Key Insights

### 1. Two Inefficiencies in Standard RL

**Advantage Collapsing:**
- Most computed advantages cluster tightly around zero
- Weak gradient updates that don't meaningfully improve the model

**Rollout Silencing:**
- By mid-training, 70-80% of samples produce zero gradient updates
- Massive computational waste despite generating these rollouts

### 2. Pairwise Trajectory Sampling (PTS)

Instead of individual advantages, pair highest-advantage with lowest-advantage responses:
- Amplifies learning signal: model learns "this is better than that" directly
- Reduces advantage variance within pairs

### 3. Advantage-based Batch Shuffle (ABS)

Dynamically adjust advantage magnitude during training:
- `method='power'`: adv * (1 - step/total)^p for emphasis decay
- `method='softmax'`: temperature increases = softer distribution

---

## Performance

| Model | MathVerse | MathVision | MathVista | WeMath | HallusionBench | ChartQA | **Avg** |
|-------|-----------|------------|----------|--------|--------------|---------|---------|
| Qwen2.5-VL-3B | 34.8 | 21.9 | 58.4 | 51.7 | 59.8 | 73.1 | 49.9 |
| Qwen2.5-VL-7B | 42.6 | 25.8 | 67.4 | 63.5 | 65.2 | 79.8 | 57.4 |
| **Shuffle-R1-3B** | 44.2 | 26.8 | 70.4 | 66.5 | 69.2 | 79.9 | **59.5** |
| **Shuffle-R1-7B** | 53.9 | 30.0 | 77.0 | 72.3 | 71.0 | 84.1 | **64.7** |

**Key finding:** Shuffle-R1-7B achieves +7.3% improvement over base Qwen2.5-VL-7B with only **50% of training steps**.

---

## Technical Details

### Core Algorithm (`shuffle_r1/custom_algos.py`, 189 lines)

**PTS Implementation:**
```python
@torch.no_grad()
def compute_pairwise_purning_grpo_advantage(
    token_level_rewards, eos_mask, index, 
    eps=1e-6, purning=False, purning_ratio=0.5
):
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    
    # Group by prompt, compute normalized advantages
    for i in range(bsz):
        id2score[index[i]].append({'score': scores[i], 'index': i})
    
    # Pair best with worst
    pairs = []
    left, right = 0, len(group_scores) - 1
    while left < right:
        pairs.append({
            'pair': [group_scores[left], group_scores[right]],
            'adv_sum': abs(adv1) + abs(adv2),
        })
        left += 1
        right -= 1
    
    return scores, scores, pairwise_list
```

**ABS Implementation:**
```python
def normalize_advantage(advantages, current_step, total_steps, 
                       method='power', p=1, t_max=2, t_min=1):
    progress = current_step / total_steps
    
    if method == 'power':
        return advantages ** p
    elif method == 'softmax':
        temperature = (t_max - t_min) * np.cos(0.5 * np.pi * progress) + t_min
        return np.exp(advantages / temperature)
```

### Training Config

```yaml
# examples/config.yaml
algorithm:
  adv_estimator: grpo
  use_kl_loss: true
  kl_coef: 1.0e-2

worker:
  actor:
    global_batch_size: 128
    model_path: Qwen/Qwen2.5-VL-3B-Instruct
  rollout:
    n: 5  # responses per query
  reward:
    compute_score: math
```

### Inference

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "XenoZLH/Shuffle-R1-Qwen-7B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# IMPORTANT: Add thinking prompt
system_prompt = """You FIRST think about the reasoning process as an internal monologue 
and then provide the final answer. The reasoning process MUST BE enclosed within 
<reasoning> </reasoning> tags. The final answer MUST BE put in \\boxed{}."""

messages = [{"role": "user", "content": [
    {"type": "image", "image": "path/to/image"},
    {"type": "text", "text": system_prompt + question},
]}]
```

---

## Open Source Implementations

| Repository | Language | Notes |
|------------|----------|-------|
| https://github.com/xiaomi-research/shuffle-r1 | Python | Official (189 lines) |
| https://huggingface.co/XenoZLH/Shuffle-R1-Qwen-3B | Model | 3B checkpoint |
| https://huggingface.co/XenoZLH/Shuffle-R1-Qwen-7B | Model | 7B checkpoint |
| https://huggingface.co/datasets/XenoZLH/MMRL30k | Dataset | Training data |

---

## Relevance to Autonomous Driving

### Potential Applications

1. **VLA Fine-tuning** — Efficient RL fine-tuning for Vision-Language-Action models
2. **Preference Learning** — Pairwise comparison for driving behavior optimization
3. **Reward Shaping** — Better advantage estimation for driving policies

### For Our Pipeline

```
VLA Training Flow:
  Pre-training → SFT → Shuffle-R1 RL → Policy
                      ↑
            (50% fewer steps than GRPO)
```

### Comparison to GRPO for Driving

| Aspect | GRPO | Shuffle-R1 | Our Use Case |
|--------|-----|------------|-------------|
| Training Steps | Full | 50% | Faster iteration |
| Sample Efficiency | Low | High | Limited data |
| Implementation | Standard | Custom | May integrate |

---

## References

- Paper: https://arxiv.org/abs/2508.05612
- Code: https://github.com/xiaomi-research/shuffle-r1
- Models: https://huggingface.co/XenoZLH/Shuffle-R1-Qwen-7B