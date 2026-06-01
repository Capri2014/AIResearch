# π0 Series 3-Day Sprint Plan
## Robotics Foundation Model Study

---

## Overall Goal

**Deliverables after 3 days:**

1. **One-page π0 series architecture diagram**
   - π0: VLM backbone + robot state/action expert + flow matching action chunking
   - π0-FAST: Continuous action trajectory tokenized → autoregressive VLA
   - π0.5: Heterogeneous co-training / semantic subtask / web & robot data / open-world generalization

2. **5-8 page technical survey**
   - RT-1 / RT-2 / OpenVLA / Diffusion Policy / ACT / Octo / GR00T / LeRobot / OpenPI
   - Focus: action representation, data mixture, generalization, fine-tuning, deployment

3. **Local runnable minimal experiment**
   - OpenPI or LeRobot π0.5 demo
   - Complete at least one: inference / tiny fine-tune / dataset format conversion

4. **30-day development roadmap link**
   - Transition from "understanding π0" → "can modify π0 / can evaluate / can deploy on own robot or simulation"

---

## Day 1: Architecture梳理 + Paper Mainline

### Morning: Build π0 Architecture Map

#### Main Paper List

| Paper | Authors | Core Contribution |
|-------|---------|------------------|
| [π0: A Vision-Language-Action Flow Model for General Robot Control](https://physicalintelligence.blog/pi0) | Physical Intelligence | VLM + flow matching + action chunking |
| [FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://physicalintelligence.blog/fast) | Physical Intelligence | Continuous action → discrete tokens |
| [π0.5: A Vision-Language-Action Model with Open-World Generalization](https://physicalintelligence.blog/pi05) | Physical Intelligence | Open-world generalization |

#### π0 Core Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           π0 ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUTS                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐                   │
│  │  Images  │  │ Language │  │  Robot State      │                   │
│  │ (visual) │  │(task desc)│  │ (joint angles,   │                   │
│  │          │  │          │  │  end-effector)   │                   │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘                   │
│       │             │                  │                                 │
│       ▼             ▼                  ▼                                 │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │           PRETRAINED VLM BACKBONE                               │        │
│  │         (PaliGemma-style late fusion)                          │        │
│  │              Visual Encoder + LLM Decoder                     │        │
│  └──────────────────────────┬────────────────────────────────────┘        │
│                            │                                            │
│                            ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │            ROBOT ACTION EXPERT                                 │        │
│  │    (state-conditioned / action-specific modules)                 │        │
│  └──────────────────────────┬────────────────────────────────────┘        │
│                            │                                            │
│                            ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │          CONDITIONAL FLOW MATCHING                             │        │
│  │  (model p(action | visual context, robot state)                │        │
│  │   - Continuous action distribution                            │        │
│  │   - Multimodal: multiple valid trajectories                  │        │
│  └──────────────────────────┬────────────────────────────────────┘        │
│                            │                                            │
│                            ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │           ACTION CHUNKING                                        │        │
│  │   Output: N timesteps × action_dim                             │        │
│  │   50Hz dexterous control possible                             │        │
│  └──────────────────────────┬────────────────────────────────────┘        │
│                            │                                            │
│                            ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │         LOW-LEVEL ROBOT CONTROLLER                              │        │
│  │    (PD control / force control / position control)           │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### Five Key Concept Understandings

| Module | Key Point |
|--------|-----------|
| **VLM backbone** | Inherits internet-scale semantic knowledge; vision-language joint pretraining |
| **Robot state proprioception** | Concatenate to latent or use dedicated encoder |
| **Action expert** | Not directly outputting action tokens from LLM; domain-specific module |
| **Flow matching** | More suitable than diffusion for continuous, high-frequency, multimodal action distribution |
| **Action chunking** | Robotics can't output step-by-step (latency too high); chunking enables 50Hz control |

**One-line understanding:**

> π0 combines VLM's semantic understanding capability with flow matching's continuous action generation, forming a robot foundation policy that can be fine-tuned across tasks and embodiments.

---

### Afternoon: π0-FAST and π0.5 Branches

#### π0 vs π0-FAST Comparison

| Dimension | π0 (Flow) | π0-FAST (Token) |
|------|-----------|----------------|
| **Action representation** | Continuous distribution (flow matching) | Discrete token sequence |
| **Training** | Diffusion-style | Autoregressive Transformer |
| **Use case** | Dexterous continuous control | Needs language following |
| **Training efficiency** | Lower | Up to 5x faster |
| **Inference cost** | Lower | 4-5x higher |
| **Infra** | Diffusion pipeline | LLM/VLM pipeline |

**FAST core technology:**
- Frequency-space / DCT-style compression
- Solves binning problem for high-frequency dexterous actions
- Achieves near diffusion VLA quality at 10k hour data scale

#### π0 vs π0.5 Comparison

| Dimension | π0 | π0.5 |
|------|-----|-------|
| **Positioning** | Physical skill foundation | Physical skill + semantic/world generalization |
| **Data** | Robot data (10k+ hours) | + Web data, semantic prediction, object detection |
| **Tasks** | Physical skills | + New rooms, new objects, new semantic tasks, long-horizon decomposition |
| **Co-training** | Homogeneous | Heterogeneous multiple robots/tasks |
| **Generalization** | Task-specific | Open-world |

**π0.5 key improvements:**

```
π0.5 Data Mix (heterogeneous co-training):
├── Robot data (multiple robots, embodiment)
├── Web data (image-video understanding)
├── High-level semantic prediction (task decomposition)
├── Object detection (open vocabulary)
├── Subtask commands (grasp, place, push...)
└── Mobile manipulation data
```

---

### Day 1 Evening Output: Architecture Comparison Table

| Version | Core Question | Technical Choice | Value | Risk |
|---------|-------------|-----------------|-------|------|
| **π0** | How to turn VLM into robot policy | VLM + flow matching + action chunking | Continuous action, dexterous control, 50Hz | Complex training/inference, needs large robot data |
| **π0-FAST** | How to fit into autoregressive VLA pipeline | Action tokenizer + AR Transformer | 5x faster training, unified LLM infra | Tokenization loss, 4-5x higher inference cost |
| **π0.5** | How to open-world generalize | Heterogeneous co-training | New environments, objects, long tasks | Complex data recipe, hard to reproduce |

---

## Day 2: Vibe Coding Intro + Minimal Experiments

### Morning: Environment Setup + Code Reading

#### OpenPI Official Resources

| Resource | Link |
|------|------|
| **GitHub** | https://github.com/physicalintelligence/openpi |
| **HuggingFace** | https://huggingface.co/collections/physicalintelligence/pi0 |
| **Blog** | https://physicalintelligence.blog/ |

#### Hardware Requirements

| Task | Minimum GPU |
|------|----------|
| Inference | 8GB+, RTX 4090 works |
| LoRA fine-tuning | 22.5GB+, RTX 4090 |
| Full fine-tuning | 70GB+, A100 80GB / H100 |

#### Environment Setup

```bash
# Clone OpenPI
git clone https://github.com/physicalintelligence/openpi.git
cd openpi

# Install dependencies
pip install -e .

# Download base checkpoint (optional)
# See HuggingFace collection
```

#### Key Code Structure

```
openpi/
├── pi0/
│   ├── modeling/
│   │   ├── pi0_model.py        # Main model definition
│   │   ├── pi0_fast_model.py  # FAST variant
│   │   └── pi0_5_model.py     # π0.5 variant
│   ├── inference/
│   │   └── inference.py       # Inference entry point
│   └── data/
│       └── dataset.py         # Data format
├── scripts/
│   ├── inference.py           # Inference script
│   └── finetune.py             # Fine-tune script
└── configs/
    └── ...                    # Config files
```

### Afternoon: Minimal Experiments

#### Experiment 1: Out-of-Box Inference

```bash
# Run inference with official checkpoint
python -m pi0.inference \
    --robot ur5e \
    --image /path/to/image.jpg \
    --language "pick up the cup" \
    --checkpoint pi0-ur5e-base
```

#### Experiment 2: LoRA Fine-tuning

```bash
# LoRA fine-tune
python -m pi0.finetune \
    --model pi0-ur5e-base \
    --dataset /path/to/my_data \
    --lora_rank 16 \
    --epochs 5
```

#### Experiment 3: Dataset Format Conversion

Convert from `bridge` format to OpenPI format:

```python
from pi0.data import convert_dataset

convert_dataset(
    source_format="bridge",
    target_format="openpi",
    input_path="/data/bridge",
    output_path="/data/openpi"
)
```

---

## Day 3: Survey Writing + 30-Day Roadmap

### Survey Outline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SURVEY OUTLINE                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Introduction (1 page)                                             │
│     - Why robot foundation models matter                               │
│     - VLA vs Diffusion Policy                                         │
│     - π0 series positioning                                           │
│                                                                      │
│  2. Related Work (2 pages)                                             │
│     ┌──────────────────────────────────────────────────────────┐       │
│     │ Model        | Year | Organization      | Action Type   │       │
│     ├──────────────────────────────────────────────────────────┤       │
│     │ RT-1         | 2022 | Google          | BC             │       │
│     │ RT-2         | 2023 | Google          | VLA            │       │
│     │ OpenVLA      | 2024 | Stanford/NVIDIA| VLA            │       │
│     │ Diffusion    | 2024 | Columbia        | Diffusion     │       │
│     │   Policy     |      |                  |               │       │
│     │ ACT          | 2024 | Stanford        | Diffusion     │       │
│     │ Octo         | 2024 | NYU/Tesla       | VLA            │       │
│     │ GR00T        | 2024 | Tesla          | VLA            │       │
│     │ LeRobot     | 2024 | LeRobot        | VLA            │       │
│     │ OpenPI      | 2025 | Physical Int.   | VLA/Flow      │       │
│     └──────────────────────────────────────────────────────────┘       │
│                                                                      │
│  3. π0 Series Technical Deep-Dive (3 pages)                            │
│     3.1 π0: Architecture, Flow Matching, Action Chunking              │
│     3.2 π0-FAST: Tokenization, Autoregressive Pipeline                 │
│     3.3 π0.5: Co-training Recipe, Generalization                     │
│                                                                      │
│  4. Comparison (2 pages)                                             │
│     - Action representation                                           │
│     - Data mixture                                                   │
│     - Generalization capability                                       │
│     - Fine-tuning                                                    │
│     - Deployment                                                     │
│                                                                      │
│  5. Experiments & Results (1 page)                                   │
│     - Benchmarks                                                    │
│     - Fine-tuning results                                            │
│                                                                      │
│  6. Discussion & Future (1 page)                                     │
│     - Limitations                                                   │
│     - 30-day roadmap                                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Action Representation Comparison

| Method | Representation Type | Pros/Cons |
|--------|-------------------|-----------|
| **BC/Actor-Critic** | Imitation learning | Simple, but needs large expert data |
| **Diffusion Policy** | Diffusion model | Multimodal, good continuous control | Slow inference, multi-step |
| **VLA (RT-2/OpenVLA)** | VLM + action token | Good language understanding | Needs large-scale pretraining |
| **Flow Matching (π0)** | Flow matching | Efficient, multimodal | Medium implementation complexity |
| **Action Chunking** | Chunk output | Low latency, high frequency | Needs synchronized controller |

### 30-Day Development Roadmap

```
Week 1-2: [Understand π0]
├── [ ] Read π0/FAST/π0.5 papers thoroughly
├── [ ] Run OpenPI inference demo
├── [ ] Understand data format and pipeline
└── [ ] Output: architecture diagram + notes

Week 3-4: [Can Modify π0]
├── [ ] LoRA fine-tune on custom data
├── [ ] Modify action chunk length
├── [ ] Test different robot embodiment
└── [ ] Output: fine-tuned checkpoint

Month 2: [Can Evaluate]
├── [ ] Establish eval pipeline
├── [ ] Design success metric
├── [ ] Compare π0 vs baseline
└── [ ] Output: evaluation report

Month 3+: [Can Deploy on Robot/Sim]
├── [ ] Connect to real robot (UR5e / Franka)
├── [ ] Sim2real transfer
└── [ ] Practical demo
```

---

## Quick Reference

| Model | Core Innovation | GitHub | Status |
|-------|----------------|-------|--------|
| **π0** | VLM + flow matching | OpenPI | ⭐ Active |
| **π0-FAST** | Action tokenization | OpenPI | ⭐ Active |
| **π0.5** | Heterogeneous co-training | OpenPI | ⭐ Active |
| **OpenVLA** | Open-source VLA | OpenVLA | Archived |
| **LeRobot** | HuggingFace-style | LeRobot | ⭐ Active |
| **Diffusion Policy** | Diffusion | diffusion_policy | Archived |
| **ACT** | Action chunking | act_torch | Archived |

---

## References

- Physical Intelligence Blog: https://physicalintelligence.blog/
- OpenPI GitHub: https://github.com/physicalintelligence
- π0 Paper: (Check blog for latest)
- LeRobot: https://github.com/huggingface/LeRobot

---

*Plan created: May 2026*
*Status: Day 1 ready, Days 2-3 to execute*