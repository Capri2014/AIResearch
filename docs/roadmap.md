# Roadmap / TODOs

This document mirrors (and can expand) the **TODO (roadmap)** section in the repo README.

## microgpt backbone (CoT for autonomous driving)

Goal: incorporate a **microgpt** component that provides the backbone for *reasoning/tracing* (CoT-style, but ideally structured) in the autonomous-driving stack.

Key decisions:
- **Integration shape:** library module (import) vs service (API/daemon).
- **Trace format:** prefer structured traces (JSON/tool traces) over free-form "CoT text".

Planned follow-up:
- **LoRA/PEFT update tricks:** train head + LoRA adapters (instead of full backbone finetune).
  - Provide a clean config switch: `--freeze-backbone`, `--lora-rank`, etc.
  - Define checkpoint format: backbone base ref + LoRA weights + head weights.

## Model/provider eval

Try **MLG / MinMax / Doubao**
- Identify official endpoints/SDKs, auth method, and pricing.
- Run a small repeatable benchmark suite:
  - coding task (repo-aware)
  - long-context summarization/survey
  - tool use / function calling / structured output
  - refusal/safety behavior quality
- Record results + choose recommendations (default + when to use each).

## Survey: Karpathy microgpt/nanoGPT gist (atomic GPT)

Status: **paused** (per Qi, 2026-02-14).

Source: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

Goal: extract reusable abstractions/patterns for our autonomy stack:
- "atomic decomposition" (params / forward / cache / loss / update / sampling)
- stepwise inference + caching semantics
- minimal eval harness mindset

Deliverables:
- a concrete **trace schema** draft for autonomy planning
- a minimal **eval harness** outline (scenarios + metrics + regression gates)

## PPO improvements for large models

Goal: improve PPO stability and explore model-based RL approaches that scale well with large foundation models.

Planned follow-up:
- **Stable PPO**: Implement advances from successful RLHF pipelines (clipping tricks, value function centering, advantage normalization, GAE tuning)
- **Model-based RL**: Explore world model integration (e.g., GAIA-2 style latent dynamics) for more sample-efficient policy learning
- **Large model compatibility**: Ensure delta-waypoint head works with scaled encoders/backbones; consider LoRA for the RL head
- **Training stability**: Add KL divergence constraints, learning rate scheduling, checkpoint selection by policy entropy

## CoT data generation and finetuning for autonomous driving

Goal: explore Chain-of-Thought reasoning data generation and finetuning for improved driving decisions.

Planned follow-up:
- **CoT data synthesis**: Generate reasoning traces from expert drivers or rule-based planners
- **Structured trace format**: Define schema for driving reasoning (perception → prediction → planning → action)
- **Finetuning**: Fine-tune vision-language or action models on CoT-augmented data
- **Evaluation**: Measure decision quality improvements from CoT reasoning

## RL algorithm upgrade: GRPO / Agent RL

Goal: move beyond basic PPO to more capable RL algorithms suitable for autonomous driving.

Planned follow-up:
- **GRPO (Group Relative Policy Optimization)**: Implement GRPO for autonomous driving tasks
- **Agent RL**: Explore agent-centric RL formulations for long-horizon driving decisions
- **Benchmark**: Compare PPO vs GRPO on waypoint prediction and control tasks
- **Scaling**: Ensure algorithm works with large foundation models and high-dimensional inputs

## DeepSeek RL pipeline: Pre-train → Reasoning SFT → RL → SFT/RLHF

Goal: survey DeepSeek's RL pipeline and upgrade our current pipeline accordingly.

Planned follow-up:
- **Survey DeepSeek RL**: Study DeepSeek's training recipe (pre-train → reasoning SFT → RL → SFT/RLHF)
- **Gap analysis**: Compare current pipeline (SFT → RL refinement) with DeepSeek's approach
- **Reasoning SFT**: Add explicit CoT/Reasoning SFT stage before RL
- **GRPO integration**: Replace or supplement PPO with GRPO in the RL stage
- **Iterative improvement**: Plan SFT/RLHF cycles for continuous refinement
- **Evaluation**: Define metrics for reasoning quality and driving safety


## Survey: Kimi RL + LengthControl (Moonshot AI)

Goal: survey Kimi's unique RL approach and LengthControl mechanism.

Planned follow-up:
- **Source**: Kimi/Moonshot AI papers or technical reports on their RL training methodology
- **LengthControl**: Understand their approach to managing output length in RL training
- **RL design**: Extract unique insights in their RL algorithm design that differ from PPO/GRPO
- **Applicability**: Assess which insights apply to autonomous driving (long-horizon planning, safety constraints)
- **Comparison**: Compare with DeepSeek's RL pipeline approach

## Survey: DeepSeek Engram (Memory-Augmented Language Models)

Goal: survey DeepSeek's Engram paper on efficient knowledge retrieval via N-gram embeddings and hash lookup.

---

## TODO: Autoregressive Decoder for Waypoints

**Question:** Can we upgrade the current decoder to Autoregressive (AR)?

### Current vs Autoregressive Decoder

| Aspect | Current (Parallel) | Autoregressive |
|--------|-------------------|----------------|
| **Output** | All waypoints at once | One at a time |
| **Speed** | Fast (parallel) | Slow (sequential) |
| **Consistency** | May not respect order | Sequential consistency |
| **Error Propagation** | None | Can accumulate |
| **Training** | Simple MSE | Teacher forcing needed |

### Why Autoregressive COULD Work

```
✓ Waypoints are SEQUENTIAL: w_{t+1} depends on w_t
✓ CoT reasoning is STEP-BY-STEP: similar pattern
✓ Natural for long sequences: avoid fixed length
✓ Variable output length: stop when confident

Example AR Generation:

Input: Images + State + CoT Text
       │
       ▼
┌─────────────────────────┐
│   Encoder (BERT/SSL)   │
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│   AR Decoder           │
│                        │
│ Step 1: w1 = f(enc)   │
│ Step 2: w2 = f(enc, w1) │
│ Step 3: w3 = f(enc, w1, w2) │
│ Step 4: w4 = f(enc, w1, w2, w3) │
│ ... until stop token   │
└─────────────────────────┘
```

### Why Autoregressive MIGHT NOT Work

```
✗ Speed: Real-time driving needs fast inference
✗ Error accumulation: Bad w1 → bad w2 → worse w3
✗ Training complexity: Teacher forcing, scheduled sampling
✗ Overfitting: May memorize training sequences
✗ Latency: Each waypoint adds delay
```

### Analysis: Can We Upgrade?

**YES, but with caveats:**

| Use Case | Recommendation |
|----------|---------------|
| **Training** | AR decoder useful for learning sequential structure |
| **Inference (real-time)** | Keep parallel decoder (speed critical) |
| **Planning (offline)** | AR decoder acceptable (can be slow) |

**Verdict:** Upgrade is possible and useful for training, but keep parallel decoder for real-time inference.

---

## TODO: Combining AR Decoder + CoT Reasoning

**Question:** If we have AR decoder + CoT, do they complement each other?

### Similarities and Differences

| Aspect | AR Decoder | CoT Reasoning |
|--------|------------|---------------|
| **Sequential** | ✅ Yes | ✅ Yes |
| **Step-by-step** | ✅ Yes | ✅ Yes |
| **Context** | Previous outputs | Full context |
| **Purpose** | Predict waypoints | Explain reasoning |
| **Input** | Encoded features | Reasoning text |

### How They Could Combine

```
┌─────────────────────────────────────────────────────────────────┐
│                    Combined AR + CoT Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Images + State                                           │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────┐                                    │
│  │   Encoder (BERT/SSL)   │                                    │
│  └─────────────────────────┘                                    │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────┐                                    │
│  │   CoT Encoder (BERT)   │                                    │
│  │                         │                                    │
│  │   Reasoning:            │                                    │
│  │   "I see a car..."    │                                    │
│  └─────────────────────────┘                                    │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    AR Decoder                             │ │
│  │                                                          │ │
│  │   At each step:                                          │ │
│  │   w_t = f(encoder_features, w_{<t}, cot_features)       │ │
│  │                                                          │ │
│  │   Benefits:                                             │ │
│  │   • Waypoint w_t uses CoT reasoning                     │ │
│  │   • CoT "explains" why w_t was chosen                 │ │
│  │   • Sequential consistency maintained                   │ │
│  │                                                          │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Is Combining a Good Idea?

**Pros:**
```
✅ Complementary strengths:
   - AR: Sequential prediction
   - CoT: Contextual reasoning

✅ Unified framework:
   - Both are step-by-step
   - Can share encoder

✅ Interpretability:
   - CoT explains each step
   - AR generates sequentially

✅ Better learning:
   - CoT guides AR
   - AR enforces consistency
```

**Cons:**
```
⚠️ Complexity:
   - More complex architecture
   - Harder to train

⚠️ Latency:
   - Both are sequential
   - Slower inference

⚠️ Overkill:
   - May be redundant
   - Simple waypoints may not need both
```

### Recommendation

| Scenario | Recommendation |
|----------|---------------|
| **Simple driving** | Parallel decoder (fast) |
| **Complex maneuvers** | AR decoder + CoT |
| **Planning tasks** | AR decoder + CoT |
| **Real-time control** | Keep parallel |

**Verdict:** Combining is a GOOD IDEA for complex/interpretable scenarios, but may be overkill for real-time control.

---

## Action Items

1. **Prototype AR decoder** on toy domain
2. **Compare** parallel vs AR decoder quality
3. **Evaluate** CoT + AR combination
4. **Benchmark** inference speed for real-time use

---

## TODO: AR Survey and Implementation Plan

### TODO 1: Survey Autoregressive Methods for Autonomous Driving and Robotics

**Goal:** Survey AR papers and approaches for driving/robotics domains.

**Survey Focus:**
- **Autoregressive LLMs for planning:** GPT-based planners, LLM planners
- **Sequential decision making:** AR policies, autoregressive action prediction
- **Robotics:** RT-2, RT-X, PaLM-E style models
- **Driving-specific:** PlannerLM, DriveGPT papers

**Papers to Survey:**
| Paper | Domain | Key Contribution |
|-------|--------|-----------------|
| RT-2 | Robotics | Vision-language-action models |
| PaLM-E | Robotics | Embodied LLM planning |
| PlannerLM | Driving | LLM-based planning |
| DriveGPT | Driving | Autoregressive driving decisions |
| VAD | Driving | Vectorized autonomous driving |
| UniAD | Driving | Unified driving perception-planning |

**Deliverable:** Survey document with comparison table and recommendations.

---

### TODO 2: Design AR Upgrade Plan for Current Pipeline

**Goal:** Create implementation roadmap for upgrading pipeline to include AR decoder.

**Current Pipeline:**
```
Waymo Data → JEPA Pre-train → SFT (CoT) → PPO RL → Deployment
                      ↓
              Parallel Decoder
```

**Proposed Pipeline (AR Upgrade):**
```
Waymo Data → JEPA Pre-train → SFT (CoT + AR Decoder) → PPO RL → Deployment
                                         ↓
                          AR Decoder (for complex scenarios)
                                         ↓
                          Parallel Decoder (for real-time)
```

**Design Questions:**
| Question | Investigation |
|----------|---------------|
| When to use AR vs parallel? | Define decision criteria |
| How to combine with CoT? | Joint training vs separate |
| What training strategy? | Teacher forcing, scheduled sampling |
| How to handle errors? | Error propagation mitigation |
| Speed-accuracy tradeoff? | Benchmark different modes |

**Implementation Phases:**
| Phase | Task | Duration |
|-------|------|----------|
| Phase 1 | Survey papers, design architecture | 2 weeks |
| Phase 2 | Implement AR decoder prototype | 3 weeks |
| Phase 3 | Train with CoT reasoning | 2 weeks |
| Phase 4 | Evaluate parallel vs AR | 1 week |
| Phase 5 | Benchmark for deployment | 1 week |

**Deliverable:** Implementation plan document with timeline.

---

### TODO 3: Code Implementation of AR Decoder (After Current Tasks)

**Goal:** Implement AR decoder for waypoint prediction (after completing current pipeline).

**Implementation Plan:**

| Step | Task | Description |
|------|------|-------------|
| 3.1 | Modify decoder architecture | Replace parallel decoder with AR |
| 3.2 | Add teacher forcing | During training |
| 3.3 | Implement stop token | Auto-regressive termination |
| 3.4 | Add CoT integration | Condition AR on reasoning |
| 3.5 | Compare quality | AR vs Parallel |
| 3.6 | Benchmark speed | Inference latency |

**Architecture Sketch:**
```python
class ARDecoder(nn.Module):
    """
    Autoregressive decoder for waypoint prediction.
    
    Generates waypoints one at a time, conditioned on:
    - Encoder features
    - CoT reasoning
    - Previous waypoints
    """
    
    def __init__(self, config):
        self.waypoint_embed = nn.Embedding(vocab_size, hidden_dim)
        self.decoder = TransformerDecoder(...)
        self.output_head = nn.Linear(hidden_dim, 3)  # x, y, heading
    
    def forward(self, encoder_out, cot_features, waypoints=None):
        # During training: teacher forcing
        # During inference: autoregressive generation
        pass
    
    def generate(self, encoder_out, cot_features, max_len=16):
        # Autoregressive generation
        for t in range(max_len):
            waypoint = self.predict_next(...)
            if stop_token: break
        return waypoints
```

**Success Metrics:**
| Metric | Target |
|--------|--------|
| ADE improvement | +5% vs parallel |
| CoT consistency | >0.8 reasoning score |
| Inference speed | <50ms (planning mode) |
| Real-time speed | <10ms (control mode) |

---

## Further Reading

| Topic | Reference |
|-------|-----------|
| Autoregressive LLMs | GPT, Transformer-XL |
| Sequential VAE | Sequential VAE for trajectories |
| CoT + AR | Language models as reasoners |

Goal: survey DeepSeek's Engram paper on efficient knowledge retrieval via N-gram embeddings and hash lookup.

Context: DeepSeek Engram builds on two research lines:
- **Memory research**: Transformer FFN as KV Memory, Product Key Memory, RETRO, External Memory
- **N-gram research**: N-Grammer, Scaling Embedding Layer

Core ideas:
- **N-gram Embedding + Hash lookup**: O(1) complexity knowledge retrieval
- **Sparsity**: Key to breaking "impossible triangle" (Performance / Compute / Model Size)
- **Gating + Memory Hierarchy**: Efficient knowledge storage and retrieval
- **Relationship to MoE**: Engram complements MoE architectures

Planned follow-up:
- **Survey Engram paper**: Study the core idea of N-gram Embedding + Hash for efficient retrieval
- **Technical points**: Understand Sparsity, Gating mechanisms, Memory Hierarchy
- **Applicability**: Assess relevance to autonomous driving (scene understanding, rule retrieval, safety knowledge)
- **Implementation**: Explore adding external memory to vision backbone for efficient knowledge access
- **Comparison**: Contrast with traditional FFN as implicit memory

Source: https://www.bilibili.com/video/BV1x3zWB6EU6/

---

## Survey: GigaBrain (VLA + World Model RL)

**Status:** **NEW** (added 2026-02-16)

**Source:** https://arxiv.org/abs/2602.12099  
**Project:** https://gigabrain05m.github.io/

**TL;DR:** VLA model trained via world model-based RL with RAMP (Reinforcement leArning via world Model-conditioned Policy)

**Key Results:**
- Pre-trained on 10,000+ hours of robotic manipulation data
- RAMP achieves ~30% improvement over RECAP baseline on Laundry Folding, Box Packing, Espresso Preparation
- Reliable long-horizon execution for complex manipulation tasks

**Core Innovation:**
```
GigaBrain-0.5 (pre-trained on 10K hours robot data)
        │
        ▼
┌─────────────────┐
│  World Model    │  Video-based world model for dynamics forecasting
│  (pre-trained)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     RAMP        │  World Model-Conditioned Policy
│                 │  (RL with world model conditioning)
└────────┬────────┘
         │
         ▼
GigaBrain-0.5M* (VLA with world model RL)
```

**Relevance to Our Pipeline:**

| Our Component | GigaBrain Application |
|--------------|----------------------|
| **World Model** | Use pre-trained world model for trajectory forecasting |
| **RAMP** | Conditioning policy on world model predictions |
| **Long-horizon** | Apply to complex driving scenarios |
| **VLA** | Vision-Language-Action for driving decisions |

**Planned follow-up:**
- **Survey RAMP**: Understand world model-conditioned policy learning
- **World Model Integration**: Explore using world models (GAIA-2 style) for driving
- **RAMP for Driving**: Adapt RAMP to autonomous driving scenarios
- **Long-horizon Planning**: Use world model predictions for trajectory planning
- **Comparison**: Contrast with our current PPO approach

---

## Survey: MoE for Autonomous Driving (Driving + Parking)

**Status:** **NEW** (added 2026-02-16)

**Goal:** Survey Mixture of Experts (MoE) approaches for supporting both **driving** and **parking** scenarios in autonomous systems.

**Motivation:**
- Driving and parking require different skills
- MoE can route to specialist experts based on scenario
- Efficient: only activate relevant experts per input

**Survey Focus:**

| Scenario | Expert Specialist | Key Differences |
|----------|------------------|-----------------|
| **Highway driving** | Highway Expert | High speed, lane keeping, lane changes |
| **Urban driving** | Urban Expert | Traffic lights, pedestrians, stop signs |
| **Parking** | Parking Expert | Fine-grained control, reverse maneuvers |
| **Emergency** | Safety Expert | Quick reactions, safety margins |

**Core Questions:**
1. **Gating Design:** How to route inputs to appropriate experts?
2. **Expert Architecture:** Shared backbone vs separate specialists?
3. **Training:** Joint training vs separate pretraining + routing?
4. **Efficiency:** How many experts to activate per forward pass?
5. **Safety:** What if routing fails? Fallback mechanism?

**Related MoE Papers to Survey:**
- **Switch Transformer** (Google): Efficient routing, top-1 expert selection
- **Mixtral** (Mistral): Open-source sparse MoE
- **DeepSeek MoE**: Expert specialization patterns
- **V-MoE** (Google): Vision MoE for perception

**Planned follow-up:**
- **Survey MoE papers**: Understand routing, expert design, training dynamics
- **Gating design**: Design gating network for scenario classification
- **Expert specialization**: Define expert architectures for each scenario
- **Implementation**: Prototype MoE architecture for driving+parking
- **Evaluation**: Compare with unified model (efficiency, quality)

---

## Survey Candidate Pipeline

The following papers are queued for future survey:

1. **GigaBrain** - VLA + World Model RL (NEW, 2026-02-16)
2. **GenAD** - Generalized predictive model for driving (CVPR 2024)
3. **VAD** - Vectorized autonomous driving planning
4. **UniAD** - Unified autonomous driving framework
5. **BEVFormer** - Bird's-eye view transformer

**Selection Criteria:**
- Relevance to driving/robotics
- Novel methodology
- Open-source implementation
- Scalability to our pipeline
