# DriveAgent-R1: VLM-Based E2E Driving with Active Perception — Digest

**Source:** arXiv:2507.20879 (v2, September 2025)  
**Paper:** https://arxiv.org/abs/2507.20879  
**Authors:** Weicheng Zheng, Xiaofei Mao, Nanfei Ye, et al. (Shanghai Qi Zhi Institute, LiAuto, Tsinghua)

---

## TL;DR

- **DriveAgent-R1** is the first E2E driving agent with **active perception** — it proactively invokes tools (RoI inspection, depth estimation, 3D detection) to gather visual evidence when uncertain, grounding decisions in verifiable perception.
- Introduces a **hybrid-thinking framework** that adaptively switches between fast text-only reasoning (simple scenes) and tool-augmented visual reasoning (complex scenes) — mimicking human driver cognition.
- Trained via a novel **three-stage progressive strategy**: SFT → Forced Contrastive Mode RL (FCM-RL) → Adaptive Mode Selection RL (AMS-RL).
- With only **3B parameters**, achieves performance competitive with GPT-5 and human drivers on long-tail driving scenarios while maintaining deployment-friendly efficiency.
- Addresses key limitations of passive VLM-based driving: shortcut tendency (relying on text cues over visual evidence) and computational inefficiency from processing redundant multi-view data.

---

## System Decomposition: E2E vs Modular

DriveAgent-R1 is a **truly end-to-end** system in the VLM sense — a single VLM (Qwen2.5-VL-3B base) processes multimodal inputs and outputs driving intentions directly. However, it has modular components:

| Component | Type | Function |
|-----------|------|----------|
| **Vision Encoder** (Qwen2.5-VL) | Frozen then fine-tuned | Encodes multi-view camera images |
| **Language Model** (3B) | Fine-tuned | Generates thoughts, tool calls, actions |
| **Vision Toolkit** (4 tools) | External functions | Active perception capabilities |
| **Meta-action Head** | Implicit in LM output | Predicts 8-second intention sequence |

**Key architectural insight:** Unlike modular pipelines (perception → prediction → planning), DriveAgent-R1 uses a **unified VLM** that can invoke tools mid-reasoning — this is a form of "tool-augmented E2E" rather than traditional sensor-to-action E2E.

---

## Inputs/Outputs + Temporal Context

### Inputs
- **Visual**: Front-view camera image (I₀) — initial visual context
- **Textual**: Vehicle speed + navigation instructions (T₀)
- **History**: Optional 5-second image memory pool for retrieving historical frames

### Outputs
- **8-second meta-action sequence**: 4 actions at 2-second intervals
- Each action = (velocity_token, trajectory_token):
  - Velocity: {Accelerate, Keep Speed, Decelerate, Stop}
  - Trajectory: {Straight, Right Turn, Left Turn}

### Temporal Context Handling
- **Single-frame input** (per decision step) — no explicit temporal recurrence
- **5-second memory pool** for historical frames (can retrieve via tools)
- **Multi-turn reasoning** (up to K steps) in tool-based mode — interleaves thought generation with tool execution

---

## Training Objectives

### Three-Stage Progressive Training

**Stage 1: Dual-Mode SFT**
- Cold-start the model with format/semantic understanding of both thinking modes
- Data: 4K CoT annotations (2K text-only, 2K tool-necessary)
- Pipeline: Partition scenes by tool necessity → Qwen2.5-VL-72B generates annotations → judge model filters

**Stage 2: Forced Contrastive Mode RL (FCM-RL)**
- Algorithm: **MP-GRPO** (Mode-Partitioned GRPO)
- Forces generation of G/2 responses per mode → unified reward normalization
- Creates intra-mode AND inter-mode contrastive learning signals
- **Reward**: R = R_acc + R_fmt
  - R_acc: weighted Levenshtein distance vs ground-truth actions
  - R_fmt: format consistency penalty

**Stage 3: Adaptive Mode Selection RL (AMS-RL)**
- Uses native GRPO to learn optimal mode selection
- Agent generates <think_text> or <think_tool> token itself
- **Additional reward**: R_tool — contrastive mechanism rewarding impactful tool use only when it outperforms text-only baseline

### Pre-training: DriveAlign
- 530K VQA pairs for autonomous-driving domain alignment
- Categories: scene description, traffic entity recognition, target localization, traffic commonsense
- Purpose: Combat VLM "shortcut tendency" — tendency to rely on text cues over visual evidence

---

## Evaluation Protocol + Metrics + Datasets

### Datasets
- **Drive-Internal**: 35K video clips, long-tail scenarios (road works, animal crossings, 40+ situation types)
- **nuScenes** (validation): 2K samples for cross-dataset generalization

### Metrics
- **Meta-action accuracy**: Weighted Levenshtein distance (position-action weights to combat action imbalance)
- **Mode selection accuracy**: % of scenes where correct mode is chosen
- **Inference latency**: Compared against passive approaches

### Key Results
- 3B DriveAgent-R1 achieves **competitive performance with GPT-5** and human drivers
- Significantly reduces inference latency vs passive VLM approaches
- Ablations confirm decisions are **grounded in visual evidence**, not textual shortcuts

---

## Tesla/Ashok Claims: What Maps and What Doesn't

### Maps Well ✅
| Tesla/Ashok Claim | DriveAgent-R1 Alignment |
|-------------------|-------------------------|
| **Camera-first** | VLM-only (no LiDAR/radar) — pure camera input |
| **Long-tail handling** | Trained on Drive-Internal (long-tail scenarios) — active perception specifically addresses uncertainty |
| **Regression testing** | RL training with outcome-based rewards functions as continuous regression against driving metrics |
| **Human-like reasoning** | Hybrid thinking mimics human driver cognition — intuitive for simple scenes, deliberate for complex |
| **Foundation model approach** | Uses Qwen2.5-VL (pretrained VLM) as foundation |

### Doesn't Map ❌
| Gap | Analysis |
|-----|----------|
| **No end-to-end motion planning** | Predicts high-level meta-actions (intent), not continuous trajectories |
| **No closed-loop evaluation** | Reports open-loop accuracy only — no CARLA/nuscenes closed-loop |
| **Single-frame input** | No temporal recurrence in base model — relies on memory pool for history |
| **No explicit safety validation** | No mention of rule-based safety layers or redundancy |
| **RL training requires ground-truth** | Outcome-based RL needs labeled data — not pure self-supervised |

---

## What to Borrow for AIResearch

### ✅ Highly Relevant

1. **Active Perception Toolkit**
   - RoI Inspection (zoom-in on regions of interest)
   - Depth Estimation (3D spatial awareness)
   - 3D Object Detection (open-vocabulary)
   - **For AIResearch**: Add similar toolkit to waypoint predictor for active uncertainty resolution

2. **Hybrid Thinking Framework**
   - Adaptive mode switching based on scene complexity
   - **For AIResearch**: Apply to waypoint head — use fast direct prediction for simple cases, engage more reasoning for complex scenarios

3. **MP-GRPO Training**
   - Mode-partitioned GRPO for balanced exploration
   - Contrastive reward for effective tool use
   - **For AIResearch**: Use similar RL setup for waypoint training with scene-complexity-conditioned rewards

4. **DriveAlign Pre-training**
   - 530K VQA pairs for driving domain alignment
   - Addresses VLM shortcut tendency
   - **For AIResearch**: Build similar VQA dataset for driving to improve visual grounding

5. **Evaluation Metrics**
   - Weighted Levenshtein for sequential actions
   - Position-action weights for class imbalance
   - **For AIResearch**: Adopt for waypoint sequence evaluation

### ⚠️ Considerations

- **Closed-loop gap**: No CARLA closed-loop results — recommend adding for AIResearch evaluation
- **Temporal modeling**: Single-frame input is a limitation — consider adding temporal encoder
- **Compute for tools**: Tool calls add latency — benchmark before deploying

---

## Action Items for AIResearch

- [ ] Adapt active perception toolkit for waypoint head (depth + detection on uncertain regions)
- [ ] Implement hybrid-thinking switch: direct waypoint prediction vs tool-augmented reasoning
- [ ] Build driving VQA dataset for visual grounding (similar to DriveAlign)
- [ ] Implement MP-GRPO or similar for waypoint RL training
- [ ] Add closed-loop CARLA evaluation for comprehensive benchmarking

---

## Citations

> "We pioneer an active perception framework for high-level driving planning. The agent proactively invokes a vision toolkit to ground decisions in visual evidence, enhancing reasoning and reliability." — DriveAgent-R1 (2025)

> "Human driving, by contrast, is an inherently active process of resolving uncertainty. Drivers will check blind spots or look again at a confusing traffic signal, demonstrating the ability to 'actively perceive,' which is the cornerstone of safe driving." — DriveAgent-R1 (2025)

> "Hybrid-Thinking framework that adaptively balances efficient text-only reasoning with robust, tool-augmented visual analysis." — DriveAgent-R1 (2025)

---

**PR:** <!-- https://github.com/openclaw/openclaw/pull/XXX -->

**Summary:**
- DriveAgent-R1 (2025) introduces **active perception** for E2E driving — VLM proactively invokes tools (RoI zoom, depth, 3D detection) to gather visual evidence when uncertain
- **Hybrid-thinking framework** switches between fast text-only reasoning (simple scenes) and tool-augmented reasoning (complex scenes), mimicking human driver cognition
- Three-stage training (SFT → FCM-RL → AMS-RL) with MP-GRPO enables balanced exploration and adaptive mode selection
- 3B parameters achieve GPT-5-competitive performance on long-tail driving while being deployment-friendly
- Key takeaways for AIResearch: active perception toolkit, hybrid mode switching, MP-GRPO training, DriveAlign visual grounding
