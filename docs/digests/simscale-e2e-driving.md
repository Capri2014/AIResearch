# SimScale: Learning to Drive via Real-World Simulation at Scale

**CVPR 2026 | Sim-Real Co-Training | Open-Source | Scalable Simulation**

**Paper:** [arXiv:2511.23369](https://arxiv.org/abs/2511.23369) | **Code:** [github.com/OpenDriveLab/SimScale](https://github.com/OpenDriveLab/SimScale) | **Models:** [HuggingFace](https://huggingface.co/datasets/OpenDriveLab/SimScale) | **Data:** [HuggingFace](https://huggingface.co/datasets/OpenDriveLab/SimScale) + [ModelScope](https://www.modelscope.cn/datasets/OpenDriveLab/SimScale)

---

## TL;DR

SimScale tackles the long-tail problem in E2E driving via **scalable simulation + sim-real co-training**. Key insight: generate diverse reactive driving scenarios with pseudo-expert planners, then co-train on simulated + real data. Achieves **48.0 EPDMS on NAVSIM v2 navhard** (+8.6 over baseline) — the highest score in the benchmark. This directly addresses Tesla's "shadow mode" philosophy but in a fully open, research-friendly way.

---

## 1. System Decomposition

### What IS End-to-End
```
Multi-View Cameras → Encoder → Planning Head → Waypoints/Trajectory
        ↑                                           ↑
    Real OR Sim                              Differentiable
    sensor input                             planning output
```

### What IS Modular (Not End-to-End)
- **Simulation pipeline:** Pseudo-expert planners (PDM, recovery) generate scenarios — not learned end-to-end
- **Metric computation:** EPDMS evaluated separately, not part of training gradient
- **Map processing:** HD prior from nuPlan — not learned from scratch
- **Sensor rendering:** Neural rendering or game-engine based — separate from policy

### Core Architecture

| Component | Type | Notes |
|-----------|------|-------|
| **Sensor Encoder** | ResNet-34 / V2-99 |冻结预训练 backbone |
| **Planning Head** | LTF / DiffusionDrive / GTRS-Dense | Supports multiple policies |
| **Simulation Engine** | Planner-based reactive scenarios | Generates diverse edge cases |
| **Co-Training Loss** | Imitation + Reward | Both real + sim data |

**Key difference from UniAD:** No unified perception-prediction-planning stack — focuses on **data scaling via simulation** rather than architectural unification.

---

## 2. Inputs & Outputs

### Inputs (Real-World)
| Input | Shape | Temporal Context |
|-------|-------|------------------|
| **6 surround cameras** | 6×H×W×3 (typical: 224×480) | 2s history @ 2Hz |
| **HD Map** | Vectorized / rasterized | nuPlan format |
| **Agent states** | Ego + other agents | 2s history |

### Inputs (Simulation)
| Input | Shape | Notes |
|-------|-------|-------|
| **Synthetic sensor blobs** | Rendered multi-view | Same format as real |
| **Pseudo-expert trajectories** | T×2 coords | From PDM or recovery planner |
| **Reward signals** | Scalar per scenario | For RL-style training |

### Outputs
| Output | Format | Planning Horizon |
|--------|--------|-----------------|
| **Future trajectory** | T×2 coordinates (4s @ 2Hz = 8 waypoints) | 4 seconds |
| **Planning score** | EPDMS (NAVSIM v2) | Metric directly |

### Temporal Handling
- **Real data:** 2s history + 4s future (6s total at 2Hz)
- **Sim data:** Same temporal structure — enables seamless co-training
- **Simulation rounds:** 5 rounds of increasing diversity (32K-65K tokens each)

---

## 3. Training Objectives

### Primary: Imitation Learning Loss

```
L_imi = ||traj_pred - traj_expert||²
```

Supervised on both real-world and pseudo-expert simulated trajectories.

### Secondary: Reward Loss (GTRS-Dense only)

```
L_reward = -E[reward(traj)]
```

For scoring-based policies, uses simulation rewards for RL-style optimization.

### Co-Training Strategy

| Data Source | Loss Type | Purpose |
|------------|-----------|---------|
| **Real-world nuPlan** | Imitation (expert) | Keep baseline driving behavior |
| **Simulated (PDM)** | Imitation (pseudo-expert) | Generalization to diverse scenarios |
| **Simulated (Recovery)** | Imitation + Reward | Handle edge cases, long-tail |

### Key Innovation: Pseudo-Expert Generation

SimScale uses **planner-based pseudo-experts** to generate diverse scenarios:

1. **PDM (Planning-based Driver Model):** Reactive planning with collision avoidance
2. **Recovery:** Recovery behaviors for challenging situations

These pseudo-experts create diverse, realistic scenarios without human annotation cost.

### Scaling Insights (from paper)

- **Sim data scales:** Adding more simulation rounds improves performance
- **Reward signal matters:** Using rewards + imitation outperforms imitation alone
- **Real data essential:** Without real-world finetuning, sim-only policy degrades

---

## 4. Evaluation Protocol & Metrics

### Primary: NAVSIM v2 EPDMS

| Metric | Description | Higher = |
|--------|-------------|----------|
| **EPDMS** | Ego-centric Planning Distance Metric Score | Better |

### Benchmark Results (NAVSIM v2 navhard)

| Method | Backbone | Config | EPDMS | Δ vs Baseline |
|--------|----------|--------|-------|---------------|
| **SimScale GTRS-Dense** | V2-99 | rewards only | **48.0** | **+8.6** |
| **SimScale GTRS-Dense** | V2-99 | pseudo-expert | 47.7 | +5.8 |
| **SimScale GTRS-Dense** | ResNet34 | rewards only | 46.9 | +8.6 |
| **SimScale GTRS-Dense** | ResNet34 | pseudo-expert | 46.1 | +7.8 |
| **SimScale DiffusionDrive** | ResNet34 | pseudo-expert | 32.6 | +5.1 |
| **SimScale LTF** | ResNet34 | pseudo-expert | 30.3 | +6.9 |

### Benchmark Results (NAVSIM v2 navtest)

| Method | Backbone | EPDMS | Δ vs Baseline |
|--------|----------|-------|---------------|
| **SimScale GTRS-Dense** | V2-99 | **84.8** | **+0.8** |
| **SimScale GTRS-Dense** | ResNet34 | 84.6 | +2.3 |
| **SimScale DiffusionDrive** | ResNet34 | 85.9 | +1.7 |
| **SimScale LTF** | ResNet34 | 84.4 | +2.9 |

### What Makes This Different

- **Sim-real co-training:** First major work showing simulation data helps real-world E2E driving
- **Scalable pseudo-experts:** No human annotation needed for edge cases
- **Full pipeline:** Simulation generation + policy training + evaluation

---

## 5. Mapping to Tesla/Ashok Claims

### What Maps Well ✓

| Tesla Claim | SimScale Alignment |
|-------------|-------------------|
| **Shadow mode / regression testing** | ✓ Pseudo-expert scenarios enable diverse testing |
| **Long-tail handling** | ✓ Simulation generates edge cases at scale |
| **Fleet data simulation** | ✓ Scalable sim pipeline (300K+ tokens across rounds) |
| **Real-world fine-tuning** | ✓ Co-training keeps real-world performance |
| **OTA updates** | ✓ Checkpoints available for easy deployment |
| **Camera-first** | ✓ Camera-only input, no LiDAR required |

### What Doesn't Map ✗

| Tesla Claim | SimScale Gap |
|-------------|-------------|
| **Massive real fleet (1M+ cars)** | Training on nuPlan (100K clips) — still smaller scale |
| **True online learning** | Static checkpoint, no continuous fleet updates |
| **Custom hardware** | General GPU training/inference |
| **End-to-end safety layers** | No explicit fallback or rule-based safety |

### Partial Alignment (Needs Work)

| Aspect | SimScale Approach | Tesla Approach |
|--------|------------------|----------------|
| **Simulation approach** | Planner-based pseudo-experts | Neural simulation (Chauffeurnet-style) |
| **Data generation** | Scalable but bounded by sim | Unlimited fleet-generated scenarios |
| **Evaluation** | NAVSIM metrics | Internal safety metrics |
| **Edge case coverage** | Recovery + PDM scenarios | Real-world fleet edge cases |

### Key Insight

SimScale validates Tesla's intuition that **simulation + real-world co-training** solves long-tail, but does so in an **open, research-friendly way**. The pseudo-expert approach is a clever workaround for expensive human annotation — directly generates diverse scenarios from planner behaviors.

---

## 6. What to Borrow for AIResearch

### Immediately Useful

| Component | Why It Matters | Implementation |
|-----------|----------------|----------------|
| **Sim-real co-training recipe** | Addresses long-tail, improves robustness | Mix real + sim data in batches |
| **Pseudo-expert generation** | Scalable edge-case data without annotation | Use PDM + recovery planners |
| **NAVSIM v2 evaluation** | Standardized benchmark, SOTA results | Integrate NAVSIM v2 metrics |
| **GTRS-Dense planning head** | Best EPDMS (48.0) with V2-99 | Adopt scoring-based policy |

### Architecture Patterns

```python
# SimScale-style co-training
class SimRealCoTrainer:
    def __init__(self, real_dataset, sim_dataset, policy):
        self.real_loader = DataLoader(real_dataset)
        self.sim_loader = DataLoader(sim_dataset)
        self.policy = policy
    
    def training_step(self):
        # Sample from both real and sim
        real_batch = next(self.real_loader)
        sim_batch = next(self.sim_loader)
        
        # Real: pure imitation
        real_loss = self.policy.imitation_loss(real_batch)
        
        # Sim: imitation + optional reward
        sim_loss = self.policy.imitation_loss(sim_batch)
        if self.use_rewards:
            sim_loss += self.policy.reward_loss(sim_batch)
        
        # Joint optimization
        total_loss = real_loss + sim_loss
        total_loss.backward()
```

### Evaluation Pipeline to Adopt

1. **NAVSIM v2 navhard** — hardest benchmark, best for research
2. **NAVSIM v2 navtest** — standard evaluation
3. **Sim-real gap analysis** — compare sim-only vs co-trained vs real-only

### Data Scaling Recipe (from SimScale)

| Round | Sim Tokens | Purpose |
|-------|------------|---------|
| 0 | 65K | Baseline reactive scenarios |
| 1 | 55K | More diversity |
| 2 | 46K | Expanding coverage |
| 3 | 38K | Long-tail focus |
| 4 | 32K | Refinement |

### AIResearch-Specific Recommendations

| Recommendation | Priority | Rationale |
|----------------|----------|-----------|
| **Start with DiffusionDrive base** | High | Open-source, well-tested, good baseline |
| **Add sim-real co-training** | High | Addresses long-tail, proven effective |
| **Use GTRS-Dense for best metrics** | Medium | Higher EPDMS but more complex |
| **Generate pseudo-expert data** | Medium | Enables scalable edge-case coverage |
| **Evaluate on NAVSIM v2** | High | Standard benchmark, reproducible |

### Not Recommended to Borrow

- **V2-99 backbone** — heavy for on-vehicle, use ResNet-34 for iteration
- **Multi-node training setup** — only needed for 100+ GPU scale
- **Full nuPlan dataset** — start with NAVSIM subset

### Minimal Viable Implementation for AIResearch

```python
# Minimal SimScale-inspired pipeline
class MinimalSimRealPlanner(nn.Module):
    """
    Simplified: DiffusionDrive + sim-real co-training
    - Use existing DiffusionDrive (from hustvl/DiffusionDrive)
    - Add synthetic scenarios from NAVSIM/SimScale
    - Co-train on real + sim
    """
    def __init__(self):
        self.encoder = ResNet34(pretrained=True)
        self.bev = LiftSplatShoot()
        self.diffusion = TruncatedDiffusionDecoder(steps=3)
        self.head = WaypointHead(horizon=8, out_dim=2)
    
    def forward(self, images, is_sim=False):
        features = self.encoder(images)
        bev_features = self.bev(features)
        waypoints = self.diffusion(bev_features)
        return self.head(waypoints)
```

---

## 7. Citations & Links

### Primary

```bibtex
@article{tian2025simscale,
  title={SimScale: Learning to Drive via Real-World Simulation at Scale},
  author={Haochen Tian and Tianyu Li and Haochen Liu and Jiazhi Yang and Yihang Qiu and Guang Li and Junli Wang and Yinfeng Gao and Zhang Zhang and Liang Wang and Hangjun Ye and Tieniu Tan and Long Chen and Hongyang Li},
  journal={arXiv preprint arXiv:2511.23369},
  year={2025}
}
```

### Related

| Paper | Venue | Relevance |
|-------|-------|-----------|
| [UniAD: Planning-Oriented AD](https://arxiv.org/abs/2212.10156) | CVPR 2023 | Baseline unified architecture |
| [NAVSIM: Neural AD Simulation](https://github.com/autonomousvision/navsim) | - | Evaluation benchmark |
| [DiffusionDrive](https://github.com/hustvl/DiffusionDrive) | CVPR 2025 | Planning head used in SimScale |
| [GTRS: Ground Truth Robot Score](https://github.com/NVlabs/GTRS) | - | Scoring-based planning |
| [nuPlan](https://nuscenes.org/nuplan) | - | Real-world driving dataset |

### Datasets

| Dataset | Size | Purpose |
|---------|------|---------|
| [nuPlan](https://nuscenes.org/nuplan) | 100K+ clips | Real-world driving, full-cycle |
| [NAVSIM](https://github.com/autonomousvision/navsim) | 500K+ | Scalable simulation benchmark |
| [SimScale Sim Data](https://huggingface.co/datasets/OpenDriveLab/SimScale) | 300K+ tokens | Synthetic reactive scenarios |

### Resources

- **Code:** [github.com/OpenDriveLab/SimScale](https://github.com/OpenDriveLab/SimScale)
- **Models:** [HuggingFace OpenDriveLab/SimScale](https://huggingface.co/datasets/OpenDriveLab/SimScale)
- **Sim Data:** [HuggingFace](https://huggingface.co/datasets/OpenDriveLab/SimScale) / [ModelScope](https://www.modelscope.cn/datasets/OpenDriveLab/SimScale)
- **Paper:** [arXiv:2511.23369](https://arxiv.org/abs/2511.23369)

---

## Summary

1. **SimScale achieves SOTA 48.0 EPDMS on NAVSIM v2 by co-training on real nuPlan data + scalable pseudo-expert simulated scenarios**, demonstrating that simulation can meaningfully improve real-world E2E driving performance.

2. **Key innovation:** Planner-based pseudo-experts (PDM + recovery) generate diverse, reactive edge-case scenarios at scale (300K+ tokens), bypassing expensive human annotation while maintaining realism and diversity.

3. **For AIResearch:** Adopt sim-real co-training recipe with DiffusionDrive base + NAVSIM evaluation; prioritize generating diverse pseudo-expert scenarios over architectural tweaks.

---

**PR Link:** https://github.com/airesearch/autonomous-driving/pull/XXX

**3-Bullet Summary:**
- SimScale (CVPR 2026) achieves 48.0 EPDMS on NAVSIM v2 by co-training E2E planners on real nuPlan data + 300K+ pseudo-expert simulated scenarios — addressing the long-tail problem via scalable simulation.
- Core insight: planner-based pseudo-experts (PDM + recovery) generate diverse reactive driving scenarios without human annotation, enabling research-friendly "shadow mode" at scale.
- Borrow: sim-real co-training recipe + NAVSIM v2 metrics + GTRS-Dense scoring head; start with DiffusionDrive backbone for fast iteration.
