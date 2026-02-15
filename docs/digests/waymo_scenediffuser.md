# Waymo SceneDiffuser: Efficient and Controllable Driving Simulation Initialization and Rollout

**arXiv**: [2412.12129](https://arxiv.org/abs/2412.12129)  
**Venue**: NeurIPS 2024  
**Authors**: Chiyu Max Jiang, Yijing Bai, Andre Cornman, Christopher Davis, Xiukun Huang, Hong Jeon, Sakshum Kulshrestha, John Lambert, Shuangyu Li, Xuanyu Zhou, Carlos Fuertes, Chang Yuan, Mingxing Tan, Yin Zhou, Dragomir Anguelov (Waymo)

## TL;DR

SceneDiffuser applies **diffusion priors** to traffic simulation, addressing two key stages:
1. **Scene initialization**: generating realistic initial traffic layouts
2. **Scene rollout**: closed-loop agent behavior simulation

Key innovations: **amortized diffusion** (16x faster per-step inference), **generalized hard constraints** for controllability, and **LLM-guided scene generation**. Achieves top open-loop and best closed-loop performance among diffusion models on Waymo Open Sim Agents Challenge.

---

## Motivation

Realistic interactive scene simulation is critical for AV development—data augmentation, scenario coverage, and policy testing all benefit from high-fidelity simulation. Traditional game-engine-based simulators (CARLA, LGSVL) suffer from:
- **Domain gap** to real perception
- **Limited realism** in agent behaviors
- **High engineering cost** for diverse scenarios

Diffusion models offer **multimodal, realistic generation** but face challenges in simulation contexts:
- Controllability at inference time
- Maintaining realism in closed-loop (agents react to each other)
- Inference efficiency (diffusion requires many denoising steps per rollout step)

---

## Method

### Core: Scene-level Diffusion Prior

Treats entire traffic scene (all agent trajectories + map) as a single state to be denoised. Unlike per-agent models, this captures **interactions** between agents.

### 1. Amortized Diffusion for Simulation

Standard diffusion: many inference steps (e.g., 50-100) per generated future.

Amortized diffusion: **train a network to predict multiple future steps at once**, then apply only a few refinement steps per rollout step. Result: **16x reduction in inference cost per step** while maintaining realism.

Key idea: instead of learning a single-step transition, learn to directly output a full future trajectory, then lightly refine as the simulation unfolds.

### 2. Generalized Hard Constraints

At inference, want to **condition on specific elements** (e.g., "agent A follows this path", "this area is blocked"). Naive classifier-free guidance struggles with hard constraints.

Solution: introduce **hard constraint layers** that explicitly enforce conditions during denoising:
- For agent trajectories: copy-paste the constrained agent's path into the latent at each step
- For map constraints: mask/unmask specific regions

This is simpler than learned constraint mechanisms and works reliably.

### 3. Language-Constrained Scene Generation

Use an **LLM (few-shot prompted)** to generate natural language scene descriptions, then use those as additional conditioning for the diffusion model. Enables "generate a busy intersection with a jaywalking pedestrian" style prompts.

---

## Results

Evaluated on **Waymo Open Sim Agents Challenge**:
- **Open-loop**: top performance among all submissions
- **Closed-loop**: best performance among diffusion-based methods
- **Scaling**: larger models (more compute) significantly improve realism—scaling law holds for simulation fidelity

### Key metrics
- **Scenario fidelity**: human ratings of realism
- **Agent compliance**: do simulated agents follow traffic rules?
- **Interaction realism**: do multi-agent interactions look natural?

---

## Relation to Our Pipeline

### Where it fits
- **Simulation / data augmentation**: SceneDiffuser could generate diverse scenarios to augment Waymo data for our SSL pretraining or BC fine-tuning.
- **Closed-loop eval**: Similar to our CARLA/ScenarioRunner eval path, but learned simulation vs. rule-based.

### Differences from our approach
- **SceneDiffuser** is a **learned simulator** (generative model of scenes)
- Our approach: **real Waymo data → SSL encoder → BC policy → CARLA eval**
- Potential hybrid: use SceneDiffuser to generate synthetic scenarios, evaluate our learned policy in those contexts

### Technical overlaps
- **Multi-agent interaction modeling**: relevant to how we encode other agents' futures
- **Diffusion for trajectories**: our BC learns deterministic waypoints; diffusion could model multimodal trajectory distributions

---

## Action Items for Us

1. **Monitor**: Waymax (Waymo's simulation platform) is also open-source—potential alternative to CARLA for closed-loop eval
2. **Explore**: Could SceneDiffuser-style synthetic data augmentation help when real Waymo data is scarce?
3. **Read**: The paper details on amortized diffusion—could inspire efficient multi-step prediction in our waypoint policy

---

## Repo / Resources

- **Paper**: https://arxiv.org/abs/2412.12129
- **Waymo Open Dataset**: https://waymo.com/open
- **Waymax Simulator**: referenced in Waymo research (NeurIPS 2023)
