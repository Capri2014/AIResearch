# DriveArena: A Closed-loop Generative Simulation Platform for Autonomous Driving

**Paper**: [arXiv:2408.00415](https://arxiv.org/abs/2408.00415)  
**Project Page**: [pjlab-adg.github.io/DriveArena](https://pjlab-adg.github.io/DriveArena/)  
**GitHub**: [PJLab-ADG/DriveArena](https://github.com/PJLab-ADG/DriveArena)  
**Authors**: Xuemeng Yang, Licheng Wen, Yukai Ma, Jianbiao Mei, Xin Li, Tiantian Wei, Wenjie Lei, Daocheng Fu, Pinlong Cai, Min Dou, Botian Shi, Liang He, Yong Liu, Yu Qiao (Shanghai AI Lab / PJLab)

## TL;DR

DriveArena is a **closed-loop generative simulation platform** for autonomous driving from Shanghai AI Lab (PJLab). It combines:
- **WorldDreamer**: generative world model for realistic scene synthesis
- **Driving agent evaluation**: both open-loop and closed-loop benchmarks
- **Modular architecture**: supports UniAD, VAD, and other vision-based agents

Key differentiator vs CARLA: **data-driven generation** (trained on nuScenes/nuPlan) rather than handcrafted assets.

---

## Motivation

Existing simulators (CARLA, LGSVL, SUMO) rely on:
- **Manual asset creation** — expensive, domain gap to real data
- **Rule-based traffic** — limited realism in agent behaviors
- **Fixed maps** — can't easily scale to new cities

DriveArena addresses this by training a **generative world model** on real data (nuScenes, nuPlan) to produce realistic, closed-loop simulation.

---

## Architecture

### WorldDreamer (Generative World Model)
- Trained on nuScenes + nuPlan datasets
- Generates realistic traffic scenes from latent queries
- Supports video autoregression (DreamForge updates)

### Traffic Manager
- Controls all agent movements dynamically
- Supports configurable scenarios

### Agent Interface
- Plug-and-play for vision-based agents: **UniAD**, **VAD**
- Both open-loop (perception metrics) and closed-loop (driving metrics) evaluation

### Evaluation Metrics
- **Open-loop**: NC (NuScenes Detection Score), DAC (Detection Average Precision), EP (Edge Perception), TTC (Time-to-Collision), C (Collision), PDMS (Planning Metric Score)
- **Closed-loop**: PDMS, RC (Route Completion), ADS (Agent Disengagement Score)

---

## Results (from Leaderboard)

### Open-loop (nuScenes)
| Agent | Env | NC | DAC | EP | TTC | C | PDMS |
|-------|-----|-----|-----|-----|-----|-----|------|
| Human | GT | 1.00 | 1.00 | 1.00 | 0.98 | 0.75 | 0.95 |
| UniAD | nuScenes | 0.99 | 0.99 | 0.91 | 0.95 | 0.85 | 0.91 |
| UniAD | DriveArena | 0.79 | 0.94 | 0.74 | 0.77 | 0.75 | 0.64 |

### Closed-loop
| Agent | Route | PDMS | RC | ADS |
|-------|-------|------|----|----|
| UniAD | sing_route_1 | 0.76 | 0.17 | 0.17 |
| UniAD | boston_route_1 | 0.50 | 0.09 | 0.04 |

**Key insight**: Performance drops in DriveArena vs GT — indicates sim-to-real gap exists; generative simulation still imperfect.

---

## Relation to Our Pipeline

### Where it fits
- **Alternative to CARLA**: Could use DriveArena for closed-loop eval (vs our CARLA/ScenarioRunner plan)
- **Data-driven**: No manual asset creation needed — world model generates scenes from real data
- **Agent eval**: Provides standardized metrics for vision-based E2E driving agents

### Differences from our approach
- **CARLA/ScenarioRunner**: Handcrafted maps, rule-based traffic, open-source
- **DriveArena**: Data-driven world model, proprietary (but Apache 2.0 license), requires nuScenes/nuPlan data
- Our current plan: Waymo data → SSL → BC → CARLA eval

### Potential reuse
- Use DriveArena as **additional eval benchmark** to compare with CARLA results
- Could generate diverse scenarios for data augmentation (like SceneDiffuser)
- Watch for sim-to-real gap: their UniAD numbers drop significantly in sim vs GT

---

## Action Items for Us

1. **Monitor**: Track DriveArena development — newer versions may reduce sim-to-real gap
2. **Compare**: Run our BC policy in both CARLA and DriveArena (if accessible) for benchmark comparison
3. **Explore**: WorldDreamer for scenario generation (complements SceneDiffuser)
4. **Consider**: DriveArena as alternative if CARLA proves insufficient for vision-based E2E eval

---

## Repo / Resources

- **Paper**: https://arxiv.org/abs/2408.00415
- **GitHub**: https://github.com/PJLab-ADG/DriveArena
- **Project Page**: https://pjlab-adg.github.io/DriveArena/
- **Google Group**: https://groups.google.com/g/drivearena
- **License**: Apache 2.0
