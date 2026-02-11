# References (seed list)

A starter set of canonical **papers / datasets / benchmarks** to anchor decisions.

> Note: this is a seed list to organize reading. We can add exact bibtex + links as we firm up priorities.

## Autonomous driving

### Logged datasets
- nuScenes
- Waymo Open Dataset (perception + motion)
- Argoverse (1/2)
- nuPlan

### Simulators / benchmarks
- CARLA (+ ScenarioRunner)

### BEV-centric perception → planning
- BEVFormer (BEV from multi-camera + temporal attention)
- BEVFusion (multi-sensor fusion into BEV)
- UniAD (unified end-to-end autonomous driving stack)
- VAD (vectorized planning / end-to-end AD variants)

### World models / video generative models for driving
- GAIA-1 (driving world model direction)

## Robotics / embodiment

### Generalist robot policies (VLA / multi-task BC)
- RT-1 / RT-2
- Open X-Embodiment / RT-X (dataset unification direction)
- Octo (generalist policy)

### Diffusion policies
- Diffusion Policy (action-sequence diffusion)

### Imitation policy baselines
- ACT (Action Chunking with Transformers)

## Sim-to-real & robustness
- Domain randomization (classic)
- System identification + dynamics randomization

## Repo-local reading
- `docs/pretrain_tech_survey.md`
- `docs/sim_training_eval.md`
