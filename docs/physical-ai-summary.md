# Physical AI summary (draft)

**Physical AI** is the study/practice of building policies that map observations (vision, proprioception, maps) and sometimes language goals into actions in the real world.

Key directions (roughly 2024–2026):

- **Vision–Language–Action (VLA)** policies: language-conditioned action policies trained on robot/driving data.
- **Diffusion policies**: generate action sequences/trajectories with strong robustness.
- **Offline RL / imitation at scale**: learn from large logged datasets, then fine-tune.
- **World models**: learn predictive models of dynamics to plan.
- **Safety + evaluation**: scenario suites, rule compliance, uncertainty estimation.

This repo focuses on *practical infrastructure*:
- scenario-driven simulation evaluation
- consistent dataset/episode format
- clean policy interface for swapping models

Next: expand `docs/references.md` with citations and add minimal baseline demos.