# Simulation for training & evaluation

Simulation is used for two distinct (but connected) purposes:

## 1) Training (data generation)
Simulation helps generate:
- diverse rollouts cheaply
- rare edge cases (cut-ins, jaywalkers, low friction, sensor dropouts)
- perfect supervision signals (depth, segmentation, poses, collisions)

Common techniques:
- **domain randomization** (textures, lighting, weather, dynamics)
- **system identification** (calibrate physics parameters to match real systems)
- **hybrid data** (sim pretrain → real log fine-tune)

## 2) Evaluation (scenario suites)
Simulation enables:
- deterministic regression tests
- controlled stress tests (vary one factor at a time)
- safe measurement of risky metrics

Example metrics:
- success rate / route completion
- collisions / near-collisions
- rule violations
- comfort (jerk, acceleration, steering rate)

## 3) The sim→real gap
Simulation is not the goal. Mitigation strategies include:
- photometric + sensor noise models
- dynamics randomization
- latency/jitter modeling
- validation on real logs/hardware

## Repo integration
- scenario specs live under `sim/**/scenarios/`
- metrics in `sim/common/metrics.py`
- record episodes in `sim/common/recorder.py`
- policies implement `models/policy_interface.py`
