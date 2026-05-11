# Learning Beyond Gradients: Heuristic Learning as the Next Paradigm

**Source:** [Jiayi Weng - Blog Post](https://trinkle23897.github.io/learning-beyond-gradients/)  
**Author:** Jiayi Weng (Creator of EnvPool)  
**Date:** May 2026  
**GitHub:** [Trinkle23897/learning-beyond-gradients](https://github.com/Trinkle23897/learning-beyond-gradients)

---

## TL;DR

What if we stop training neural networks and start writing code instead? This blog post introduces **Heuristic Learning (HL)** — using coding agents (GPT-5.4) to iteratively write and maintain programmatic policies, instead of gradient-based training.

The results are startling:
- **Breakout:** 387 → 507 → 839 → **864** (theoretical max!)
- **MuJoCo Ant:** Reached **6000+** (comparable to Deep RL)
- **MuJoCo HalfCheetah:** **11836.7** (comparable to Deep RL)
- **VizDoom D3 Battle:** Mean **557.0** without any neural network
- **Atari57:** Median HNS **0.83** (vs PPO's 0.80, CleanRL's 0.98)

The key insight: coding agents reduce heuristic maintenance cost, making previously impractical rules worth owning. This could be the next paradigm after pretraining → RLHF → RLVR.

---

## 1. Intuition

### The Anomaly

While maintaining EnvPool, the author wanted a cheap way to test if game environments were behaving correctly. Running a neural network every time was too expensive for CI.

The question:
> Can we write cheap, reproducible heuristics that are much stronger than a random policy, and use them to drive environments into informative states?

The experiment: Use Codex (GPT-5.4) to write rule-based policies, no neural networks, iterate like software.

The results were far more surprising than expected:

| Environment | Score | Comparable to Deep RL? |
|--------------|-------|----------------------|
| Atari Breakout | 864 | ✅ Yes (max possible!) |
| MuJoCo Ant | 6000+ | ✅ Yes |
| MuJoCo HalfCheetah | 11836.7 | ✅ Yes |
| VizDoom D3 Battle | 557.0 | ✅ Yes |
| Atari57 (median) | 0.83 HNS | ⚠️ Near |

### Why This Matters

The thing being updated is **no longer just a policy function**. It became a **software system** with:
- Memory (trials, summaries, failure logs)
- Feedback channels (videos, replays, tests)
- Regression mechanisms

This felt like a new concept.

---

## 2. The Problem It Solves

### What's Broken in Current Deep RL?

| Issue | Description | Impact |
|-------|------------|--------|
| **Catastrophic Forgetting** | New weights overwrite old capabilities | Can't learn continuously |
| **Sample Inefficiency** | Need millions of env steps | Expensive |
| **Unexplainable** | Neural network = black box | Hard to debug |
| **Overfitting** | Policy learns environment loopholes | Doesn't generalize |
| **Maintenance Cost** | Can't easily fix specific failures | One-off patches pile up |

### The Expert Systems Problem

Before coding agents, expert systems and rule-based AI were abandoned because:

```
Day 1: Add one rule to fix case A
Day 2: Case B breaks
Day 3: Add if-statement
Day 4: Nobody dares delete anything
```

**The problem wasn't that heuristics were useless. The problem was that humans couldn't afford to maintain them.**

### The Paradigm Shift

Current trajectory:
1. Pretraining (next-token prediction)
2. RLHF (human feedback)
3. Large-scale RL / RLVR (verification)

The proposed next step: **Heuristic Learning**

> Anything that can be continuously iterated on starts to become solvable.

---

## 3. How It Works

### Heuristic Learning Definition

**Heuristic Learning (HL)** = Learning where the update is **direct code editing** by a coding agent, instead of gradient updates to neural network weights.

```
Deep RL:     state → action → reward �� gradient → policy weights
Heuristic:  state → action → feedback → code edit → policy code
```

### The Heuristic System (HS)

An HS is more than `policy.py`. It contains:

| Component | Description |
|-----------|-------------|
| **Programmatic Policy** | Rules, state machines, controllers, MPC |
| **State Representation** | Detectors, caches, readable variables |
| **Feedback Channels** | Tests, environment rewards, logs, videos |
| **Experiment Records** | trials.jsonl, summary.csv |
| **Memory** | Failed directions, version diffs, replays |
| **Update Mechanism** | Coding agent editing code |

### The Iteration Loop

```
┌─────────────────────────────────────────────────┐
│         Heuristic Learning Loop                │
├─────────────────────────────────────────────────┤
│  1. Probe actions + observations             │
│  2. Write state detectors                  │
│  3. Write policy                          │
│  4. Run full episodes                      │
│  5. Record trials.jsonl + summary.csv       │
│  6. Generate videos/curves               │
│  7. Inspect failure modes                 │
│  8. Edit policy / add tests              │
│  9. Simplify + run regressions           │
│  10. Loop back to 1                       │
└─────────────────────────────────────────────────┘
```

### Example: Breakout Evolution

| Trial | Score | Mechanism |
|-------|-------|-----------|
| baseline_v0 | 99 | Initial RAM intercept |
| tunnel0_v1 | 387 | No tunnel offset |
| + stuck breaker | 507 | Loop perturbation when no reward |
| + fast lead | 839 | Fast ball lead compensation |
| + late release | **864** | Release stuck offset near paddle |

Each iteration wasn't training — it was code editing.

### The Feedback Loop in Action

```
feature request 
  → agent writes code 
  → tests pass 
  → human gives feedback 
  → next patch

environment feedback / test failure / log anomaly
  → coding agent reads context
  → edits policy / test / memory
  → reruns
  → writes results back into trials/summaries
  → continues to next round
```

---

## 4. The Math

### Conceptual Comparison: Deep RL vs Heuristic Learning

| Axis | Deep RL | Heuristic Learning |
|------|--------|-------------------|
| **Policy** | Neural network parameters | Code: rules, state machines, MPC |
| **State** | Explicit observations | Explicit variables, detectors, caches |
| **Action** | Neural network forward pass | Execute code logic |
| **Feedback** | Fixed reward | Tests, environment, logs, replays |
| **Update** | Gradient descent | Direct code edits |
| **Memory** | Replay buffer (implicit) | Explicit: trials, failures, version diffs |

### Explainability

**Deep RL:** Policy = 10M parameters → hard to explain

**Heuristic Learning:**
```python
# Breakout: predict where ball will land
if vy > 0.1 and ball_y <= paddle_y:
    steps_to_paddle = max((paddle_y - ball_y) / vy, 0.0)
    intercept_x = reflect_position(ball_x + vx * steps_to_paddle)
    target_x = intercept_x + stuck_offset
```

This is **plain language**.

### Sample Efficiency Comparison

| Method | Atari57 Median HNS (@ 10M steps) |
|--------|------------------------------|
| PPO2 (OpenAI) | 0.80 |
| CleanRL EnvPool PPO | 0.98 |
| **HL (Codex, unattended)** | **0.83** |
| HL (best single run) | 1.18 |

One effective code update can jump directly to a new policy, rather than slowly climbing through learning rate tuning.

### Coupling Complexity

Defined as: *how many interdependent states, rules, tests, feedback signals, and historical constraints an update has to account for at once.*

```
Coupling Complexity = f(
    module boundaries,
    interface stability,
    test coverage,
    observability,
    rollback cost,
    state reproducibility,
    model capability,
    context length,
    memory quality,
    tool quality
)
```

Working hypotheses:
- Clearer feedback → higher coupling complexity
- Stronger models → handle more interactions
- Modularity/tests → reduce coupling
- Memory + tools → increase effective context

---

## 5. Code Examples

### 5.1 Breakout State Detector

```python
# heuristic_breakout.py - State detection from RAM/frame
def find_paddle_and_ball(ram):
    """Scan RAM bytes to find paddle and ball positions."""
    # Paddle at y=194 (fixed position)
    paddle_x = ram[78]  # direct RAM mapping
    
    # Ball: scan for brick-like colors
    ball_x, ball_y = scan_for_ball(frame)
    vx, vy = estimate_velocity(ball_x, ball_y)
    
    return paddle_x, ball_x, ball_y, vx, vy

def scan_for_ball(frame):
    """RGB segmentation for ball."""
    # Threshold for ball color (orange)
    mask = cv2.inRange(frame, BALL_LOW, BALL_HIGH)
    # Find contours
    contours, _ = cv2.findContours(mask, ...)
    # Return centroid
    return centroid(contours[0])
```

### 5.2 Breakout Policy with Stuck Detection

```python
def policy(state, config):
    paddle_x, ball_x, ball_y, vx, vy = state
    steps_since_reward = state['steps_since_reward']
    
    # Stuck loop detection
    if steps_since_reward >= config.stuck_trigger_steps:
        phase = (steps_since_reward // config.stuck_switch_steps) % 4
        if phase == 0:
            offset = +config.stuck_offset_px
        elif phase == 1:
            offset = -config.stuck_offset_px
        ...
    else:
        offset = 0.0
    
    # Predict landing point
    if vy > 0 and ball_y <= paddle_y:
        steps = (paddle_y - ball_y) / vy
        intercept_x = reflect(ball_x + vx * steps)
        target_x = intercept_x + offset
    else:
        target_x = ball_x + config.chase_lead_steps * vx
    
    # Act
    if target_x < paddle_x - deadband:
        return ACTION_LEFT
    elif target_x > paddle_x + deadband:
        return ACTION_RIGHT
    else:
        return ACTION_NOOP
```

### 5.3 Ant CPG + PD Controller

```python
# MuJoCo Ant: rhythmic gait with CPG
import numpy as np

class AntCPGPolicy:
    """Central Pattern Generator + PD Controller."""
    
    def __init__(self):
        self.phase = 0.0
        
    def step(self, obs, config):
        # Extract observations
        vx, yaw, q, dq = self.parse_obs(obs)
        
        # Phase oscillator
        phase = self.phase + config.phase_speed
        stance = phase < np.pi
        
        # Left/right anti-phase
        leg_phase = phase + (np.pi if stance else 0)
        
        # Hip + ankle PD terms
        hip_wave = config.HIP_BIAS + config.stance_scale * np.sin(leg_phase)
        action[0::2] = config.KP * (hip_wave - q[0::2])
        action[1::2] = config.KP * (config.ankle_wave - q[1::2])
        
        self.phase = phase % (2 * np.pi)
        return action
```

### 5.4 Ant with Residual MPC

```python
# Add short-horizon model planning
def ant_policy_mpc(obs, config):
    # Base CPG reflex
    base = cpg_action(phase, q, dq, obs)
    
    # Residual MPC: think ahead
    best_plan = previous_plan.copy()
    best_obj = rollout_objective(obs, best_plan)
    
    for _ in range(config.CANDIDATES - 1):
        residuals = clip(
            best_plan + np.random.normal(0, config.MPC_SIGMA, size=(HORIZON, 8)),
            -config.MPC_CLIP, config.MPC_CLIP
        )
        residuals[1:] = 0.6 * residuals[1:] + 0.4 * residuals[:-1]
        obj = rollout_objective(obs, residuals)
        if obj > best_obj:
            best_obj = obj
            best_plan = residuals
    
    # Decay previous plan
    plan[:-1] = config.PLAN_DECAY * plan[1:]
    
    return clip(base + best_plan[0], -1, 1)
```

### 5.5 VizDoom Screen CV

```python
# Visual control without neural networks
def vizdoom_cv_policy(screen, info):
    # Extract health/ammo from game info
    health = info['HEALTH']
    ammo = info['AMMO2']
    
    # Screen CV: extract objects
    enemies = find_enemies(screen)  # cv2 color threshold
    items = find_items(screen)
    
    # Behavior decomposition
    if enemies.visible:
        return aim_and_shoot(enemies)
    elif health < 30 or ammo < 10:
        return seek_supplies(items)
    else:
        return explore_quickly()
```

---

## 6. Cross-Comparison

### 6.1 Learning Paradigms

| Paradigm | Update Type | Sample Efficiency | Explainability | Continual Learning |
|----------|-----------|-----------------|---------------|------------------|
| **Pretraining** | Gradient | Low | ❌ | ❌ |
| **RLHF** | Gradient + RL | Medium | ❌ | ❌ |
| **RLVR** | Gradient + V&V | Medium-High | ❌ | ⚠️ Partial |
| **Heuristic Learning** | Code edit | **Very High** | ✅ | ✅ |

### 6.2 Deep RL vs Heuristic Learning

| Aspect | Deep RL (PPO) | Heuristic Learning |
|--------|--------------|----------------|
| **Policy** | Neural network | Code |
| **Training data** | Millions of env steps | Iterative edits |
| **Sample efficiency** | ~10M steps | ~100K steps |
| **Explainable** | ❌ Black box | ✅ Readable |
| **Catastrophic forgetting** | ❌ Yes | ✅ Write tests |
| **Regression testing** | ❌ Hard | ✅ Tests, replays |
| **Maintenance** | Retrain | Edit code |
| **Hardware** | GPU required | CPU only |

### 6.3 Results Comparison

| Environment | Deep RL (PPO) | Heuristic Learning | Gap |
|--------------|---------------|-------------------|-----|
| Breakout | ~400 | 864 | HL wins |
| Ant | ~6000 | 6146 | Tie |
| HalfCheetah | ~10000 | 11836 | HL wins |
| VizDoom D3 | N/A | 557 | - |
| Atari57 median | 0.80-0.98 | 0.83 | Near |

---

## 7. When to Use Heuristic Learning

### Decision Guide

**Use Heuristic Learning when:**
- [ ] Environment is well-understood (clear geometry/physics)
- [ ] Sample efficiency is critical
- [ ] Explainability matters
- [ ] Specific failure modes need targeted fixes
- [ ] Continuous learning without retraining
- [ ] Running on CPU (no GPU budget)
- [ ] You have a coding agent (GPT-4+, Claude, etc.)

**Stick with Deep RL when:**
- [ ] High-dimensional perception (ImageNet-level)
- [ ] Complex, unknown dynamics
- [ ] Human labels available for imitation
- [ ] You need to learn from massive data
- [ ] The environment is too complex for rules

### Hybrid: Neural + Heuristic

The most promising direction:

```
System 1 (Fast):
  - Shallow NNs for perception
  - Heuristics for safety/tests

System 2 (Slow):
  - Coding agent improving heuristics
  - Extracting trainable data from HL runs
  - Periodically updating neural network
```

Example for robotics:
- **Joint-level HL**: Safety, low-latency
- **Limb-level HL**: Gait, contact
- **Whole-body HL**: Balance
- **Task-level HL**: Long-term memory

---

## 8. Applications to Our Work

### 8.1 Current Pipeline Integration

Our current approach (GRPO + waypoint prediction):
- Neural network predicts waypoints
- RL optimizes via rewards
- No explicit rules

### 8.2 HL Integration Options

**Option A: Heuristic Failures → Training Data**
- Run HL to find failure modes
- Convert HL-generated data to training examples
- Use for offline fine-tuning

**Option B: Hybrid Safety Layer**
- Keep our neural policy
- Add heuristic sanity checks
- Override when heuristic is confident

**Option C: Full HL for Simple Tasks**
- Use HL for simpler navigation scenarios
- Neural for complex ones

**Option D: Regression Test Suite**
- Convert HL policies to tests
- Use for validation after neural updates

### 8.3 Code: Safety Check Wrapper

```python
class HybridPolicy:
    """Neural policy + heuristic safety."""
    
    def __init__(self, neural_policy, heuristic_check):
        self.neural = neural_policy
        self.heuristic = heuristic_check
    
    def act(self, obs):
        # Run heuristic check
        if self.heuristic.is_safe(obs):
            return self.neural.act(obs)
        else:
            # Fall back to safe heuristic action
            return self.heuristic.safe_action(obs)
    
    def add_failure_case(self, obs, failed_action, correct_action):
        """Add to heuristic safety rules."""
        self.heuristic.add_rule(obs, correct_action)
```

---

## 9. Honest Pros and Cons

### Pros

✅ **Sample efficiency** — One code edit >> millions of gradient steps  
✅ **Explainable** — Policies are readable code  
✅ **Regression testable** — Old capabilities become tests  
✅ **No catastrophic forgetting** — Write old rules into tests  
✅ **CPU-only** — No GPU needed  
✅ **Targeted fixes** — Fix specific failure modes with code  
✅ **Continuous iteration** — Software-like development

### Cons

❌ **Limited to code-expressible tasks** — Can't solve ImageNet  
❌ **Maintenance burden** — Still needs thoughtful design  
❌ **Requires strong coding agent** — GPT-5.4 used (not cheap)  
❌ **Coupling complexity** — System can become unmaintainable  
❌ **No neural network generalization** — Can't learn features  
❌ **Some environments need new forms** — Montezuma needs macro-actions

### When to Use

| Use Case | Recommendation |
|---------|-------------|
| Simple physics/games | ✅ HL works great |
| Complex robotics | ⚠️ Hybrid |
| Image classification | ❌ Use neural |
| Language tasks | ❌ Use neural |
| Continuous control | ⚠️ Try HL first |
| Debugging/validation | ⚠️ HL as sanity check |

---

## References

- [Learning Beyond Gradients - Blog](https://trinkle23897.github.io/learning-beyond-gradients/)
- [GitHub Repo](https://github.com/Trinkle23897/learning-beyond-gradients)
- [EnvPool](https://github.com/sail-sg/envpool)
- [Jiayi Weng CV](https://trinkle23897.github.io/cv/)

---

## Appendix A: Experiment Results Summary

### A.1 Breakout

| Version | Score | Key Mechanism |
|---------|-------|---------------|
| baseline_v0 | 99 | Initial |
| tunnel0_v1 | 387 | Basic intercept |
| + stuck breaker | 507 | Loop perturbation |
| + fast lead | 839 | Fast ball compensation |
| **final** | **864** | Late-game release |

### A.2 Ant (MuJoCo)

| Version | Score | Key Mechanism |
|---------|-------|---------------|
| ant_lr_cpgpd_v1 | 2291 | CPG + PD |
| ant_yawaxis | 2857 | Yaw feedback |
| ant_h3 | 3162 | Harmonics |
| ant_mpc_residual_v1 | 3635 | + MPC |
| **final** | **6146** | + Full MPC |

### A.3 Atari57 Results

| Metric | HL (native_obs) | HL (ram) | PPO2 | CleanRL |
|-------|-----------------|----------|------|--------|
| Median HNS | 0.32 | 0.26 | ~0.80 | ~0.98 |
| Median HNS @ 10M | 0.81 | 0.59 | 0.80 | 0.98 |
| Best single run | 1.18 | - | - | - |

### A.4 Montezuma (Counterexample)

| Version | Score | Notes |
|---------|-------|-------|
| HL | 400 | 86 macro-actions |
| Deep RL | Varies | Hard task |

**Note:** HL exposed an expressivity problem — some environments need new program forms (macro-actions, recoverable state, long-term memory).

---

## Quick Reference

| Aspect | Details |
|--------|---------|
| **Paradigm** | Code-based learning (no gradients) |
| **Agent** | Codex GPT-5.4 |
| **Key insight** | Maintenance cost is now the bottleneck |
| **Best results** | Breakout 864, Ant 6146 |
| **Limitation** | Code expressivity |
| **Future** | Neural + HL hybrid |

---

*Survey completed: 2026-05-10*
*Based on blog post by Jiayi Weng*