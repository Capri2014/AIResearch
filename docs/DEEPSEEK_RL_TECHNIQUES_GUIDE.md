# DeepSeek-R1 / V3.2 RL Design Patterns — Complete Guide

**Last Updated:** 2026-03-29  
**Status:** Active development  
**Focus:** Autonomous driving waypoint prediction with RL

---

This document explains every RL technique in our `train_deepseek_grpo.py` (v1) and `train_deepseek_grpo_v2.py` (v2) implementations, with intuition, equations, code, and comparison to alternatives.

---

## Table of Contents

1. [GRPO — Group Relative Policy Optimization](#1-grpo)
2. [Termination Reward Shaping](#2-termination-reward-shaping)
3. [Self-Correction Head](#3-self-correction-head)
4. [Self-Verification Head](#4-self-verification-head)
5. [Process + Outcome Reward System](#5-process-outcome-reward)
6. [Multi-Token Prediction (MTP)](#6-multi-token-prediction)
7. [Length-Normalized Advantages](#7-length-normalized-advantages)
8. [Dynamic Strategy Adaptation](#8-dynamic-strategy-adaptation)
9. [Curriculum Learning](#9-curriculum-learning)
10. [Self-Evolution Tracking](#10-self-evolution-tracking)
11. [Comparison Table — All Methods](#11-comparison-table)

---

<a name="1-grpo"></a>
## 1. GRPO — Group Relative Policy Optimization

### What is it?

GRPO is DeepSeek-Math's (2024) alternative to PPO for LLM reasoning. Instead of training a separate value network (critic), it uses **group-relative advantages** — comparing each response to the group's mean.

### The core problem GRPO solves

Standard PPO needs a value function `V(s)` to estimate "how good is this state?" For waypoint prediction, this means:
- Training an extra network just to estimate expected return
- Error accumulation in the value estimate
- 2x computation for every update

GRPO eliminates the value function entirely.

### How it works (intuition)

```
For each prompt (waypoint planning task):
  1. Sample G different waypoint trajectories
  2. Compute total return for each
  3. Advantage = (my return) - (group mean return)
  4. Update policy to favor higher-advantage trajectories
```

Think of it as "self-play with population": you're comparing yourself against similar versions of yourself, rather than against a fixed baseline.

### The math

**Standard PPO advantage:**
```
A_t = Q(s_t, a_t) - V(s_t)
```
Requires a trained value function `V`.

**GRPO advantage (DeepSeek-Math):**
```
A_i = r_i - mean(r_group)
```
Where `r_i` is the return of trajectory `i` in the group of size G.

No value function needed — just the rewards!

**GRPO objective:**
```
L = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]
```
Where `r(θ) = π_θ(a|s) / π_old(a|s)` is the probability ratio.

### Why group size matters

| Group Size (G) | Advantage Estimation Quality | Computation |
|---------------|------------------------------|-------------|
| G=2 | Very noisy — just 1 comparison | Fast |
| G=4 | Moderate (our v1 default) | Medium |
| G=8 | Good (DeepSeek-R1 default) | Medium |
| G=16 | Excellent (our v2 default) | Slower |

Larger groups = more stable advantage estimates = better gradient direction.

### Code (v2 implementation)

```python
# From train_deepseek_grpo_v2.py — LengthNormalizedAdvantage.compute()
def compute(rewards: torch.Tensor, group_ids: torch.Tensor,
            length_normalize: bool = True) -> torch.Tensor:
    """
    Compute GRPO group-relative advantages.
    
    Args:
        rewards: [B] per-step rewards (already shaped)
        group_ids: [B] which group each step belongs to
    
    Returns:
        advantages: [B] — positive = better than group average
    """
    advantages = torch.zeros_like(rewards)
    
    for gid in group_ids.unique():
        mask = group_ids == gid
        group_rewards = rewards[mask]
        
        if len(group_rewards) <= 1:
            advantages[mask] = 0.0
            continue
        
        # Group-relative baseline
        mean_reward = group_rewards.mean()
        std_reward = group_rewards.std() + 1e-8
        
        raw_adv = (group_rewards - mean_reward) / std_reward
        
        # Length normalization prevents short-trajectory bias
        if length_normalize:
            group_size = mask.sum().float()
            raw_adv = raw_adv / (group_size ** 0.5)
        
        advantages[mask] = raw_adv
    
    return advantages
```

### Comparison with other methods

| Method | Value Function | Group Size | Reward Type | Complexity |
|--------|---------------|------------|-------------|------------|
| PPO | Yes (critic) | 1 (on-policy) | Dense | High |
| REINFORCE | No | 1 | Episode only | Low |
| A2C/A3C | Optional | 1 (parallel) | Dense | Medium |
| **GRPO** | **No** | **G (8-16)** | **Episode** | **Medium** |
| GAE | Yes | 1 | Dense | Medium |

### When to use GRPO vs PPO

**Use GRPO when:**
- Reward signal is sparse (episodic success only)
- Computing a good value function is hard
- Sample efficiency is important (no off-policy correction needed)
- You're doing reasoning/planning tasks

**Use PPO when:**
- Reward signal is dense (every step has feedback)
- You have a pre-trained, accurate value function
- You need stable updates for very large policy networks

For our waypoint task: **GRPO wins** because we only get reward at episode end.

---

<a name="2-termination-reward-shaping"></a>
## 2. Termination Reward Shaping

### What is it?

A reward engineering technique from DeepSeek-R1: give a **large bonus** when the episode ends in success, and a smaller penalty for failure. This creates a clear learning target for the policy.

### The core problem it solves

Without termination reward shaping:
```
Episode 1: return = -5.2  (failed)
Episode 2: return = -4.8  (failed)
Episode 3: return = -6.1  (failed)
...
```
All failures look the same — the policy gets no signal about *how close* it was to success.

With termination reward shaping:
```
Episode 1: return = -5.2  + shaped = 2.3  (got 3 waypoints)
Episode 2: return = -4.8  + shaped = 88.5  (SUCCESS! +100 bonus)
Episode 3: return = -6.1  + shaped = -41.3  (fail but less bad)
```
The sparse success signal becomes loud and learnable.

### The intuition

Think of it like learning to throw a basketball:
- **Without shaping:** You only know the ball went in or missed — very hard to learn
- **With shaping:** You know how close you were (how many waypoints you hit)

DeepSeek-R1 discovered this for LLM reasoning: the sparse "correct answer" reward is the key signal that enables self-evolution.

### The math

```python
shaped_reward = raw_reward + progress_reward + termination_bonus

# Each step: progress reward
progress = (waypoint_idx + 1) / total_waypoints
shaped += progress_coef * progress  # e.g., +2 per step

# Episode end: termination bonus
if done and not truncated:  # SUCCESS
    shaped += success_reward  # e.g., +100
    shaped += per_waypoint_reward * waypoints_reached  # +5 per wp

if truncated and not done:  # FAILURE
    shaped -= success_reward * 0.3  # -30 penalty
```

### Why +100 specifically?

DeepSeek-R1's insight: the termination reward must be **large enough to dominate all other rewards combined**. If success_reward ≈ sum_of_all_step_rewards × γ, the policy learns "get to success no matter what."

For our toy waypoint env:
- ~100 steps at -0.05 per step = ~-5 raw reward
- success_reward = 100 means: one success ≈ twenty failures
- The policy learns to prioritize reaching the goal

### Code (v1 implementation)

```python
# From train_deepseek_grpo.py — TerminationRewardShaper.shape()

def shape(self, reward, info, done, trunc):
    """Shape raw reward with termination signal."""
    self.step_count += 1
    shaped = float(reward)
    
    wp_idx = info.get('current_waypoint_idx', 0)
    total_wp = info.get('total_waypoints', 20)
    
    # 1. Progress reward
    progress = wp_idx / max(total_wp, 1)
    shaped += self.progress_coef * progress  # +1.0 * progress
    
    # 2. Success termination bonus (DeepSeek-R1 key technique!)
    if done and not trunc:
        key = f"wp{wp_idx}"
        shaped += self.success_reward         # +100
        shaped += wp_idx * self.per_wp_reward  # +5 per waypoint
        if key not in self.historical:
            self.historical[key] = []
        self.historical[key].append(shaped)
    
    # 3. Failure penalty
    elif trunc and not done:
        shaped -= self.success_reward * 0.3   # -30
    
    # 4. Self-evolution bonus (see Section 10)
    if self.use_self_evolution:
        ...
    
    return shaped
```

### Comparison: reward shaping approaches

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Dense shaping** | Reward every step | Fast learning | Biased, collapses to local optimum |
| **Potential-based** | r' = r + γ*φ(s') - φ(s) | Theoretically sound | Requires defining potential function |
| **Hindsight** | Relabel failed episodes as successful | Works for discrete actions | Complex for continuous |
| **Termination shaping** | Large bonus on success | Simple, powerful | Requires careful magnitude tuning |
| **Shaped + termination** | Dense + large end bonus | Best of both | Most complex to tune |

Our v2 uses **Process + Outcome Reward Shaping** — the last row.

---

<a name="3-self-correction-head"></a>
## 3. Self-Correction Head

### What is it?

A trainable neural network that learns **when and how much to correct** its own waypoint predictions. This is inspired by DeepSeek-R1's observation that large models spontaneously develop "rethink" tokens.

### The core problem it solves

A single-pass waypoint prediction is suboptimal because:
1. The encoder has limited information at decision time
2. Trajectories can deviate from the planned path
3. No mechanism to "undo" wrong predictions

The self-correction head provides a **learned feedback loop** within the forward pass.

### How it works (intuition)

```
Step 1: Base policy predicts waypoints (8 waypoints, 3D each)
         ↓
Step 2: Self-correction head examines the prediction + encoder state
         ↓
Step 3: Decides: "should I correct?" based on confidence
         ↓
Step 4: If yes, outputs a delta to add to the base prediction
```

The head learns:
- **Confidence:** How good does the prediction look?
- **Delta:** What correction to apply if confidence is low?

### Why this is better than just training longer

Without self-correction:
- The policy must learn to predict correctly on the first try
- Errors compound through the trajectory
- Corrections require multiple RL episodes to learn

With self-correction:
- Each episode provides signal for both prediction AND correction
- The model learns to "second-guess" itself
- Errors can be corrected mid-trajectory

### The architecture

```python
class SelfCorrectionHead(nn.Module):
    """
    Confidence-based correction — learns when to revise predictions.
    
    Input: encoder features + base waypoint prediction
    Output: confidence score + correction delta
    """
    def __init__(self, encoder_dim=128, num_waypoints=16, waypoint_dim=3):
        super().__init__()
        
        # Confidence: is the current prediction trustworthy?
        self.confidence_mlp = nn.Sequential(
            nn.Linear(encoder_dim + num_waypoints * waypoint_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        # Correction: what delta to apply?
        self.correction_mlp = nn.Sequential(
            nn.Linear(encoder_dim + num_waypoints * waypoint_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_waypoints * waypoint_dim),
        )
        
        # Learnable threshold for when to correct
        self.register_parameter('trigger_threshold', nn.Parameter(torch.tensor(0.5)))
    
    def forward(self, encoder_features, base_waypoints):
        """
        Args:
            encoder_features: [B, 128] — hidden state from encoder
            base_waypoints: [B, 16, 3] — waypoint predictions from base policy
        
        Returns:
            dict with confidence, delta, should_correct flags
        """
        # Concatenate for joint decision
        wp_flat = base_waypoints.flatten(-2)  # [B, 48]
        concat = torch.cat([encoder_features, wp_flat], dim=-1)  # [B, 176]
        
        confidence = self.confidence_mlp(concat)  # [B, 1]
        correction = self.correction_mlp(concat)    # [B, 48]
        correction = correction.reshape(-1, 16, 3)  # [B, 16, 3]
        
        threshold = torch.sigmoid(self.trigger_threshold)  # Learnable [0,1]
        should_correct = (confidence < threshold).float()
        
        return {
            'confidence': confidence,      # How good does it look?
            'delta': correction,             # Correction to apply
            'should_correct': should_correct,  # Binary decision
        }
```

### How it's trained

The correction head is trained alongside the base policy via the GRPO loss. The key insight: **high reward → want lower confidence** (we want the model to correct more when it's actually doing well).

```python
def get_correction_loss(self, correction_info, rewards):
    """Train the correction head using rewards as supervision."""
    if not correction_info:
        return torch.tensor(0.0)
    
    confidence = correction_info['confidence']  # [B, 1]
    
    # Normalize rewards to [0, 1]
    reward_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    reward_norm = torch.sigmoid(reward_norm * 3)  # Map to [0, 1]
    
    # High reward → want low confidence (we should correct more)
    # Low reward → want high confidence (we were already right)
    target_confidence = 1.0 - reward_norm.unsqueeze(-1)
    
    return F.mse_loss(confidence, target_confidence)
```

### Comparison: self-correction approaches

| Method | When | How | Complexity |
|--------|------|-----|------------|
| **ReAct-style** | After action | Think again | Prompt-based |
| **Tree search** | Before action | Explore branches | High |
| **Beam search** | At generation | Keep best K | Medium |
| **Self-correction head** | During forward pass | Learned delta | Low |
| **Refinement loop** | After prediction | Iterate | Medium |

Self-correction head is the most efficient for real-time inference.

---

<a name="4-self-verification-head"></a>
## 4. Self-Verification Head

### What is it?

An extension of self-correction that predicts **quality scores** for each waypoint prediction. This is DeepSeek-R1's "self-verification" applied to waypoint prediction.

### The core problem it solves

With self-correction alone, the model only knows "correct" or "incorrect." With verification, it knows **which waypoint is wrong** and **how wrong**.

### How it works (intuition)

```
1. Predict waypoints: [wp0, wp1, wp2, wp3, wp4, wp5, wp6, wp7]
                                    ↓
2. Verifier scores each: [0.9, 0.85, 0.3, 0.4, 0.7, 0.8, 0.9, 0.95]
                                    ↓
3. Focus correction on lowest-scored waypoints (wp2, wp3)
```

This is analogous to a teacher grading each answer, not just pass/fail.

### Code

```python
class WaypointVerifierHead(nn.Module):
    """
    Trainable verifier that judges waypoint prediction quality.
    
    This is the "reflection" mechanism from DeepSeek-R1:
    - After predicting waypoints, the verifier scores how good they are
    - Low verification score → trigger correction
    - Trained using rewards as supervision signal
    """
    def __init__(self, encoder_dim=128, num_waypoints=16, waypoint_dim=3):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # Overall quality score
        total_dim = encoder_dim + num_waypoints * waypoint_dim
        self.verifier_net = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # [0, 1]
        )
        
        # Per-waypoint scores (which waypoint is most likely wrong?)
        self.per_wp_verifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_waypoints),
        )
    
    def forward(self, encoder_features, waypoints):
        """
        Returns:
            quality_score: [B, 1] — overall quality of prediction
            per_waypoint_scores: [B, num_waypoints] — score per waypoint
            should_verify: [B, 1] — whether to trigger verification step
        """
        wp_flat = waypoints.flatten(-2)
        concat = torch.cat([encoder_features, wp_flat], dim=-1)
        
        quality_score = self.verifier_net(concat)  # [B, 1]
        per_wp_scores = self.per_wp_verifier(concat)  # [B, 16]
        
        should_verify = (quality_score < 0.7).float()
        
        return {
            'quality_score': quality_score,
            'per_waypoint_scores': per_wp_scores,
            'should_verify': should_verify,
        }
    
    def compute_verification_loss(self, ver_out, rewards):
        """
        Train verifier: high reward → high quality expected.
        """
        quality = ver_out['quality_score']
        reward_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        reward_norm = torch.sigmoid(reward_norm * 2)  # Map to [0, 1]
        
        # High reward → expect high quality
        loss = F.mse_loss(quality, reward_norm.unsqueeze(-1))
        return loss
```

---

<a name="5-process-outcome-reward"></a>
## 5. Process + Outcome Reward System

### What is it?

A two-tier reward system where:
- **Process reward:** Given at each step based on immediate progress
- **Outcome reward:** Large bonus on episode completion

This is the v2 improvement over v1's simple termination shaping.

### The problem with pure termination reward

With only termination reward, all intermediate steps look the same. The policy doesn't know if it's making progress or regressing until the episode ends.

### The solution: dense + sparse

```python
class ProcessOutcomeRewardShaper:
    def shape(self, reward, info, done, trunc, prev_pos=None, target_wp=None):
        shaped = float(reward)
        
        # ---- PROCESS REWARD (each step) ----
        # Progress reward
        wp_idx = info.get('current_waypoint_idx', 0)
        total_wp = info.get('total_waypoints', 20)
        progress = (wp_idx + 1) / max(total_wp, 1)
        shaped += self.per_waypoint_reward * progress  # +2 per step
        
        # Step efficiency penalty
        shaped += self.step_penalty  # -0.05 per step
        
        # Approach bonus (getting closer)
        if prev_pos is not None and target_wp is not None:
            curr_pos = info.get('position', prev_pos)
            prev_dist = np.linalg.norm(prev_pos - target_wp[:2])
            curr_dist = np.linalg.norm(curr_pos - target_wp[:2])
            if curr_dist < prev_dist:
                shaped += self.approach_bonus  # +0.3
            elif curr_dist > prev_dist * 1.1:
                shaped += self.wrong_direction_penalty  # -0.5
        
        # ---- OUTCOME REWARD (episode end) ----
        if done and not trunc:  # SUCCESS
            shaped += self.success_reward  # +100
            # Efficiency bonus: fewer steps is better
            efficiency = (max_steps - step_count) / max_steps
            shaped += self.success_reward * 0.2 * efficiency
        
        return shaped
```

### Why this is better

| Reward Type | Signal | Learning Speed | Bias |
|-------------|--------|--------------|------|
| Pure termination | Sparse (end only) | Slow | Low |
| Dense shaping | Every step | Fast | High (local optima) |
| **Process + Outcome** | Every step + end | Fast | **Low** |

Process + Outcome gives you the best of both: fast learning without the bias of pure dense shaping.

---

<a name="6-multi-token-prediction"></a>
## 6. Multi-Token Prediction (MTP)

### What is it?

Instead of predicting one step at a time, predict the next N waypoints simultaneously. This is DeepSeek-V3's multi-token prediction objective.

### The core problem it solves

Standard auto-regressive prediction has error accumulation:
```
Step 1: predict wp0 → small error
Step 2: predict wp1 using wp0 → error compounds
Step 3: predict wp2 using wp1 → error grows
...
```

With MTP, you predict multiple waypoints in a single forward pass, reducing error propagation.

### How it works (intuition)

```
Encoder features
       ↓
Head 0: predict [wp0, wp1, ..., wp15]  ← base prediction
       ↓
Head 1: predict [wp0, wp1, ..., wp15] conditioned on Head 0
       ↓
Head 2: predict [wp0, wp1, ..., wp15] conditioned on Head 0 + Head 1
```

Each head predicts the full trajectory, but later heads get to "see" earlier predictions and refine.

### The architecture

```python
class MultiTokenPredictionHead(nn.Module):
    """
    Predicts multiple waypoint horizons simultaneously.
    
    DeepSeek-V3: "MTP helps the model pre-plan by predicting multiple 
    tokens at once, which reduces error accumulation."
    """
    def __init__(self, encoder_dim=128, num_waypoints=16,
                 waypoint_dim=3, num_ahead=3):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.num_ahead = num_ahead
        
        # MTP heads for different prediction horizons
        self.mtp_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim + i * num_waypoints * waypoint_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_waypoints * waypoint_dim),
            )
            for i in range(num_ahead)  # 3 heads
        ])
    
    def forward(self, encoder_features, base_waypoints):
        """
        Returns list of N horizon predictions:
            predictions[i] = [B, 16, 3] for horizon i+1
        """
        predictions = []
        
        for i, head in enumerate(self.mtp_heads):
            if i == 0:
                concat = encoder_features
            else:
                # Condition on previous predictions
                prev_preds = torch.cat([p.flatten(-2) for p in predictions[:i]], dim=-1)
                concat = torch.cat([encoder_features, prev_preds], dim=-1)
            
            pred = head(concat)
            pred = pred.reshape(-1, self.num_waypoints, self.waypoint_dim)
            predictions.append(pred)
        
        return predictions  # [pred_1, pred_2, pred_3]
```

### Comparison: prediction strategies

| Method | Predictions per Forward | Error Accumulation | Compute |
|--------|----------------------|-------------------|---------|
| Auto-regressive | 1 | High | Linear |
| **MTP (N=3)** | 3 | Medium | ~3x |
| Direct multi-step | N | Low | N× larger |
| Hierarchical | Variable | Medium | Variable |

---

<a name="7-length-normalized-advantages"></a>
## 7. Length-Normalized Advantages

### What is it?

Advantage normalization by trajectory length to prevent bias toward short trajectories.

### The problem it solves

Standard GRPO can favor short trajectories because they accumulate less total (possibly negative) reward:

```
Trajectory A: 50 steps, total reward = -2.5  (fast but imperfect)
Trajectory B: 100 steps, total reward = -5.0  (slow but accurate)
Trajectory C: 20 steps, total reward = -1.0   (short, so best)
```

Without normalization, short trajectories always look better because they accumulate less negative reward.

### The solution

```python
# Length-normalized GRPO advantage
raw_adv = (group_rewards - mean_reward) / std_reward
length_normalized_adv = raw_adv / (group_size ** 0.5)
```

By dividing by √(group_size), we account for the fact that longer trajectories have more chances to accumulate reward. Now:
```
Trajectory A: 50 steps, normalized_adv = -2.5 / √50 = -0.35
Trajectory B: 100 steps, normalized_adv = -5.0 / √100 = -0.50
Trajectory C: 20 steps, normalized_adv = -1.0 / √20 = -0.22
```

The advantage still favors C (shortest), but the difference is proportional to efficiency, not just length.

### Why 0.5 power?

The √ comes from the variance scaling of sum of random variables:
- Var(sum of n i.i.d. variables) ∝ n
- Std(sum) ∝ √n
- So we divide by √n to normalize

---

<a name="8-dynamic-strategy-adaptation"></a>
## 8. Dynamic Strategy Adaptation

### What is it?

Different correction strategies for different scenarios — the policy adapts its behavior based on context.

### How it works

```python
class DynamicCorrectionStrategy:
    """
    Different correction strategies based on scenario:
    - SAFE: conservative (near final waypoints)
    - NORMAL: standard corrections
    - AGGRESSIVE: large corrections (open road)
    - RECOVERY: backtracking (wrong direction)
    """
    SAFE, NORMAL, AGGRESSIVE, RECOVERY = 0, 1, 2, 3
    
    @staticmethod
    def detect_mode(info, encoder_features):
        """Detect correction strategy from environment info."""
        wp_idx = info.get('current_waypoint_idx', 0)
        total_wp = info.get('total_waypoints', 20)
        
        if wp_idx < 1:
            return DynamicCorrectionStrategy.RECOVERY
        if wp_idx >= total_wp - 2:
            return DynamicCorrectionStrategy.SAFE
        if wp_idx <= 2:
            return DynamicCorrectionStrategy.AGGRESSIVE
        
        return DynamicCorrectionStrategy.NORMAL
    
    @staticmethod
    def get_correction_scale(mode):
        """Get correction magnitude scale for each mode."""
        scales = {
            DynamicCorrectionStrategy.SAFE: 0.3,       # Conservative
            DynamicCorrectionStrategy.NORMAL: 1.0,     # Standard
            DynamicCorrectionStrategy.AGGRESSIVE: 2.0,  # Large corrections
            DynamicCorrectionStrategy.RECOVERY: 0.5,    # Backtracking
        }
        return scales.get(mode, 1.0)
```

### When each mode triggers

| Mode | Trigger Condition | Correction Scale | Why |
|------|------------------|-------------------|-----|
| RECOVERY | waypoint_idx < 1 | 0.5× | Already wrong, be careful |
| SAFE | waypoint_idx >= total - 2 | 0.3× | Almost done, don't risk it |
| NORMAL | 2 < idx < total - 2 | 1.0× | Standard behavior |
| AGGRESSIVE | waypoint_idx <= 2 | 2.0× | Lots of room to correct |

---

<a name="9-curriculum-learning"></a>
## 9. Curriculum Learning

### What is it?

Start with easy tasks (few waypoints, slow speed), gradually increase difficulty. This is a best practice for RL in robotics.

### The problem it solves

Starting with hard tasks (20 waypoints at max speed) is inefficient because:
- Most episodes fail immediately
- No learning signal for the majority of state space
- High variance in early training

With curriculum:
- Early: 3 waypoints at 30% speed → high success rate → good learning
- Late: 20 waypoints at 100% speed → full difficulty → specialization

### Implementation

```python
class CurriculumScheduler:
    """
    Progressive curriculum: start easy, increase difficulty.
    
    For waypoint env:
    - Early: fewer waypoints, shorter horizon, slower speed
    - Late: more waypoints, longer horizon, faster speed
    """
    def __init__(self, max_waypoints=20, max_speed=5.0):
        self.max_waypoints = max_waypoints
        self.max_speed = max_speed
        self.step = 0
    
    def get_difficulty(self):
        """
        Returns difficulty parameters for current step.
        """
        progress = min(self.step / 2000, 1.0)  # 2000 steps to full difficulty
        
        return {
            'num_waypoints': max(3, int(3 + progress * (self.max_waypoints - 3))),
            'speed_factor': 0.3 + 0.7 * progress,   # 0.3 → 1.0
            'noise_std': 0.1 * (1 - progress),       # Less noise as you improve
            'episode_length_factor': 0.5 + 0.5 * progress,
        }
    
    def step_update(self):
        self.step += 1
```

### Curriculum schedule

| Step | Waypoints | Speed | Noise |
|------|-----------|-------|-------|
| 0 | 3 | 30% | 0.1 |
| 500 | 7 | 47% | 0.07 |
| 1000 | 11 | 65% | 0.05 |
| 1500 | 16 | 85% | 0.02 |
| 2000+ | 20 | 100% | 0.0 |

---

<a name="10-self-evolution-tracking"></a>
## 10. Self-Evolution Tracking

### What is it?

A reward bonus for beating your own historical performance. This is DeepSeek-R1's key technique for continuous improvement.

### The core problem it solves

Standard RL only compares trajectories to each other. It doesn't track *improvement over time*. A policy can plateau without the algorithm knowing.

### How it works

```python
class TerminationRewardShaper:
    def __init__(self, ...):
        self.historical = {}  # Per scenario: rolling reward history
        self.evolution_bonus_coef = 0.5  # How much bonus for beating history
    
    def shape(self, reward, info, done, trunc):
        shaped = float(reward)
        key = f"wp{wp_idx}"  # Scenario key
        
        if done and not trunc:  # SUCCESS
            shaped += self.success_reward
            if key not in self.historical:
                self.historical[key] = []
            self.historical[key].append(shaped)
        
        # Self-evolution bonus: reward for beating your own average
        if self.use_self_evolution and key in self.historical:
            hist = self.historical[key]
            if len(hist) >= 5:  # Need minimum history
                rolling_avg = np.mean(hist[-20:])  # Last 20 episodes
                if shaped > rolling_avg:
                    bonus = (shaped - rolling_avg) * self.evolution_bonus_coef
                    shaped += bonus
                    # "You beat your own average — here's extra reward!"
        
        return shaped
```

### Why this is powerful

1. **Prevents plateau:** If the policy stops improving, the bonus disappears
2. **Encourages consistency:** High variance gets penalized relative to the rolling average
3. **Scenario-specific:** Different waypoint configurations have different baselines
4. **No extra computation:** Just track history, no extra forward passes

---

<a name="11-comparison-table"></a>
## 11. Comparison Table — All Methods

### RL Algorithms

| Method | Value Function | Off-Policy | Sample Efficiency | Stability |
|--------|---------------|------------|-------------------|-----------|
| PPO | Required | No | Medium | High |
| TRPO | Required | No | Medium | Very High |
| A2C | Optional | No | Low | Medium |
| DQN | Required | Yes | Medium | Medium |
| SAC | Required | Yes | High | High |
| **GRPO** | **No** | **No** | **High** | **High** |
| REINFORCE | No | No | Low | Low |

### Reward Shaping Methods

| Method | Signal Density | Bias | Convergence | Complexity |
|--------|---------------|------|-------------|------------|
| Sparse | Very sparse | Low | Slow | None |
| Dense | Very dense | High | Fast | Low |
| Potential-based | Variable | Low | Good | Medium |
| Hindsight | Dense | Low | Good | High |
| **Termination** | **Sparse + end** | **Low** | **Fast** | **Low** |
| **Process+Outcome** | **Dense + end** | **Low** | **Fastest** | **Medium** |

### Self-Correction Methods

| Method | When | How | Latency | Complexity |
|--------|------|-----|---------|------------|
| None | — | — | 0ms | 0 |
| ReAct | After action | Think again | High | Prompt |
| Tree search | Before action | Explore | Very high | High |
| Self-correction head | During forward | Learned delta | ~1ms | Low |
| Verification head | During forward | Quality score | ~1ms | Low |

---

## Quick Reference: Which Technique to Use?

| Scenario | Recommended Techniques |
|----------|----------------------|
| Sparse rewards only | GRPO + Termination Reward |
| Dense rewards available | PPO + dense shaping |
| Long-horizon planning | GRPO + MTP + Curriculum |
| Need stability | GRPO + length-normalization |
| Want maximum performance | All v2 techniques |
| Real-time inference needed | Self-correction head (no extra passes) |
| Offline RL | GRPO with off-policy correction |

---

## Glossary

| Term | Definition |
|------|------------|
| **GRPO** | Group Relative Policy Optimization — RL without value function |
| **Advantage** | How much better is this action than average? |
| **Return** | Sum of discounted rewards for an episode |
| **Shaping** | Modifying rewards to make learning easier |
| **MTP** | Multi-Token Prediction — predict multiple steps at once |
| **Self-evolution** | Bonus for beating your own historical performance |
| **Curriculum** | Progressive difficulty from easy to hard |
| **Clip epsilon** | How much the policy can change per update (PPO/GRPO) |
| **Entropy bonus** | Encourages exploration by rewarding diverse actions |

---

## References

- DeepSeek-R1 (arXiv:2501.12948) — Reinforcement learning for reasoning
- DeepSeek-Math (arXiv:2408.07142) — GRPO original paper
- DeepSeek-V3 (arXiv:2412.15115) — Multi-token prediction, auxiliary-loss-free MoE
- DeepSeek-V3.2 (2025) — Thinking in tool-use, agent training
- PPO (Schulman et al. 2017) — Proximal Policy Optimization

---

*Last updated: 2026-03-29 by OpenClaw agent*