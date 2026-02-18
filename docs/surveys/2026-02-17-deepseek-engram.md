# Survey: DeepSeek Engram & Reasoning Paper Series

**Date:** 2026-02-17  
**Status:** Complete Survey  
**Author:** OpenClaw (Pipeline Agent)

---

## TL;DR

| Paper/Technique | Key Contribution | Relevance to Driving |
|----------------|-----------------|---------------------|
| **DeepSeek-R1** | RL-induced reasoning chains | CoT reasoning for driving decisions |
| **DeepSeek-V3** | Efficient MoE architecture | Specialist routing for scenarios |
| **GRPO** | Group-based advantage estimation | Stable RL training without critic |
| **SFT→RL→RLHF** | Reasoning emergence pipeline | Bootstrapping driving policies |

---

## 1. DeepSeek-R1: Reinforcement-Induced Reasoning

### 1.1 Core Idea

DeepSeek-R1 demonstrates that **reasoning chains can emerge from RL training** without explicit CoT supervision.

```
Traditional: SFT with CoT → RL (supervised reasoning)
     ↓
DeepSeek-R1: SFT → RL (discovers reasoning) → SFT (distills reasoning)
```

### 1.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DeepSeek-R1 Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Cold Start: SFT on standard reasoning data                 │
│     → Model has basic CoT capability                            │
│                                                                 │
│  2. RL Training: GRPO on reasoning tasks                        │
│     → Model discovers better reasoning patterns                  │
│     → Reasoning chains emerge naturally                         │
│                                                                 │
│  3. Distillation: SFT on RL-generated reasoning                 │
│     → Compact model with distilled reasoning                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Findings

1. **Reasoning emerges progressively** - Longer CoT appears as training progresses
2. **GRPO is stable** - No value function needed, group-based advantages work well
3. **Distillation is effective** - Small models can inherit reasoning from large models

### 1.4 Code Pattern for Driving

```python
# DeepSeek-R1 inspired reasoning for driving decisions

class ReasoningDriver:
    """
    Driver that generates reasoning chains before acting.
    
    Inspired by DeepSeek-R1's RL-induced reasoning.
    """
    
    def __init__(self, sft_model, tokenizer):
        self.sft_model = sft_model  # Frozen SFT checkpoint
        self.tokenizer = tokenizer
        self.reasoning_steps = []
    
    def think_before_act(self, state: Dict) -> Tuple[str, np.ndarray]:
        """
        Generate reasoning chain, then predict waypoints.
        
        Args:
            state: Dict with 'image', 'speed', 'heading', etc.
            
        Returns:
            reasoning: str - reasoning chain
            waypoints: np.ndarray - [T, 3] predicted trajectory
        """
        # 1. Encode state
        state_tokens = self.encode_state(state)
        
        # 2. Generate reasoning (like DeepSeek-R1)
        reasoning_prompt = f"""\
Current state:
- Speed: {state['speed']:.2f} m/s
- Heading: {state['heading']:.2f} rad
- Nearby agents: {len(state['agents'])}
- Distance to goal: {state['distance_to_goal']:.2f} m

Let's plan this driving decision step by step:
1. Analyze current situation:
2. Predict other agents' behavior:
3. Plan trajectory:
4. Verify safety:
"""
        reasoning = self.sft_model.generate(
            reasoning_prompt,
            max_tokens=256,
            temperature=0.7,
        )
        
        # 3. Extract waypoint prediction from reasoning
        waypoints = self.extract_waypoints(reasoning, state)
        
        return reasoning, waypoints


# Alternative: Pure RL approach (like R1-Zero)
class RLReasoningDriver:
    """
    Driver trained with RL to discover reasoning.
    
    Like DeepSeek-R1-Zero: no SFT CoT, pure RL.
    """
    
    def __init__(self, config):
        self.policy = PolicyNetwork(config)
        self.value = ValueNetwork(config)  # For advantage estimation
        self.grpo = GRPO(config)  # Group Relative Policy Optimization
    
    def train_step(self, trajectories: List[Trajectory]) -> Dict:
        """
        Train on batch of driving trajectories using GRPO.
        
        Returns advantage based on driving outcomes:
        - Route completion (+)
        - Collision (--)
        - Comfort violation (-)
        - Deviation from goal (-)
        """
        # Group trajectories by scenario type
        groups = self.group_by_scenario(trajectories)
        
        # Compute group-relative advantages
        advantages = self.grpo.compute_advantages(groups)
        
        # Update policy
        loss = self.policy.update(advantages)
        
        return {'loss': loss, 'advantages': advantages}
```

---

## 2. DeepSeek-V3: Mixture of Experts for Scaling

### 2.1 Key Architecture

DeepSeek-V3 uses **sparse MoE** with expert routing:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DeepSeek-V3 MoE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: [B, D]                                                  │
│      ↓                                                          │
│  ┌───────────────────┐                                          │
│  │ Expert Router     │ → learns to route tokens to experts     │
│  │ (top-k gating)    │ → balanced load + specialization       │
│  └─────────┬─────────┘                                          │
│            ↓                                                    │
│  ┌─────────┴─────────┐                                          │
│  │   Expert 1       │  → Highway driving specialist           │
│  │   Expert 2       │  → Intersection handling               │
│  │   Expert 3       │  → Parking/lane change                 │
│  │   ...            │  → Weather adaptation                   │
│  │   Expert N       │  → Emergency response                  │
│  └─────────┬─────────┘                                          │
│            ↓                                                    │
│  Output: [B, D] (sparse combination of expert outputs)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Techniques

1. **Auxiliary loss-free load balancing** - Router learns without auxiliary losses
2. **Expert affinity routing** - Tokens with similar routing affinity grouped
3. **Shared experts** - Some experts always active for common knowledge

### 2.3 Code Pattern for Driving Scenarios

```python
class MoEDrivingPolicy(nn.Module):
    """
    Mixture of Experts for different driving scenarios.
    
    Each expert specializes in a driving regime:
    - Highway: fast, lane-keeping
    - Urban: stop signs, pedestrians
    - Parking: precise control
    - Emergency: quick reactions
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k  # Usually 2-4
        
        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            SharedExpert(self.hidden_dim) for _ in range(2)
        ])
        
        # Specialized experts (sparse)
        self.experts = nn.ModuleList([
            Expert(self.hidden_dim) for _ in range(self.num_experts)
        ])
        
        # Router (learnable gating)
        self.router = TopKRouter(
            input_dim=self.hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
        )
        
        # Final output layer
        self.output_layer = nn.Linear(self.hidden_dim, 3)  # x, y, heading
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D] fused sensor features
            
        Returns:
            waypoints: [B, T, 3]
        """
        B, D = features.shape
        
        # Route through shared experts
        shared_out = features
        for expert in self.shared_experts:
            shared_out = expert(shared_out)
        
        # Route through specialized experts
        routing_weights, expert_indices = self.router(features)
        
        # Combine expert outputs
        expert_outputs = []
        for b in range(B):
            expert_out = torch.zeros(D, device=features.device)
            for k in range(self.top_k):
                expert_idx = expert_indices[b, k]
                weight = routing_weights[b, k]
                expert_out += weight * self.experts[expert_idx](features[b])
            expert_outputs.append(expert_out)
        
        expert_out = torch.stack(expert_outputs)  # [B, D]
        
        # Combine shared + expert
        fused = shared_out + expert_out
        
        # Predict waypoints
        return self.output_layer(fused)
    
    def loss_fn(self, pred_waypoints, target_waypoints, routing_weights):
        """
        MoE-specific losses:
        1. Task loss (MSE on waypoints)
        2. Load balancing loss (encourage even expert usage)
        3. Importance loss (router quality)
        """
        task_loss = F.mse_loss(pred_waypoints, target_waypoints)
        
        # Load balancing: encourage uniform expert distribution
        expert_counts = torch.bincount(
            routing_weights.argmax(dim=1),
            num_experts=self.num_experts
        ).float()
        target_counts = torch.ones_like(expert_counts) * (B / self.num_experts)
        load_loss = F.mse_loss(expert_counts, target_counts)
        
        return task_loss + 0.01 * load_loss


class TopKRouter(nn.Module):
    """
    Router that sends each token to top-k experts.
    
    Key: Use noise for exploration, then reduce noise.
    """
    
    def __init__(self, input_dim, num_experts, top_k):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_stddev = nn.Parameter(torch.zeros(num_experts))
    
    def forward(self, x):
        logits = self.gate(x)  # [B, num_experts]
        
        # Add noise during training for load balancing
        if self.training:
            noise = torch.randn_like(logits) * self.noise_stddev
            logits = logits + noise
        
        # Top-k selection
        weights, indices = logits.topk(self.top_k, dim=1)
        
        # Softmax for weights
        weights = F.softmax(weights, dim=1)
        
        return weights, indices
```

---

## 3. GRPO: Group Relative Policy Optimization

### 3.1 DeepSeek's RL Innovation

GRPO eliminates the value function by comparing within a group:

```
Traditional PPO (requires value function):
┌─────────────────────────────────────────┐
│  Advantage = Q(s,a) - V(s)             │
│  Need: policy + value function         │
│  Risk: value function drift             │
└─────────────────────────────────────────┘

GRPO (no value function needed):
┌─────────────────────────────────────────┐
│  For each state, sample G actions       │
│  Score each action with reward function  │
│  Advantage = action_reward - mean(group)│
│  Only need: policy network              │
└─────────────────────────────────────────┘
```

### 3.2 Why GRPO Works for Driving

1. **Driving is naturally groupable** - Similar states can be grouped
2. **Reward shaping is learnable** - Group relative advantages adapt
3. **Stable training** - No value function drift

### 3.3 Code Implementation

```python
class GRPO:
    """
    Group Relative Policy Optimization.
    
    From DeepSeek-Math and DeepSeek-R1.
    """
    
    def __init__(self, policy: nn.Module, config):
        self.policy = policy
        self.config = config
        self.clip_epsilon = config.clip_epsilon
        self.entropy_coef = config.entropy_coef
        self.gamma = config.gamma  # Discount factor
        self.lam = config.lam  # GAE lambda
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute advantages using GAE + group normalization.
        
        Args:
            rewards: [B, T] - Reward at each timestep
            values: [B, T] - Value function estimates
            dones: [B, T] - Done flags
            
        Returns:
            advantages: [B, T] - Normalized advantages
        """
        advantages = []
        
        for b in range(rewards.size(0)):
            # Extract trajectory for this episode
            traj_rewards = rewards[b]
            traj_values = values[b]
            traj_dones = dones[b]
            
            # GAE
            deltas = []
            last_gae = 0
            for t in reversed(range(len(traj_rewards))):
                if traj_dones[t]:
                    last_gae = 0
                delta = traj_rewards[t] + self.gamma * last_gae - traj_values[t]
                deltas.append(delta)
                last_gae = traj_values[t] + self.gamma * self.lam * delta
            
            advantages_t = torch.tensor(list(reversed(deltas)))
            
            # Normalize within trajectory
            if advantages_t.numel() > 1:
                advantages_t = (advantages_t - advantages_t.mean()) / (
                    advantages_t.std() + 1e-8
                )
            
            advantages.append(advantages_t)
        
        return torch.stack(advantages)
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """
        PPO update with GRPO-style advantages.
        """
        # Get current policy outputs
        logits = self.policy(states)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        
        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        
        # Total loss
        loss = policy_loss - self.entropy_coef * entropy
        
        # Backward pass
        loss.backward()
        
        return {
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'ratio_mean': ratio.mean().item(),
        }


# Driving-specific GRPO with scenario grouping
class ScenarioGRPO:
    """
    GRPO with scenario-based grouping.
    
    Like DeepSeek: group similar trajectories together.
    """
    
    def __init__(self, policy, config):
        self.policy = policy
        self.grpo = GRPO(policy, config)
    
    def group_trajectories(
        self,
        trajectories: List[Trajectory]
    ) -> Dict[str, List[Trajectory]]:
        """Group trajectories by driving scenario."""
        groups = defaultdict(list)
        
        for traj in trajectories:
            # Classify scenario
            if traj.highway:
                groups['highway'].append(traj)
            elif traj.urban:
                groups['urban'].append(traj)
            elif traj.parking:
                groups['parking'].append(traj)
            else:
                groups['general'].append(traj)
        
        return groups
    
    def compute_group_advantages(
        self,
        groups: Dict[str, List[Trajectory]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute advantages relative to group mean.
        
        This is DeepSeek's key insight:
        - Within-group: compute relative ranking
        - Between-group: normalize separately
        """
        group_advantages = {}
        
        for name, trajs in groups.items():
            if len(trajs) < 2:
                # Not enough for relative comparison
                group_advantages[name] = self.grpo.compute_advantages(...)
                continue
            
            # Collect rewards
            rewards = torch.stack([t.rewards for t in trajs])
            
            # Group-relative normalization
            group_mean = rewards.mean(dim=0, keepdim=True)
            group_std = rewards.std(dim=0, keepdim=True) + 1e-8
            
            # Normalized relative to group
            relative_rewards = (rewards - group_mean) / group_std
            
            group_advantages[name] = relative_rewards
        
        return group_advantages
```

---

## 4. Combined Pipeline: What We Can Adopt

### 4.1 DeepSeek-Inspired Driving Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│            DeepSeek × Driving Pipeline (Our Adoption)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. PRETRAIN (JEPA SSL)                                         │
│     Waymo episodes → SSL encoder                                │
│     → Latent features for driving                               │
│                                                                 │
│  2. SFT (AR Decoder)                                            │
│     Features → ARDecoder + CoT                                  │
│     → Waypoint predictions with reasoning                       │
│                                                                 │
│  3. RL (GRPO)                                                   │
│     Trajectories → Group by scenario                            │
│     → Compute relative advantages                               │
│     → Update policy with PPO clipped objective                  │
│                                                                 │
│  4. MoE SPECIALIZATION                                          │
│     Highway expert, Urban expert, Parking expert                │
│     → Router learns to dispatch                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Implementation Priority

| Priority | Component | DeepSeek Source | Effort |
|----------|-----------|-----------------|--------|
| 1 | GRPO implementation | DeepSeek-R1 | Medium |
| 2 | AR Decoder + CoT | DeepSeek reasoning | Done |
| 3 | MoE specialists | DeepSeek-V3 | High |
| 4 | RL-induced reasoning | R1-Zero | Research |

### 4.3 Code Structure

```
training/
├── sft/
│   ├── ar_decoder.py          # Done
│   ├── train_waypoint_bc_cot.py # Done
│   └── sft_cot_reasoning.py    # To do: reasoning distillation
├── rl/
│   ├── grpo_waypoint.py       # To do: GRPO implementation
│   └── moe_driving.py          # To do: MoE specialists
└── unified/
    └── deepseek_pipeline.py    # To do: Combined training script
```

---

## 5. Key Takeaways

1. **GRPO eliminates value function** - Simpler RL, stable training
2. **Reasoning can emerge from RL** - R1-Zero shows CoT without supervision
3. **MoE enables specialization** - Different experts for different scenarios
4. **Distillation compresses reasoning** - Small models can reason well

## 6. Action Items

- [ ] Implement GRPO in `training/rl/grpo_waypoint.py`
- [ ] Add MoE specialist routing to policy network
- [ ] Create SFT→RL→RLHF pipeline for driving
- [ ] Experiment with RL-induced reasoning chains

---

## References

1. DeepSeek-V3: https://arxiv.org/abs/2401.02994
2. DeepSeek-R1: https://arxiv.org/abs/2501.12948
3. DeepSeek-Math (GRPO): https://arxiv.org/abs/2408.07142
