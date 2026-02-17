# Survey: GRPO, MoE, and Autoregressive Methods for Autonomous Driving

**Date:** 2026-02-16  
**Status:** Complete Survey

---

## TL;DR

| Topic | Key Finding | Action Item |
|-------|-------------|------------|
| **GRPO** | DeepSeek's GRPO outperforms PPO on reasoning tasks with group-based advantage estimation | Implement GRPO for driving RL |
| **MoE** | Sparse Mixture of Experts enables specialist routing for different driving scenarios | Design MoE for driving+parking |
| **Autoregressive** | AR decoders improve sequential consistency but add latency | AR for planning, parallel for control |
| **Combined** | AR + CoT + MoE can create specialist reasoning systems | Prototype combined architecture |

---

## 1. GRPO: Group Relative Policy Optimization

### 1.1 What is GRPO?

**GRPO (Group Relative Policy Optimization)** is a reinforcement learning algorithm introduced by DeepSeek that eliminates the need for a value function by using group-based relative advantages.

### 1.2 Key Innovation

```
Traditional PPO:
┌─────────────────────────────────────────────────┐
│  • Requires value function (critic)            │
│  • Computes advantage: A = Q(s,a) - V(s)       │
│  • Two networks: policy + value                 │
│  • Can be unstable with large models           │
└─────────────────────────────────────────────────┘

GRPO:
┌─────────────────────────────────────────────────┐
│  • No value function needed                    │
│  • Advantage from group comparison             │
│  • Sample multiple actions per state          │
│  • Relative ranking within group               │
│  • Only one network: policy                    │
└─────────────────────────────────────────────────┘
```

### 1.3 GRPO Algorithm

```python
class GRPO:
    """
    Group Relative Policy Optimization.
    
    Key idea: For each state s, sample a group of actions {a_1, ..., a_G}
    Compute rewards for all actions
    Compute advantage as relative performance within group
    
    Advantages:
    - No value function needed (memory efficient)
    - Group relative advantage is more stable
    - Scales well to large models
    """
    
    def __init__(self, model, config):
        self.model = model
        self.group_size = config.group_size  # e.g., 8
        self.beta = config.beta  # KL penalty coefficient
    
    def compute_advantages(self, states, actions_group, rewards_group):
        """
        Compute group-relative advantages.
        
        For each state:
        1. Sample G actions
        2. Get rewards for all G actions
        3. Compute mean and std of rewards in group
        4. Advantage = (reward - group_mean) / group_std
        """
        advantages = []
        
        for i in range(len(states)):
            group_rewards = rewards_group[i]  # [G]
            
            # Relative advantage within group
            mean_r = np.mean(group_rewards)
            std_r = np.std(group_rewards) + 1e-8
            
            # Normalized relative reward
            group_adv = (group_rewards - mean_r) / std_r
            
            advantages.append(group_adv)
        
        return np.array(advantages)
    
    def update(self, states, actions_group, rewards_group):
        """
        Perform GRPO update step.
        """
        advantages = self.compute_advantages(states, actions_group, rewards_group)
        
        # Policy gradient with relative advantages
        for g in range(self.group_size):
            action = actions_group[:, g]  # [B]
            adv = advantages[:, g]  # [B]
            
            # Compute log prob ratio
            log_probs = self.model.get_log_prob(states, action)
            
            # PPO-style clipped objective
            # ... (standard PPO clipping with GRPO advantages)
        
        return {"policy_loss": ..., "kl": ...}
```

### 1.4 GRPO vs PPO Comparison

| Aspect | PPO | GRPO |
|--------|-----|------|
| **Value Function** | Required | Not needed |
| **Parameters** | Policy + Critic | Policy only |
| **Sample Efficiency** | Moderate | Higher (group sampling) |
| **Stability** | Can be unstable | More stable |
| **Memory** | 2x model size | 1x model size |
| **Scaling** | Good | Excellent for large models |
| **Best For** | Simple tasks | Reasoning tasks |

### 1.5 GRPO for Autonomous Driving

```python
class GRPOForDriving:
    """
    GRPO adapted for autonomous driving waypoint prediction.
    
    Driving-specific considerations:
    - Multi-modal action space (continuous waypoints)
    - Safety constraints
    - Hierarchical decisions
    """
    
    def __init__(self, config):
        self.model = WaypointModel(config)
        self.group_size = 8  # Sample 8 trajectory proposals
        self.safety_threshold = 0.9  # Safety filter
    
    def sample_trajectory_group(self, state, obs):
        """
        Sample a group of trajectory proposals.
        
        Each proposal is a sequence of waypoints.
        """
        trajectories = []
        
        for g in range(self.group_size):
            # Add noise to base trajectory for diversity
            base = self.model.get_base_trajectory(obs)
            noise = np.random.randn(*base.shape) * 0.5
            traj = base + noise
            
            # Safety filter
            if self.check_safety(traj):
                trajectories.append(traj)
        
        return trajectories
    
    def evaluate_trajectories(self, obs, trajectories):
        """
        Evaluate each trajectory with driving-specific reward.
        """
        rewards = []
        
        for traj in trajectories:
            reward = 0.0
            
            # Progress reward (distance to goal)
            reward += self.progress_reward(traj)
            
            # Comfort reward (low acceleration, smooth turns)
            reward += self.comfort_reward(traj)
            
            # Safety reward (distance to obstacles)
            reward += self.safety_reward(traj, obs)
            
            # Rule compliance (traffic lights, lanes)
            reward += self.rule_reward(traj, obs)
            
            rewards.append(reward)
        
        return np.array(rewards)
```

### 1.6 GRPO Implementation Plan

```python
# File: training/rl/grpo_waypoint.py

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np


class GRPOConfig:
    """GRPO Configuration for waypoint prediction."""
    
    def __init__(self):
        self.group_size = 8  # Number of trajectories per group
        self.gamma = 0.99  # Discount factor
        self.lam = 0.95  # GAE lambda (if used)
        self.clip_ratio = 0.2  # PPO clipping
        self.learning_rate = 3e-4
        self.kl_coef = 0.01  # KL divergence penalty
        self.entropy_coef = 0.01  # Entropy bonus
        self.safety_threshold = 0.9
        self.hidden_dim = 128
        self.waypoint_dim = 3  # x, y, heading
        self.horizon_steps = 16


class GRPOWaypointModel(nn.Module):
    """
    Waypoint prediction model for GRPO.
    
    Outputs trajectory proposals for group sampling.
    """
    
    def __init__(self, config: GRPOConfig):
        super().__init__()
        self.config = config
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(256, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        
        # Waypoint head (predicts mean trajectory)
        self.waypoint_head = nn.Linear(
            config.hidden_dim,
            config.horizon_steps * config.waypoint_dim
        )
        
        # Log std (learned, broadcast to all waypoints)
        self.log_std = nn.Parameter(torch.zeros(
            config.horizon_steps, config.waypoint_dim
        ))
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict trajectory parameters.
        
        Returns:
            mean: [B, H, 3] mean waypoints
            std: [B, H, 3] standard deviation (from log_std)
        """
        enc = self.encoder(z)  # [B, hidden_dim]
        mean = self.waypoint_head(enc)  # [B, H*3]
        mean = mean.view(-1, self.config.horizon_steps, self.config.waypoint_dim)
        
        std = torch.exp(self.log_std)  # [H, 3]
        std = std.unsqueeze(0).expand(mean.shape[0], -1, -1)
        
        return mean, std
    
    def sample_trajectory(self, z: torch.Tensor) -> torch.Tensor:
        """
        Sample a single trajectory from the policy.
        """
        mean, std = self.forward(z)
        
        # Sample from Gaussian
        noise = torch.randn_like(mean)
        traj = mean + std * noise
        
        return traj
    
    def sample_trajectory_group(
        self,
        z: torch.Tensor,
        group_size: int
    ) -> torch.Tensor:
        """
        Sample a group of trajectories for GRPO.
        
        Returns:
            trajectories: [G, B, H, 3]
        """
        mean, std = self.forward(z)  # [B, H, 3]
        
        group = []
        for _ in range(group_size):
            noise = torch.randn_like(mean)
            traj = mean + std * noise
            group.append(traj)
        
        return torch.stack(group, dim=0)  # [G, B, H, 3]


class GRPOTrainer:
    """
    GRPO Trainer for waypoint prediction.
    
    Implements group-relative advantage estimation.
    """
    
    def __init__(self, model: GRPOWaypointModel, config: GRPOConfig):
        self.model = model
        self.config = config
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
    
    def compute_group_advantages(
        self,
        rewards_group: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute group-relative advantages.
        
        Args:
            rewards_group: [G, B] rewards for each trajectory in group
            
        Returns:
            advantages: [G, B] normalized relative advantages
        """
        G, B = rewards_group.shape
        
        # Compute group statistics
        mean_r = rewards_group.mean(dim=0, keepdim=True)  # [1, B]
        std_r = rewards_group.std(dim=0, keepdim=True) + 1e-8  # [1, B]
        
        # Normalized relative reward = advantage
        advantages = (rewards_group - mean_r) / std_r  # [G, B]
        
        return advantages
    
    def update(
        self,
        z: torch.Tensor,
        rewards_group: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform GRPO update step.
        """
        self.model.train()
        
        # Get trajectory group
        trajectories = self.model.sample_trajectory_group(z, self.config.group_size)
        
        # Compute group-relative advantages
        advantages = self.compute_group_advantages(rewards_group)
        
        # Flatten for policy update
        traj_flat = trajectories.view(self.config.group_size, -1)
        adv_flat = advantages.view(-1)
        
        # Get log probabilities
        mean, std = self.model.forward(z)
        
        log_probs = []
        for g in range(self.config.group_size):
            traj_g = trajectories[g]
            diff = (traj_g - mean) / (std + 1e-8)
            log_prob = -0.5 * (diff ** 2) - torch.log(std + 1e-8)
            log_prob = log_prob.sum(dim=(1, 2))
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs, dim=0)  # [G, B]
        
        # PPO-style clipped objective
        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * adv_flat.unsqueeze(0)
        surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * adv_flat.unsqueeze(0)
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus
        entropy = (-log_probs).mean()
        kl = (log_probs - log_probs.detach()).mean()
        
        loss = policy_loss - self.config.entropy_coef * entropy + self.config.kl_coef * kl
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "kl": kl.item(),
            "mean_reward": rewards_group.mean().item(),
        }


def train_grpo_waypoint(config: GRPOConfig):
    """Main training loop for GRPO waypoint prediction."""
    model = GRPOWaypointModel(config)
    trainer = GRPOTrainer(model, config)
    
    for epoch in range(1000):
        z = ...  # [B, D] encoded state
        
        trajectories = model.sample_trajectory_group(z, config.group_size)
        rewards = evaluate_trajectories(z, trajectories)  # [G, B]
        
        metrics = trainer.update(z, rewards)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss={metrics['policy_loss']:.4f}")
    
    return model
```

---

## 2. Mixture of Experts (MoE)

### 2.1 What is MoE?

**Mixture of Experts (MoE)** is a model architecture that uses a gating mechanism to route inputs to different expert sub-networks. Only a subset of experts is activated per forward pass, enabling sparse computation.

### 2.2 MoE Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MoE Architecture                              │
├─────────────────────────────────────────────────────────────────┤
│  Input: [B, D] features                                         │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Gating Network                              │    │
│  │   Input: [B, D] → Output: [B, E] (E = number experts) │    │
│  │   Selects top-k experts per input                        │    │
│  └───────────────────────────┬─────────────────────────────┘    │
│                              │                                    │
│              ┌───────────────┼───────────────┐                  │
│              │               │               │                   │
│              ▼               ▼               ▼                   │
│        ┌─────────┐    ┌─────────┐    ┌─────────┐              │
│        │ Expert1 │    │ Expert2 │    │ Expert3 │              │
│        │ Highway │    │  Urban  │    │ Parking │              │
│        │ Driving │    │ Driving │    │         │              │
│        └────┬────┘    └────┬────┘    └────┬────┘              │
│             │               │               │                   │
│             └───────────────┼───────────────┘                   │
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Output Combination                           │    │
│  │   weighted_sum = Σ (gate_i * expert_i(output))         │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 MoE Components

```python
class MoEConfig:
    """Configuration for MoE."""
    
    def __init__(self):
        self.num_experts = 8
        self.top_k = 2
        self.expert_hidden_dim = 256
        self.gate_hidden_dim = 128
        self.noise_std = 1.0


class GatingNetwork(nn.Module):
    """Gating network that routes inputs to experts."""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        self.gate = nn.Sequential(
            nn.Linear(config.input_dim, config.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gate_hidden_dim, config.num_experts),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        
        if self.training:
            noise = torch.randn_like(logits) * self.config.noise_std
            logits = logits + noise
        
        top_k_values, top_k_indices = torch.topk(logits, k=self.config.top_k, dim=-1)
        
        gate_weights = torch.zeros_like(logits)
        gate_weights.scatter_(-1, top_k_indices, top_k_values)
        
        gate_weights = F.softmax(gate_weights, dim=-1)
        
        importance = gate_weights.mean(dim=0)
        load_loss = torch.var(importance)
        
        return gate_weights, load_loss


class Expert(nn.Module):
    """Individual expert network."""
    
    def __init__(self, config: MoEConfig, expert_type: str = "shared"):
        super().__init__()
        self.config = config
        self.expert_type = expert_type
        
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, config.expert_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.expert_hidden_dim, config.output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MoELayer(nn.Module):
    """Complete MoE layer with gating and experts."""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        self.experts = nn.ModuleList([
            Expert(config) for _ in range(config.num_experts)
        ])
        
        self.gate = GatingNetwork(config)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        gate_weights, load_loss = self.gate(x)
        
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        output = torch.bmm(gate_weights.unsqueeze(1), expert_outputs)
        output = output.squeeze(1)
        
        info = {
            "gate_weights": gate_weights,
            "load_loss": load_loss,
            "top_experts": gate_weights.argmax(dim=-1),
        }
        
        return output, info
```

### 2.4 MoE for Driving + Parking

```python
class MoEForDriving(nn.Module):
    """
    MoE for autonomous driving with scenario-specific experts.
    
    Experts:
    - Highway Expert: High-speed lane keeping, lane changes
    - Urban Expert: Traffic lights, pedestrians, stop signs
    - Parking Expert: Fine-grained control, reverse maneuvers
    - Emergency Expert: Quick reactions, safety margins
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        self.shared_backbone = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        self.moe = MoELayer(config)
        
        self.waypoint_head = nn.Linear(128, 48)
        self.control_head = nn.Linear(128, 3)
    
    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        features = self.shared_backbone(images)
        moe_out, routing_info = self.moe(features)
        
        waypoints = self.waypoint_head(moe_out)
        control = self.control_head(moe_out)
        
        return {
            "waypoints": waypoints,
            "control": control,
            "routing_info": routing_info,
        }
```

### 2.5 MoE Comparison

| Aspect | Unified Model | MoE |
|--------|--------------|-----|
| **Parameters** | All shared | Sparse |
| **Specialization** | Limited | Experts can specialize |
| **Inference Speed** | Fast | Slower routing overhead |
| **Training Stability** | Higher | Lower (load balancing) |
| **Scalability** | Limited | Excellent |

---

## 3. Autoregressive Methods for Waypoints

### 3.1 AR vs Parallel Decoding

| Aspect | Parallel (Current) | Autoregressive |
|--------|-------------------|----------------|
| **Output** | All at once | One at a time |
| **Speed** | O(1) forward | O(T) forwards |
| **Consistency** | May not respect order | Sequential consistency |
| **Error Propagation** | None | Can accumulate |
| **Training** | Simple MSE | Teacher forcing |

### 3.2 AR Decoder Implementation

```python
class ARDecoder(nn.Module):
    """
    Autoregressive decoder for waypoint prediction.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.wp_embed = nn.Embedding(1000, config.hidden_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.max_waypoints, config.hidden_dim) * 0.1
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=8,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        self.x_head = nn.Linear(config.hidden_dim, 1)
        self.y_head = nn.Linear(config.hidden_dim, 1)
        self.heading_head = nn.Linear(config.hidden_dim, 1)
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        waypoints: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.training:
            return self._teacher_forcing(encoder_out, waypoints)
        else:
            return self._autoregressive_generate(encoder_out)
    
    def _teacher_forcing(self, encoder_out, waypoints):
        B, T, _ = waypoints.shape
        
        wp_emb = self.wp_embed(waypoints)
        wp_emb = wp_emb + self.pos_embed[:, :T, :]
        
        memory = encoder_out.unsqueeze(1).expand(-1, T, -1)
        decoder_out = self.decoder(wp_emb, memory)
        
        x_pred = self.x_head(decoder_out).squeeze(-1)
        y_pred = self.y_head(decoder_out).squeeze(-1)
        heading_pred = self.heading_head(decoder_out).squeeze(-1)
        
        return {
            "waypoints": torch.stack([x_pred, y_pred, heading_pred], dim=-1),
        }
    
    def _autoregressive_generate(self, encoder_out, max_steps=16):
        B = encoder_out.shape[0]
        generated = []
        
        memory = encoder_out.unsqueeze(1)
        current_wp = torch.zeros(B, 1, device=encoder_out.device, dtype=torch.long)
        
        for t in range(max_steps):
            wp_emb = self.wp_embed(current_wp)
            wp_emb = wp_emb + self.pos_embed[:, t:t+1, :]
            
            decoder_out = self.decoder(wp_emb, memory)
            
            x = self.x_head(decoder_out).squeeze(-1)
            y = self.y_head(decoder_out).squeeze(-1)
            heading = self.heading_head(decoder_out).squeeze(-1)
            
            next_wp = torch.stack([x, y, heading], dim=-1)
            generated.append(next_wp)
            
            current_wp = self._discretize(next_wp)
        
        waypoints = torch.cat(generated, dim=1)
        return {"waypoints": waypoints}
```

### 3.3 AR Recommendation

| Use Case | Recommendation |
|----------|---------------|
| **Real-time control** | Keep parallel decoder (speed critical) |
| **Planning tasks** | AR decoder acceptable |
| **Complex maneuvers** | AR + CoT |
| **Training** | AR decoder useful for learning sequential structure |

---

## 4. Recommendations

### 4.1 Priority Order

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 1 | Run CoT training (existing infrastructure) | Low | High |
| 2 | Implement GRPO | Medium | High |
| 3 | Survey MoE (design phase) | Low | Medium |
| 4 | AR Decoder upgrade (training only) | Medium | Medium |

### 4.2 Implementation Roadmap

```
Phase 1 (Week 1-2):
├── Run CoT training with existing infrastructure
├── Collect baseline metrics
└── Implement GRPO prototype

Phase 2 (Week 3-4):
├── Complete GRPO implementation
├── Compare GRPO vs PPO
└── Design MoE architecture

Phase 3 (Week 5-6):
├── Prototype MoE for driving+parking
├── Implement AR decoder for training
└── Evaluate combined approaches
```

### 4.3 Success Metrics

| Metric | Target |
|--------|--------|
| CoT training ADE improvement | +5% |
| GRPO vs PPO performance | Equal or better |
| MoE specialization | >80% routing accuracy |
| AR decoder quality | Comparable to parallel |

---

## 5. Related Papers

| Paper | Citation | Relevance |
|-------|----------|-----------|
| GRPO | DeepSeek, 2024 | RL algorithm |
| Switch Transformer | Google, 2021 | MoE routing |
| Mixtral | Mistral, 2023 | Open-source MoE |
| GAIA-2 | Wayve, 2025 | World model |
| GigaBrain | arXiv:2602.12099 | VLA + World Model RL |

---

## References

- DeepSeek GRPO Paper (2024)
- Switch Transformer (Google, 2021)
- Mixtral (Mistral, 2023)
- GigaBrain: VLA + World Model RL (arXiv:2602.12099)
