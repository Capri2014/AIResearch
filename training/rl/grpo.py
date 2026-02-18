"""
GRPO (Group Relative Policy Optimization) Implementation
======================================================

GRPO is a reinforcement learning algorithm from DeepSeek that eliminates
the need for a value function by using group-based relative advantages.

Key Features:
- No value function needed (simpler, more stable)
- Group-based advantage estimation
- Sample multiple actions per state
- Relative ranking within group

Reference: DeepSeek-Math (arXiv:2408.07142)

Usage:
    from training.rl.grpo import GRPO, GRPOTrainer
    
    grpo = GRPO(policy, config)
    advantages = grpo.compute_advantages(rewards, group_ids)
    loss = grpo.update(states, actions, old_log_probs, advantages)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GRPOConfig:
    """
    Configuration for GRPO algorithm.
    
    Attributes:
        - policy: The policy network to optimize
        - optimizer: Optimizer for policy parameters
        - clip_epsilon: PPO clipping parameter
        - entropy_coef: Entropy bonus coefficient
        - batch_size: Number of samples per batch
        - group_size: Number of samples per group (for relative advantages)
        - update_epochs: Number of gradient steps per batch
        - max_grad_norm: Gradient clipping threshold
        - temperature: Softmax temperature for action selection
        - sample_temperature: Higher = more exploration
    """
    # Core
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    batch_size: int = 64
    group_size: int = 4  # Samples per group for relative comparison
    update_epochs: int = 4
    max_grad_norm: float = 0.5
    
    # Sampling
    temperature: float = 1.0
    sample_temperature: float = 1.0
    
    # KL divergence (optional, for regularization)
    use_kl: bool = True
    kl_target: float = 0.01
    kl_horizon: int = 10
    
    # Advantage estimation
    advantage_normalize: bool = True
    advantage_clip: Optional[float] = None  # Clip advantages
    
    # Logging
    log_interval: int = 100
    
    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.clip_epsilon <= 1.0, "clip_epsilon must be in (0, 1]"
        assert self.group_size >= 2, "group_size must be at least 2"
        assert 0 <= self.entropy_coef <= 1.0, "entropy_coef must be in [0, 1]"


# ============================================================================
# Core GRPO Algorithm
# ============================================================================

class GRPO:
    """
    Group Relative Policy Optimization.
    
    GRPO eliminates the need for a value function by computing advantages
    relative to a group of samples at each state.
    
    Algorithm Overview:
    ```
    For each update:
        1. Sample groups of trajectories
        2. Compute rewards for each trajectory
        3. Compute group-relative advantages
        4. Update policy with PPO-style clipped objective
    ```
    
    Advantages over PPO:
    - No value function needed (simpler)
    - More stable training
    - Better for sparse reward environments
    
    Limitations:
    - Requires grouping (natural grouping or arbitrary)
    - May need more samples per update
    """
    
    def __init__(self, policy: nn.Module, config: GRPOConfig):
        self.policy = policy
        self.config = config
    
    @torch.no_grad()
    def sample(
        self,
        states: torch.Tensor,
        num_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy.
        
        Args:
            states: [B, state_dim] states
            num_samples: Number of action samples per state
            
        Returns:
            actions: [B, num_samples, action_dim] sampled actions
            log_probs: [B, num_samples] log probability of each action
            entropies: [B, num_samples] entropy of action distribution
        """
        B = states.size(0)
        
        # Get action distribution from policy
        action_dist = self.policy.get_action_distribution(states)
        
        # Sample actions
        actions = action_dist.sample((num_samples,))  # [num_samples, B, action_dim]
        actions = actions.permute(1, 0, 2)  # [B, num_samples, action_dim]
        
        # Compute log probabilities
        log_probs = action_dist.log_prob(actions)  # [B, num_samples]
        
        # Compute entropies
        entropies = action_dist.entropy()  # [B] or [B, num_samples]
        if entropies.dim() == 1:
            entropies = entropies.unsqueeze(1).expand(-1, num_samples)
        
        return actions, log_probs, entropies
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute group-relative advantages.
        
        For each group, the advantage is computed as:
            A_i = r_i - mean(r_group)
            
        This makes the advantages relative within each group, eliminating
        the need for a baseline/value function.
        
        Args:
            rewards: [B] or [B, T] rewards per sample
            group_ids: [B] group assignment for each sample
            
        Returns:
            advantages: [B] or [B, T] normalized advantages
        """
        if rewards.dim() == 2:
            # Per-timestep advantages
            advantages = []
            for t in range(rewards.size(1)):
                adv_t = self._compute_group_advantages(rewards[:, t], group_ids)
                advantages.append(adv_t)
            return torch.stack(advantages, dim=1)
        else:
            # Per-sample advantages
            return self._compute_group_advantages(rewards, group_ids)
    
    def _compute_group_advantages(
        self,
        rewards: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute advantages relative to group mean.
        
        Args:
            rewards: [B] rewards per sample
            group_ids: [B] group assignment
            
        Returns:
            advantages: [B] normalized relative advantages
        """
        B = rewards.size(0)
        
        if group_ids is None:
            # Use all samples as one group
            advantages = rewards - rewards.mean()
        else:
            # Compute mean per group
            group_means = defaultdict(list)
            for i, gid in enumerate(group_ids.cpu().numpy()):
                group_means[int(gid)].append(i)
            
            # Compute advantages relative to group mean
            advantages = torch.zeros(B, device=rewards.device)
            for gid, indices in group_means.items():
                indices = torch.tensor(indices, device=rewards.device, dtype=torch.long)
                group_rewards = rewards[indices]
                group_mean = group_rewards.mean()
                advantages[indices] = group_rewards - group_mean
        
        # Normalize advantages
        if self.config.advantage_normalize:
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Clip advantages
        if self.config.advantage_clip is not None:
            advantages = torch.clamp(
                advantages,
                -self.config.advantage_clip,
                self.config.advantage_clip
            )
        
        return advantages
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Update policy using GRPO objective.
        
        The GRPO objective is similar to PPO but uses relative advantages:
            L = min(r * A, clip(r, 1-ε, 1+ε) * A)
            
        where r = π_θ(a|s) / π_θ_old(a|s)
        
        Args:
            states: [B, state_dim] states
            actions: [B, action_dim] actions
            old_log_probs: [B] log probability under old policy
            advantages: [B] computed advantages
            returns: [B] returns (optional, for logging)
            
        Returns:
            Dictionary of loss metrics
        """
        B = states.size(0)
        
        # Get action distribution
        action_dist = self.policy.get_action_distribution(states)
        
        # Compute new log probabilities
        new_log_probs = action_dist.log_prob(actions)
        
        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus
        entropy = action_dist.entropy().mean()
        entropy_loss = -self.config.entropy_coef * entropy
        
        # Total loss
        loss = policy_loss + entropy_loss
        
        # KL divergence (optional)
        kl_loss = torch.tensor(0.0, device=states.device)
        if self.config.use_kl:
            with torch.no_grad():
                ratio_kl = new_log_probs - old_log_probs
                kl_loss = ratio_kl.mean()
        
        # Compute metrics
        with torch.no_grad():
            clip_fraction = ((ratio - 1).abs() > self.config.clip_epsilon).float().mean()
            approx_kl = (ratio - 1) - torch.log(ratio)
            kl = approx_kl.mean()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'kl': kl.item(),
            'clip_fraction': clip_fraction.item(),
            'mean_advantage': advantages.mean().item(),
        }


# ============================================================================
# Waypoint-Specific GRPO
# ============================================================================

class WaypointGRPOPolicy(nn.Module):
    """
    GRPO-compatible waypoint prediction policy.
    
    This wraps the waypoint prediction model to provide:
    - Action distribution for sampling
    - Group-based advantage computation
    - Integration with GRPO algorithm
    """
    
    def __init__(
        self,
        ar_decoder: nn.Module,
        config,
    ):
        super().__init__()
        self.ar_decoder = ar_decoder
        self.config = config
        
        # Waypoint prediction head
        self.waypoint_head = nn.Linear(
            config.hidden_dim,
            config.waypoint_dim * config.num_waypoints
        )
        
        # Logits head for discrete action selection (if using discrete)
        self.logits_head = nn.Linear(
            config.hidden_dim,
            config.num_discrete_actions
        )
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def get_action_distribution(
        self,
        features: torch.Tensor,
    ) -> torch.distributions.Distribution:
        """
        Get action distribution for waypoint prediction.
        
        Returns a continuous distribution (Normal) for waypoint coordinates.
        """
        B = features.size(0)
        
        # Get waypoint predictions
        waypoints = self.forward(features)  # [B, T, 3]
        
        # For continuous action space, we use a Normal distribution
        # with predicted mean and learnable/variable std
        T = waypoints.size(1)
        
        # Predict mean waypoints
        mean = waypoints  # [B, T, 3]
        
        # Learnable standard deviation (same for all)
        if not hasattr(self, 'action_std'):
            self.register_parameter(
                'action_std',
                nn.Parameter(torch.zeros(1, 1, 3) + 0.1)
            )
        
        std = self.action_std.exp().expand(B, T, -1)
        
        return torch.distributions.Normal(mean, std + 1e-8)
    
    def forward(
        self,
        features: torch.Tensor,
        waypoints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: [B, feature_dim] conditioning features
            waypoints: [B, T, waypoint_dim] target waypoints (for training)
            
        Returns:
            waypoints: [B, T, waypoint_dim] predicted waypoints
        """
        # Get AR decoder output
        if hasattr(self.ar_decoder, 'ar_decoder'):
            # ARCoTDecoder
            ar_output = self.ar_decoder(features)
            if waypoints is not None:
                wp_features = ar_output.get('embeddings')
            else:
                wp_features = features
        else:
            # ARDecoder
            wp_features = features
        
        # Predict waypoints
        waypoints = self.waypoint_head(wp_features)
        
        # Reshape to [B, T, waypoint_dim]
        T = self.config.num_waypoints
        D = self.config.waypoint_dim
        waypoints = waypoints.view(-1, T, D)
        
        return waypoints
    
    def get_log_prob(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get log probability of actions given features.
        
        Args:
            features: [B, feature_dim]
            actions: [B, T, waypoint_dim]
            
        Returns:
            log_prob: [B]
        """
        dist = self.get_action_distribution(features)
        return dist.log_prob(actions).sum(dim=(-2, -1))


# ============================================================================
# Scenario-Based GRPO for Driving
# ============================================================================

class ScenarioGRPO:
    """
    GRPO with scenario-based grouping.
    
    Groups trajectories by driving scenario for more meaningful
    relative comparisons (e.g., highway vs urban vs parking).
    
    Benefits:
    - Meaningful comparisons within scenario types
    - Can learn scenario-specific baselines
    - Natural grouping for driving data
    """
    
    def __init__(
        self,
        policy: nn.Module,
        config: GRPOConfig,
        scenario_encoder: Optional[nn.Module] = None,
    ):
        self.policy = policy
        self.config = config
        self.scenario_encoder = scenario_encoder
    
    def classify_scenario(
        self,
        state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Classify the current driving scenario.
        
        Args:
            state: Dict with 'features', 'speed', 'context', etc.
            
        Returns:
            scenario_id: [B] scenario ID for each sample
        """
        if self.scenario_encoder is not None:
            # Learned scenario classification
            features = state.get('features')
            if features is not None:
                with torch.no_grad():
                    scenario_logits = self.scenario_encoder(features)
                    scenario_id = scenario_logits.argmax(dim=-1)
                    return scenario_id
        
        # Heuristic-based classification
        speed = state.get('speed', torch.zeros(1))
        if speed.dim() == 1:
            speed = speed.unsqueeze(0)
        
        # Simple speed-based categorization
        scenario_ids = torch.zeros(speed.size(0), device=speed.device, dtype=torch.long)
        
        # Highway: speed > 20 m/s
        scenario_ids[speed > 20] = 0
        
        # Urban: 5 m/s < speed <= 20 m/s
        scenario_ids[(speed > 5) & (speed <= 20)] = 1
        
        # Low speed: speed <= 5 m/s (intersection, parking)
        scenario_ids[speed <= 5] = 2
        
        return scenario_ids
    
    def group_trajectories(
        self,
        trajectories: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        """
        Group trajectories by scenario.
        
        Args:
            trajectories: List of trajectory dicts
            
        Returns:
            groups: Dict[scenario_id, List[trajectory]]
        """
        groups = defaultdict(list)
        
        for traj in trajectories:
            scenario_id = self.classify_scenario(traj)
            groups[int(scenario_id)].append(traj)
        
        return dict(groups)
    
    def compute_scenario_advantages(
        self,
        trajectories: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute advantages within scenario groups.
        
        Args:
            trajectories: List of trajectory dicts with 'reward', 'scenario_id'
            
        Returns:
            advantages: [B] advantages per trajectory
        """
        # Group by scenario
        groups = self.group_trajectories(trajectories)
        
        advantages = []
        
        for scenario_id, group_trajs in groups.items():
            if len(group_trajs) < 2:
                # Not enough for relative comparison
                for traj in group_trajs:
                    advantages.append(torch.tensor(0.0))
                continue
            
            # Get rewards within group
            rewards = torch.stack([t['reward'] for t in group_trajs])
            
            # Compute relative advantages
            group_mean = rewards.mean()
            group_std = rewards.std() + 1e-8
            
            for i, traj in enumerate(group_trajs):
                adv = (rewards[i] - group_mean) / group_std
                advantages.append(adv)
        
        return torch.stack(advantages)


# ============================================================================
# GRPO Trainer
# ============================================================================

class GRPOTrainer:
    """
    Trainer for GRPO algorithm.
    
    Handles:
    - Trajectory collection
    - Advantage computation
    - Policy update
    - Logging and checkpointing
    """
    
    def __init__(
        self,
        policy: nn.Module,
        config: GRPOConfig,
        env,  # Environment with step() and reset() methods
        device: str = 'cuda',
    ):
        self.policy = policy.to(device)
        self.config = config
        self.env = env
        self.device = device
        
        # GRPO algorithm
        self.grpo = GRPO(policy, config)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
        )
        
        # Logging
        self.global_step = 0
        self.episode_count = 0
        self.rewards_history = []
    
    def collect_trajectories(
        self,
        num_episodes: int = 10,
        max_steps: int = 100,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Collect trajectories by interacting with environment.
        
        Args:
            num_episodes: Number of episodes to collect
            max_steps: Maximum steps per episode
            
        Returns:
            trajectories: List of trajectory dicts
        """
        trajectories = []
        
        for _ in range(num_episodes):
            # Reset environment
            state, info = self.env.reset()
            
            episode_rewards = []
            episode_states = []
            episode_actions = []
            episode_log_probs = []
            episode_dones = []
            
            for step in range(max_steps):
                # Convert state to tensor
                state_tensor = self._state_to_tensor(state)
                state_tensor = state_tensor.unsqueeze(0).to(self.device)
                
                # Sample action
                with torch.no_grad():
                    action_dist = self.policy.get_action_distribution(state_tensor)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
                
                action_np = action.cpu().numpy().squeeze()
                log_prob_np = log_prob.cpu().numpy().item()
                
                # Step environment
                next_state, reward, done, info = self.env.step(action_np)
                
                # Store transition
                episode_states.append(state_tensor)
                episode_actions.append(action)
                episode_log_probs.append(log_prob)
                episode_rewards.append(reward)
                episode_dones.append(done)
                
                # Move to next state
                state = next_state
                
                if done:
                    break
            
            # Compute episode return
            episode_return = sum(episode_rewards)
            
            # Store trajectory
            trajectory = {
                'states': episode_states,
                'actions': episode_actions,
                'log_probs': episode_log_probs,
                'rewards': episode_rewards,
                'return': episode_return,
                'length': len(episode_rewards),
            }
            trajectories.append(trajectory)
            self.rewards_history.append(episode_return)
        
        self.episode_count += num_episodes
        
        return trajectories
    
    def _state_to_tensor(self, state: Dict[str, np.ndarray]) -> torch.Tensor:
        """Convert state dict to tensor."""
        features = state.get('features', state.get('image'))
        if isinstance(features, np.ndarray):
            return torch.tensor(features, dtype=torch.float32)
        return torch.zeros(10)
    
    def compute_returns_and_advantages(
        self,
        trajectories: List[Dict[str, Any]],
        gamma: float = 0.99,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute returns and advantages for trajectories.
        
        Args:
            trajectories: List of trajectory dicts
            gamma: Discount factor
            
        Returns:
            returns: List of return tensors per trajectory
            advantages: List of advantage tensors per trajectory
        """
        returns = []
        advantages = []
        
        for traj in trajectories:
            rewards = torch.tensor(traj['rewards'], dtype=torch.float32)
            
            # Compute discounted returns
            R = 0
            returns_traj = []
            for r in reversed(rewards):
                R = r + gamma * R
                returns_traj.insert(0, R)
            returns.append(torch.stack(returns_traj))
            
            # Compute advantages (simple: return - baseline)
            baseline = returns_traj[-1]  # Last return as baseline
            advantages_traj = [r - baseline for r in returns_traj]
            advantages.append(torch.stack(advantages_traj))
        
        return returns, advantages
    
    def update(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Update policy using collected trajectories.
        
        Args:
            trajectories: Collected trajectories
            
        Returns:
            metrics: Dictionary of training metrics
        """
        # Concatenate all trajectories
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []
        
        for traj in trajectories:
            all_states.extend(traj['states'])
            all_actions.extend(traj['actions'])
            all_old_log_probs.extend(traj['log_probs'])
            all_returns.extend(traj['returns'])
            all_advantages.extend(traj['advantages'])
        
        # Stack to tensors
        states = torch.cat(all_states, dim=0)
        actions = torch.cat(all_actions, dim=0)
        old_log_probs = torch.cat(all_old_log_probs, dim=0)
        advantages = torch.cat(all_advantages, dim=0)
        returns = torch.cat(all_returns, dim=0)
        
        # Compute metrics before update
        with torch.no_grad():
            old_kl = (old_log_probs * 0).mean()  # Placeholder
        
        # Update for multiple epochs
        total_loss = 0
        metrics_list = []
        
        # Shuffle data
        indices = torch.randperm(len(states))
        
        for epoch in range(self.config.update_epochs):
            # Mini-batch update
            for start in range(0, len(states), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Compute GRPO loss
                metrics = self.grpo.update(
                    batch_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_advantages,
                    batch_returns,
                )
                
                metrics_list.append(metrics)
                total_loss += metrics['loss']
                
                # Gradient step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
        
        # Average metrics
        avg_metrics = {
            'loss': total_loss / len(metrics_list),
            'policy_loss': np.mean([m['policy_loss'] for m in metrics_list]),
            'entropy': np.mean([m['entropy'] for m in metrics_list]),
            'kl': np.mean([m['kl'] for m in metrics_list]),
            'clip_fraction': np.mean([m['clip_fraction'] for m in metrics_list]),
        }
        
        return avg_metrics
    
    def train(
        self,
        num_iterations: int = 100,
        episodes_per_iteration: int = 10,
        max_steps: int = 100,
        eval_interval: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            num_iterations: Number of training iterations
            episodes_per_iteration: Episodes per iteration
            max_steps: Max steps per episode
            eval_interval: Evaluate every N iterations
            
        Returns:
            training_history: Dict of metric histories
        """
        history = {
            'returns': [],
            'loss': [],
            'entropy': [],
            'eval_returns': [],
        }
        
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")
            
            # Collect trajectories
            trajectories = self.collect_trajectories(episodes_per_iteration, max_steps)
            
            # Compute returns and advantages
            returns, advantages = self.compute_returns_and_advantages(trajectories)
            
            # Add to trajectories
            for i, traj in enumerate(trajectories):
                traj['returns'] = [r for r in returns[i].cpu().numpy()]
                traj['advantages'] = [a for a in advantages[i].cpu().numpy()]
            
            # Update policy
            metrics = self.update(trajectories)
            
            # Log
            avg_return = np.mean([t['return'] for t in trajectories])
            history['returns'].append(avg_return)
            history['loss'].append(metrics['loss'])
            history['entropy'].append(metrics['entropy'])
            
            print(f"  Avg Return: {avg_return:.2f}")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
            
            # Evaluation
            if (iteration + 1) % eval_interval == 0:
                eval_return = self.evaluate(num_episodes=5, max_steps=max_steps)
                history['eval_returns'].append(eval_return)
                print(f"  Eval Return: {eval_return:.2f}")
        
        return history
    
    def evaluate(
        self,
        num_episodes: int = 5,
        max_steps: int = 100,
    ) -> float:
        """
        Evaluate current policy.
        
        Returns:
            Average return over episodes
        """
        self.policy.eval()
        
        total_returns = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_return = 0
            
            for _ in range(max_steps):
                state_tensor = self._state_to_tensor(state)
                state_tensor = state_tensor.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    dist = self.policy.get_action_distribution(state_tensor)
                    action = dist.mean  # Greedy action
                
                action_np = action.cpu().numpy().squeeze()
                next_state, reward, done, _ = self.env.step(action_np)
                
                episode_return += reward
                state = next_state
                
                if done:
                    break
            
            total_returns.append(episode_return)
        
        self.policy.train()
        
        return np.mean(total_returns)
    
    def save_checkpoint(self, path: str, episode: int):
        """Save training checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': episode,
            'global_step': self.global_step,
            'rewards_history': self.rewards_history,
        }, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.rewards_history = checkpoint['rewards_history']
        print(f"Loaded checkpoint from {path}")


# ============================================================================
# Integration with Waypoint BC
# ============================================================================

class WaypointBC_GRPOSplit:
    """
    Combined SFT + GRPO training for waypoint prediction.
    
    Pattern:
    1. Pre-train with SFT (teacher forcing)
    2. Fine-tune with GRPO (no critic, group-based advantages)
    
    Benefits:
    - SFT provides good initialization
    - GRPO improves beyond SFT baseline
    - No value function needed for RL phase
    """
    
    def __init__(
        self,
        sft_model: nn.Module,
        grpo_config: GRPOConfig,
        device: str = 'cuda',
    ):
        self.sft_model = sft_model.to(device)
        self.grpo_config = grpo_config
        self.device = device
        
        # Freeze SFT model
        for param in sft_model.parameters():
            param.requires_grad = False
        
        # GRPO policy (wraps SFT)
        self.grpo_policy = WaypointGRPOPolicy(
            sft_model.ar_decoder if hasattr(sft_model, 'ar_decoder') else sft_model,
            grpo_config,
        ).to(device)
        
        # GRPO trainer
        self.grpo = GRPO(self.grpo_policy, grpo_config)
        
        # Optimizer for GRPO policy
        self.optimizer = torch.optim.Adam(
            self.grpo_policy.parameters(),
            lr=1e-4,
        )
    
    def sft_pretrain(
        self,
        dataloader,
        epochs: int = 10,
    ) -> Dict[str, float]:
        """
        Pre-train with SFT (teacher forcing).
        """
        self.sft_model.train()
        total_loss = 0
        
        for epoch in range(epochs):
            for batch in dataloader:
                features = batch['features'].to(self.device)
                waypoints = batch['waypoints'].to(self.device)
                
                # Forward
                output = self.sft_model(features, waypoints=waypoints)
                pred_waypoints = output['waypoints']
                
                # MSE loss
                loss = F.mse_loss(pred_waypoints, waypoints)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return {'sft_loss': avg_loss}
    
    def grpo_finetune(
        self,
        env,
        num_iterations: int = 100,
        episodes_per_iteration: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune with GRPO.
        """
        # Create GRPO trainer
        trainer = GRPOTrainer(
            policy=self.grpo_policy,
            config=self.grpo_config,
            env=env,
            device=self.device,
        )
        
        # Train
        history = trainer.train(
            num_iterations=num_iterations,
            episodes_per_iteration=episodes_per_iteration,
        )
        
        return history
    
    def combined_train(
        self,
        sft_dataloader,
        env,
        sft_epochs: int = 10,
        grpo_iterations: int = 100,
        episodes_per_iteration: int = 10,
    ) -> Dict[str, Any]:
        """
        Combined SFT + GRPO training.
        """
        print("Phase 1: SFT Pre-training")
        print("=" * 50)
        sft_metrics = self.sft_pretrain(sft_dataloader, epochs=sft_epochs)
        print(f"SFT Loss: {sft_metrics['sft_loss']:.4f}")
        
        print("\nPhase 2: GRPO Fine-tuning")
        print("=" * 50)
        grpo_history = self.grpo_finetune(
            env,
            num_iterations=grpo_iterations,
            episodes_per_iteration=episodes_per_iteration,
        )
        
        return {
            'sft_metrics': sft_metrics,
            'grpo_history': grpo_history,
        }


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example of using GRPO for waypoint prediction."""
    
    # Configuration
    config = GRPOConfig(
        clip_epsilon=0.2,
        entropy_coef=0.01,
        batch_size=64,
        group_size=4,
        update_epochs=4,
    )
    
    # Mock policy (replace with actual AR decoder)
    class MockPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(256, 30)  # 10 waypoints * 3
        
        def get_action_distribution(self, x):
            mean = self.fc(x).view(-1, 10, 3)
            std = torch.ones_like(mean) * 0.1
            return torch.distributions.Normal(mean, std)
    
    policy = MockPolicy()
    
    # GRPO
    grpo = GRPO(policy, config)
    
    # Sample actions
    states = torch.randn(32, 256)
    actions, log_probs, entropies = grpo.sample(states, num_samples=4)
    print(f"Actions shape: {actions.shape}")  # [32, 4, 3]
    
    # Compute advantages
    rewards = torch.randn(32)
    advantages = grpo.compute_advantages(rewards)
    print(f"Advantages shape: {advantages.shape}")  # [32]
    
    # Update
    metrics = grpo.update(states, actions[:, 0], log_probs[:, 0], advantages)
    print(f"Loss: {metrics['loss']:.4f}")


if __name__ == "__main__":
    example_usage()
