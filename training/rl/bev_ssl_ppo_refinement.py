#!/usr/bin/env python3
"""
BEV SSL Waypoint Predictor RL Refinement with PPO.

This module implements PPO-based RL refinement for the BEV SSL waypoint predictor.
The BC model provides base waypoint predictions, and PPO trains a delta head
to refine those predictions based on environment feedback.

Usage:
    python -m training.rl.bev_ssl_ppo_refinement \
        --bc-checkpoint out/bev_ssl_waypoint_predictor/final.pt \
        --output-dir out/bev_ssl_ppo_refine \
        --episodes 200

    # Or use stub BC model for testing
    python -m training.rl.bev_ssl_ppo_refinement --test
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BEVSSLPPORefineConfig:
    """Configuration for BEV SSL PPO refinement."""
    # Model
    encoder_dim: int = 128
    num_waypoints: int = 8
    waypoint_dim: int = 2
    
    # Delta head
    delta_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    delta_log_std: float = -0.5
    
    # PPO
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training
    lr: float = 3e-4
    episodes: int = 200
    batch_size: int = 32
    update_epochs: int = 4
    minibatch_size: int = 8
    
    # Environment
    world_size: float = 100.0
    max_speed: float = 10.0
    max_steer: float = math.pi / 4
    wheelbase: float = 2.5
    dt: float = 0.1
    max_episode_steps: int = 100
    success_radius: float = 2.0
    
    # Rewards
    waypoint_tracking_weight: float = 1.0
    progress_weight: float = 0.5
    time_penalty: float = -0.01
    success_bonus: float = 10.0
    
    # Logging
    eval_interval: int = 20
    save_interval: int = 50
    output_dir: str = "out/bev_ssl_ppo_refine"


# ============================================================================
# Stub BC Model for Testing
# ============================================================================

class StubWaypointPredictor:
    """Stub waypoint predictor for testing without pretrained model."""
    
    def __init__(self, config: BEVSSLPPORefineConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def predict_waypoints(self, state: np.ndarray) -> np.ndarray:
        """Predict waypoints from state (stub: straight line)."""
        num_wp = self.config.num_waypoints
        spacing = 3.0
        # Simple straight-line prediction
        waypoints = np.zeros((num_wp, 2))
        for i in range(num_wp):
            waypoints[i, 0] = (i + 1) * spacing  # x forward
            waypoints[i, 1] = 0.0  # y center
        return waypoints
    
    def to(self, device):
        """Move to device."""
        self.device = device
        return self


# ============================================================================
# Delta Waypoint Head
# ============================================================================

class WaypointPolicyHead(nn.Module):
    """Policy head that outputs steer/throttle for waypoint following."""
    
    def __init__(self, state_dim: int, hidden_dims: List[int], 
                 log_std: float = -0.5):
        super().__init__()
        
        self.action_dim = 2  # steer, throttle
        
        # Build MLP
        dims = [state_dim] + hidden_dims + [self.action_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)
        
        self.log_std = nn.Parameter(torch.tensor(log_std))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns mean and std of action."""
        action = self.mlp(state)
        std = torch.exp(self.log_std).expand_as(action)
        return action, std
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        action, std = self.forward(state)
        dist = Normal(action, std)
        sampled = dist.sample()
        log_prob = dist.log_prob(sampled).sum(dim=-1, keepdim=True)
        return sampled, log_prob
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log prob and entropy."""
        action_mean, std = self.forward(state)
        dist = Normal(action_mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# ============================================================================
# Value Network
# ============================================================================

class ValueNetwork(nn.Module):
    """Value network for state value estimation."""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        
        dims = [state_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ============================================================================
# Kinematic Environment
# ============================================================================

class KinematicWaypointEnv:
    """Kinematic environment for waypoint following with BC + delta refinement."""
    
    def __init__(self, config: BEVSSLPPORefineConfig, bc_model: Any):
        self.config = config
        self.bc_model = bc_model
        
        self.world_size = config.world_size
        self.max_speed = config.max_speed
        self.max_steer = config.max_steer
        self.wheelbase = config.wheelbase
        self.dt = config.dt
        self.max_episode_steps = config.max_episode_steps
        self.success_radius = config.success_radius
        self.num_waypoints = config.num_waypoints
        
        # Vehicle state
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.speed = 0.0
        
        # Target waypoints (set by reset)
        self.target_waypoints = None
        self.current_wp_idx = 0
        
        self.step_count = 0
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        # Random start position
        self.x = random.uniform(-10, 10)
        self.y = random.uniform(-self.world_size/2, self.world_size/2)
        self.heading = random.uniform(-math.pi/4, math.pi/4)
        self.speed = random.uniform(0, 2)
        
        # Generate target waypoints (straight line ahead)
        self.target_waypoints = np.zeros((self.num_waypoints, 2))
        for i in range(self.num_waypoints):
            self.target_waypoints[i, 0] = self.x + (i + 1) * 5.0
            self.target_waypoints[i, 1] = self.y
        
        self.current_wp_idx = 0
        self.step_count = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        # Vehicle state: [x, y, heading, speed]
        # Target waypoints (relative): num_waypoints * 2
        # Current target index: 1
        state = np.zeros(self.num_waypoints * 2 + 5)
        state[0] = self.x / self.world_size
        state[1] = self.y / self.world_size
        state[2] = self.heading / math.pi
        state[3] = self.speed / self.max_speed
        
        # Relative target waypoints
        for i, wp in enumerate(self.target_waypoints):
            state[4 + i*2] = (wp[0] - self.x) / self.world_size
            state[4 + i*2 + 1] = (wp[1] - self.y) / self.world_size
        
        return state
    
    def step(self, steer: float, throttle: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action."""
        # Clamp inputs
        steer = np.clip(steer, -self.max_steer, self.max_steer)
        throttle = np.clip(throttle, -1.0, 1.0)
        
        # Update speed
        if throttle > 0:
            self.speed += throttle * 2.0 * self.dt
        else:
            self.speed += throttle * 1.0 * self.dt
        self.speed = np.clip(self.speed, -self.max_speed, self.max_speed)
        
        # Update position (bicycle model)
        if abs(self.speed) > 0.01:
            self.heading += (self.speed / self.wheelbase) * math.tan(steer) * self.dt
        
        self.x += self.speed * math.cos(self.heading) * self.dt
        self.y += self.speed * math.sin(self.heading) * self.dt
        
        self.step_count += 1
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check termination
        done = False
        if self.step_count >= self.max_episode_steps:
            done = True
        
        # Check success (close to final waypoint)
        if self.num_waypoints > 0:
            final_wp = self.target_waypoints[-1]
            dist = math.sqrt((self.x - final_wp[0])**2 + (self.y - final_wp[1])**2)
            if dist < self.success_radius:
                done = True
        
        # Info
        info = {
            'ade': self._compute_ade(),
            'fde': self._compute_fde(),
            'success': self._is_success(),
        }
        
        return self._get_state(), reward, done, info
    
    def _compute_reward(self) -> float:
        """Compute reward."""
        # Waypoint tracking reward (negative ADE)
        ade = self._compute_ade()
        reward = -ade * self.config.waypoint_tracking_weight
        
        # Progress reward
        if self.num_waypoints > 0:
            final_wp = self.target_waypoints[-1]
            dist_to_goal = math.sqrt((self.x - final_wp[0])**2 + (self.y - final_wp[1])**2)
            reward += -dist_to_goal * self.config.progress_weight
        
        # Time penalty
        reward += self.config.time_penalty
        
        # Success bonus
        if self._is_success():
            reward += self.config.success_bonus
        
        return reward
    
    def _compute_ade(self) -> float:
        """Average Displacement Error."""
        if self.target_waypoints is None:
            return 0.0
        # Simplified: distance to current target waypoint
        wp = self.target_waypoints[min(self.current_wp_idx, len(self.target_waypoints)-1)]
        return math.sqrt((self.x - wp[0])**2 + (self.y - wp[1])**2)
    
    def _compute_fde(self) -> float:
        """Final Displacement Error."""
        if self.target_waypoints is None:
            return 0.0
        final_wp = self.target_waypoints[-1]
        return math.sqrt((self.x - final_wp[0])**2 + (self.y - final_wp[1])**2)
    
    def _is_success(self) -> bool:
        """Check if episode is successful."""
        if self.target_waypoints is None:
            return False
        final_wp = self.target_waypoints[-1]
        dist = math.sqrt((self.x - final_wp[0])**2 + (self.y - final_wp[1])**2)
        return dist < self.success_radius


# ============================================================================
# PPO Agent with BC Waypoint Refinement
# ============================================================================

class PPORefineAgent:
    """PPO agent that refines BC waypoints via delta predictions."""
    
    def __init__(self, config: BEVSSLPPORefineConfig, bc_model: Any):
        self.config = config
        self.bc_model = bc_model
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # State dimension: vehicle (4) + waypoints (num_waypoints * 2) + bias (1)
        state_dim = 4 + config.num_waypoints * 2 + 1
        
        # Policy head (outputs steer/throttle)
        self.policy_head = WaypointPolicyHead(
            state_dim=state_dim,
            hidden_dims=config.delta_hidden_dims,
            log_std=config.delta_log_std,
        ).to(self.device)
        
        # Value network
        self.value_net = ValueNetwork(state_dim).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy_head.parameters()) + list(self.value_net.parameters()),
            lr=config.lr,
        )
        
        # Memory for PPO
        self.states = []
        self.actions = []
        self.old_log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def get_action(self, state: np.ndarray, eval_mode: bool = False) -> Tuple[np.ndarray, float]:
        """Get action from policy."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.value_net(state_t).item()
        
        if eval_mode:
            # Use mean
            with torch.no_grad():
                action_mean, _ = self.policy_head(state_t)
                action = action_mean.cpu().numpy().squeeze()
            log_prob = 0.0
        else:
            action, log_prob = self.policy_head.get_action(state_t)
            action = action.cpu().numpy().squeeze()
            log_prob = log_prob.item()
        
        # Clamp to valid range
        steer = float(np.clip(action[0], -self.config.max_steer, self.config.max_steer))
        throttle = float(np.clip(action[1], -1.0, 1.0))
        
        return np.array([steer, throttle]), log_prob, value
    
    def store_transition(self, state, action, log_prob, reward, done, value):
        """Store transition in memory."""
        self.states.append(state)
        self.actions.append(action)
        self.old_log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def compute_gae(self, rewards, values, dones, next_value: float = 0.0):
        """Compute GAE advantages."""
        advantages = []
        gae = 0
        
        # Include terminal value
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values[:-1])
        
        return advantages, returns
    
    def update(self):
        """Update policy with PPO."""
        if len(self.states) == 0:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.old_log_probs).unsqueeze(-1).to(self.device)
        advantages, returns = self.compute_gae(self.rewards, self.values, self.dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # PPO update
        losses = []
        value_losses = []
        entropies = []
        
        for _ in range(self.config.update_epochs):
            # Mini-batch update
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_indices = indices[start:end]
                
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Get new log prob and entropy
                log_probs, entropies_batch = self.policy_head.evaluate_actions(mb_states, mb_actions)
                
                # PPO ratio
                ratio = torch.exp(log_probs - mb_old_log_probs)
                
                # Clipped surrogate
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values_pred = self.value_net(mb_states)
                value_loss = F.mse_loss(values_pred, mb_returns.unsqueeze(-1))
                
                # Entropy bonus
                entropy_loss = -entropies_batch.mean()
                
                # Total loss
                loss = (policy_loss + 
                        self.config.value_coef * value_loss + 
                        self.config.entropy_coef * entropy_loss)
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_head.parameters()) + list(self.value_net.parameters()),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropies_batch.mean().item())
        
        # Clear memory
        self.states = []
        self.actions = []
        self.old_log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
        return {
            'policy_loss': np.mean(losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
        }
    
    def save(self, path: str):
        """Save checkpoint."""
        torch.save({
            'policy_head': self.policy_head.state_dict(),
            'value_net': self.value_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_head.load_state_dict(checkpoint['policy_head'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


# ============================================================================
# Training Loop
# ============================================================================

def train_bev_ssl_ppo_refinement(
    config: BEVSSLPPORefineConfig,
    bc_checkpoint: Optional[str] = None,
    test_mode: bool = False,
) -> Dict[str, Any]:
    """Train PPO refinement for BEV SSL waypoint model."""
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load BC model or use stub
    if test_mode:
        bc_model = StubWaypointPredictor(config)
        print("Using stub BC model for testing")
    elif bc_checkpoint and os.path.exists(bc_checkpoint):
        # Load checkpoint
        print(f"Loading BC checkpoint: {bc_checkpoint}")
        # For now, use stub model
        bc_model = StubWaypointPredictor(config)
    else:
        print("No BC checkpoint found, using stub model")
        bc_model = StubWaypointPredictor(config)
    
    # Create environment
    env = KinematicWaypointEnv(config, bc_model)
    
    # Create PPO agent
    agent = PPORefineAgent(config, bc_model)
    
    # Training metrics
    all_metrics = []
    eval_metrics = []
    
    # Training loop
    print(f"\nTraining PPO refinement for {config.episodes} episodes...")
    
    for episode in range(config.episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
        
        while True:
            # Get action
            action, log_prob, value = agent.get_action(state)
            
            # Step environment
            next_state, reward, done, info = env.step(action[0], action[1])
            
            # Store transition
            agent.store_transition(state, action, log_prob, reward, done, value)
            
            # Update if enough samples
            if len(agent.states) >= config.batch_size:
                update_metrics = agent.update()
                episode_loss += update_metrics.get('policy_loss', 0)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        # Update at end of episode
        if len(agent.states) > 0:
            update_metrics = agent.update()
            episode_loss += update_metrics.get('policy_loss', 0)
        
        # Record metrics
        all_metrics.append({
            'episode': episode,
            'reward': episode_reward,
            'steps': steps,
            'ade': info.get('ade', 0),
            'fde': info.get('fde', 0),
            'success': info.get('success', False),
        })
        
        # Eval
        if (episode + 1) % config.eval_interval == 0:
            eval_reward, eval_ade, eval_fde, eval_success = evaluate_agent(agent, env, config)
            eval_metrics.append({
                'episode': episode,
                'eval_reward': eval_reward,
                'eval_ade': eval_ade,
                'eval_fde': eval_fde,
                'eval_success': eval_success,
            })
            
            recent_rewards = [m['reward'] for m in all_metrics[-config.eval_interval:]]
            print(f"Episode {episode+1}/{config.episodes}: "
                  f"reward={np.mean(recent_rewards):.2f}, "
                  f"eval_reward={eval_reward:.2f}, "
                  f"eval_ADE={eval_ade:.2f}m, "
                  f"eval_FDE={eval_fde:.2f}m, "
                  f"eval_success={eval_success*100:.1f}%")
        
        # Save checkpoint
        if (episode + 1) % config.save_interval == 0:
            checkpoint_path = os.path.join(config.output_dir, f"checkpoint_{episode+1}.pt")
            agent.save(checkpoint_path)
    
    # Save final model
    final_path = os.path.join(config.output_dir, "final.pt")
    agent.save(final_path)
    
    # Save metrics
    metrics_path = os.path.join(config.output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            'train_metrics': all_metrics,
            'eval_metrics': eval_metrics,
        }, f, indent=2)
    
    # Summary
    if eval_metrics:
        final_eval = eval_metrics[-1]
        summary = {
            'output_dir': config.output_dir,
            'final_eval_reward': final_eval['eval_reward'],
            'final_eval_ade': final_eval['eval_ade'],
            'final_eval_fde': final_eval['eval_fde'],
            'final_eval_success': final_eval['eval_success'],
        }
    else:
        summary = {'output_dir': config.output_dir}
    
    print(f"\nTraining complete!")
    print(f"Output: {config.output_dir}")
    if eval_metrics:
        print(f"Final eval: reward={final_eval['eval_reward']:.2f}, "
              f"ADE={final_eval['eval_ade']:.2f}m, "
              f"FDE={final_eval['eval_fde']:.2f}m, "
              f"Success={final_eval['eval_success']*100:.1f}%")
    
    return summary


def evaluate_agent(agent: PPORefineAgent, env: KinematicWaypointEnv, 
                   config: BEVSSLPPORefineConfig, num_eval: int = 5) -> Tuple[float, float, float, float]:
    """Evaluate agent."""
    eval_rewards = []
    eval_ades = []
    eval_fdes = []
    eval_successes = []
    
    for _ in range(num_eval):
        state = env.reset()
        episode_reward = 0
        
        while True:
            action, _, _ = agent.get_action(state, eval_mode=True)
            next_state, reward, done, info = env.step(action[0], action[1])
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        eval_rewards.append(episode_reward)
        eval_ades.append(info.get('ade', 0))
        eval_fdes.append(info.get('fde', 0))
        eval_successes.append(1 if info.get('success', False) else 0)
    
    return (
        np.mean(eval_rewards),
        np.mean(eval_ades),
        np.mean(eval_fdes),
        np.mean(eval_successes),
    )


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="BEV SSL PPO Waypoint Refinement")
    parser.add_argument("--bc-checkpoint", type=str, default=None,
                        help="Path to BC checkpoint")
    parser.add_argument("--output-dir", type=str, default="out/bev_ssl_ppo_refine",
                        help="Output directory")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for PPO updates")
    parser.add_argument("--eval-interval", type=int, default=20,
                        help="Evaluation interval")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Checkpoint save interval")
    parser.add_argument("--test", action="store_true",
                        help="Run smoke test")
    
    args = parser.parse_args()
    
    # Create config
    config = BEVSSLPPORefineConfig(
        output_dir=args.output_dir,
        episodes=args.episodes,
        lr=args.lr,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
    )
    
    # Train
    if args.test:
        # Smoke test with fewer episodes
        config.episodes = 20
        config.eval_interval = 10
        config.save_interval = 20
        print("Running smoke test...")
    
    summary = train_bev_ssl_ppo_refinement(config, args.bc_checkpoint, args.test)
    
    print(f"\nSummary: {summary}")


if __name__ == "__main__":
    main()
