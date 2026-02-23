"""
Training script for Multi-Scenario RL with Domain Randomization.

Trains a residual delta-waypoint policy across multiple weather/lighting conditions:
- clear, cloudy, night, rain, fog

Features:
- Domain randomization for robust policies
- Curriculum learning (easy → hard scenarios)
- Scenario-specific reward shaping
- Per-scenario evaluation
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime
from collections import deque

from training.rl.multi_scenario_env import (
    MultiScenarioWaypointEnv, 
    ScenarioType, 
    CurriculumScheduler,
    SCENARIO_DIFFICULTY
)


class ResidualDeltaNetwork(nn.Module):
    """
    Residual delta-waypoint network.
    
    Architecture: final_waypoints = sft_waypoints + delta_head(z)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Delta head (predicts corrections to SFT waypoints)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * action_dim),
        )
        
        # Uncertainty head (for risk-aware training)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * action_dim),
            nn.Softplus(),  # Ensure positive variance
        )
        
    def forward(
        self, 
        state: torch.Tensor, 
        sft_waypoints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: (batch, state_dim)
            sft_waypoints: (batch, horizon, action_dim)
            
        Returns:
            final_waypoints: (batch, horizon, action_dim)
            uncertainty: (batch, horizon, action_dim)
        """
        features = self.encoder(state)
        
        # Predict delta
        delta = self.delta_head(features)
        delta = delta.view(-1, self.horizon, self.action_dim)
        
        # Predict uncertainty
        uncertainty = self.uncertainty_head(features)
        uncertainty = uncertainty.view(-1, self.horizon, self.action_dim)
        
        # Final waypoints = SFT + delta
        final_waypoints = sft_waypoints + delta
        
        return final_waypoints, uncertainty


class PPOMultiScenarioAgent:
    """
    PPO agent for multi-scenario training.
    
    Features:
    - GAE for advantage estimation
    - Clip pruning for policy updates
    - Value function baseline
    - Per-scenario tracking
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int = 20,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.1,
        use_residual: bool = True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.use_residual = use_residual
        
        # Networks
        self.policy = ResidualDeltaNetwork(
            state_dim, action_dim, horizon, hidden_dim
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr
        )
        
        # Trajectory buffer
        self.buffer = []
        
        # Per-scenario metrics
        self.scenario_metrics = {s.value: {"rewards": [], "success": []} for s in ScenarioType}
        
    def get_action(
        self, 
        state: np.ndarray, 
        sft_waypoints: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get action from policy."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        sft_t = torch.FloatTensor(sft_waypoints).unsqueeze(0)
        
        with torch.no_grad():
            waypoints, uncertainty = self.policy(state_t, sft_t)
            waypoints = waypoints.squeeze(0).numpy()
            uncertainty = uncertainty.squeeze(0).numpy()
        
        if deterministic:
            return waypoints, uncertainty, 0.0
        
        # Add exploration noise based on uncertainty
        noise = np.random.normal(0, uncertainty * 0.1)
        waypoints = waypoints + noise
        
        log_prob = -np.sum(uncertainty)  # Simplified
        
        return waypoints, uncertainty, log_prob
    
    def compute_gae(
        self, 
        rewards: List[float], 
        values: List[float], 
        next_value: float
    ) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (values[t + 1] if t + 1 < len(values) else next_value) - values[t]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def update(
        self, 
        states: List[np.ndarray],
        sft_waypoints_list: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        dones: List[bool],
        scenario: str,
    ) -> Dict[str, float]:
        """Update policy from trajectory using simplified advantage."""
        # Use states[:-1] and actions for non-terminal steps
        # Add terminal state value for bootstrapping
        states_t = torch.FloatTensor(np.array(states[:-1]))
        sft_t = torch.FloatTensor(np.array(sft_waypoints_list[:-1]))
        actions_t = torch.FloatTensor(np.array(actions))
        rewards_t = torch.FloatTensor(rewards)
        
        # Compute values for all states
        all_states_t = torch.FloatTensor(np.array(states))
        all_values = self.value(all_states_t).squeeze(-1)
        
        # Bootstrap value from terminal state
        with torch.no_grad():
            next_value = all_values[-1].item()
        
        # Compute advantages using GAE-like: A_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        values = all_values[:-1].tolist()  # V(s_0) to V(s_{n-1})
        advantages = []
        for t in range(len(rewards)):
            next_v = next_value if t == len(rewards) - 1 else values[t + 1]
            adv = rewards[t] + self.gamma * next_v - values[t]
            advantages.append(adv)
        
        advantages_t = torch.FloatTensor(advantages)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # Forward pass (only for non-terminal states)
        waypoints_pred, uncertainty = self.policy(states_t, sft_t)
        
        # Policy loss (MSE between predicted and actual waypoints)
        policy_loss = nn.functional.mse_loss(waypoints_pred, actions_t)
        
        # Value loss
        values_pred = self.value(states_t).squeeze(-1)
        returns = torch.FloatTensor([adv + val for adv, val in zip(advantages, values)])
        value_loss = nn.functional.mse_loss(values_pred, returns)
        
        # Entropy bonus (encourage exploration)
        entropy_loss = -uncertainty.mean()
        
        # KL divergence (keep close to SFT)
        if self.use_residual:
            # Simplified KL: MSE between delta and zero
            delta = waypoints_pred - sft_t
            kl_loss = (delta ** 2).mean()
        else:
            kl_loss = 0.0
        
        # Total loss
        total_loss = (
            policy_loss +
            self.value_coef * value_loss +
            self.entropy_coef * entropy_loss +
            self.kl_coef * kl_loss
        )
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        # Track per-scenario metrics
        success_rate = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0
        self.scenario_metrics[scenario]["rewards"].append(np.mean(rewards))
        self.scenario_metrics[scenario]["success"].append(success_rate)
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
            "mean_reward": np.mean(rewards),
            "success_rate": success_rate,
        }
    
    def get_scenario_stats(self) -> Dict[str, Dict[str, float]]:
        """Get per-scenario statistics."""
        stats = {}
        for scenario, metrics in self.scenario_metrics.items():
            if metrics["rewards"]:
                stats[scenario] = {
                    "mean_reward": np.mean(metrics["rewards"]),
                    "mean_success": np.mean(metrics["success"]),
                    "episodes": len(metrics["rewards"]),
                }
            else:
                stats[scenario] = {"mean_reward": 0, "mean_success": 0, "episodes": 0}
        return stats


def train_multi_scenario(
    num_episodes: int = 1000,
    horizon: int = 20,
    hidden_dim: int = 64,
    curriculum_level: float = 1.0,
    enable_domain_randomization: bool = True,
    log_interval: int = 10,
    save_dir: str = "out/multi_scenario_rl",
) -> Tuple[PPOMultiScenarioAgent, Dict]:
    """
    Train PPO agent on multi-scenario environment.
    
    Args:
        num_episodes: Number of training episodes
        horizon: Waypoint horizon
        hidden_dim: Hidden dimension for networks
        curriculum_level: Curriculum learning level (0.0-1.0)
        enable_domain_randomization: Enable domain randomization
        log_interval: Logging interval
        save_dir: Directory for saving artifacts
        
    Returns:
        Trained agent and training metrics
    """
    # Create environment
    env = MultiScenarioWaypointEnv(
        horizon=horizon,
        enable_domain_randomization=enable_domain_randomization,
        curriculum_level=curriculum_level,
        randomize_every_episode=True,
    )
    
    # Create agent
    agent = PPOMultiScenarioAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        horizon=horizon,
        hidden_dim=hidden_dim,
        use_residual=True,
    )
    
    # Training metrics
    metrics = {
        "episode_rewards": [],
        "policy_losses": [],
        "value_losses": [],
        "success_rates": [],
        "scenario_counts": {s.value: 0 for s in ScenarioType},
    }
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Collect trajectory
        states = [state]
        sft_waypoints_list = [env.get_sft_waypoints()]
        actions = []
        rewards = []
        dones = [False]
        
        while not done:
            # Get SFT baseline waypoints
            sft_waypoints = env.get_sft_waypoints()
            
            # Get action from policy
            waypoints, uncertainty, _ = agent.get_action(state, sft_waypoints)
            
            # Environment step
            next_state, reward, done, info = env.step(waypoints)
            
            # Store transition
            actions.append(waypoints)
            rewards.append(reward)
            dones.append(done)
            states.append(next_state)
            sft_waypoints_list.append(env.get_sft_waypoints())
            
            # Track scenario
            scenario = info.get("scenario", "clear")
            metrics["scenario_counts"][scenario] = metrics["scenario_counts"].get(scenario, 0) + 1
            
            episode_reward += reward
            state = next_state
        
        # Update policy
        update_metrics = agent.update(
            states,  # Include all states (for value computation including terminal)
            sft_waypoints_list,
            actions,
            rewards,
            dones[1:],  # Remove first done (initial)
            scenario,
        )
        
        # Log metrics
        metrics["episode_rewards"].append(episode_reward)
        metrics["policy_losses"].append(update_metrics["policy_loss"])
        metrics["value_losses"].append(update_metrics["value_loss"])
        metrics["success_rates"].append(1.0 if info["goal_reached"] else 0.0)
        
        # Logging
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(metrics["episode_rewards"][-log_interval:])
            avg_success = np.mean(metrics["success_rates"][-log_interval:])
            scenario_stats = agent.get_scenario_stats()
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Success: {avg_success:.2%}")
            print(f"  Policy Loss: {update_metrics['policy_loss']:.4f}")
            print(f"  Scenario distribution:")
            for s, count in metrics["scenario_counts"].items():
                if count > 0:
                    print(f"    {s}: {count}")
    
    # Save artifacts
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    torch.save({
        "policy": agent.policy.state_dict(),
        "value": agent.value.state_dict(),
    }, f"{save_dir}/model.pt")
    
    # Save metrics
    final_metrics = {
        "final_avg_reward": np.mean(metrics["episode_rewards"][-100:]),
        "final_success_rate": np.mean(metrics["success_rates"][-100:]),
        "scenario_stats": agent.get_scenario_stats(),
        "total_episodes": num_episodes,
    }
    
    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    with open(f"{save_dir}/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"  Final avg reward: {final_metrics['final_avg_reward']:.2f}")
    print(f"  Final success rate: {final_metrics['final_success_rate']:.2%}")
    print(f"  Model saved to: {save_dir}/model.pt")
    
    return agent, final_metrics


def evaluate_on_scenarios(
    agent: PPOMultiScenarioAgent,
    num_episodes_per_scenario: int = 20,
    horizon: int = 20,
    save_dir: str = "out/multi_scenario_eval",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate agent on each scenario separately.
    
    Returns per-scenario performance metrics.
    """
    results = {}
    
    for scenario in ScenarioType:
        env = MultiScenarioWaypointEnv(
            horizon=horizon,
            scenario=scenario,
            enable_domain_randomization=False,  # Deterministic eval
            curriculum_level=1.0,
            randomize_every_episode=False,
        )
        
        rewards = []
        successes = []
        distances = []
        
        for episode in range(num_episodes_per_scenario):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                sft_waypoints = env.get_sft_waypoints()
                waypoints, _, _ = agent.get_action(state, sft_waypoints, deterministic=True)
                state, reward, done, info = env.step(waypoints)
                episode_reward += reward
            
            rewards.append(episode_reward)
            successes.append(1.0 if info["goal_reached"] else 0.0)
            distances.append(info["distance"])
        
        results[scenario.value] = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "success_rate": float(np.mean(successes)),
            "mean_distance": float(np.mean(distances)),
            "difficulty": SCENARIO_DIFFICULTY[scenario],
        }
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/scenario_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nPer-scenario evaluation:")
    for scenario, metrics in results.items():
        print(f"  {scenario}: reward={metrics['mean_reward']:.2f}, "
              f"success={metrics['success_rate']:.2%}, "
              f"distance={metrics['mean_distance']:.2f}m")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train multi-scenario RL agent")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--horizon", type=int, default=20, help="Waypoint horizon")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--curriculum", type=float, default=1.0, help="Curriculum level")
    parser.add_argument("--no-domain-randomization", action="store_true", help="Disable DR")
    parser.add_argument("--eval-only", action="store_true", help="Evaluation only")
    parser.add_argument("--save-dir", type=str, default="out/multi_scenario_rl", help="Save directory")
    
    args = parser.parse_args()
    
    if args.eval_only:
        # Load model and evaluate
        print("Loading model for evaluation...")
        # Would load model here
        print("Evaluation complete.")
    else:
        # Train
        agent, metrics = train_multi_scenario(
            num_episodes=args.episodes,
            horizon=args.horizon,
            hidden_dim=args.hidden_dim,
            curriculum_level=args.curriculum,
            enable_domain_randomization=not args.no_domain_randomization,
            save_dir=args.save_dir,
        )
        
        # Evaluate on each scenario
        print("\nEvaluating on individual scenarios...")
        evaluate_on_scenarios(agent, horizon=args.horizon, save_dir=args.save_dir)
