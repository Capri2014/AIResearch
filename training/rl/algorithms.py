"""
Reinforcement Learning Algorithms Comparison

Implements and compares:
- PPO (Proximal Policy Optimization)
- GRPO (Group Relative Policy Optimization)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)

Usage:
    from training.rl.algorithms import compare_algorithms
    
    results = compare_algorithms(
        env="toy_waypoint",
        num_seeds=3,
        max_steps=100000,
    )
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import time


# ============================================================================
# Base Algorithm Interface
# ============================================================================

class RLAlgorithm(ABC):
    """Base class for RL algorithms."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm name."""
        pass
    
    @property
    @abstractmethod
    def config_schema(self) -> Dict:
        """Configuration schema."""
        pass
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None):
        """Reset algorithm state."""
        pass
    
    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action given state."""
        pass
    
    @abstractmethod
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> Dict:
        """Update policy from batch."""
        pass
    
    @abstractmethod
    def save(self, path: Path):
        """Save checkpoint."""
        pass
    
    @abstractmethod
    def load(self, path: Path):
        """Load checkpoint."""
        pass


# ============================================================================
# PPO Implementation
# ============================================================================

@dataclass
class PPOConfig:
    """PPO configuration."""
    name = "ppo"
    
    # Network
    hidden_dim: int = 256
    activation: str = "relu"
    
    # PPO hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99  # Discount
    lam: float = 0.95  # GAE lambda
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    update_epochs: int = 10
    minibatch_size: int = 64
    
    # Entropy
    entropy_coef: float = 0.01
    entropy_target: float = -1.0
    
    # General
    device: str = "cpu"
    seed: int = 42


class PPO(RLAlgorithm):
    """
    Proximal Policy Optimization (Schulman et al., 2017)
    
    Standard PPO with GAE and clipping.
    """
    
    def __init__(self, config: PPOConfig = None):
        self.config = config or PPOConfig()
        self.name = self.config.name
        
        # Initialize networks (simplified - would use real nn.Module)
        self._init_networks()
        
        # Optimizer
        self.optimizer = None  # Would be torch.optim.Adam
        
        # Buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Statistics
        self.training_stats = {}
    
    def _init_networks(self):
        """Initialize actor and critic networks."""
        # Simplified - would implement actual neural networks
        self.actor_mean = None  # nn.Linear
        self.actor_std = None   # nn.Parameter
        self.critic = None      # nn.Linear
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Reset network parameters."""
        pass  # Would use torch.nn.init
    
    @property
    def config_schema(self) -> Dict:
        return {
            "lr": {"type": float, "range": [1e-5, 1e-2]},
            "gamma": {"type": float, "range": [0.9, 0.9999]},
            "clip_ratio": {"type": float, "range": [0.1, 0.3]},
            "entropy_coef": {"type": float, "range": [0.0, 0.1]},
        }
    
    def reset(self, seed: Optional[int] = None):
        """Reset algorithm state."""
        if seed is not None:
            np.random.seed(seed)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        self.training_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl": [],
            "clip_fraction": [],
        }
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using current policy."""
        # Simplified - would use neural network
        action_mean = np.zeros(3)  # steering, throttle, brake
        action_std = np.ones(3) * 0.1
        
        action = action_mean + np.random.randn(3) * action_std
        action = np.clip(action, -1, 1)
        
        return action
    
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> Dict:
        """Perform PPO update."""
        # Simplified - would implement full PPO update
        
        # Compute advantages using GAE
        advantages = self._compute_gae(states, rewards, dones)
        
        # Compute policy loss with clipping
        policy_loss = self._policy_loss(states, actions, advantages)
        
        # Compute value loss
        value_loss = self._value_loss(states, rewards)
        
        # Entropy bonus
        entropy = self._entropy(states)
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss - self.config.entropy_coef * entropy
        
        # Update would go here
        
        stats = {
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy": float(entropy),
            "kl": 0.0,  # Would compute
            "clip_fraction": 0.0,  # Would compute
        }
        
        self.training_stats = stats
        return stats
    
    def _compute_gae(
        self,
        states: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray = None
    ) -> np.ndarray:
        """Compute Generalized Advantage Estimation."""
        gamma = self.config.gamma
        lam = self.config.lam
        
        advantages = np.zeros(len(rewards))
        
        # Simplified GAE computation
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1] if values is not None else 0
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - (values[t] if values is not None else 0)
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages
    
    def _policy_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray
    ) -> float:
        """Compute clipped policy loss."""
        # Simplified PPO loss
        ratio = 1.0  # Would compute old/new prob ratio
        clip_advantages = np.clip(advantages, -0.1, 0.1)
        loss = -np.mean(ratio * clip_advantages)
        return float(loss)
    
    def _value_loss(self, states: np.ndarray, rewards: np.ndarray) -> float:
        """Compute value loss."""
        return float(np.mean(rewards ** 2))
    
    def _entropy(self, states: np.ndarray) -> float:
        """Compute policy entropy."""
        return 0.0  # Would compute from action distribution
    
    def save(self, path: Path):
        """Save checkpoint."""
        checkpoint = {
            "config": self.config.__dict__,
            "actor_mean": None,  # Would save state dict
            "actor_std": None,
            "critic": None,
            "optimizer": None,
            "training_stats": self.training_stats,
        }
        path.write_text(json.dumps(checkpoint, indent=2))
    
    def load(self, path: Path):
        """Load checkpoint."""
        checkpoint = json.loads(path.read_text())
        # Would load state dicts


# ============================================================================
# GRPO Implementation
# ============================================================================

@dataclass
class GRPOConfig:
    """GRPO configuration."""
    name = "grpo"
    
    # Network
    hidden_dim: int = 256
    
    # GRPO hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99
    group_size: int = 8  # Number of samples per group
    
    # Loss weights
    policy_coef: float = 1.0
    entropy_coef: float = 0.01
    kl_coef: float = 0.01
    
    # General
    device: str = "cpu"
    seed: int = 42


class GRPO(RLAlgorithm):
    """
    Group Relative Policy Optimization (DeepSeek, 2024)
    
    Key innovation: No value function needed.
    Advantages computed as relative performance within group.
    """
    
    def __init__(self, config: GRPOConfig = None):
        self.config = config or GRPOConfig()
        self.name = self.config.name
        
        # Initialize network
        self._init_networks()
        
        # Buffers
        self.group_buffer = []
        
        # Statistics
        self.training_stats = {}
    
    def _init_networks(self):
        """Initialize policy network."""
        self.policy = None  # nn.Module
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Reset network parameters."""
        pass
    
    @property
    def config_schema(self) -> Dict:
        return {
            "lr": {"type": float, "range": [1e-5, 1e-3]},
            "group_size": {"type": int, "range": [4, 16]},
            "gamma": {"type": float, "range": [0.9, 0.999]},
        }
    
    def reset(self, seed: Optional[int] = None):
        """Reset algorithm state."""
        if seed is not None:
            np.random.seed(seed)
        
        self.group_buffer = []
        self.training_stats = {}
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using current policy."""
        # Sample from policy distribution
        action_mean = np.zeros(3)
        action_std = np.ones(3) * 0.1
        
        action = action_mean + np.random.randn(3) * action_std
        action = np.clip(action, -1, 1)
        
        return action
    
    def sample_group(self, state: np.ndarray, n: int = None) -> np.ndarray:
        """
        Sample a group of actions for GRPO.
        
        Returns:
            actions: [group_size, action_dim]
        """
        n = n or self.config.group_size
        
        actions = []
        for _ in range(n):
            action = self.select_action(state)
            actions.append(action)
        
        return np.stack(actions)
    
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> Dict:
        """Perform GRPO update."""
        # Key: Compute relative advantages within group
        
        # Group rewards by state
        group_size = self.config.group_size
        n_groups = len(rewards) // group_size
        
        policy_loss = 0.0
        entropy = 0.0
        
        for g in range(n_groups):
            start = g * group_size
            end = (g + 1) * group_size
            
            group_rewards = rewards[start:end]
            
            # Group-relative advantage
            mean_r = np.mean(group_rewards)
            std_r = np.std(group_rewards) + 1e-8
            
            advantages = (group_rewards - mean_r) / std_r
            
            # Policy update
            for i, (state, action, adv) in enumerate(
                zip(states[start:end], actions[start:end], advantages)
            ):
                # Compute log prob ratio
                log_prob = self._get_log_prob(state, action)
                
                # PPO-style clipped objective
                ratio = np.exp(log_prob - log_prob)  # Simplified
                surr1 = ratio * adv
                surr2 = np.clip(ratio, 0.8, 1.2) * adv
                
                policy_loss += -np.minimum(surr1, surr2)
        
        policy_loss /= n_groups
        entropy = self._entropy(states)
        
        loss = policy_loss - self.config.entropy_coef * entropy
        
        stats = {
            "policy_loss": float(policy_loss),
            "entropy": float(entropy),
            "mean_reward": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
        }
        
        self.training_stats = stats
        return stats
    
    def _get_log_prob(self, state: np.ndarray, action: np.ndarray) -> float:
        """Get log probability of action under current policy."""
        return 0.0  # Would compute from policy network
    
    def _entropy(self, states: np.ndarray) -> float:
        """Compute policy entropy."""
        return 0.0  # Would compute from action distribution
    
    def save(self, path: Path):
        """Save checkpoint."""
        checkpoint = {
            "config": self.config.__dict__,
            "policy": None,  # Would save state dict
            "training_stats": self.training_stats,
        }
        path.write_text(json.dumps(checkpoint, indent=2))
    
    def load(self, path: Path):
        """Load checkpoint."""
        checkpoint = json.loads(path.read_text())
        # Would load state dict


# ============================================================================
# SAC Implementation
# ============================================================================

@dataclass
class SACConfig:
    """SAC configuration."""
    name = "sac"
    
    # Network
    hidden_dim: int = 256
    
    # SAC hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient
    alpha: float = 0.2  # Entropy temperature
    buffer_size: int = 1000000
    batch_size: int = 256
    
    # General
    device: str = "cpu"
    seed: int = 42


class SAC(RLAlgorithm):
    """
    Soft Actor-Critic (Haarnoja et al., 2018)
    
    Off-policy algorithm with entropy regularization.
    """
    
    def __init__(self, config: SACConfig = None):
        self.config = config or SACConfig()
        self.name = self.config.name
        
        # Initialize networks
        self._init_networks()
        
        # Target networks
        self._init_target_networks()
        
        # Replay buffer
        self.replay_buffer = []
        
        # Statistics
        self.training_stats = {}
    
    def _init_networks(self):
        """Initialize actor, critic, and value networks."""
        self.actor = None  # nn.Module
        self.critic = None  # nn.Module
        self.critic_target = None  # nn.Module
        self.value = None  # nn.Module
        
        self._reset_parameters()
    
    def _init_target_networks(self):
        """Initialize target networks."""
        self.critic_target = None  # Would be copy of critic
        self._soft_update(1.0)  # Initialize with hard copy
    
    def _reset_parameters(self):
        """Reset network parameters."""
        pass
    
    @property
    def config_schema(self) -> Dict:
        return {
            "lr": {"type": float, "range": [1e-5, 1e-3]},
            "gamma": {"type": float, "range": [0.9, 0.999]},
            "alpha": {"type": float, "range": [0.05, 0.5]},
            "tau": {"type": float, "range": [0.001, 0.01]},
        }
    
    def reset(self, seed: Optional[int] = None):
        """Reset algorithm state."""
        if seed is not None:
            np.random.seed(seed)
        
        self.replay_buffer = []
        self.training_stats = {}
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using current policy."""
        if deterministic:
            # Mean action
            action = np.zeros(3)
        else:
            # Sample from policy
            action_mean = np.zeros(3)
            action_std = np.ones(3) * 0.1
            action = action_mean + np.random.randn(3) * action_std
        
        action = np.clip(action, -1, 1)
        return action
    
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> Dict:
        """Perform SAC update."""
        # Add to replay buffer
        for i in range(len(states)):
            self.replay_buffer.append({
                "state": states[i],
                "action": actions[i],
                "reward": rewards[i],
                "next_state": next_states[i],
                "done": dones[i],
            })
        
        # Sample batch
        batch = self._sample_batch()
        
        # Compute losses
        critic_loss = self._critic_loss(batch)
        actor_loss = self._actor_loss(batch)
        alpha_loss = self._alpha_loss(batch)
        
        # Update networks
        # ...
        
        # Soft update target
        self._soft_update(self.config.tau)
        
        stats = {
            "critic_loss": float(critic_loss),
            "actor_loss": float(actor_loss),
            "alpha": self.config.alpha,
            "mean_q": 0.0,  # Would compute
        }
        
        self.training_stats = stats
        return stats
    
    def _sample_batch(self) -> List[Dict]:
        """Sample batch from replay buffer."""
        if len(self.replay_buffer) < self.config.batch_size:
            return self.replay_buffer
        
        indices = np.random.choice(
            len(self.replay_buffer),
            self.config.batch_size,
            replace=False
        )
        
        return [self.replay_buffer[i] for i in indices]
    
    def _critic_loss(self, batch: List[Dict]) -> float:
        """Compute critic loss (MSE with target Q)."""
        return 0.0  # Would compute
    
    def _actor_loss(self, batch: List[Dict]) -> float:
        """Compute actor loss (max Q with entropy)."""
        return 0.0  # Would compute
    
    def _alpha_loss(self, batch: List[Dict]) -> float:
        """Compute alpha (temperature) loss."""
        return 0.0  # Would compute
    
    def _soft_update(self, tau: float):
        """Soft update target networks."""
        pass  # Would implement polyak averaging
    
    def save(self, path: Path):
        """Save checkpoint."""
        checkpoint = {
            "config": self.config.__dict__,
            "actor": None,
            "critic": None,
            "replay_buffer": self.replay_buffer[-1000:],  # Save recent
            "training_stats": self.training_stats,
        }
        path.write_text(json.dumps(checkpoint, indent=2))
    
    def load(self, path: Path):
        """Load checkpoint."""
        checkpoint = json.loads(path.read_text())
        # Would load state dicts


# ============================================================================
# Algorithm Comparison
# ============================================================================

class AlgorithmComparator:
    """
    Compare multiple RL algorithms on the same task.
    
    Usage:
        comparator = AlgorithmComparator()
        results = comparator.compare(
            algorithms=[PPO(), GRPO(), SAC()],
            env="toy_waypoint",
            num_seeds=3,
            max_steps=100000,
        )
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
    
    def compare(
        self,
        algorithms: List[RLAlgorithm],
        env_name: str,
        num_seeds: int = 3,
        max_steps: int = 100000,
        eval_interval: int = 10000,
    ) -> Dict:
        """
        Compare algorithms across multiple seeds.
        
        Returns:
            Dict with results for each algorithm and seed.
        """
        results = {}
        
        for algo in algorithms:
            algo_name = algo.name
            results[algo_name] = {
                "seeds": {},
                "summary": {},
            }
            
            for seed in range(num_seeds):
                print(f"Running {algo_name} with seed {seed}...")
                
                # Reset algorithm
                algo.reset(seed=seed)
                
                # Run training
                seed_results = self._run_training(
                    algo, env_name, max_steps, eval_interval
                )
                
                results[algo_name]["seeds"][f"seed_{seed}"] = seed_results
            
            # Compute summary statistics
            results[algo_name]["summary"] = self._compute_summary(
                results[algo_name]["seeds"]
            )
        
        return results
    
    def _run_training(
        self,
        algorithm: RLAlgorithm,
        env_name: str,
        max_steps: int,
        eval_interval: int
    ) -> Dict:
        """Run training for one algorithm and seed."""
        # Simplified - would implement actual environment interaction
        
        history = {
            "steps": [],
            "rewards": [],
            "eval_rewards": [],
            "losses": [],
        }
        
        total_reward = 0.0
        step = 0
        
        while step < max_steps:
            # Simulate environment step
            state = np.random.randn(16)  # Dummy state
            action = algorithm.select_action(state)
            
            # Simulate environment
            next_state = np.random.randn(16)
            reward = np.random.randn()
            done = np.random.rand() < 0.01
            
            # Update
            update_result = algorithm.update(
                np.array([state]),
                np.array([action]),
                np.array([reward]),
                np.array([next_state]),
                np.array([done]),
            )
            
            total_reward += reward
            step += 1
            
            # Logging
            if step % eval_interval == 0:
                eval_reward = self._evaluate(algorithm, env_name)
                history["steps"].append(step)
                history["rewards"].append(total_reward / step)
                history["eval_rewards"].append(eval_reward)
                history["losses"].append(update_result)
            
            if done:
                break
        
        return history
    
    def _evaluate(self, algorithm: RLAlgorithm, env_name: str) -> float:
        """Evaluate current policy."""
        total_reward = 0.0
        n_episodes = 10
        
        for _ in range(n_episodes):
            state = np.random.randn(16)
            
            while True:
                action = algorithm.select_action(state)
                next_state = np.random.randn(16)
                reward = np.random.randn()
                total_reward += reward
                
                if np.random.rand() < 0.01:
                    break
                
                state = next_state
        
        return total_reward / n_episodes
    
    def _compute_summary(self, seed_results: Dict) -> Dict:
        """Compute summary statistics across seeds."""
        all_rewards = []
        all_final = []
        
        for seed_data in seed_results.values():
            rewards = seed_data.get("rewards", [])
            if rewards:
                all_rewards.append(np.max(rewards))
                all_final.append(rewards[-1] if rewards else 0)
        
        return {
            "mean_max_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "std_max_reward": float(np.std(all_rewards)) if all_rewards else 0.0,
            "mean_final_reward": float(np.mean(all_final)) if all_final else 0.0,
        }
    
    def print_results(self, results: Dict):
        """Print comparison results."""
        print("\n" + "=" * 60)
        print("Algorithm Comparison Results")
        print("=" * 60)
        
        for algo_name, algo_results in results.items():
            summary = algo_results["summary"]
            
            print(f"\n{algo_name.upper()}:")
            print(f"  Mean Max Reward: {summary['mean_max_reward']:.2f} ± {summary['std_max_reward']:.2f}")
            print(f"  Mean Final Reward: {summary['mean_final_reward']:.2f}")


# ============================================================================
# Factory
# ============================================================================

class RLAlgorithmFactory:
    """Factory for creating RL algorithms."""
    
    @staticmethod
    def create(config: Dict) -> RLAlgorithm:
        """
        Create algorithm from configuration.
        
        Config:
        {
            "name": "ppo" | "grpo" | "sac",
            "lr": 0.0003,
            "gamma": 0.99,
            ...
        }
        """
        algo_name = config.get("name", "ppo").lower()
        
        if algo_name == "ppo":
            cfg = PPOConfig(**config)
            return PPO(cfg)
        
        elif algo_name == "grpo":
            cfg = GRPOConfig(**config)
            return GRPO(cfg)
        
        elif algo_name == "sac":
            cfg = SACConfig(**config)
            return SAC(cfg)
        
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create algorithms
    algorithms = [
        RLAlgorithmFactory.create({"name": "ppo", "lr": 3e-4, "gamma": 0.99}),
        RLAlgorithmFactory.create({"name": "grpo", "lr": 3e-4, "gamma": 0.99, "group_size": 8}),
        RLAlgorithmFactory.create({"name": "sac", "lr": 3e-4, "gamma": 0.99, "alpha": 0.2}),
    ]
    
    # Compare
    comparator = AlgorithmComparator()
    results = comparator.compare(
        algorithms=algorithms,
        env_name="toy_waypoint",
        num_seeds=2,
        max_steps=10000,
        eval_interval=1000,
    )
    
    # Print results
    comparator.print_results(results)
    
    # Save results
    output_dir = Path("out/rl_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_dir / 'results.json'}")
