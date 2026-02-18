"""
SAC (Soft Actor-Critic) Implementation
====================================

SAC is an off-policy RL algorithm with automatic entropy regularization.

Key Features:
- Automatic entropy tuning
- Twin Q-networks (Clipped Double Q)
- Target networks for stability
- Off-policy with replay buffer
- Entropy maximization for exploration

Reference: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy RL" (2018)

Usage:
    from training.rl.sac import SAC, SACTrainer, ReplayBuffer
    
    sac = SAC(policy, q_network, config)
    sac.update(replay_buffer, batch_size=256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import copy


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SACConfig:
    """
    Configuration for SAC algorithm.
    
    Attributes:
        - policy: The policy network
        - q_network: Q-value network
        - optimizer: Optimizer for policy and Q-network
        - gamma: Discount factor
        - tau: Target network update rate (soft update)
        - alpha: Entropy temperature (auto-tuned if None)
        - target_entropy: Target entropy for automatic alpha tuning
        - learning_rate: Learning rate for all networks
        - batch_size: Batch size for updates
        - buffer_size: Replay buffer size
        - start_steps: Number of random steps before learning
        - update_after: Steps before starting updates
        - target_update_interval: Steps between target updates
        - clip_grad: Gradient clipping threshold
    """
    # Core
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient
    alpha: Optional[float] = None  # None = auto-tune
    
    # Entropy
    target_entropy: Optional[float] = None  # None = -action_dim
    
    # Learning
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1000000
    start_steps: int = 10000
    update_after: int = 1000
    target_update_interval: int = 1
    
    # Gradient
    clip_grad: float = 1.0
    
    # Q-network
    hidden_dim: int = 256
    q_hidden_dim: int = 256
    
    # Logging
    log_interval: int = 100
    
    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.gamma < 1.0, "gamma must be in (0, 1)"
        assert 0 < self.tau <= 1.0, "tau must be in (0, 1]"
        assert self.batch_size > 0, "batch_size must be positive"


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """
    Experience replay buffer for off-policy RL.
    
    Stores transitions (s, a, r, s', d) and samples mini-batches.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int = 1000000,
        device: str = 'cuda',
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device
        
        # Allocate buffers
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        # Indices
        self.ptr = 0
        self.size = 0
        
        # For priority (optional)
        self.priorities = np.ones(buffer_size, dtype=np.float32)
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ):
        """Add transition to buffer."""
        idx = self.ptr
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        
        # Priority (TD error, will be updated during learning)
        self.priorities[idx] = 1.0
        
        # Update pointer and size
        self.ptr = (idx + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample mini-batch from buffer.
        
        Returns:
            states, actions, rewards, next_states, dones, weights, indices
        """
        # Sample uniformly
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # Get data
        states = torch.tensor(self.states[indices], dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions[indices], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(self.next_states[indices], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones[indices], dtype=torch.float32, device=self.device)
        
        # Weights (for PER, default = 1)
        weights = torch.tensor(self.priorities[indices], dtype=torch.float32, device=self.device)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray,
    ):
        """Update priorities after learning."""
        self.priorities[indices] = priorities + 1e-6
    
    def __len__(self):
        return self.size


# ============================================================================
# Q-Network (Critic)
# ============================================================================

class QNetwork(nn.Module):
    """
    Q-value network (Critic).
    
    Predicts Q(s, a) for given state and action.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: [B, state_dim]
            action: [B, action_dim]
            
        Returns:
            q1: [B] Q-value from first network
            q2: [B] Q-value from second network
        """
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class TwinnedQNetwork(nn.Module):
    """
    Twin Q-networks with shared state encoder.
    
    Used in SAC for more stable Q-learning.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Separate Q-heads
        self.q1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        
        self.q1_out = nn.Linear(hidden_dim, 1)
        self.q2_out = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            q1, q2: [B] Q-values from each network
        """
        h = self.encoder(state)
        x = torch.cat([h, action], dim=-1)
        
        q1 = F.relu(self.q1(x))
        q2 = F.relu(self.q2(x))
        
        return self.q1_out(q1), self.q2_out(q2)
    
    def q1_forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Q1 only forward pass (used for policy update)."""
        h = self.encoder(state)
        x = torch.cat([h, action], dim=-1)
        q1 = F.relu(self.q1(x))
        return self.q1_out(q1)


# ============================================================================
# Policy Network (Actor)
# ============================================================================

class GaussianPolicy(nn.Module):
    """
    Gaussian policy network for continuous action spaces.
    
    Outputs mean and std of action distribution.
    Uses reparameterization trick for training.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        
        # Network
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Mean output
        self.mean = nn.Linear(hidden_dim, action_dim)
        
        # Log std output (learnable)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Action bounds
        self.register_buffer('action_scale', torch.tensor(1.0))
        self.register_buffer('action_bias', torch.tensor(0.0))
        
        if action_bounds is not None:
            action_low, action_high = action_bounds
            self.action_scale = (action_high - action_low) / 2
            self.action_bias = (action_high + action_low) / 2
    
    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            mean: [B, action_dim] mean of action distribution
            log_std: [B, action_dim] log std of action distribution
        """
        h = self.net(state)
        mean = self.mean(h)
        log_std = self.log_std.clamp(-20, 2)
        
        return mean, log_std
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action from state.
        
        Args:
            state: [state_dim] or [B, state_dim]
            deterministic: If True, return mean (exploitation)
            
        Returns:
            action: [action_dim] or [B, action_dim]
            log_prob: [1] log probability of selected action
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        mean, log_std = self.forward(state)
        
        if deterministic:
            action = mean
        else:
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()  # Reparameterized sample
        
        # Squash to bounds
        action = torch.tanh(action)
        action = action * self.action_scale + self.action_bias
        
        # Compute log probability
        log_prob = self.get_log_prob(state, action)
        
        if state.dim() == 1:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
        
        return action, log_prob
    
    def get_log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of action.
        
        Includes tanh squashing correction.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Normal distribution
        dist = torch.distributions.Normal(mean, std)
        
        # Pre-tanh action
        action_pre_tanh = torch.atanh(
            torch.clamp((action - self.action_bias) / self.action_scale, -0.999, 0.999)
        )
        
        # Log probability with tanh correction
        log_prob = dist.log_prob(action_pre_tanh)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        
        return log_prob


class TanhGaussianPolicy(GaussianPolicy):
    """
    Tanh-Gaussian policy (same as GaussianPolicy in SAC).
    
    Uses tanh squashing for bounded actions.
    """
    
    def sample_with_temperature(
        self,
        state: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample with temperature scaling.
        
        Higher temperature = more exploration.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp() * temperature
        
        # Reparameterized sample
        dist = torch.distributions.Normal(mean, std)
        action_raw = dist.rsample()
        
        # Tanh squashing
        action = torch.tanh(action_raw)
        action = action * self.action_scale + self.action_bias
        
        # Log probability
        log_prob = self.get_log_prob(state, action)
        
        return action, log_prob


# ============================================================================
# SAC Core Algorithm
# ============================================================================

class SAC:
    """
    Soft Actor-Critic algorithm.
    
    SAC maximizes expected return + entropy:
        J(π) = E_{s,a ~ π} [r(s,a) + α * H(π(·|s))]
    
    The entropy coefficient α is automatically tuned.
    """
    
    def __init__(
        self,
        policy: nn.Module,
        q_network: nn.Module,
        config: SACConfig,
        device: str = 'cuda',
    ):
        self.policy = policy
        self.q_network = q_network
        self.config = config
        self.device = device
        
        # Target Q-network (for stability)
        self.q_target = copy.deepcopy(q_network)
        for param in self.q_target.parameters():
            param.requires_grad = False
        
        # Entropy coefficient (auto-tuned)
        if config.alpha is None:
            self.log_alpha = nn.Parameter(torch.tensor(0.0))
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.alpha = torch.tensor(config.alpha, device=device, dtype=torch.float32)
        
        # Target entropy
        if config.target_entropy is None:
            # Default: -action_dim (continuous action space)
            action_dim = policy.action_dim
            self.target_entropy = -action_dim
        else:
            self.target_entropy = config.target_entropy
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate
        )
        self.q_optimizer = torch.optim.Adam(
            q_network.parameters(), lr=config.learning_rate
        )
        
        # Training parameters
        self.global_step = 0
    
    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select action from state.
        
        Args:
            state: [state_dim]
            deterministic: If True, exploit
            
        Returns:
            action: [action_dim]
        """
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        action, _ = self.policy.get_action(state_tensor, deterministic=deterministic)
        
        return action.cpu().numpy().squeeze()
    
    def update(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """
        Update networks using mini-batch from replay buffer.
        
        Args:
            replay_buffer: Experience replay buffer
            batch_size: Mini-batch size
            
        Returns:
            Dictionary of training metrics
        """
        if len(replay_buffer) < batch_size:
            return {}
        
        if len(replay_buffer) < self.config.update_after:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(batch_size)
        
        # Compute target Q-values
        with torch.no_grad():
            # Next actions from policy
            next_actions, next_log_probs = self.policy.sample_with_temperature(next_states)
            
            # Target Q-values (minimum of two networks)
            next_q1, next_q2 = self.q_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            
            # Target
            target_q = rewards.unsqueeze(1) + self.config.gamma * (1 - dones.unsqueeze(1)) * next_q
        
        # Compute current Q-values
        q1, q2 = self.q_network(states, actions)
        
        # Q-loss (MSE with Huber loss)
        q_loss = F.smooth_l1_loss(q1, target_q, reduction='none') + \
                 F.smooth_l1_loss(q2, target_q, reduction='none')
        
        q_loss = (q_loss * weights.unsqueeze(1)).mean()
        
        # Update Q-network
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.clip_grad)
        self.q_optimizer.step()
        
        # Policy loss
        actions_new, log_probs = self.policy.sample_with_temperature(states)
        q1_new, q2_new = self.q_network(states, actions_new)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.clip_grad)
        self.policy_optimizer.step()
        
        # Alpha loss (if auto-tuning)
        alpha_loss = torch.tensor(0.0)
        if self.config.alpha is None:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().detach()
        
        # Soft update target network
        if self.global_step % self.config.target_update_interval == 0:
            self._soft_update(self.q_network, self.q_target)
        
        # Update priorities in buffer
        priorities = (q_loss.detach() + 1e-6).cpu().numpy()
        replay_buffer.update_priorities(indices, priorities)
        
        self.global_step += 1
        
        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha.item(),
            'mean_q': q1.mean().item(),
            'mean_action': actions_new.mean().item(),
        }
    
    def _soft_update(
        self,
        network: nn.Module,
        target_network: nn.Module,
    ):
        """Soft update target network."""
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
    
    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q_network_state_dict': self.q_network.state_dict(),
            'q_target_state_dict': self.q_target.state_dict(),
            'global_step': self.global_step,
            'log_alpha': self.log_alpha if self.config.alpha is None else None,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.q_target.load_state_dict(checkpoint['q_target_state_dict'])
        self.global_step = checkpoint['global_step']
        
        if self.config.alpha is None and checkpoint.get('log_alpha') is not None:
            self.log_alpha = checkpoint['log_alpha']


# ============================================================================
# SAC Trainer
# ============================================================================

class SACTrainer:
    """
    Trainer for SAC algorithm.
    
    Handles:
    - Environment interaction
    - Replay buffer management
    - Network updates
    - Logging and checkpointing
    """
    
    def __init__(
        self,
        sac: SAC,
        replay_buffer: ReplayBuffer,
        config,
        device: str = 'cuda',
    ):
        self.sac = sac
        self.replay_buffer = replay_buffer
        self.config = config
        self.device = device
        
        # Logging
        self.episode_count = 0
        self.episode_reward = 0
        self.rewards_history = []
        self.eval_rewards = []
    
    def collect_trajectories(
        self,
        env,
        num_steps: int = 10000,
        start_steps: Optional[int] = None,
    ):
        """
        Collect trajectories by interacting with environment.
        
        Args:
            env: Environment with step() and reset() methods
            num_steps: Total steps to collect
            start_steps: Random action steps before using policy
        """
        if start_steps is None:
            start_steps = self.config.start_steps
        
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(num_steps):
            # Random action in early steps
            if step < start_steps:
                action = env.action_space.sample() if hasattr(env, 'action_space') else np.random.randn(self.sac.policy.action_dim)
            else:
                action = self.sac.select_action(state, deterministic=False)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            self.replay_buffer.add(state, action, reward, next_state, float(done))
            
            # Update
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Update SAC
            update_metrics = self.sac.update(self.replay_buffer, self.config.batch_size)
            
            # Handle episode end
            if done or episode_length >= env.max_steps if hasattr(env, 'max_steps') else done:
                self.episode_count += 1
                self.rewards_history.append(episode_reward)
                
                if self.episode_count % self.config.log_interval == 0:
                    avg_reward = np.mean(self.rewards_history[-100:])
                    print(f"Episode {self.episode_count}: avg_reward={avg_reward:.2f}")
                
                # Reset
                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0
    
    def train(
        self,
        env,
        num_iterations: int = 1000,
        eval_env = None,
        eval_interval: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            env: Training environment
            num_iterations: Number of training iterations
            eval_env: Evaluation environment (optional)
            eval_interval: Evaluate every N iterations
            
        Returns:
            training_history: Dict of metric histories
        """
        history = {
            'rewards': [],
            'eval_rewards': [],
            'q_loss': [],
            'policy_loss': [],
            'alpha': [],
        }
        
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")
            
            # Collect trajectories and update
            self.collect_trajectories(env, num_steps=1000)
            
            # Average metrics from recent updates
            recent_rewards = self.rewards_history[-100:] if self.rewards_history else [0]
            history['rewards'].append(np.mean(recent_rewards))
            
            # Evaluation
            if eval_env is not None and (iteration + 1) % eval_interval == 0:
                eval_reward = self.evaluate(eval_env)
                history['eval_rewards'].append(eval_reward)
                print(f"Eval Reward: {eval_reward:.2f}")
        
        return history
    
    def evaluate(
        self,
        env,
        num_episodes: int = 5,
    ) -> float:
        """
        Evaluate current policy.
        
        Returns:
            Average return over episodes
        """
        self.sac.policy.eval()
        
        total_returns = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_return = 0
            
            while True:
                action = self.sac.select_action(state, deterministic=True)
                next_state, reward, done, _ = env.step(action)
                
                episode_return += reward
                state = next_state
                
                if done:
                    break
            
            total_returns.append(episode_return)
        
        self.sac.policy.train()
        
        return np.mean(total_returns)
    
    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.sac.save_checkpoint(path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        self.sac.load_checkpoint(path)
        print(f"Loaded checkpoint from {path}")


# ============================================================================
# Waypoint-Specific SAC
# ============================================================================

class WaypointSACPolicy(nn.Module):
    """
    SAC-compatible waypoint prediction policy.
    
    Wraps the AR decoder for SAC training.
    """
    
    def __init__(
        self,
        ar_decoder: nn.Module,
        config,
        action_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        super().__init__()
        self.ar_decoder = ar_decoder
        self.config = config
        
        # Waypoint prediction head
        self.waypoint_head = nn.Linear(
            config.hidden_dim,
            config.waypoint_dim * config.num_waypoints
        )
        
        # Action bounds
        if action_bounds is not None:
            action_low = torch.tensor(action_bounds[0], dtype=torch.float32)
            action_high = torch.tensor(action_bounds[1], dtype=torch.float32)
            self.action_bounds = (action_low, action_high)
        else:
            self.action_bounds = (-1.0, 1.0)  # Default [-1, 1]
        
        # Gaussian policy
        self.policy = GaussianPolicy(
            state_dim=config.hidden_dim,
            action_dim=config.waypoint_dim * config.num_waypoints,
            hidden_dim=256,
            action_bounds=self.action_bounds,
        )
    
    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            mean: [B, action_dim]
            log_std: [B, action_dim]
        """
        # Get features from AR decoder
        if hasattr(self.ar_decoder, 'forward'):
            features = self.ar_decoder.get_features(state)
        else:
            features = state
        
        return self.policy(features)
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action from state."""
        return self.policy.get_action(state, deterministic=deterministic)
    
    def get_log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get log probability of action."""
        return self.policy.get_log_prob(state, action)


# ============================================================================
# Integration: SFT + SAC
# ============================================================================

class WaypointBC_SACSplit(nn.Module):
    """
    Combined SFT + SAC training for waypoint prediction.
    
    Pattern:
    1. Pre-train with SFT (teacher forcing)
    2. Fine-tune with SAC (off-policy, entropy maximization)
    
    Benefits:
    - SFT provides good initialization
    - SAC provides exploration + entropy
    - Off-policy = more sample efficient
    """
    
    def __init__(
        self,
        sft_model: nn.Module,
        sac_config: SACConfig,
        state_dim: int,
        action_dim: int,
        device: str = 'cuda',
    ):
        super().__init__()
        
        self.sft_model = sft_model.to(device)
        self.sac_config = sac_config
        self.device = device
        
        # Freeze SFT model
        for param in sft_model.parameters():
            param.requires_grad = False
        
        # SAC components
        self.policy = GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,
        )
        
        self.q_network = TwinnedQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,
        )
        
        self.sac = SAC(
            policy=self.policy,
            q_network=self.q_network,
            config=sac_config,
            device=device,
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=sac_config.buffer_size,
            device=device,
        )
    
    def sft_pretrain(
        self,
        dataloader,
        epochs: int = 10,
    ) -> Dict[str, float]:
        """
        Pre-train with SFT.
        """
        self.sft_model.train()
        total_loss = 0
        
        for epoch in range(epochs):
            for batch in dataloader:
                features = batch['features'].to(self.device)
                waypoints = batch['waypoints'].to(self.device)
                
                output = self.sft_model(features, waypoints=waypoints)
                pred_wp = output['waypoints']
                
                loss = F.mse_loss(pred_wp, waypoints)
                
                self.sac.policy_optimizer.zero_grad()
                loss.backward()
                self.sac.policy_optimizer.step()
                
                total_loss += loss.item()
        
        return {'sft_loss': total_loss / len(dataloader)}
    
    def sac_finetune(
        self,
        env,
        num_iterations: int = 1000,
        batch_size: int = 256,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune with SAC.
        """
        # Copy SFT weights to policy
        self._copy_sft_to_policy()
        
        history = {
            'rewards': [],
            'q_loss': [],
            'policy_loss': [],
            'alpha': [],
        }
        
        trainer = SACTrainer(
            sac=self.sac,
            replay_buffer=self.replay_buffer,
            config=self.sac_config,
            device=self.device,
        )
        
        history = trainer.train(env, num_iterations=num_iterations)
        
        return history
    
    def _copy_sft_to_policy(self):
        """Copy SFT weights to SAC policy."""
        # Get SFT prediction as policy initialization
        with torch.no_grad():
            # Copy waypoint head weights
            if hasattr(self.sft_model, 'waypoint_head'):
                self.policy.mean.weight.copy_(self.sft_model.waypoint_head.weight[:self.policy.mean.weight.size(0)])
                self.policy.mean.bias.copy_(self.sft_model.waypoint_head.bias[:self.policy.mean.bias.size(0)])


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example of using SAC."""
    
    import gymnasium as gym
    
    # Configuration
    config = SACConfig(
        gamma=0.99,
        tau=0.005,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=1000000,
        start_steps=10000,
    )
    
    # Environment
    env = gym.make('HalfCheetah-v4')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Policy
    policy = GaussianPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        action_bounds=(env.action_space.low, env.action_space.high),
    )
    
    # Q-network
    q_network = TwinnedQNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
    )
    
    # SAC
    sac = SAC(policy, q_network, config, device='cpu')
    
    # Replay buffer
    buffer = ReplayBuffer(state_dim, action_dim, buffer_size=100000)
    
    # Collect some random data
    state, _ = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)
        buffer.add(state, action, reward, next_state, float(done))
        state = next_state
        if done:
            state, _ = env.reset()
    
    # Update
    for _ in range(10):
        metrics = sac.update(buffer, batch_size=64)
        print(f"Q Loss: {metrics.get('q_loss', 0):.4f}")
    
    print("SAC example completed!")


if __name__ == "__main__":
    example_usage()
