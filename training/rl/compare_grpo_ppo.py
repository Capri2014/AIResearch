#!/usr/bin/env python3
"""
GRPO vs PPO Comparison Utility

Compares Group Relative Policy Optimization (GRPO) against PPO
on the toy waypoint environment for RL after SFT.

Usage:
    python compare_grpo_ppo.py --episodes 200 --seeds 3
    python compare_grpo_ppo.py --smoke --episodes 50
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.rl.toy_kinematics_env import ToyKinematicsEnv
from training.rl.toy_kinematics_gym import ToyKinematicsEnvGym


@dataclass
class TrainingResult:
    """Results from a training run."""
    algorithm: str
    seed: int
    episodes: int
    final_reward: float
    reward_std: float
    final_goal_rate: float
    training_time: float
    checkpoint_path: str
    metrics: dict


def run_ppo_training(env: ToyKinematicsEnv, episodes: int, seed: int, lr: float = 3e-4) -> TrainingResult:
    """Run PPO training on the toy environment."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    state_dim = env.state_dim
    waypoint_dim = env.waypoint_dim
    action_dim = 2  # delta waypoint adjustment
    
    # Policy network: state -> mean, log_std
    policy_net = nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, action_dim * 2)  # mean + log_std
    )
    
    # Value network
    value_net = nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr)
    optimizer_value = optim.Adam(value_net.parameters(), lr=lr)
    
    gamma = 0.99
    epsilon = 0.2
    lam = 0.95
    horizon = 20
    updates_per_episode = 4
    
    rewards_history = []
    start_time = time.time()
    
    for episode in range(episodes):
        states, actions, rewards, values, log_probs = [], [], [], [], []
        state = env.reset(seed=seed + episode)
        
        for t in range(horizon):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            
            # Policy forward
            policy_out = policy_net(state_t)
            mean = policy_out[:, :action_dim]
            log_std = policy_out[:, action_dim:]
            log_std = torch.tanh(log_std)
            std = torch.exp(log_std).clamp(1e-4, 1.0)
            
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Value forward
            value = value_net(state_t).item()
            
            # Environment step
            next_state, reward, done, _ = env.step(action[0].numpy())
            
            states.append(state)
            actions.append(action.detach().numpy())
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob.item())
            
            state = next_state
            
            if done:
                break
        
        # Compute returns and advantages
        returns = []
        advantages = []
        R = 0
        V_next = 0
        
        for r, v in zip(reversed(rewards), reversed(values + [V_next])):
            R = r + gamma * R
            advantage = R - v
            advantages.insert(0, advantage)
            returns.insert(0, R)
        
        advantages = np.array(advantages)
        returns = np.array(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(updates_per_episode):
            states_t = torch.FloatTensor(np.array(states))
            actions_t = torch.FloatTensor(np.array(actions))
            returns_t = torch.FloatTensor(returns)
            advantages_t = torch.FloatTensor(advantages)
            old_log_probs = torch.FloatTensor(log_probs)
            
            # Policy loss
            policy_out = policy_net(states_t)
            mean = policy_out[:, :action_dim]
            log_std = policy_out[:, action_dim:]
            log_std = torch.tanh(log_std)
            std = torch.exp(log_std).clamp(1e-4, 1.0)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions_t).sum(dim=-1)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values_pred = value_net(states_t).squeeze()
            value_loss = nn.functional.mse_loss(values_pred, returns_t)
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Update
            optimizer_policy.zero_grad()
            (policy_loss - 0.01 * entropy).backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
            optimizer_policy.step()
            
            optimizer_value.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            optimizer_value.step()
        
        # Track metrics
        episode_reward = sum(rewards)
        rewards_history.append(episode_reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"  PPO Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}")
    
    # Final evaluation
    eval_rewards = []
    eval_goals = 0
    eval_episodes = 20
    
    for _ in range(eval_episodes):
        state = env.reset(seed=seed + 10000)
        episode_reward = 0
        done = False
        
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                policy_out = policy_net(state_t)
                mean = policy_out[:, :action_dim]
                action = mean[0].numpy()
            
            state, reward, done, info = env.step(action)
            episode_reward += reward
            
            if info.get('goal_reached', False):
                eval_goals += 1
        
        eval_rewards.append(episode_reward)
    
    training_time = time.time() - start_time
    
    # Save checkpoint
    os.makedirs('out/compare_grpo_ppo', exist_ok=True)
    checkpoint_path = f'out/compare_grpo_ppo/ppo_seed{seed}.pt'
    torch.save({
        'policy_state_dict': policy_net.state_dict(),
        'value_state_dict': value_net.state_dict(),
    }, checkpoint_path)
    
    return TrainingResult(
        algorithm='PPO',
        seed=seed,
        episodes=episodes,
        final_reward=np.mean(eval_rewards),
        reward_std=np.std(eval_rewards),
        final_goal_rate=eval_goals / eval_episodes,
        training_time=training_time,
        checkpoint_path=checkpoint_path,
        metrics={
            'rewards_history': rewards_history,
            'final_ade': np.mean([abs(r) for r in eval_rewards]),  # proxy
        }
    )


def run_grpo_training(env: ToyKinematicsEnv, episodes: int, seed: int, lr: float = 3e-4) -> TrainingResult:
    """Run GRPO training on the toy environment."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    state_dim = env.state_dim
    action_dim = 2  # delta waypoint adjustment
    group_size = 4  # Sample 4 actions per state
    
    # Policy network (no value function needed for GRPO)
    policy_net = nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, action_dim * 2)  # mean + log_std
    )
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    gamma = 0.99
    epsilon = 0.2
    horizon = 20
    
    rewards_history = []
    start_time = time.time()
    
    for episode in range(episodes):
        # GRPO: sample multiple actions per state
        states = []
        all_rewards = []
        all_log_probs = []
        
        state = env.reset(seed=seed + episode)
        
        for t in range(horizon):
            # Sample group of actions
            state_t = torch.FloatTensor(state).unsqueeze(0).expand(group_size, -1)
            
            policy_out = policy_net(state_t)
            mean = policy_out[:, :action_dim]
            log_std = policy_out[:, action_dim:]
            log_std = torch.tanh(log_std)
            std = torch.exp(log_std).clamp(1e-4, 1.0)
            
            dist = Normal(mean, std)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # Get rewards for all actions
            group_rewards = []
            for i in range(group_size):
                # Clone state for each action
                next_state, reward, done, _ = env.step(actions[i].numpy())
                group_rewards.append(reward)
                
                if i == 0:  # Use first action's next state
                    state = next_state
                    done_flag = done
            
            # Store group data
            states.append(state_t.cpu().numpy())
            all_rewards.append(group_rewards)
            all_log_probs.append(log_probs.detach().cpu().numpy())
            
            if done_flag:
                break
        
        # Compute group-relative advantages
        for t in range(len(all_rewards)):
            group_rewards = np.array(all_rewards[t])
            group_log_probs = np.array(all_log_probs[t])
            
            # Normalize rewards within group
            if len(group_rewards) > 1:
                group_advantages = (group_rewards - group_rewards.mean()) / (group_rewards.std() + 1e-8)
            else:
                group_advantages = np.zeros_like(group_rewards)
            
            # GRPO policy update
            states_t = torch.FloatTensor(states[t])
            actions_t = torch.FloatTensor(actions[t].numpy() if t < len(actions) else np.zeros((group_size, action_dim)))
            
            policy_out = policy_net(states_t)
            mean = policy_out[:, :action_dim]
            log_std = policy_out[:, action_dim:]
            log_std = torch.tanh(log_std)
            std = torch.exp(log_std).clamp(1e-4, 1.0)
            
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions_t).sum(dim=-1)
            
            # Group-relative advantage
            ratio = torch.exp(new_log_probs - torch.FloatTensor(group_log_probs))
            advantages = torch.FloatTensor(group_advantages)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
            optimizer.step()
        
        # Track metrics
        episode_reward = np.mean([np.mean(r) for r in all_rewards]) if all_rewards else 0
        rewards_history.append(episode_reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"  GRPO Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}")
    
    # Final evaluation
    eval_rewards = []
    eval_goals = 0
    eval_episodes = 20
    
    for _ in range(eval_episodes):
        state = env.reset(seed=seed + 10000)
        episode_reward = 0
        done = False
        
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                policy_out = policy_net(state_t)
                mean = policy_out[:, :action_dim]
                action = mean[0].numpy()
            
            state, reward, done, info = env.step(action)
            episode_reward += reward
            
            if info.get('goal_reached', False):
                eval_goals += 1
        
        eval_rewards.append(episode_reward)
    
    training_time = time.time() - start_time
    
    # Save checkpoint
    os.makedirs('out/compare_grpo_ppo', exist_ok=True)
    checkpoint_path = f'out/compare_grpo_ppo/grpo_seed{seed}.pt'
    torch.save({
        'policy_state_dict': policy_net.state_dict(),
    }, checkpoint_path)
    
    return TrainingResult(
        algorithm='GRPO',
        seed=seed,
        episodes=episodes,
        final_reward=np.mean(eval_rewards),
        reward_std=np.std(eval_rewards),
        final_goal_rate=eval_goals / eval_episodes,
        training_time=training_time,
        checkpoint_path=checkpoint_path,
        metrics={
            'rewards_history': rewards_history,
        }
    )


def main():
    parser = argparse.ArgumentParser(description='Compare GRPO vs PPO on toy waypoint environment')
    parser.add_argument('--episodes', type=int, default=200, help='Training episodes per seed')
    parser.add_argument('--seeds', type=int, default=3, help='Number of seeds to average over')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--smoke', action='store_true', help='Quick smoke test')
    parser.add_argument('--output', type=str, default='out/compare_grpo_ppo/metrics.json',
                        help='Output metrics file')
    
    args = parser.parse_args()
    
    if args.smoke:
        args.episodes = 50
        args.seeds = 2
    
    print(f"Comparing GRPO vs PPO on Toy Waypoint Environment")
    print(f"=" * 60)
    print(f"Episodes: {args.episodes}, Seeds: {args.seeds}, LR: {args.lr}")
    print()
    
    env = ToyKinematicsEnv()
    
    # Run PPO
    print("\n--- Training PPO ---")
    ppo_results = []
    for seed in range(args.seeds):
        print(f"Seed {seed}:")
        result = run_ppo_training(env, args.episodes, seed, args.lr)
        ppo_results.append(result)
    
    # Run GRPO
    print("\n--- Training GRPO ---")
    env = ToyKinematicsEnv()  # Fresh environment
    grpo_results = []
    for seed in range(args.seeds):
        print(f"Seed {seed}:")
        result = run_grpo_training(env, args.episodes, seed, args.lr)
        grpo_results.append(result)
    
    # Aggregate results
    ppo_rewards = [r.final_reward for r in ppo_results]
    ppo_goal_rates = [r.final_goal_rate for r in ppo_results]
    ppo_times = [r.training_time for r in ppo_results]
    
    grpo_rewards = [r.final_reward for r in grpo_results]
    grpo_goal_rates = [r.final_goal_rate for r in grpo_results]
    grpo_times = [r.training_time for r in grpo_results]
    
    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"\nPPO ({args.seeds} seeds):")
    print(f"  Final Reward: {np.mean(ppo_rewards):.3f} ± {np.std(ppo_rewards):.3f}")
    print(f"  Goal Rate:    {np.mean(ppo_goal_rates)*100:.1f}% ± {np.std(ppo_goal_rates)*100:.1f}%")
    print(f"  Time:         {np.mean(ppo_times):.1f}s ± {np.std(ppo_times):.1f}s")
    
    print(f"\nGRPO ({args.seeds} seeds):")
    print(f"  Final Reward: {np.mean(grpo_rewards):.3f} ± {np.std(grpo_rewards):.3f}")
    print(f"  Goal Rate:    {np.mean(grpo_goal_rates)*100:.1f}% ± {np.std(grpo_goal_rates)*100:.1f}%")
    print(f"  Time:         {np.mean(grpo_times):.1f}s ± {np.std(grpo_times):.1f}s")
    
    # Compute deltas
    reward_delta = np.mean(grpo_rewards) - np.mean(ppo_rewards)
    goal_delta = (np.mean(grpo_goal_rates) - np.mean(ppo_goal_rates)) * 100
    time_delta = np.mean(grpo_times) - np.mean(ppo_times)
    
    print(f"\nGRPO vs PPO:")
    print(f"  Reward Δ: {reward_delta:+.3f} ({'GRPO better' if reward_delta > 0 else 'PPO better'})")
    print(f"  Goal Δ:    {goal_delta:+.1f}% ({'GRPO better' if goal_delta > 0 else 'PPO better'})")
    print(f"  Time Δ:    {time_delta:+.1f}s ({'GRPO faster' if time_delta < 0 else 'PPO faster'})")
    
    # Save metrics
    metrics = {
        'comparison': 'GRPO vs PPO',
        'config': {
            'episodes': args.episodes,
            'seeds': args.seeds,
            'lr': args.lr,
        },
        'ppo': {
            'final_reward_mean': float(np.mean(ppo_rewards)),
            'final_reward_std': float(np.std(ppo_rewards)),
            'goal_rate_mean': float(np.mean(ppo_goal_rates)),
            'goal_rate_std': float(np.std(ppo_goal_rates)),
            'time_mean': float(np.mean(ppo_times)),
            'time_std': float(np.std(ppo_times)),
            'seeds': [r.final_reward for r in ppo_results],
        },
        'grpo': {
            'final_reward_mean': float(np.mean(grpo_rewards)),
            'final_reward_std': float(np.std(grpo_rewards)),
            'goal_rate_mean': float(np.mean(grpo_goal_rates)),
            'goal_rate_std': float(np.std(grpo_goal_rates)),
            'time_mean': float(np.mean(grpo_times)),
            'time_std': float(np.std(grpo_times)),
            'seeds': [r.final_reward for r in grpo_results],
        },
        'delta': {
            'reward': float(reward_delta),
            'goal_rate': float(goal_delta),
            'time': float(time_delta),
        }
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {args.output}")
    
    return metrics


if __name__ == '__main__':
    main()
