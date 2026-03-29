#!/usr/bin/env python3
"""
DeepSeek-R1 Style GRPO Training — Day 3 Long Run
================================================
Implements Option A: Termination Reward Shaping + Self-Correction Head + GRPO

Usage:
    python training/rl/train_deepseek_grpo.py --iterations 50 --episodes 16
"""
import os, sys, json, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

WORKSPACE = "/data/.openclaw/workspace"
sys.path.insert(0, WORKSPACE)

from training.rl.toy_waypoint_env import ToyWaypointEnv
from training.rl.grpo import GRPOConfig


class TerminationRewardConfig:
    success_reward: float = 100.0
    success_per_waypoint: float = 5.0
    progress_coef: float = 1.0
    reward_clip: float = 10.0
    use_self_evolution: bool = True
    evolution_bonus_coef: float = 0.5
    evolution_window: int = 20


class TerminationRewardShaper:
    def __init__(self, config):
        self.config = config
        self.historical_rewards = {}
        self.global_step = 0
    
    def shape_reward(self, reward, info, done, truncated):
        self.global_step += 1
        shaped = float(reward)
        wp_idx = info.get('current_waypoint_idx', 0)
        total_wp = info.get('total_waypoints', 20)
        progress = wp_idx / max(total_wp, 1)
        shaped += self.config.progress_coef * progress
        
        key = f"wp{wp_idx}"
        
        if done and not truncated:
            shaped += self.config.success_reward + wp_idx * self.config.success_per_waypoint
            if key not in self.historical_rewards:
                self.historical_rewards[key] = []
            self.historical_rewards[key].append(shaped)
        elif truncated and not done:
            shaped -= self.config.success_reward * 0.5
        
        if self.config.use_self_evolution and key in self.historical_rewards:
            hist = self.historical_rewards[key]
            if len(hist) >= 5:
                rolling_avg = np.mean(hist[-self.config.evolution_window:])
                if shaped > rolling_avg:
                    shaped += (shaped - rolling_avg) * self.config.evolution_bonus_coef
        
        if self.config.reward_clip:
            shaped = np.clip(shaped, -self.config.reward_clip, self.config.reward_clip)
        return shaped
    
    def get_stats(self):
        return {'global_step': self.global_step}


class SelfCorrectionHead(nn.Module):
    def __init__(self, hidden_dim=128, horizon_steps=16, waypoint_dim=3):
        super().__init__()
        self.confidence_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 1), nn.Sigmoid()
        )
        self.correction_mlp = nn.Sequential(
            nn.Linear(hidden_dim + horizon_steps*waypoint_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, horizon_steps*waypoint_dim)
        )
        self.register_parameter('trigger_threshold', nn.Parameter(torch.tensor(0.5)))
    
    def forward(self, z, current_waypoints):
        confidence = self.confidence_mlp(z)
        wp_flat = current_waypoints.reshape(z.size(0), -1)
        concat = torch.cat([z, wp_flat], dim=-1)
        correction = self.correction_mlp(concat).reshape(z.size(0), 16, 3)
        threshold = torch.sigmoid(self.trigger_threshold)
        should_correct = (confidence < threshold).float()
        return {
            'confidence': confidence,
            'correction': correction,
            'should_correct': should_correct,
        }


class SimpleWaypointPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.waypoint_head = nn.Linear(128, 16 * 3)
        self.action_std = nn.Parameter(torch.zeros(1, 16, 3) + 0.1)
    
    def forward(self, z, waypoints=None, return_hidden=False):
        h = self.fc(z)
        out = self.waypoint_head(h).reshape(-1, 16, 3)
        return (out, h) if return_hidden else out
    
    def get_action_distribution(self, z):
        mean = self.forward(z)
        std = self.action_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std + 1e-8)


class WaypointGRPOPolicyWithCorrection(nn.Module):
    def __init__(self, base_policy):
        super().__init__()
        self.base_policy = base_policy
        self.correction_head = SelfCorrectionHead(128, 16, 3)
    
    def forward(self, z, waypoints=None, use_correction=True):
        base_waypoints, z_hidden = self.base_policy(z, waypoints, return_hidden=True)
        if not use_correction:
            return {'final_waypoints': base_waypoints, 'base_waypoints': base_waypoints, 'hidden': z_hidden, 'correction_info': {}}
        corr_info = self.correction_head(z_hidden, base_waypoints)
        should_correct = corr_info['should_correct']
        correction = corr_info['correction']
        blend = should_correct.unsqueeze(-1).unsqueeze(-1)
        final_waypoints = base_waypoints + blend * correction
        return {'final_waypoints': final_waypoints, 'base_waypoints': base_waypoints, 'hidden': z_hidden, 'correction_info': corr_info}
    
    def get_correction_loss(self, corr_info, rewards):
        if not corr_info: return torch.tensor(0.0)
        confidence = corr_info['confidence']
        reward_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        reward_norm = reward_norm.unsqueeze(-1)
        target_conf = 1.0 - torch.sigmoid(reward_norm * 3)
        return F.mse_loss(confidence, target_conf)


class DeepSeekGRPO:
    def __init__(self, policy, config, term_config, optimizer=None):
        self.policy = policy
        self.config = config
        self.optimizer = optimizer
        self.term_shaper = TerminationRewardShaper(term_config)
    
    def compute_advantages(self, shaped_rewards, group_ids=None):
        if group_ids is None:
            return shaped_rewards - shaped_rewards.mean()
        advantages = torch.zeros_like(shaped_rewards)
        for gid in group_ids.unique():
            mask = group_ids == gid
            if mask.sum() > 1:
                advantages[mask] = (shaped_rewards[mask] - shaped_rewards[mask].mean()) / (shaped_rewards[mask].std() + 1e-8)
            else:
                advantages[mask] = 0.0
        return advantages
    
    def update(self, states, actions, old_log_probs, advantages, shaped_rewards=None):
        B = states.size(0)
        std = 0.1 + 1e-8
        log_std_val = float(np.log(std))
        
        with torch.no_grad():
            out = self.policy.forward(states, use_correction=False)
            mean = out['final_waypoints']
        
        log_ll = -0.5 * (((actions - mean) / std) ** 2 + 2 * log_std_val + np.log(2 * np.pi))
        new_log_probs = log_ll.reshape(B, -1).sum(dim=-1)
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        clip_eps = self.config.clip_epsilon
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy = 0.5 * (1 + np.log(2 * np.pi) + 2 * log_std_val)
        entropy_loss = -self.config.entropy_coef * entropy
        total_loss = policy_loss + entropy_loss
        
        corr_metrics = {'correction_loss': 0.0, 'mean_confidence': 0.5, 'correction_rate': 0.0}
        if hasattr(self.policy, 'get_correction_loss') and shaped_rewards is not None:
            corr_out = self.policy.forward(states, use_correction=True)
            corr_info = corr_out.get('correction_info', {})
            if corr_info:
                corr_loss = self.policy.get_correction_loss(corr_info, shaped_rewards)
                total_loss = total_loss + self.config.entropy_coef * corr_loss
                corr_metrics = {
                    'correction_loss': corr_loss.item(),
                    'mean_confidence': corr_info['confidence'].mean().item(),
                    'correction_rate': corr_info['should_correct'].float().mean().item(),
                }
        
        if self.optimizer:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        with torch.no_grad():
            clip_frac = ((ratio - 1).abs() > clip_eps).float().mean()
        
        metrics = {
            'loss': total_loss.item(), 'policy_loss': policy_loss.item(), 'entropy': entropy,
            'kl': (new_log_probs - old_log_probs).mean().item(),
            'clip_fraction': clip_frac.item(), 'mean_advantage': advantages.mean().item(),
            'term_global_step': self.term_shaper.get_stats()['global_step'],
        }
        metrics.update(corr_metrics)
        return metrics


def collect_trajectories(env, policy, num_episodes, max_steps, term_shaper, device, use_correction=True):
    trajectories = []
    for ep_idx in range(num_episodes):
        obs, info = env.reset()
        wp_total = len(info.get('waypoints', []))
        episode_states, episode_actions, episode_rewards, episode_shaped = [], [], [], []
        done, trunc, steps = False, False, 0
        
        while not (done or trunc) and steps < max_steps:
            state_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                if use_correction and hasattr(policy, 'forward'):
                    out = policy.forward(state_tensor, use_correction=True)
                    waypoints_pred = out['final_waypoints']
                else:
                    waypoints_pred = torch.zeros(1, 16, 3).to(device)
            
            wp_idx = info.get('current_waypoint_idx', 0)
            waypoints = info.get('waypoints', [])
            if wp_idx < len(waypoints):
                target = waypoints[wp_idx]
                pos, direction = obs[:2], target[:2] - obs[:2]
                dist = np.linalg.norm(direction)
                if dist > 0.1: direction = direction / dist
                steer = np.arctan2(direction[1], direction[0]) * 0.3
                throttle = min(1.0, dist / 10.0) * 0.8
            else:
                steer, throttle = 0.0, 0.0
            
            info['total_waypoints'] = wp_total
            obs, reward, done, trunc, info = env.step(np.array([throttle, steer]))
            shaped = term_shaper.shape_reward(reward, info, done, trunc)
            
            episode_states.append(state_tensor)
            episode_actions.append(waypoints_pred.detach().clone())
            episode_rewards.append(float(reward))
            episode_shaped.append(float(shaped))
            steps += 1
        
        trajectories.append({
            'states': episode_states, 'actions': episode_actions,
            'rewards': episode_rewards, 'shaped_rewards': episode_shaped,
            'return': sum(episode_rewards), 'shaped_return': sum(episode_shaped),
            'steps': steps, 'group_id': ep_idx % 8,
        })
    return trajectories


def evaluate(policy, env, device, use_correction=True):
    seeds = list(range(42, 52))
    results = []
    for seed in seeds:
        np.random.seed(seed); torch.manual_seed(seed)
        obs, info = env.reset()
        positions = [obs[:2].copy()]
        done, trunc, steps, total_r = False, False, 0, 0.0
        wp_total = len(info.get('waypoints', []))
        
        while not (done or trunc) and steps < 100:
            state_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                if use_correction and hasattr(policy, 'forward'):
                    out = policy.forward(state_tensor, use_correction=True)
                    waypoints_pred = out['final_waypoints']
                else:
                    waypoints_pred = policy(state_tensor)
                wp_np = waypoints_pred.cpu().numpy().squeeze()
                if wp_np.ndim == 3: wp_np = wp_np[0]
            
            wp_idx = info.get('current_waypoint_idx', 0)
            waypoints = info.get('waypoints', [])
            if wp_idx < len(waypoints):
                target, pos = waypoints[wp_idx], obs[:2]
                direction = target[:2] - pos
                dist = np.linalg.norm(direction)
                if dist > 0.1: direction = direction / dist
                steer = np.arctan2(direction[1], direction[0]) * 0.3
                throttle = min(1.0, dist / 10.0) * 0.8
            else:
                steer, throttle = 0.0, 0.0
            
            obs, reward, done, trunc, info = env.step(np.array([throttle, steer]))
            total_r += reward
            positions.append(obs[:2].copy())
            steps += 1
        
        waypoints = info.get('waypoints', [])
        pos_arr = np.array(positions); wp_arr = np.array(waypoints)
        min_len = min(len(pos_arr), len(wp_arr))
        ade = np.linalg.norm(pos_arr[:min_len] - wp_arr[:min_len], axis=1).mean() if min_len > 0 else 999.0
        fde = np.linalg.norm(pos_arr[-1] - wp_arr[-1]) if len(wp_arr) > 0 else 999.0
        success = bool(info.get('current_waypoint_idx', 0) >= len(waypoints) - 1)
        results.append({'ade': float(ade), 'fde': float(fde), 'return': float(total_r), 'success': success, 'steps': steps})
    
    return {
        'success_rate': sum(r['success'] for r in results) / len(results),
        'ade': float(np.mean([r['ade'] for r in results])),
        'fde': float(np.mean([r['fde'] for r in results])),
        'return_mean': float(np.mean([r['return'] for r in results])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output or f"out/day3_deepseek_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = ToyWaypointEnv()
    
    grpo_cfg = GRPOConfig(clip_epsilon=0.2, entropy_coef=0.01, group_size=8, update_epochs=4, batch_size=32, max_grad_norm=0.5, advantage_normalize=True)
    term_cfg = TerminationRewardConfig()
    
    base = SimpleWaypointPolicy().to(device)
    policy = WaypointGRPOPolicyWithCorrection(base).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    grpo = DeepSeekGRPO(policy, grpo_cfg, term_cfg, optimizer)
    
    history = {'iterations': [], 'eval_success_rate': [], 'eval_ade': [], 'eval_fde': [], 'mean_reward': [], 'mean_shaped': [], 'correction_rate': []}
    
    print("=" * 60)
    print("Day 3: DeepSeek-GRPO Long Training Run")
    print("=" * 60)
    print(f"Iterations: {args.iterations}, Episodes/iter: {args.episodes}")
    print(f"Output: {output_dir}")
    print()
    
    for iteration in range(args.iterations):
        trajs = collect_trajectories(env, policy, args.episodes, 100, grpo.term_shaper, device, True)
        
        for traj in trajs:
            traj['shaped_returns'] = torch.tensor(traj['shaped_rewards'], dtype=torch.float32)
        
        all_states = torch.cat([t['states'][s] for t in trajs for s in range(len(t['states']))])
        all_actions = torch.cat([t['actions'][s] for t in trajs for s in range(len(t['actions']))])
        all_shaped = torch.tensor(sum([t['shaped_rewards'] for t in trajs], []), dtype=torch.float32).to(device)
        group_ids = torch.tensor([t['group_id'] for t in trajs for _ in range(len(t['states']))], dtype=torch.long).to(device)
        
        advantages = grpo.compute_advantages(all_shaped, group_ids)
        
        for epoch in range(4):
            indices = torch.randperm(len(all_states))
            for start in range(0, len(all_states), 32):
                end = start + 32
                batch_idx = indices[start:end]
                batch_states = all_states[batch_idx]
                batch_actions = all_actions[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_rewards = all_shaped[batch_idx]
                
                with torch.no_grad():
                    out = policy.forward(batch_states, use_correction=False)
                    mean = out['final_waypoints']
                    std = 0.1 + 1e-8
                    log_std_val = float(np.log(std))
                    ll = -0.5 * (((batch_actions - mean) / std) ** 2 + 2 * log_std_val + np.log(2 * np.pi))
                    old_log_probs = ll.reshape(len(batch_states), -1).sum(dim=-1)
                
                metrics = grpo.update(batch_states, batch_actions, old_log_probs, batch_adv, batch_rewards)
        
        mean_r = np.mean([t['return'] for t in trajs])
        mean_s = np.mean([t['shaped_return'] for t in trajs])
        conf = metrics.get('mean_confidence', 0)
        corr = metrics.get('correction_rate', 0)
        
        print(f"Iter {iteration+1:3d}/{args.iterations} | Reward={mean_r:.2f} | Shaped={mean_s:.2f} | Conf={conf:.2f} | Corr={corr:.2f}")
        
        if (iteration + 1) % 10 == 0:
            eval_res = evaluate(policy, env, device, True)
            print(f"  => Eval: success={eval_res['success_rate']:.0%} | ADE={eval_res['ade']:.3f} | FDE={eval_res['fde']:.3f}")
            history['iterations'].append(iteration + 1)
            history['eval_success_rate'].append(eval_res['success_rate'])
            history['eval_ade'].append(eval_res['ade'])
            history['eval_fde'].append(eval_res['fde'])
            history['mean_reward'].append(mean_r)
            history['mean_shaped'].append(mean_s)
            history['correction_rate'].append(corr)
        
        if (iteration + 1) % 25 == 0:
            ckpt = os.path.join(output_dir, f"checkpoint_iter{iteration+1}.pt")
            torch.save({'iteration': iteration+1, 'policy_state_dict': policy.state_dict()}, ckpt)
            print(f"  => Saved {ckpt}")
    
    final_path = os.path.join(output_dir, "final_model.pt")
    torch.save({'policy_state_dict': policy.state_dict()}, final_path)
    
    hist_path = os.path.join(output_dir, "history.json")
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final eval
    eval_res = evaluate(policy, env, device, True)
    print(f"\n✅ Final: success={eval_res['success_rate']:.0%} | ADE={eval_res['ade']:.3f} | FDE={eval_res['fde']:.3f}")
    print(f"Output: {output_dir}")
    return history, eval_res


if __name__ == '__main__':
    history, eval_res = main()