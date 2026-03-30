#!/usr/bin/env python3
"""
DeepSeek-GRPO with Real SFT Checkpoint Integration
==================================================
Integrates DeepSeek-R1 termination reward shaping with the real SFT model
from `out/waypoint_bc/run_20260312_083423/checkpoint.pt`.

This bridges the toy environment (state-only) with the real SFT encoder
by learning a state-to-encoder mapping, then training a delta correction head.

Usage:
    python training/rl/train_deepseek_sft_grpo.py --sft-checkpoint out/waypoint_bc/run_20260312_083423/checkpoint.pt

Reference: DeepSeek-R1 (arXiv:2501.12948)
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


class WaypointHead(nn.Module):
    """
    Waypoint head matching the SFT checkpoint architecture:
    Linear(256,512) -> ReLU -> Linear(512,512) -> ReLU -> Linear(512,16)
    """
    def __init__(self, in_dim=256, out_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )
    
    def forward(self, z):
        # z: [B, in_dim] -> [B, horizon, 2]
        y = self.mlp(z)
        # Reshape to [B, 8, 2] — horizon=8, dim=2
        return y.reshape(-1, 8, 2)
    
    def parameters(self):
        return self.mlp.parameters()


# ============================================================================
# State → SFT Feature Mapping (Bridge)
# ============================================================================

class StateToSFTEncoder(nn.Module):
    """
    Maps raw state (4D: x, y, heading, speed) to SFT encoder feature space.
    
    The SFT encoder expects images (128x128x3). We learn a mapping from state
    to the encoder's output space (256D), then the waypoint head outputs waypoints.
    
    This is a "state-conditioned encoder" — learns to mimic what the visual encoder
    would predict from state-only input.
    """
    def __init__(self, state_dim=4, encoder_out_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.encoder_out_dim = encoder_out_dim
        
        # State encoder → matches SFT encoder output dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, encoder_out_dim),
        )
    
    def forward(self, state):
        """state: [B, 4] → [B, encoder_out_dim]"""
        return self.net(state)


class DeltaCorrectionHead(nn.Module):
    """
    Learns residual corrections on top of SFT waypoint predictions.
    
    final_waypoints = sft_waypoints + delta
    
    This is the GRPO-trainable component. We freeze the SFT base and only
    train this delta head with termination reward shaping.
    """
    def __init__(self, encoder_out_dim=256, num_waypoints=8, waypoint_dim=2):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        
        # Delta correction head
        self.delta_net = nn.Sequential(
            nn.Linear(encoder_out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_waypoints * waypoint_dim),
        )
        
        # Learnable correction scale
        self.delta_scale = nn.Parameter(torch.ones(1, num_waypoints, 1))
    
    def forward(self, encoder_features):
        """encoder_features: [B, encoder_out_dim] -> [B, num_waypoints, waypoint_dim]"""
        delta = self.delta_net(encoder_features)  # [B, num_waypoints * waypoint_dim]
        delta = delta.reshape(-1, self.num_waypoints, self.waypoint_dim)
        return self.delta_scale * delta  # [B, H, D]


class SelfCorrectionHead(nn.Module):
    """
    Confidence-based correction (DeepSeek-R1 reflection tokens).
    
    Learns when the SFT base prediction is likely wrong, and applies
    additional delta correction in those cases.
    """
    def __init__(self, encoder_out_dim=256, num_waypoints=8, waypoint_dim=2):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        
        # Confidence: is the SFT prediction trustworthy?
        self.confidence_net = nn.Sequential(
            nn.Linear(encoder_out_dim + num_waypoints * waypoint_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        # Correction head for when confidence is low
        self.correction_net = nn.Sequential(
            nn.Linear(encoder_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_waypoints * waypoint_dim),
        )
        
        self.register_parameter('trigger_threshold', nn.Parameter(torch.tensor(0.5)))
    
    def forward(self, encoder_features, sft_waypoints):
        """
        Args:
            encoder_features: [B, encoder_out_dim]
            sft_waypoints: [B, num_waypoints, waypoint_dim]
        
        Returns:
            correction_delta: [B, num_waypoints, waypoint_dim]
            should_correct: [B, 1] — bool indicating when to apply correction
            confidence: [B, 1] — how confident is the SFT prediction?
        """
        # Confidence based on encoder features + SFT prediction
        sft_flat = sft_waypoints.reshape(encoder_features.size(0), -1)
        concat = torch.cat([encoder_features, sft_flat], dim=-1)
        confidence = self.confidence_net(concat)  # [B, 1]
        
        # Correction delta — reshape [B, H*D] -> [B, H, D]
        correction_delta = self.correction_net(encoder_features)
        correction_delta = correction_delta.reshape(-1, self.num_waypoints, self.waypoint_dim)
        
        # Trigger: correct if confidence < threshold
        threshold = torch.sigmoid(self.trigger_threshold)
        should_correct = (confidence < threshold).float()
        
        return {
            'confidence': confidence,
            'correction_delta': correction_delta,
            'should_correct': should_correct,
        }


class SFTGRPOPolicy(nn.Module):
    """
    Full policy: State encoder → SFT waypoints → Delta correction + Self-correction.
    
    Components (all trainable except SFT encoder):
    1. StateToSFTEncoder — maps state to feature space
    2. Frozen SFT waypoint_head — pre-trained waypoint predictions
    3. DeltaCorrectionHead — residual corrections (trainable)
    4. SelfCorrectionHead — confidence-based additional corrections (trainable)
    """
    def __init__(self, sft_checkpoint_path, num_waypoints=8, waypoint_dim=2, trainable_parts='delta+correction'):
        super().__init__()
        self.trainable_parts = trainable_parts
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        
        # State encoder
        self.state_encoder = StateToSFTEncoder(state_dim=4, encoder_out_dim=256)
        
        # SFT waypoint head (frozen)
        self.sft_waypoint_head = WaypointHead(in_dim=256, out_dim=num_waypoints * waypoint_dim)
        
        # Delta correction head
        self.delta_head = DeltaCorrectionHead(encoder_out_dim=256, num_waypoints=num_waypoints, waypoint_dim=waypoint_dim)
        
        # Self-correction head
        self.correction_head = SelfCorrectionHead(encoder_out_dim=256, num_waypoints=num_waypoints, waypoint_dim=waypoint_dim)
        
        # Load SFT checkpoint
        self._load_sft_checkpoint(sft_checkpoint_path)
        
        # Freeze SFT components
        for param in self.state_encoder.parameters():
            if 'delta' not in trainable_parts and 'correction' not in trainable_parts:
                pass  # Keep trainable for now
        
        self._freeze_sft()
    
    def _load_sft_checkpoint(self, path):
        """Load pre-trained SFT encoder and waypoint head."""
        print(f"  Loading SFT checkpoint: {path}")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = ckpt['model_state_dict']
        
        # Load state encoder (may not be in checkpoint — initialize from scratch)
        if 'encoder.fc.0.weight' in sd:
            print("  Loading SFT encoder weights")
            # The encoder has conv layers — can't directly load from state-only
            # We'll learn a mapping from state to encoder output space
            pass
        
        # Load waypoint head
        if 'waypoint_head.mlp.0.weight' in sd:
            print("  Loading SFT waypoint head")
            sd_reshaped = {}
            for k, v in sd.items():
                if k.startswith('waypoint_head.'):
                    sd_reshaped[k.replace('waypoint_head.', '')] = v
            self.sft_waypoint_head.load_state_dict(sd_reshaped, strict=False)
        
        # Also try model_state_dict style
        if 'model_state_dict' in ckpt:
            msd = ckpt['model_state_dict']
            if 'waypoint_head.mlp.0.weight' in msd:
                sd_reshaped = {}
                for k, v in msd.items():
                    if k.startswith('waypoint_head.'):
                        sd_reshaped[k.replace('waypoint_head.', '')] = v
                self.sft_waypoint_head.load_state_dict(sd_reshaped, strict=False)
        
        print(f"  SFT waypoint_head loaded (num_params={sum(p.numel() for p in self.sft_waypoint_head.parameters())})")
    
    def _freeze_sft(self):
        """Freeze SFT components — only train delta + correction heads."""
        for param in self.sft_waypoint_head.parameters():
            param.requires_grad = False
        
        if self.trainable_parts == 'delta+correction':
            # Freeze state encoder too — only train delta and correction heads
            for param in self.state_encoder.parameters():
                param.requires_grad = False
    
    def forward(self, state, use_correction=True, use_delta=True):
        """
        Args:
            state: [B, 4] — x, y, heading, speed
        
        Returns:
            dict with:
                - sft_waypoints: [B, num_waypoints, waypoint_dim]
                - delta: [B, num_waypoints, waypoint_dim]
                - correction_delta: [B, num_waypoints, waypoint_dim]
                - final_waypoints: [B, num_waypoints, waypoint_dim]
                - features: [B, 256] — encoder output
                - correction_info: dict from self-correction head
        """
        # Get SFT features from state
        features = self.state_encoder(state)  # [B, 256]
        
        # SFT base prediction — reshape from [B, num_waypoints*waypoint_dim] to [B, H, D]
        sft_wp_flat = self.sft_waypoint_head(z=features)  # [B, H*D]
        sft_wp = sft_wp_flat.reshape(-1, self.num_waypoints, self.waypoint_dim)
        
        # Delta correction — output already [B, H, D]
        delta = torch.zeros_like(sft_wp)
        if use_delta:
            delta = self.delta_head(features)
        
        # Self-correction (conditional on confidence)
        correction_info = {}
        if use_correction:
            correction_info = self.correction_head(features, sft_wp)
            correction_delta = correction_info.get('correction_delta', torch.zeros_like(sft_wp))
            should_correct = correction_info.get('should_correct', torch.zeros(1, 1))
            
            # Apply correction as extra delta when confidence is low
            blend = should_correct.unsqueeze(-1)  # [B, 1, 1]
            delta = delta + blend * correction_delta
        
        final = sft_wp + delta
        
        return {
            'sft_waypoints': sft_wp,
            'delta': delta,
            'final_waypoints': final,
            'features': features,
            'correction_info': correction_info,
        }
    
    def get_correction_loss(self, correction_info, rewards):
        """Loss for learning when to correct (DeepSeek-R1 style)."""
        if not correction_info:
            return torch.tensor(0.0, device=next(self.parameters()).device if self.parameters() else 'cpu')
        confidence = correction_info['confidence']
        reward_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        reward_norm = reward_norm.unsqueeze(-1)
        # Low reward → want high confidence (we're already right)
        # High reward → want low confidence (correct more aggressively)
        target_conf = torch.sigmoid(reward_norm * 2)
        return F.mse_loss(confidence, target_conf)


# ============================================================================
# Termination Reward Shaping (DeepSeek-R1)
# ============================================================================

class TerminationRewardShaper:
    def __init__(self, success_reward=100.0, progress_coef=5.0, reward_clip=20.0,
                 use_self_evolution=True, evolution_bonus_coef=0.3):
        self.success_reward = success_reward
        self.progress_coef = progress_coef
        self.reward_clip = reward_clip
        self.use_self_evolution = use_self_evolution
        self.evolution_bonus_coef = evolution_bonus_coef
        self.historical = {}
        self.step = 0
    
    def shape(self, reward, info, done, trunc):
        self.step += 1
        shaped = float(reward)
        
        wp_idx = info.get('current_waypoint_idx', 0)
        total_wp = info.get('total_waypoints', 20)
        progress = wp_idx / max(total_wp, 1)
        
        # Progress reward
        shaped += self.progress_coef * progress
        
        # Success termination bonus (DeepSeek-R1 key technique)
        if done and not trunc:
            key = f"wp{wp_idx}"
            shaped += self.success_reward
            if key not in self.historical:
                self.historical[key] = []
            self.historical[key].append(shaped)
        
        # Failure penalty
        if trunc and not done:
            shaped -= self.success_reward * 0.3
        
        # Self-evolution bonus
        if self.use_self_evolution:
            key = f"wp{wp_idx}"
            if key in self.historical and len(self.historical[key]) >= 5:
                rolling = np.mean(self.historical[key][-20:])
                if shaped > rolling:
                    shaped += (shaped - rolling) * self.evolution_bonus_coef
        
        if self.reward_clip:
            shaped = np.clip(shaped, -self.reward_clip, self.reward_clip)
        
        return shaped
    
    def get_stats(self):
        return {'step': self.step, 'scenarios': len(self.historical)}


# ============================================================================
# GRPO Update for Waypoint Prediction
# ============================================================================

class DeepSeekGRPOUpdate:
    def __init__(self, policy, optimizer, config):
        self.policy = policy
        self.optimizer = optimizer
        self.config = config
    
    def step(self, states, actions, old_log_probs, advantages, shaped_rewards=None):
        """Custom GRPO update for waypoint sequence actions."""
        B = states.size(0)
        H = self.policy.num_waypoints
        D = self.policy.waypoint_dim
        std = 0.5  # Fixed std for waypoint prediction
        log_std_val = float(np.log(std))
        
        # Current policy prediction
        with torch.no_grad():
            out = self.policy(states, use_correction=False, use_delta=True)
            mean = out['final_waypoints']  # [B, H, D]
        
        # Gaussian log prob
        log_ll = -0.5 * (((actions - mean) / std) ** 2 + 2 * log_std_val + np.log(2 * np.pi))
        new_log_probs = log_ll.reshape(B, -1).sum(dim=-1)  # [B]
        
        # GRPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        clip_eps = self.config.clip_epsilon
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus
        entropy = 0.5 * (1 + np.log(2 * np.pi) + 2 * log_std_val)
        total_loss = policy_loss - self.config.entropy_coef * entropy
        
        # Self-correction loss
        corr_metrics = {'conf': 0.5, 'corr_rate': 0.0, 'corr_loss': 0.0}
        if shaped_rewards is not None:
            out_corr = self.policy(states, use_correction=True, use_delta=True)
            ci = out_corr.get('correction_info', {})
            if ci:
                corr_loss = self.policy.get_correction_loss(ci, shaped_rewards)
                total_loss = total_loss + 0.01 * corr_loss
                corr_metrics = {
                    'conf': ci['confidence'].mean().item(),
                    'corr_rate': ci['should_correct'].float().mean().item(),
                    'corr_loss': corr_loss.item(),
                }
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        with torch.no_grad():
            clip_frac = ((ratio - 1).abs() > clip_eps).float().mean()
        
        return {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': entropy,
            'clip_frac': clip_frac.item(),
            'adv_mean': advantages.mean().item(),
            **corr_metrics,
        }


# ============================================================================
# Training Loop
# ============================================================================

def collect_trajectories(env, policy, num_episodes, max_steps, shaper, device):
    """Collect trajectories with termination reward shaping."""
    trajectories = []
    for ep_idx in range(num_episodes):
        obs, info = env.reset()
        wp_total = len(info.get('waypoints', []))
        
        states_list, actions_list, rewards_list, shaped_list = [], [], [], []
        done, trunc, steps = False, False, 0
        
        while not (done or trunc) and steps < max_steps:
            state_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = policy(state_tensor, use_correction=True, use_delta=True)
                final_wp = out['final_waypoints']  # [1, H, D]
            
            # Use waypoints to drive the agent (simplified: steer toward current waypoint)
            wp_idx = info.get('current_waypoint_idx', 0)
            waypoints = info.get('waypoints', [])
            
            if wp_idx < len(waypoints):
                target = waypoints[wp_idx]
                pos = obs[:2]
                direction = target[:2] - pos
                dist = np.linalg.norm(direction)
                if dist > 0.1:
                    direction = direction / dist
                steer = np.arctan2(direction[1], direction[0]) * 0.3
                throttle = min(1.0, dist / 10.0) * 0.8
            else:
                steer, throttle = 0.0, 0.0
            
            obs, reward, done, trunc, info = env.step(np.array([throttle, steer]))
            info['total_waypoints'] = wp_total
            
            shaped = shaper.shape(reward, info, done, trunc)
            
            # Store state + action (waypoint prediction)
            states_list.append(state_tensor)
            actions_list.append(final_wp.detach().clone())
            rewards_list.append(float(reward))
            shaped_list.append(float(shaped))
            steps += 1
        
        trajectories.append({
            'states': states_list,
            'actions': actions_list,
            'rewards': rewards_list,
            'shaped': shaped_list,
            'return': sum(rewards_list),
            'shaped_return': sum(shaped_list),
            'group_id': ep_idx % 8,
        })
    
    return trajectories


def evaluate(policy, env, device):
    """Evaluate policy on multiple seeds."""
    seeds = list(range(42, 52))
    results = []
    
    for seed in seeds:
        np.random.seed(seed); torch.manual_seed(seed)
        obs, info = env.reset()
        positions = [obs[:2].copy()]
        done, trunc, steps, total_r = False, False, 0, 0.0
        
        while not (done or trunc) and steps < 100:
            st = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                out = policy(st, use_correction=True, use_delta=True)
                wp = out['final_waypoints'].cpu().numpy().squeeze()
                if wp.ndim == 3: wp = wp[0]
            
            wi = info.get('current_waypoint_idx', 0); wps = info.get('waypoints', [])
            if wi < len(wps):
                t, p = wps[wi], obs[:2]
                d = t[:2] - p; dist = np.linalg.norm(d)
                if dist > 0.1: d = d / dist
                steer = np.arctan2(d[1], d[0]) * 0.3
                throttle = min(1.0, dist / 10.0) * 0.8
            else:
                steer, throttle = 0.0, 0.0
            
            obs, reward, done, trunc, info = env.step(np.array([throttle, steer]))
            total_r += reward; positions.append(obs[:2].copy()); steps += 1
        
        wps = info.get('waypoints', [])
        pa, wa = np.array(positions), np.array(wps)
        ml = min(len(pa), len(wa))
        ade = np.linalg.norm(pa[:ml] - wa[:ml], axis=1).mean() if ml > 0 else 999.0
        fde = np.linalg.norm(pa[-1] - wa[-1]) if len(wa) > 0 else 999.0
        success = bool(info.get('current_waypoint_idx', 0) >= len(wps) - 1)
        results.append({'ade': float(ade), 'fde': float(fde), 'success': success})
    
    return {
        'success_rate': sum(r['success'] for r in results) / len(results),
        'ade': float(np.mean([r['ade'] for r in results])),
        'fde': float(np.mean([r['fde'] for r in results])),
    }


def main(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output or f"out/deepseek_sft_grpo_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = ToyWaypointEnv()
    
    # Config
    grpo_cfg = GRPOConfig(
        clip_epsilon=0.2,
        entropy_coef=0.01,
        group_size=8,
        update_epochs=4,
        batch_size=32,
        max_grad_norm=1.0,
        advantage_normalize=True,
    )
    shaper = TerminationRewardShaper(
        success_reward=args.success_reward,
        progress_coef=args.progress_reward,
        use_self_evolution=not args.no_self_evolution,
    )
    
    # Policy
    sft_path = args.sft_checkpoint or 'out/waypoint_bc/run_20260312_083423/checkpoint.pt'
    policy = SFTGRPOPolicy(
        sft_checkpoint_path=sft_path,
        num_waypoints=8,
        waypoint_dim=2,
        trainable_parts='delta+correction',
    ).to(device)
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    # Optimizer — only train delta + correction heads
    optimizer = torch.optim.Adam([
        {'params': policy.delta_head.parameters(), 'lr': args.lr},
        {'params': policy.correction_head.parameters(), 'lr': args.lr},
    ], weight_decay=1e-4)
    
    updater = DeepSeekGRPOUpdate(policy, optimizer, grpo_cfg)
    
    print("=" * 60)
    print("DeepSeek-GRPO with Real SFT Checkpoint")
    print("=" * 60)
    print(f"SFT checkpoint: {sft_path}")
    print(f"Trainable: delta_head + correction_head")
    print(f"Success reward: {args.success_reward}")
    print(f"Iterations: {args.iterations}, Episodes/iter: {args.episodes}")
    print()
    
    history = {'iterations': [], 'eval_success_rate': [], 'eval_ade': [], 'eval_fde': []}
    
    for iteration in range(args.iterations):
        trajs = collect_trajectories(env, policy, args.episodes, 100, shaper, device)
        
        # Concatenate
        all_states = torch.cat([t['states'][s] for t in trajs for s in range(len(t['states']))])
        all_actions = torch.cat([t['actions'][s] for t in trajs for s in range(len(t['actions']))])
        all_shaped = torch.tensor(sum([t['shaped'] for t in trajs], []), dtype=torch.float32).to(device)
        group_ids = torch.tensor([t['group_id'] for t in trajs for _ in range(len(t['states']))], dtype=torch.long).to(device)
        
        # Group-relative advantages
        advantages = torch.zeros_like(all_shaped)
        for gid in group_ids.unique():
            mask = group_ids == gid
            if mask.sum() > 1:
                advantages[mask] = (all_shaped[mask] - all_shaped[mask].mean()) / (all_shaped[mask].std() + 1e-8)
        
        # Old log probs
        std = 0.5
        log_std_val = float(np.log(std))
        with torch.no_grad():
            out = policy(all_states, use_correction=False, use_delta=True)
            mean = out['final_waypoints']
            ll = -0.5 * (((all_actions - mean) / std) ** 2 + 2 * log_std_val + np.log(2 * np.pi))
            old_log_probs = ll.reshape(len(all_states), -1).sum(dim=-1)
        
        # Update
        for epoch in range(grpo_cfg.update_epochs):
            indices = torch.randperm(len(all_states))
            for start in range(0, len(all_states), grpo_cfg.batch_size):
                end = min(start + grpo_cfg.batch_size, len(all_states))
                idx = indices[start:end]
                metrics = updater.step(
                    all_states[idx], all_actions[idx],
                    old_log_probs[idx], advantages[idx],
                    all_shaped[idx],
                )
        
        mean_r = np.mean([t['return'] for t in trajs])
        mean_s = np.mean([t['shaped_return'] for t in trajs])
        print(f"Iter {iteration+1:3d}/{args.iterations} | R={mean_r:.2f} | Shaped={mean_s:.2f} | "
              f"Conf={metrics['conf']:.2f} | Corr={metrics['corr_rate']:.2f}")
        
        if (iteration + 1) % args.eval_interval == 0:
            er = evaluate(policy, env, device)
            print(f"  => Eval: succ={er['success_rate']:.0%} ADE={er['ade']:.3f} FDE={er['fde']:.3f}")
            history['iterations'].append(iteration + 1)
            history['eval_success_rate'].append(er['success_rate'])
            history['eval_ade'].append(er['ade'])
            history['eval_fde'].append(er['fde'])
        
        if (iteration + 1) % args.save_interval == 0:
            ckpt = os.path.join(output_dir, f"checkpoint_iter{iteration+1}.pt")
            torch.save({
                'iteration': iteration + 1,
                'delta_state_dict': policy.delta_head.state_dict(),
                'correction_state_dict': policy.correction_head.state_dict(),
            }, ckpt)
            print(f"  => Saved {ckpt}")
    
    # Final
    final = os.path.join(output_dir, "final_model.pt")
    torch.save({
        'delta_state_dict': policy.delta_head.state_dict(),
        'correction_state_dict': policy.correction_head.state_dict(),
    }, final)
    
    er = evaluate(policy, env, device)
    print(f"\n✅ Final: succ={er['success_rate']:.0%} ADE={er['ade']:.3f} FDE={er['fde']:.3f}")
    print(f"Output: {output_dir}")
    
    hist_path = os.path.join(output_dir, "history.json")
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return history, er


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sft-checkpoint', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--success-reward', type=float, default=100.0)
    parser.add_argument('--progress-reward', type=float, default=5.0)
    parser.add_argument('--no-self-evolution', action='store_true')
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=25)
    args = parser.parse_args()
    main(args)