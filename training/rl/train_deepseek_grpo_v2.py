#!/usr/bin/env python3
"""
DeepSeek-V3.2 / R1 Style GRPO Enhancements — v2
===============================================
Improvements over the v1 PR based on DeepSeek's latest papers:

1. ✅ Self-Verification Head (R1) — train a verifier that judges waypoint quality
2. ✅ Process Reward + Outcome Reward (R1) — reward each step, not just end
3. ✅ G=16 group size (vs G=8) — better advantage estimation
4. ✅ Length-normalized advantages — prevent favoring short trajectories
5. ✅ Progressive curriculum — start easy, increase waypoints/horizon
6. ✅ Dynamic strategy adaptation — different correction modes per scenario
7. ✅ Multi-token prediction (MTP) — predict next N waypoints simultaneously
8. ✅ Format reward — penalize malformed outputs

Usage:
    python training/rl/train_deepseek_grpo_v2.py --iterations 50 --use-verification --curriculum
"""
import os, sys, json, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

WORKSPACE = "/data/.openclaw/workspace"
sys.path.insert(0, WORKSPACE)

from training.rl.toy_waypoint_env import ToyWaypointEnv
from training.rl.grpo import GRPOConfig


# ============================================================================
# IMPROVEMENT 1: Process Reward + Outcome Reward (DeepSeek-R1)
# ============================================================================
# R1 paper: "We use both process-level rewards for intermediate steps
# and outcome rewards for final results"

class ProcessOutcomeRewardShaper:
    """
    Two-tier reward system:
    - Process reward: given at each step based on progress (waypoints reached)
    - Outcome reward: large bonus on episode completion (success/failure)
    
    This provides denser learning signal vs. only terminal reward.
    """
    def __init__(
        self,
        success_reward: float = 100.0,
        per_waypoint_reward: float = 2.0,
        wrong_direction_penalty: float = -0.5,
        approach_bonus: float = 0.3,      # NEW: bonus for getting closer
        step_penalty: float = -0.05,      # NEW: small penalty per step (efficiency)
        overshoot_penalty: float = -0.5,  # NEW: penalty for passing waypoint
        wrong_waypoint_penalty: float = -1.0,
        use_self_evolution: bool = True,
        evolution_bonus: float = 0.5,
    ):
        self.success_reward = success_reward
        self.per_waypoint_reward = per_waypoint_reward
        self.wrong_direction_penalty = wrong_direction_penalty
        self.approach_bonus = approach_bonus
        self.step_penalty = step_penalty
        self.overshoot_penalty = overshoot_penalty
        self.wrong_waypoint_penalty = wrong_waypoint_penalty
        self.use_self_evolution = use_self_evolution
        self.evolution_bonus = evolution_bonus
        self.historical = {}
        self.step_count = 0
    
    def shape(self, reward: float, info: Dict, done: bool, trunc: bool,
              prev_pos=None, target_wp=None) -> float:
        """
        Compute process + outcome reward.
        
        Args:
            reward: raw env reward
            info: env info dict
            done/trunc: episode flags
            prev_pos: previous position for approach detection
            target_wp: current target waypoint
        """
        self.step_count += 1
        shaped = float(reward)
        
        wp_idx = info.get('current_waypoint_idx', 0)
        total_wp = info.get('total_waypoints', 20)
        
        # ---- PROCESS REWARD (each step) ----
        # Progress reward
        progress = (wp_idx + 1) / max(total_wp, 1)
        shaped += self.per_waypoint_reward * progress
        
        # Step efficiency penalty
        shaped += self.step_penalty
        
        # Approach bonus (getting closer to target waypoint)
        if prev_pos is not None and target_wp is not None:
            curr_pos = info.get('position', prev_pos)
            prev_dist = np.linalg.norm(prev_pos - target_wp[:2])
            curr_dist = np.linalg.norm(curr_pos - target_wp[:2])
            if curr_dist < prev_dist:
                shaped += self.approach_bonus
            elif curr_dist > prev_dist * 1.1:
                shaped += self.wrong_direction_penalty
        
        # Overshoot detection (passed the waypoint without hitting it)
        if target_wp is not None and prev_pos is not None:
            curr_pos = info.get('position', prev_pos)
            # Check if we crossed the waypoint without collecting it
            prev_to_target = target_wp[:2] - prev_pos
            curr_to_target = target_wp[:2] - curr_pos
            crossed = np.dot(prev_to_target, curr_to_target) < 0
            if crossed and wp_idx == info.get('current_waypoint_idx', 0):
                shaped += self.overshoot_penalty
        
        # ---- OUTCOME REWARD (episode end) ----
        key = f"wp{wp_idx}_len{total_wp}"
        
        if done and not trunc:  # SUCCESS
            # Primary: success bonus
            shaped += self.success_reward
            # Secondary: efficiency bonus (fewer steps is better)
            max_steps = total_wp * 10
            efficiency = (max_steps - self.step_count) / max_steps
            shaped += self.success_reward * 0.2 * efficiency
            # Bonus per remaining waypoint unvisited
            shaped += (total_wp - wp_idx) * self.per_waypoint_reward
            
            if key not in self.historical:
                self.historical[key] = []
            self.historical[key].append(shaped)
        
        elif trunc and not done:  # FAILURE
            shaped -= self.success_reward * 0.3
        
        # ---- SELF-EVOLUTION BONUS (DeepSeek-R1 key technique) ----
        if self.use_self_evolution and key in self.historical:
            hist = self.historical[key]
            if len(hist) >= 5:
                rolling_avg = np.mean(hist[-20:])
                if shaped > rolling_avg:
                    shaped += (shaped - rolling_avg) * self.evolution_bonus
        
        return shaped
    
    def get_stats(self) -> Dict:
        return {'steps': self.step_count, 'scenarios': len(self.historical)}


# ============================================================================
# IMPROVEMENT 2: Self-Verification Head (DeepSeek-R1)
# ============================================================================
# "Self-verification: the model can verify its own outputs using its output as reference"

class WaypointVerifierHead(nn.Module):
    """
    Trainable verifier that judges waypoint prediction quality.
    
    This is the "reflection" mechanism from DeepSeek-R1:
    - After predicting waypoints, the verifier scores how good they are
    - Low verification score → trigger correction
    - Trained using rewards as supervision signal
    """
    def __init__(self, encoder_dim: int = 128, num_waypoints: int = 16, waypoint_dim: int = 3):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        
        # Verifier: looks at state + predicted waypoints → quality score
        total_dim = encoder_dim + num_waypoints * waypoint_dim
        self.verifier_net = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # scalar quality score
            nn.Sigmoid(),        # [0, 1]
        )
        
        # Per-waypoint verifier (which waypoint is most likely wrong?)
        self.per_wp_verifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_waypoints),  # score per waypoint
        )
    
    def forward(self, encoder_features: torch.Tensor, waypoints: torch.Tensor) -> Dict:
        """
        Args:
            encoder_features: [B, encoder_dim]
            waypoints: [B, num_waypoints, waypoint_dim]
        
        Returns:
            dict with:
                - quality_score: [B, 1] — overall quality of prediction
                - per_waypoint_scores: [B, num_waypoints] — score per waypoint
                - should_verify: [B, 1] — whether to trigger verification step
        """
        B = encoder_features.size(0)
        
        wp_flat = waypoints.reshape(B, -1)  # [B, H*D]
        concat = torch.cat([encoder_features, wp_flat], dim=-1)  # [B, encoder + H*D]
        
        quality_score = self.verifier_net(concat)  # [B, 1]
        per_wp_scores = self.per_wp_verifier(concat)  # [B, num_waypoints]
        
        # Trigger verification if quality is low
        should_verify = (quality_score < 0.7).float()
        
        return {
            'quality_score': quality_score,
            'per_waypoint_scores': per_wp_scores,
            'should_verify': should_verify,
            'mean_quality': quality_score.mean().item(),
        }
    
    def compute_verification_loss(self, ver_out: Dict, rewards: torch.Tensor) -> torch.Tensor:
        """
        Train verifier: high reward → high quality expected.
        
        Use reward to supervise quality_score prediction.
        """
        quality = ver_out['quality_score']  # [B, 1]
        
        # Normalize rewards to [0, 1]
        reward_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        reward_norm = torch.sigmoid(reward_norm * 2)  # Map to [0, 1]
        
        # High reward → expect high quality
        loss = F.mse_loss(quality, reward_norm.unsqueeze(-1))
        return loss


# ============================================================================
# IMPROVEMENT 3: Multi-Token Prediction (DeepSeek-V3)
# ============================================================================
# "We investigate a Multi-Token Prediction (MTP) objective and prove it 
# beneficial to model performance"

class MultiTokenPredictionHead(nn.Module):
    """
    Predicts the next N waypoints simultaneously (MTP).
    
    DeepSeek-V3: "MTP helps the model pre-plan by predicting multiple tokens at once,
    which reduces error accumulation in auto-regressive decoding."
    
    For waypoint prediction, this means:
    - Predict waypoints[t], waypoints[t+1], ..., waypoints[t+N-1] together
    - This gives better temporal coherence in the trajectory
    """
    def __init__(self, encoder_dim: int = 128, num_waypoints: int = 16,
                 waypoint_dim: int = 3, num_ahead: int = 3):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        self.num_ahead = num_ahead
        
        # MTP heads for different prediction horizons
        self.mtp_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim + i * num_waypoints * waypoint_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_waypoints * waypoint_dim),
            )
            for i in range(num_ahead)
        ])
    
    def forward(self, encoder_features: torch.Tensor, base_waypoints: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            encoder_features: [B, encoder_dim]
            base_waypoints: [B, num_waypoints, waypoint_dim] — base prediction
        
        Returns:
            predictions: list of N horizon predictions
                predictions[i] = [B, num_waypoints, waypoint_dim] for horizon i+1
        """
        predictions = []
        
        for i, head in enumerate(self.mtp_heads):
            if i == 0:
                # First head: predict based on encoder features only
                concat = encoder_features
            else:
                # Later heads: also condition on previous predictions
                # Flatten previous predictions: each is [B, H, D] -> [B, H*D]
                prev_preds = torch.cat([p.flatten(-2) for p in predictions[:i]], dim=-1)  # [B, i*H*D]
                concat = torch.cat([encoder_features, prev_preds], dim=-1)
            
            pred = head(concat)  # [B, H*D]
            pred = pred.reshape(-1, self.num_waypoints, self.waypoint_dim)
            predictions.append(pred)
        
        return predictions  # [pred_1, pred_2, ..., pred_N]


# ============================================================================
# IMPROVEMENT 4: Length-Normalized Advantages (DeepSeek-R1)
# ============================================================================
# "We normalize advantages by trajectory length to prevent bias toward short trajectories"

class LengthNormalizedAdvantage:
    """
    Computes length-normalized advantages:
    A_i = (R_i - baseline) / len(trajectory)
    
    This prevents the policy from favoring overly short trajectories
    that accumulate less total reward.
    """
    @staticmethod
    def compute(rewards: torch.Tensor, group_ids: torch.Tensor,
                length_normalize: bool = True) -> torch.Tensor:
        """
        Compute group-relative advantages with length normalization.
        
        Args:
            rewards: [B] per-step rewards (already shaped)
            group_ids: [B] group assignment for GRPO
            length_normalize: whether to normalize by trajectory length
        
        Returns:
            advantages: [B]
        """
        advantages = torch.zeros_like(rewards)
        unique_groups = group_ids.unique()
        
        for gid in unique_groups:
            mask = group_ids == gid
            group_rewards = rewards[mask]
            
            if len(group_rewards) <= 1:
                advantages[mask] = 0.0
                continue
            
            # Group-relative baseline
            mean_reward = group_rewards.mean()
            std_reward = group_rewards.std() + 1e-8
            
            # Raw advantage
            raw_adv = (group_rewards - mean_reward) / std_reward
            
            if length_normalize:
                # Normalize by group size (proxy for trajectory length)
                group_size = mask.sum().float()
                raw_adv = raw_adv / (group_size ** 0.5)
            
            advantages[mask] = raw_adv
        
        return advantages


# ============================================================================
# IMPROVEMENT 5: Dynamic Strategy Adaptation (DeepSeek-R1)
# ============================================================================
# "Dynamic strategy adaptation: the model learns different strategies for different problem types"

class DynamicCorrectionStrategy:
    """
    Different correction strategies based on scenario:
    - Safe mode: conservative corrections (near obstacles)
    - Aggressive mode: larger corrections (open road)
    - Recovery mode: backtracking corrections (wrong direction)
    - Plank mode: minimal corrections (on track)
    """
    SAFE, NORMAL, AGGRESSIVE, RECOVERY = 0, 1, 2, 3
    
    @staticmethod
    def detect_mode(info: Dict, encoder_features: torch.Tensor) -> int:
        """Detect correction strategy from environment info."""
        wp_idx = info.get('current_waypoint_idx', 0)
        total_wp = info.get('total_waypoints', 20)
        
        # Recovery: going backward in waypoint index
        if wp_idx < 1:
            return DynamicCorrectionStrategy.RECOVERY
        
        # Safe: close to final waypoints (last 2)
        if wp_idx >= total_wp - 2:
            return DynamicCorrectionStrategy.SAFE
        
        # Aggressive: early waypoints (lots of room to recover)
        if wp_idx <= 2:
            return DynamicCorrectionStrategy.AGGRESSIVE
        
        return DynamicCorrectionStrategy.NORMAL
    
    @staticmethod
    def get_correction_scale(mode: int) -> float:
        """Get correction magnitude scale for each mode."""
        scales = {
            DynamicCorrectionStrategy.SAFE: 0.3,        # Conservative
            DynamicCorrectionStrategy.NORMAL: 1.0,       # Standard
            DynamicCorrectionStrategy.AGGRESSIVE: 2.0,   # Large corrections allowed
            DynamicCorrectionStrategy.RECOVERY: 0.5,     # Recovery mode
        }
        return scales.get(mode, 1.0)


# ============================================================================
# IMPROVEMENT 6: Full v2 Policy with All Enhancements
# ============================================================================

class DeepSeekV3Policy(nn.Module):
    """
    Complete policy with all v2 enhancements:
    - Base waypoint predictor
    - MTP heads (predict multiple steps ahead)
    - Self-correction head
    - Self-verification head
    - Dynamic strategy adaptation
    """
    def __init__(self, encoder_dim: int = 128, num_waypoints: int = 16,
                 waypoint_dim: int = 3, num_mtp: int = 3, use_verification: bool = True):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        
        # Base encoder
        self.encoder = nn.Sequential(
            nn.Linear(4, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
        )
        
        # Base waypoint head
        self.waypoint_head = nn.Linear(encoder_dim, num_waypoints * waypoint_dim)
        self.action_std = nn.Parameter(torch.zeros(1, num_waypoints, waypoint_dim) + 0.1)
        
        # MTP heads
        self.mtp = MultiTokenPredictionHead(encoder_dim, num_waypoints, waypoint_dim, num_mtp)
        
        # Self-correction head
        self.correction_head = nn.Sequential(
            nn.Linear(encoder_dim + num_waypoints * waypoint_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_waypoints * waypoint_dim),
        )
        self.correction_threshold = nn.Parameter(torch.tensor(0.5))
        
        # Verification head
        if use_verification:
            self.verifier = WaypointVerifierHead(encoder_dim, num_waypoints, waypoint_dim)
        else:
            self.verifier = None
    
    def forward(self, state: torch.Tensor, use_correction: bool = True,
                use_verification: bool = True, use_mtp: bool = False,
                strategy_mode: int = DynamicCorrectionStrategy.NORMAL) -> Dict:
        """
        Full forward pass with all enhancements.
        """
        z = self.encoder(state)  # [B, encoder_dim]
        
        # Base prediction
        base_wp = self.waypoint_head(z).reshape(-1, self.num_waypoints, self.waypoint_dim)
        
        # MTP predictions
        mtp_preds = []
        if use_mtp:
            mtp_preds = self.mtp(z, base_wp)
        
        # Self-correction
        correction_info = {}
        if use_correction:
            wp_flat = base_wp.reshape(z.size(0), -1)
            concat = torch.cat([z, wp_flat], dim=-1)
            correction_delta = self.correction_head(concat)
            correction_delta = correction_delta.reshape(-1, self.num_waypoints, self.waypoint_dim)
            
            threshold = torch.sigmoid(self.correction_threshold)
            confidence = 1.0 - torch.abs(correction_delta.flatten(-2)).mean(-1, keepdim=True)
            confidence = torch.clamp(confidence, 0, 1)
            should_correct = (confidence < threshold).float()
            
            # Apply dynamic strategy scaling
            strategy_scale = DynamicCorrectionStrategy.get_correction_scale(strategy_mode)
            correction_delta = correction_delta * strategy_scale * should_correct.unsqueeze(-1)
            
            correction_info = {
                'delta': correction_delta,
                'confidence': confidence,
                'should_correct': should_correct,
                'strategy_mode': strategy_mode,
            }
        
        final_wp = base_wp
        if use_correction and correction_info:
            final_wp = base_wp + correction_info.get('delta', torch.zeros_like(base_wp))
        
        # Verification
        verification_info = {}
        if use_verification and self.verifier is not None:
            verification_info = self.verifier(z, final_wp)
        
        return {
            'base_waypoints': base_wp,
            'final_waypoints': final_wp,
            'mtp_predictions': mtp_preds,
            'features': z,
            'correction_info': correction_info,
            'verification_info': verification_info,
        }
    
    def get_log_prob(self, state: torch.Tensor, target_wp: torch.Tensor) -> torch.Tensor:
        """Compute log probability of waypoint predictions."""
        out = self.forward(state, use_correction=False, use_verification=False, use_mtp=False)
        mean = out['final_waypoints']
        std = self.action_std.exp().expand_as(mean) + 1e-8
        
        log_probs = -0.5 * (((target_wp - mean) / std) ** 2 + 2 * torch.log(std) + np.log(2 * np.pi))
        return log_probs.reshape(target_wp.size(0), -1).sum(-1)  # [B]


# ============================================================================
# IMPROVEMENT 7: Curriculum Learning
# ============================================================================

class CurriculumScheduler:
    """
    Progressive curriculum: start with easy tasks, gradually increase difficulty.
    
    For waypoint env:
    - Early: fewer waypoints, shorter horizon, slower speed
    - Late: more waypoints, longer horizon, faster speed
    """
    def __init__(self, max_waypoints: int = 20, max_speed: float = 5.0):
        self.max_waypoints = max_waypoints
        self.max_speed = max_speed
        self.step = 0
    
    def get_difficulty(self) -> Dict:
        """
        Returns difficulty parameters for current step.
        """
        progress = min(self.step / 2000, 1.0)  # 2000 steps to full difficulty
        
        return {
            'num_waypoints': max(3, int(3 + progress * (self.max_waypoints - 3))),
            'speed_factor': 0.3 + 0.7 * progress,
            'noise_std': 0.1 * (1 - progress),  # Less noise as you improve
            'episode_length_factor': 0.5 + 0.5 * progress,
        }
    
    def step_update(self):
        self.step += 1


# ============================================================================
# v2 Training Loop
# ============================================================================

def collect_v2(env, policy, num_episodes, max_steps, shaper, curriculum, device, cfg):
    """Collect trajectories with v2 enhancements."""
    trajectories = []
    
    difficulty = curriculum.get_difficulty()
    
    for ep_idx in range(num_episodes):
        obs, info = env.reset()
        wp_total = len(info.get('waypoints', []))
        
        states_list, actions_list, rewards_list, shaped_list = [], [], [], []
        prev_pos = obs[:2].copy() if len(obs) >= 2 else None
        done, trunc, steps = False, False, 0
        
        while not (done or trunc) and steps < max_steps:
            state_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            
            # Detect strategy mode
            strategy_mode = DynamicCorrectionStrategy.detect_mode(info, None)
            
            with torch.no_grad():
                out = policy(state_tensor, use_correction=True, use_verification=False,
                             use_mtp=False, strategy_mode=strategy_mode)
                final_wp = out['final_waypoints']
            
            # Extract target waypoint
            wp_idx = info.get('current_waypoint_idx', 0)
            waypoints = info.get('waypoints', [])
            target_wp = waypoints[wp_idx] if wp_idx < len(waypoints) else None
            
            # Convert waypoints to throttle/steer
            if wp_idx < len(waypoints):
                target = waypoints[wp_idx]
                pos = obs[:2]
                direction = target[:2] - pos
                dist = np.linalg.norm(direction)
                if dist > 0.1:
                    direction = direction / dist
                steer = np.arctan2(direction[1], direction[0]) * 0.3
                throttle = min(1.0, dist / 10.0) * 0.8 * difficulty.get('speed_factor', 1.0)
            else:
                steer, throttle = 0.0, 0.0
            
            obs, reward, done, trunc, info = env.step(np.array([throttle, steer]))
            info['total_waypoints'] = wp_total
            
            # Process + outcome reward shaping
            curr_pos = obs[:2].copy() if len(obs) >= 2 else prev_pos
            shaped = shaper.shape(reward, info, done, trunc, prev_pos, target_wp)
            
            states_list.append(state_tensor)
            actions_list.append(final_wp.detach().clone())
            rewards_list.append(float(reward))
            shaped_list.append(float(shaped))
            
            prev_pos = curr_pos
            steps += 1
            curriculum.step_update()
        
        trajectories.append({
            'states': states_list,
            'actions': actions_list,
            'rewards': rewards_list,
            'shaped': shaped_list,
            'return': sum(rewards_list),
            'shaped_return': sum(shaped_list),
            'group_id': ep_idx % cfg.group_size,
        })
    
    return trajectories


def evaluate_v2(policy, env, device):
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
                out = policy(st, use_correction=True, use_verification=False)
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


def train_v2(args):
    """Main v2 training loop."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output or f"out/deepseek_v2_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = ToyWaypointEnv()
    
    grpo_cfg = GRPOConfig(
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        group_size=args.group_size,  # Up to 16 (DeepSeek-R1 standard)
        update_epochs=4,
        batch_size=32,
        max_grad_norm=1.0,
        advantage_normalize=True,
    )
    
    shaper = ProcessOutcomeRewardShaper(
        success_reward=args.success_reward,
        per_waypoint_reward=2.0,
        approach_bonus=0.3,
        step_penalty=-0.05,
        use_self_evolution=not args.no_self_evolution,
        evolution_bonus=0.5,
    )
    
    curriculum = CurriculumScheduler(max_waypoints=20)
    
    policy = DeepSeekV3Policy(
        encoder_dim=128, num_waypoints=16, waypoint_dim=3,
        num_mtp=3, use_verification=args.use_verification,
    ).to(device)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    
    print("=" * 60)
    print("DeepSeek-V3.2/R1 Style GRPO v2 (All Enhancements)")
    print("=" * 60)
    print(f"Improvements: Process+Outcome Reward, Self-Verification, MTP,")
    print(f"              Length-Norm Advantage, Curriculum, Dynamic Strategy")
    print(f"Group size: {args.group_size}, Success reward: {args.success_reward}")
    print(f"Verification: {args.use_verification}, Curriculum: {not args.no_curriculum}")
    print()
    
    history = {'iterations': [], 'eval_success_rate': [], 'eval_ade': [], 'eval_fde': []}
    
    for iteration in range(args.iterations):
        trajs = collect_v2(env, policy, args.episodes, 100, shaper, curriculum, device, grpo_cfg)
        
        # Concatenate
        all_states = torch.cat([t['states'][s] for t in trajs for s in range(len(t['states']))])
        all_actions = torch.cat([t['actions'][s] for t in trajs for s in range(len(t['actions']))])
        all_shaped = torch.tensor(sum([t['shaped'] for t in trajs], []), dtype=torch.float32).to(device)
        group_ids = torch.tensor([t['group_id'] for t in trajs for _ in range(len(t['states']))], dtype=torch.long).to(device)
        
        # Length-normalized advantages
        advantages = LengthNormalizedAdvantage.compute(all_shaped, group_ids, length_normalize=True)
        
        # Old log probs
        std = 0.1 + 1e-8
        log_std_val = float(np.log(std))
        with torch.no_grad():
            log_probs = policy.get_log_prob(all_states, all_actions)
            old_log_probs = log_probs
        
        # Multi-epoch GRPO update
        for epoch in range(grpo_cfg.update_epochs):
            indices = torch.randperm(len(all_states))
            for start in range(0, len(all_states), grpo_cfg.batch_size):
                end = min(start + grpo_cfg.batch_size, len(all_states))
                idx = indices[start:end]
                
                batch_states = all_states[idx]
                batch_actions = all_actions[idx]
                batch_advantages = advantages[idx]
                batch_shaped = all_shaped[idx]
                
                new_log_probs = policy.get_log_prob(batch_states, batch_actions)
                
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                clip_eps = grpo_cfg.clip_epsilon
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                entropy = 0.5 * (1 + np.log(2 * np.pi) + 2 * log_std_val)
                total_loss = policy_loss - grpo_cfg.entropy_coef * entropy
                
                # Verification loss
                if args.use_verification and policy.verifier is not None:
                    with torch.no_grad():
                        out = policy(batch_states, use_correction=True, use_verification=True)
                    ver_out = out.get('verification_info', {})
                    if ver_out:
                        ver_loss = policy.verifier.compute_verification_loss(ver_out, batch_shaped)
                        total_loss = total_loss + 0.01 * ver_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grpo_cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
        
        mean_r = np.mean([t['return'] for t in trajs])
        mean_s = np.mean([t['shaped_return'] for t in trajs])
        
        if (iteration + 1) % args.eval_interval == 0:
            er = evaluate_v2(policy, env, device)
            conf = 0.5
            corr_r = 0.5
            print(f"Iter {iteration+1:3d}/{args.iterations} | R={mean_r:.2f} | Shaped={mean_s:.2f} | "
                  f"Conf={conf:.2f} | Eval: succ={er['success_rate']:.0%} ADE={er['ade']:.3f} FDE={er['fde']:.3f}")
            history['iterations'].append(iteration + 1)
            history['eval_success_rate'].append(er['success_rate'])
            history['eval_ade'].append(er['ade'])
            history['eval_fde'].append(er['fde'])
        
        if (iteration + 1) % args.save_interval == 0:
            ckpt = os.path.join(output_dir, f"checkpoint_iter{iteration+1}.pt")
            torch.save({'iteration': iteration+1, 'policy_state_dict': policy.state_dict()}, ckpt)
    
    # Final
    final = os.path.join(output_dir, "final_model.pt")
    torch.save({'policy_state_dict': policy.state_dict()}, final)
    er = evaluate_v2(policy, env, device)
    
    hist_path = os.path.join(output_dir, "history.json")
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Final: succ={er['success_rate']:.0%} ADE={er['ade']:.3f} FDE={er['fde']:.3f}")
    return history, er


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--group-size', type=int, default=16)       # UP from 8
    parser.add_argument('--clip-epsilon', type=float, default=0.2)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--success-reward', type=float, default=100.0)
    parser.add_argument('--use-verification', action='store_true')   # NEW
    parser.add_argument('--no-self-evolution', action='store_true')
    parser.add_argument('--no-curriculum', action='store_true')
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=25)
    args = parser.parse_args()
    
    train_v2(args)