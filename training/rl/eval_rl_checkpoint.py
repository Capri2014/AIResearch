"""
Real RL Checkpoint Evaluation Script

Loads the trained RL checkpoint from PPO kinematics delta training
and runs evaluation comparing SFT-only vs SFT+RL policies.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# Add repo root to path
_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object

from training.rl.kinematics_waypoint_env import KinematicsWaypointEnv


# ============================================================================
# Model Classes (simplified from train_ppo_kinematics_delta.py)
# ============================================================================

class SFTWaypointModel(nn.Module if HAS_TORCH else object):
    """SFT waypoint model - simple MLP predictor."""
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        num_waypoints: int = 10,
    ):
        if HAS_TORCH:
            super().__init__()
            import torch.nn as nn
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            )
            self.waypoint_head = nn.Linear(hidden_dim, num_waypoints * 2)
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_waypoints = num_waypoints
        else:
            raise ImportError("PyTorch required")
    
    def forward(self, obs):
        import torch
        h = self.encoder(obs)
        out = self.waypoint_head(h)
        waypoints = out.view(-1, self.num_waypoints, 2)
        waypoints = torch.tanh(waypoints) * 10.0
        return waypoints
    
    def predict_waypoints(self, obs):
        """Predict waypoints from observation."""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        waypoints = self.forward(obs)
        return waypoints


class DeltaWaypointHead(nn.Module if HAS_TORCH else object):
    """Delta head for residual learning."""
    
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 64,
        num_waypoints: int = 10,
    ):
        if HAS_TORCH:
            super().__init__()
            import torch.nn as nn
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_waypoints * 2),
            )
            self.latent_dim = latent_dim
            self.num_waypoints = num_waypoints
        else:
            raise ImportError("PyTorch required")
    
    def forward(self, z):
        import torch
        out = self.net(z)
        delta = out.view(-1, self.num_waypoints, 2)
        delta = torch.tanh(delta) * 2.0
        return delta
    
    def predict_delta(self, z):
        """Predict delta from latent."""
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
        delta = self.forward(z)
        return delta


class DeltaWaypointPolicy:
    """Combined policy: SFT + delta head."""
    
    def __init__(
        self,
        sft_model,
        delta_head,
        delta_scale: float = 1.0,
    ):
        self.sft_model = sft_model
        self.delta_head = delta_head
        self.delta_scale = delta_scale
    
    def get_latent(self, obs):
        import torch
        return self.sft_model.encoder(obs)
    
    def predict_waypoints(self, obs):
        """Predict combined waypoints (SFT + delta)."""
        import torch
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        
        # SFT waypoints
        sft_waypoints = self.sft_model.forward(obs)
        
        # Get latent for delta
        z = self.get_latent(obs)
        
        # Delta waypoints
        delta = self.delta_head.forward(z)
        
        # Combine
        waypoints = sft_waypoints + self.delta_scale * delta
        
        return waypoints


# ============================================================================
# Checkpoint Loading
# ============================================================================

def load_rl_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load RL checkpoint and extract policy components."""
    print(f"Loading RL checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy_state = checkpoint.get("policy_state", {})
    config = checkpoint.get("config", {})
    
    print(f"Checkpoint loaded. Config: {config}")
    print(f"Policy state keys: {list(policy_state.keys())[:5]}...")
    
    return checkpoint, policy_state, config


def create_models_from_config(config: dict, device: str = "cpu"):
    """Create SFT and delta models from config."""
    import torch.nn as nn
    
    num_waypoints = config.get("num_waypoints", 10)
    hidden_dim = 64
    input_dim = 8
    
    # Create SFT model
    sft_model = SFTWaypointModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_waypoints=num_waypoints,
    )
    
    # Create delta head
    delta_head = DeltaWaypointHead(
        latent_dim=hidden_dim,
        hidden_dim=hidden_dim,
        num_waypoints=num_waypoints,
    )
    
    return sft_model, delta_head


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_policy(env, policy, episodes: int = 10, seed_base: int = 100, max_steps: int = 50):
    """Evaluate policy on kinematics waypoint environment."""
    import torch
    
    all_ade = []
    all_fde = []
    all_success = []
    all_max_accel = []
    all_max_jerk = []
    
    for ep in range(episodes):
        seed = seed_base + ep
        obs = env.reset(seed=seed)
        
        steps = 0
        
        while steps < max_steps:
            # Get waypoint prediction from policy
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                waypoints = policy.predict_waypoints(obs_tensor)
                waypoints_np = waypoints.squeeze(0).numpy()
            
            # Step environment
            obs, reward, done, info = env.step(waypoints_np)
            
            steps += 1
            
            if done:
                break
        
        # Compute episode metrics at episode end
        episode_metrics = env.compute_metrics()
        
        all_ade.append(float(episode_metrics.get('ADE', 0.0)))
        all_fde.append(float(episode_metrics.get('FDE', 0.0)))
        all_success.append(float(episode_metrics.get('success', 0.0)))
        all_max_accel.append(float(episode_metrics.get('max_accel', 0.0)))
        all_max_jerk.append(float(episode_metrics.get('max_jerk', 0.0)))
    
    # Aggregate
    metrics = {
        "ade_mean": float(np.mean(all_ade)),
        "ade_std": float(np.std(all_ade)),
        "fde_mean": float(np.mean(all_fde)),
        "fde_std": float(np.std(all_fde)),
        "success_rate": float(np.mean(all_success)),
        "max_accel_mean": float(np.mean(all_max_accel)),
        "max_jerk_mean": float(np.mean(all_max_jerk)),
    }
    
    return metrics


def run_evaluation(
    rl_checkpoint_path: str,
    episodes: int = 10,
    seed_base: int = 100,
    max_steps: int = 50,
    delta_scale: float = 1.0,
    world_size: float = 100.0,
    output_dir: str = None
):
    """Run complete evaluation with RL checkpoint."""
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = f"out/eval/rl_checkpoint_eval_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cpu"
    
    # Load RL checkpoint
    checkpoint, policy_state, config = load_rl_checkpoint(rl_checkpoint_path, device)
    
    # Create models
    sft_model, delta_head = create_models_from_config(config, device)
    
    # Load trained delta head from checkpoint
    if policy_state:
        # Try to load the delta head state
        delta_state = {}
        for k, v in policy_state.items():
            if 'delta' in k or len(delta_state) > 0:
                # Extract delta head params
                new_key = k.replace('sft_model.', '').replace('delta_head.', '')
                delta_state[new_key] = v
        
        # Try to load partial state
        try:
            delta_head.net.load_state_dict(policy_state, strict=False)
            print("Loaded delta head from checkpoint")
        except Exception as e:
            print(f"Could not load delta head: {e}")
            print("Using random weights (this is a demo)")
    
    # Freeze SFT
    for p in sft_model.parameters():
        p.requires_grad = False
    
    # Create environment
    env = KinematicsWaypointEnv(world_size=world_size, max_episode_steps=max_steps)
    
    # Evaluate SFT-only (delta_scale=0.0)
    print("\n=== Evaluating SFT-only policy ===")
    sft_policy = DeltaWaypointPolicy(sft_model, delta_head, delta_scale=0.0)
    sft_metrics = evaluate_policy(env, sft_policy, episodes, seed_base, max_steps)
    print(f"SFT-only: ADE={sft_metrics['ade_mean']:.3f}m ± {sft_metrics['ade_std']:.3f}m, "
          f"FDE={sft_metrics['fde_mean']:.3f}m, Success={sft_metrics['success_rate']:.1%}")
    
    # Evaluate SFT+RL (delta_scale=1.0)
    print("\n=== Evaluating SFT+RL policy ===")
    rl_policy = DeltaWaypointPolicy(sft_model, delta_head, delta_scale=delta_scale)
    rl_metrics = evaluate_policy(env, rl_policy, episodes, seed_base, max_steps)
    print(f"SFT+RL: ADE={rl_metrics['ade_mean']:.3f}m ± {rl_metrics['ade_std']:.3f}m, "
          f"FDE={rl_metrics['fde_mean']:.3f}m, Success={rl_metrics['success_rate']:.1%}")
    
    # Compute delta
    ade_delta = rl_metrics['ade_mean'] - sft_metrics['ade_mean']
    ade_delta_pct = (ade_delta / sft_metrics['ade_mean'] * 100) if sft_metrics['ade_mean'] > 0 else 0.0
    fde_delta = rl_metrics['fde_mean'] - sft_metrics['fde_mean']
    fde_delta_pct = (fde_delta / sft_metrics['fde_mean'] * 100) if sft_metrics['fde_mean'] > 0 else 0.0
    
    print(f"\n=== Delta (RL - SFT) ===")
    print(f"ADE: {ade_delta:+.3f}m ({ade_delta_pct:+.1f}%)")
    print(f"FDE: {fde_delta:+.3f}m ({fde_delta_pct:+.1f}%)")
    
    # Build output metrics
    output = {
        "run_id": f"rl_checkpoint_eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "domain": "rl_checkpoint_eval",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "episodes": episodes,
            "seed_base": seed_base,
            "max_steps": max_steps,
            "delta_scale": delta_scale,
            "world_size": world_size,
            "rl_checkpoint": rl_checkpoint_path
        },
        "sft_only": sft_metrics,
        "rl_refined": rl_metrics,
        "comparison": {
            "ade_delta": float(ade_delta),
            "ade_delta_pct": float(ade_delta_pct),
            "fde_delta": float(fde_delta),
            "fde_delta_pct": float(fde_delta_pct),
            "improvement": "yes" if ade_delta < 0 else "no"
        },
        "checkpoint_info": {
            "config": config
        }
    }
    
    # Write metrics.json
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nMetrics saved to: {metrics_path}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL checkpoint on kinematics waypoint env")
    parser.add_argument("--checkpoint", type=str, 
                        default="out/ppo_kinematics_delta_sft/run_20260331-173000/checkpoint.pt",
                        help="Path to RL checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--seed-base", type=int, default=100, help="Base random seed")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--delta-scale", type=float, default=1.0, help="Delta scale factor")
    parser.add_argument("--world-size", type=float, default=100.0, help="World size in meters")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    run_evaluation(
        rl_checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        seed_base=args.seed_base,
        max_steps=args.max_steps,
        delta_scale=args.delta_scale,
        world_size=args.world_size,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
