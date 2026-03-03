"""
Complete SFT → RL Pipeline for Driving-First Approach.

This script provides an end-to-end pipeline that:
1. Trains SFT waypoint model (or loads existing checkpoint)
2. Trains RL delta head on top of frozen SFT using PPO
3. Evaluates the combined SFT + RL model

Usage:
    # Full pipeline (train SFT, then train RL):
    python -m training.sft_rl_pipeline \
        --output-dir out/sft_rl_pipeline \
        --sft-epochs 50 \
        --rl-episodes 100
    
    # Use existing SFT checkpoint, train RL:
    python -m training.sft_rl_pipeline \
        --output-dir out/sft_rl_pipeline \
        --sft-checkpoint out/waypoint_bc_sft/best_checkpoint.pt \
        --rl-episodes 100
        
    # Smoke test:
    python -m training.sft_rl_pipeline --smoke

The driving-first pipeline:
    Waymo episodes → SSL pretrain → waypoint BC (SFT) → RL refinement → CARLA eval
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))

from rl.ppo_residual_delta_train import (
    PPOResidualDeltaAgent, 
    load_sft_checkpoint, 
    SFTWaypointModel,
    DeltaWaypointHead
)
from rl.waypoint_rl_env import WaypointRLEnv


def generate_synthetic_data(
    n_samples: int = 10000,
    state_dim: int = 6,
    waypoint_dim: int = 2,
    horizon: int = 20,
    noise_std: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic waypoint data for SFT training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    states = []
    waypoints = []
    
    for _ in range(n_samples):
        x = np.random.randn() * 50
        y = np.random.randn() * 50
        heading = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(0, 10)
        
        goal_distance = np.random.uniform(20, 100)
        goal_angle = heading + np.random.uniform(-0.5, 0.5)
        goal_x = x + goal_distance * np.cos(goal_angle)
        goal_y = y + goal_distance * np.sin(goal_angle)
        
        state = np.array([x, y, heading, speed, goal_x, goal_y], dtype=np.float32)
        
        t = np.linspace(0, 1, horizon)
        dx = (goal_x - x) * t + np.random.randn(horizon) * noise_std
        dy = (goal_y - y) * t + np.random.randn(horizon) * noise_std
        
        curvature = np.random.uniform(-0.1, 0.1)
        dx += curvature * t ** 2 * 10
        
        # Flatten for SFTWaypointModel (horizon * 2)
        waypoint = np.concatenate([dx, dy]).astype(np.float32)
        
        states.append(state)
        waypoints.append(waypoint)
    
    return np.array(states, dtype=np.float32), np.array(waypoints, dtype=np.float32)


def train_sft(
    output_dir: str,
    sft_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    horizon: int = 20,
    state_dim: int = 6,
    hidden_dim: int = 64,
    eval_interval: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
) -> Dict:
    """
    Train SFT waypoint model (compatible with PPO residual delta training).
    
    Returns:
        Dict with training metrics and checkpoint path
    """
    print(f"\n{'='*60}")
    print("PHASE 1: SFT Waypoint Training")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate training data
    print("Generating synthetic training data...")
    train_states, train_waypoints = generate_synthetic_data(
        n_samples=10000, horizon=horizon, seed=seed
    )
    val_states, val_waypoints = generate_synthetic_data(
        n_samples=2000, horizon=horizon, seed=seed + 1
    )
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(train_states),
        torch.from_numpy(train_waypoints)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_states),
        torch.from_numpy(val_waypoints)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model (compatible with SFTWaypointModel architecture)
    model = SFTWaypointModel(
        state_dim=state_dim,
        horizon=horizon,
        hidden_dim=hidden_dim,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Training loop
    best_val_loss = float('inf')
    sft_metrics = {"train_loss": [], "val_loss": []}
    
    for epoch in range(1, sft_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for batch_states, batch_waypoints in train_loader:
            batch_states = batch_states.to(device)
            batch_waypoints = batch_waypoints.to(device)
            
            optimizer.zero_grad()
            pred_waypoints = model(batch_states)
            loss = criterion(pred_waypoints, batch_waypoints)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_states, batch_waypoints in val_loader:
                batch_states = batch_states.to(device)
                batch_waypoints = batch_waypoints.to(device)
                
                pred_waypoints = model(batch_states)
                loss = criterion(pred_waypoints, batch_waypoints)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        sft_metrics["train_loss"].append(train_loss)
        sft_metrics["val_loss"].append(val_loss)
        
        if epoch % eval_interval == 0 or epoch == sft_epochs:
            print(f"Epoch {epoch}/{sft_epochs}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(output_dir, "sft_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"  → Saved best model (val_loss={val_loss:.4f})")
    
    # Save final model
    final_checkpoint_path = os.path.join(output_dir, "sft_final.pt")
    torch.save({
        "epoch": sft_epochs,
        "model_state": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": best_val_loss,
    }, final_checkpoint_path)
    
    print(f"\nSFT Training Complete!")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Checkpoint: {checkpoint_path}")
    
    return {
        "checkpoint_path": checkpoint_path,
        "best_val_loss": best_val_loss,
        "metrics": sft_metrics,
    }


def train_rl(
    sft_checkpoint_path: str,
    output_dir: str,
    rl_episodes: int = 100,
    horizon: int = 20,
    hidden_dim: int = 64,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
) -> Dict:
    """
    Train RL delta head on top of frozen SFT model.
    
    Returns:
        Dict with training metrics
    """
    print(f"\n{'='*60}")
    print("PHASE 2: RL Delta Head Training (PPO)")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load SFT model from checkpoint
    print(f"Loading SFT checkpoint: {sft_checkpoint_path}")
    sft_model = load_sft_checkpoint(
        sft_checkpoint_path,
        state_dim=6,
        horizon=horizon,
        hidden_dim=hidden_dim,
    )
    
    # Create environment
    env = WaypointRLEnv(horizon=horizon)
    
    # Create PPO agent with SFT model
    print(f"Creating PPO Residual Delta Agent...")
    agent = PPOResidualDeltaAgent(
        state_dim=6,
        horizon=horizon,
        hidden_dim=hidden_dim,
        lr=lr,
        sft_model=sft_model,
    )
    
    # Count parameters
    sft_params = sum(p.numel() for p in sft_model.parameters()) if sft_model else 0
    delta_params = sum(p.numel() for p in agent.delta_head.parameters() if p.requires_grad)
    value_params = sum(p.numel() for p in agent.value_fn.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in agent.parameters())
    
    print(f"Model parameters:")
    print(f"  SFT (frozen): {sft_params:,}")
    print(f"  Delta head (trainable): {delta_params:,}")
    print(f"  Value function (trainable): {value_params:,}")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable ratio: {(delta_params + value_params)/total_params*100:.1f}%")
    
    # Training loop
    print(f"\nTraining RL delta head ({rl_episodes} episodes)...")
    episode_rewards = []
    episode_goals = []
    
    for episode in range(1, rl_episodes + 1):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get action from agent
            action = agent.get_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, done)
            
            state = next_state
            episode_reward += reward
        
        # Update policy
        if episode % agent.update_interval == 0:
            agent.update()
        
        episode_rewards.append(episode_reward)
        episode_goals.append(info.get("goal_reached", False))
        
        if episode % 20 == 0 or episode == rl_episodes:
            avg_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
            goal_rate = np.mean(episode_goals[-20:]) if len(episode_goals) >= 20 else np.mean(episode_goals)
            print(f"Episode {episode}/{rl_episodes}: "
                  f"avg_reward={avg_reward:.2f}, goal_rate={goal_rate*100:.1f}%")
    
    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, "rl_checkpoint.pt")
    agent.save(checkpoint_path)
    
    rl_metrics = {
        "avg_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "goal_rate": np.mean(episode_goals),
        "total_episodes": rl_episodes,
    }
    
    print(f"\nRL Training Complete!")
    print(f"  Avg reward: {rl_metrics['avg_reward']:.2f} ± {rl_metrics['std_reward']:.2f}")
    print(f"  Goal rate: {rl_metrics['goal_rate']*100:.1f}%")
    print(f"  Checkpoint: {checkpoint_path}")
    
    return {
        "checkpoint_path": checkpoint_path,
        "metrics": rl_metrics,
    }


def evaluate_combined(
    sft_checkpoint_path: str,
    rl_checkpoint_path: str,
    output_dir: str,
    horizon: int = 20,
    hidden_dim: int = 64,
    num_episodes: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
) -> Dict:
    """
    Evaluate SFT-only vs SFT+RL models.
    
    Returns:
        Dict with evaluation metrics for both models
    """
    print(f"\n{'='*60}")
    print("PHASE 3: Evaluation (SFT vs SFT+RL)")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load SFT model
    sft_model = load_sft_checkpoint(
        sft_checkpoint_path,
        state_dim=6,
        horizon=horizon,
        hidden_dim=hidden_dim,
    )
    
    # Create environment
    env = WaypointRLEnv(horizon=horizon)
    
    # SFT-only evaluation
    print("\nEvaluating SFT-only model...")
    sft_rewards = []
    sft_goals = []
    
    for ep in range(num_episodes):
        state = env.reset(seed=seed + ep)
        episode_reward = 0
        done = False
        
        while not done:
            # Get SFT waypoints (no delta)
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                sft_waypoints = sft_model(state_tensor).squeeze(0).numpy()
            
            # Take first waypoint as action
            action = sft_waypoints[:2]  # First 2 values are first waypoint
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
        
        sft_rewards.append(episode_reward)
        sft_goals.append(info.get("goal_reached", False))
    
    sft_results = {
        "avg_reward": np.mean(sft_rewards),
        "std_reward": np.std(sft_rewards),
        "goal_rate": np.mean(sft_goals),
    }
    
    print(f"SFT-only: reward={sft_results['avg_reward']:.2f}±{sft_results['std_reward']:.2f}, "
          f"goal_rate={sft_results['goal_rate']*100:.1f}%")
    
    # SFT+RL evaluation
    print("\nEvaluating SFT+RL model...")
    
    # Load agent
    agent = PPOResidualDeltaAgent(
        state_dim=6,
        horizon=horizon,
        hidden_dim=hidden_dim,
        lr=3e-4,
        sft_model=sft_model,
    )
    agent.load(rl_checkpoint_path)
    agent.eval()  # Evaluation mode (no exploration)
    
    rl_rewards = []
    rl_goals = []
    
    for ep in range(num_episodes):
        state = env.reset(seed=seed + ep)
        episode_reward = 0
        done = False
        
        while not done:
            # Get action from RL agent
            action = agent.get_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
        
        rl_rewards.append(episode_reward)
        rl_goals.append(info.get("goal_reached", False))
    
    rl_results = {
        "avg_reward": np.mean(rl_rewards),
        "std_reward": np.std(rl_rewards),
        "goal_rate": np.mean(rl_goals),
    }
    
    print(f"SFT+RL:   reward={rl_results['avg_reward']:.2f}±{rl_results['std_reward']:.2f}, "
          f"goal_rate={rl_results['goal_rate']*100:.1f}%")
    
    # Compute improvement
    reward_improvement = rl_results['avg_reward'] - sft_results['avg_reward']
    goal_improvement = (rl_results['goal_rate'] - sft_results['goal_rate']) * 100
    
    print(f"\nImprovement:")
    print(f"  Reward: {reward_improvement:+.2f}")
    print(f"  Goal rate: {goal_improvement:+.1f}%")
    
    return {
        "sft": sft_results,
        "rl": rl_results,
        "improvement": {
            "reward": reward_improvement,
            "goal_rate_pct": goal_improvement,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Complete SFT → RL Pipeline")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="out/sft_rl_pipeline",
                        help="Output directory for all artifacts")
    
    # SFT arguments
    parser.add_argument("--sft-epochs", type=int, default=50,
                        help="Number of SFT training epochs")
    parser.add_argument("--sft-checkpoint", type=str, default=None,
                        help="Path to existing SFT checkpoint (skip SFT training)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="SFT batch size")
    parser.add_argument("--sft-lr", type=float, default=1e-3,
                        help="SFT learning rate")
    
    # RL arguments
    parser.add_argument("--rl-episodes", type=int, default=100,
                        help="Number of RL training episodes")
    parser.add_argument("--delta-lr", type=float, default=3e-4,
                        help="Delta head learning rate")
    
    # Model arguments
    parser.add_argument("--horizon", type=int, default=20,
                        help="Waypoint horizon")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Hidden dimension")
    
    # Eval arguments
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    
    # Other
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--smoke", action="store_true",
                        help="Run smoke test with minimal training")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation phase")
    
    args = parser.parse_args()
    
    # Smoke test overrides
    if args.smoke:
        args.sft_epochs = 5
        args.rl_episodes = 20
        args.eval_episodes = 20
        print("SMOKE TEST MODE: Using minimal training")
    
    print(f"\n{'='*60}")
    print("Complete SFT → RL Pipeline for Driving-First Approach")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    pipeline_results = {}
    start_time = time.time()
    
    # Phase 1: SFT Training (or use existing checkpoint)
    if args.sft_checkpoint:
        sft_results = {
            "checkpoint_path": args.sft_checkpoint,
            "skipped": True,
        }
        print(f"\nUsing existing SFT checkpoint: {args.sft_checkpoint}")
    else:
        sft_results = train_sft(
            output_dir=args.output_dir,
            sft_epochs=args.sft_epochs,
            batch_size=args.batch_size,
            lr=args.sft_lr,
            horizon=args.horizon,
            state_dim=6,
            hidden_dim=args.hidden_dim,
            device=args.device,
            seed=args.seed,
        )
    
    pipeline_results["sft"] = sft_results
    sft_checkpoint_path = sft_results["checkpoint_path"]
    
    # Phase 2: RL Training
    rl_results = train_rl(
        sft_checkpoint_path=sft_checkpoint_path,
        output_dir=args.output_dir,
        rl_episodes=args.rl_episodes,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        lr=args.delta_lr,
        device=args.device,
        seed=args.seed,
    )
    
    pipeline_results["rl"] = rl_results
    
    # Phase 3: Evaluation
    if not args.skip_eval:
        eval_results = evaluate_combined(
            sft_checkpoint_path=sft_checkpoint_path,
            rl_checkpoint_path=rl_results["checkpoint_path"],
            output_dir=args.output_dir,
            horizon=args.horizon,
            hidden_dim=args.hidden_dim,
            num_episodes=args.eval_episodes,
            device=args.device,
            seed=args.seed,
        )
        pipeline_results["eval"] = eval_results
    
    # Save pipeline results
    total_time = time.time() - start_time
    pipeline_results["total_time_seconds"] = total_time
    
    results_path = os.path.join(args.output_dir, "pipeline_results.json")
    with open(results_path, "w") as f:
        json.dump(pipeline_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {results_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    if "eval" in pipeline_results:
        eval_res = pipeline_results["eval"]
        print(f"SFT-only:  reward={eval_res['sft']['avg_reward']:.2f}, goal_rate={eval_res['sft']['goal_rate']*100:.1f}%")
        print(f"SFT+RL:    reward={eval_res['rl']['avg_reward']:.2f}, goal_rate={eval_res['rl']['goal_rate']*100:.1f}%")
        print(f"Improvement: reward={eval_res['improvement']['reward']:+.2f}, "
              f"goal_rate={eval_res['improvement']['goal_rate_pct']:+.1f}%")


if __name__ == "__main__":
    main()
