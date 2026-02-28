#!/usr/bin/env python3
"""
Generate animated GIF showing contingency planning in action.

Shows three approaches:
1. Baseline - aggressive, no contingency
2. Tree-Based - branches over possibilities
3. Model-Based - adaptive learned behavior
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# Create output directory
os.makedirs("out/contingency_animation", exist_ok=True)


def simulate_trajectory(planner_type, scenario="pedestrian", seed=42):
    """
    Simulate a single trajectory for a given planner.
    
    Returns list of (x, y) positions over time.
    """
    np.random.seed(seed)
    
    # Scenario parameters
    if scenario == "pedestrian":
        goal = np.array([50, 0])
        obstacle_pos = np.array([25, 0])
        obstacle_radius = 3
        initial_v = 8
    elif scenario == "cutin":
        goal = np.array([80, 0])
        obstacle_pos = np.array([40, 3.5])  # adjacent lane
        obstacle_radius = 2
        initial_v = 15
    else:
        goal = np.array([40, 0])
        obstacle_pos = np.array([20, 0])
        obstacle_radius = 2
        initial_v = 6
    
    # Initial state: [x, y, v, heading]
    state = np.array([0, 0, initial_v, 0])
    dt = 0.1
    
    positions = [(state[0], state[1])]
    states = [state.copy()]
    
    for t in range(200):  # max steps
        dist_to_goal = np.linalg.norm(state[:2] - goal)
        dist_to_obs = np.linalg.norm(state[:2] - obstacle_pos)
        
        if planner_type == "baseline":
            # Aggressive - no contingency
            desired_v = 12
            acc = (desired_v - state[2]) * 0.4
            steer = -state[3] * 0.1
            
        elif planner_type == "tree":
            # Conservative - branches, slows early
            if dist_to_obs < 25:
                desired_v = 4
            elif dist_to_obs < 40:
                desired_v = 8
            else:
                desired_v = 10
            acc = (desired_v - state[2]) * 0.2
            steer = -state[3] * 0.15
            
        else:  # model
            # Adaptive - learns to modulate
            progress = t / 50.0
            if dist_to_obs < 20:
                desired_v = 5 + progress * 2
            else:
                desired_v = 10 + progress * 3
            desired_v = min(desired_v, 12)
            acc = (desired_v - state[2]) * 0.3
            steer = -state[3] * 0.12
        
        # Dynamics
        v = max(0, state[2] + acc * dt)
        heading = state[3] + steer * dt
        x = state[0] + v * np.cos(heading) * dt
        y = state[1] + v * np.sin(heading) * dt
        
        state = np.array([x, y, v, heading])
        positions.append((x, y))
        states.append(state.copy())
        
        # Check done
        if dist_to_goal < 2:
            break
        if dist_to_obs < obstacle_radius:
            positions.append((x, y))  # Collision marker
            break
    
    return positions, obstacle_pos, obstacle_radius, goal


def create_animation():
    """Create animated GIF."""
    
    # Set up figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Contingency Planning: Three Approaches", fontsize=16, fontweight='bold')
    
    planner_types = ["baseline", "tree", "model"]
    titles = [
        "Baseline\n(No Contingency)",
        "Tree-Based\n(Classical Control-Tree)",
        "Model-Based\n(Neural + Safety Filter)"
    ]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    
    # Pre-compute trajectories
    trajectories = {}
    for ptype in planner_types:
        traj, obs, rad, goal = simulate_trajectory(ptype, "pedestrian")
        trajectories[ptype] = (traj, obs, rad, goal)
    
    # Find max time
    max_len = max(len(t[0]) for t in trajectories.values())
    
    def init():
        return []
    
    def animate(frame):
        artists = []
        
        for ax, (ptype, title) in enumerate(zip(planner_types, titles)):
            ax = axes[ax]
            ax.clear()
            ax.set_xlim(-5, 55)
            ax.set_ylim(-8, 8)
            ax.set_title(title, fontsize=11, color=colors[ax])
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            traj, obs, rad, goal = trajectories[ptype]
            
            # Draw obstacle
            circle = patches.Circle(obs, rad, color='red', alpha=0.4, label='Obstacle')
            ax.add_patch(circle)
            
            # Draw goal
            ax.scatter(goal[0], goal[1], marker='*', s=200, color='gold', 
                      edgecolors='black', linewidths=1, zorder=5, label='Goal')
            
            # Draw trajectory up to current frame
            n = min(frame + 1, len(traj))
            if n > 0:
                traj_arr = np.array(traj[:n])
                ax.plot(traj_arr[:, 0], traj_arr[:, 1], 
                       color=colors[ax], linewidth=2, alpha=0.8)
                
                # Draw ego vehicle
                x, y = traj[n-1]
                
                # Check collision
                dist_to_obs = np.linalg.norm(np.array([x, y]) - obs)
                if dist_to_obs < rad:
                    vehicle = patches.Circle((x, y), 1.5, color='black', alpha=0.8)
                    ax.text(x, y+3, "💥", ha='center', fontsize=14)
                else:
                    vehicle = patches.Circle((x, y), 1.5, color=colors[ax], alpha=0.9)
                ax.add_patch(vehicle)
                
                # Draw start
                ax.scatter(traj[0][0], traj[0][1], marker='o', s=100, 
                          color='gray', edgecolors='black', zorder=5)
            
            # Legend
            if ax == 0:
                ax.legend(loc='upper right', fontsize=8)
        
        # Add time indicator
        fig.text(0.5, 0.02, f"Time step: {frame}", ha='center', fontsize=12)
        
        return []
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=max_len, interval=100, blit=False)
    
    # Save
    output_path = "out/contingency_animation/contingency_planning.gif"
    writer = PillowWriter(fps=10)
    anim.save(output_path, writer=writer, dpi=100)
    print(f"✓ Saved: {output_path}")
    
    plt.close()
    
    return output_path


def create_comparison_animation():
    """Create side-by-side comparison at key moments."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Contingency Planning: Key Moments Comparison", fontsize=16, fontweight='bold')
    
    planner_types = ["baseline", "tree", "model"]
    titles = ["Baseline", "Tree-Based", "Model-Based"]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    key_frames = [10, 25, 45]  # Early, Middle, Late
    
    # Pre-compute trajectories
    trajectories = {}
    for ptype in planner_types:
        traj, obs, rad, goal = simulate_trajectory(ptype, "pedestrian")
        trajectories[ptype] = (traj, obs, rad, goal)
    
    for col, frame in enumerate(key_frames):
        for row, (ptype, title) in enumerate(zip(planner_types, titles)):
            ax = axes[row, col]
            
            traj, obs, rad, goal = trajectories[ptype]
            
            # Draw obstacle
            circle = patches.Circle(obs, rad, color='red', alpha=0.4)
            ax.add_patch(circle)
            
            # Draw goal
            ax.scatter(goal[0], goal[1], marker='*', s=200, color='gold', 
                      edgecolors='black', linewidths=1, zorder=5)
            
            # Draw trajectory
            n = min(frame, len(traj))
            if n > 0:
                traj_arr = np.array(traj[:n])
                ax.plot(traj_arr[:, 0], traj_arr[:, 1], 
                       color=colors[col], linewidth=2, alpha=0.8)
                
                # Ego
                x, y = traj[n-1]
                dist = np.linalg.norm(np.array([x, y]) - obs)
                c = 'black' if dist < rad else colors[col]
                vehicle = patches.Circle((x, y), 1.5, color=c, alpha=0.9)
                ax.add_patch(vehicle)
                
                # Speed annotation
                if n < len(traj):
                    ax.text(x+2, y+2, f"t={frame}", fontsize=9, alpha=0.7)
            
            ax.set_xlim(-5, 55)
            ax.set_ylim(-8, 8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            if row == 0:
                ax.set_title(f"t = {frame}", fontsize=12)
            if col == 0:
                ax.set_ylabel(title, fontsize=12, color=colors[col])
    
    plt.tight_layout()
    
    output_path = "out/contingency_animation/comparison.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    return output_path


if __name__ == "__main__":
    print("Creating contingency planning animation...")
    print("=" * 50)
    
    # Create animations
    gif_path = create_animation()
    png_path = create_comparison_animation()
    
    print("=" * 50)
    print("Done! Files created:")
    print(f"  - {gif_path}")
    print(f"  - {png_path}")
