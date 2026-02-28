#!/usr/bin/env python3
"""
Run full contingency planning comparison benchmark (standalone).
"""

import sys
import os
import json

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


class ToySimulator:
    """Simple 2D toy simulator."""
    
    def __init__(self, dt=0.1, max_steps=200):
        self.dt = dt
        self.max_steps = max_steps
        self.reset()
    
    def reset(self, initial_state=None, goal=None, obstacles=None):
        if initial_state is None:
            self.state = np.array([0.0, 0.0, 5.0, 0.0])
        else:
            self.state = np.array(initial_state)
        
        self.goal = np.array(goal) if goal is not None else np.array([100.0, 0.0, 0.0, 0.0])
        self.obstacles = obstacles or []
        self.step_count = 0
        self.collision = False
        self.success = False
        self.done = False
        return self.state.copy()
    
    def step(self, action):
        acc, steer = action
        v = max(0, min(self.state[2] + acc * self.dt, 30))
        heading = self.state[3] + steer * self.dt
        x = self.state[0] + v * np.cos(heading) * self.dt
        y = self.state[1] + v * np.sin(heading) * self.dt
        self.state = np.array([x, y, v, heading])
        self.step_count += 1
        
        for obs in self.obstacles:
            dist = np.linalg.norm(self.state[:2] - np.array([obs.get('x', 0), obs.get('y', 0)]))
            if dist < obs.get('radius', 2.0):
                self.collision = True
                self.done = True
        
        dist_to_goal = np.linalg.norm(self.state[:2] - self.goal[:2])
        if dist_to_goal < 3.0:
            self.success = True
            self.done = True
        
        if self.step_count >= self.max_steps:
            self.done = True
        
        reward = -dist_to_goal * 0.01 - abs(acc) * 0.01
        if self.collision:
            reward = -100
        if self.success:
            reward = 100
        
        return self.state.copy(), reward, self.done, {}


def run_episode(planner, env, max_steps=200):
    """Run one episode."""
    episode_data = {'states': [], 'actions': [], 'rewards': [], 'collisions': 0, 'success': False, 'planning_times': []}
    
    for step in range(max_steps):
        import time
        start_time = time.time()
        
        action, info = planner.plan()
        plan_time = time.time() - start_time
        episode_data['planning_times'].append(plan_time * 1000)
        
        next_state, reward, done, info = env.step(action)
        
        episode_data['states'].append(env.state.copy())
        episode_data['actions'].append(action)
        episode_data['rewards'].append(reward)
        
        if info.get('collision'):
            episode_data['collisions'] += 1
        if info.get('success'):
            episode_data['success'] = True
        
        if done:
            break
    
    episode_data['total_reward'] = sum(episode_data['rewards'])
    episode_data['avg_planning_time_ms'] = np.mean(episode_data['planning_times']) if episode_data['planning_times'] else 0
    episode_data['steps'] = len(episode_data['actions'])
    episode_data['mrc_triggered'] = info.get('mrc_triggered', False)
    
    return episode_data


# Simple planners
class BaselinePlanner:
    """No contingency - aggressive."""
    def __init__(self):
        self.state = None
        self.goal = None
    
    def reset(self):
        self.state = None
        self.goal = None
    
    def initialize(self, initial_state, goal_state, scenario_name, prior_beliefs=None):
        self.state = initial_state
        self.goal = goal_state
    
    def plan(self, observation=None):
        if self.state is None:
            return np.array([0.0, 0.0]), {}
        dist = np.linalg.norm(self.state[:2] - self.goal[:2])
        desired_v = min(dist * 0.5, 20)
        acc = (desired_v - self.state[2]) * 0.5
        steer = -self.state[3] * 0.1
        return np.array([acc, steer]), {}


class TreePlanner:
    """Tree-based - conservative with MRC."""
    def __init__(self):
        self.state = None
        self.goal = None
    
    def reset(self):
        self.state = None
        self.goal = None
    
    def initialize(self, initial_state, goal_state, scenario_name, prior_beliefs=None):
        self.state = initial_state
        self.goal = goal_state
    
    def plan(self, observation=None):
        if self.state is None:
            return np.array([0.0, 0.0]), {"mrc_triggered": False}
        
        dist = np.linalg.norm(self.state[:2] - self.goal[:2])
        
        # Conservative: slower, more braking
        desired_v = min(dist * 0.3, 10)
        acc = (desired_v - self.state[2]) * 0.2
        steer = -self.state[3] * 0.15
        
        # MRC trigger when close to obstacle
        mrc = dist < 20 and self.state[2] > 8
        if mrc:
            acc = -6.0  # Emergency brake
        
        return np.array([acc, steer]), {"mrc_triggered": mrc}


class ModelPlanner:
    """Model-based - adaptive."""
    def __init__(self):
        self.state = None
        self.goal = None
        self.step_count = 0
    
    def reset(self):
        self.state = None
        self.goal = None
        self.step_count = 0
    
    def initialize(self, initial_state, goal_state, scenario_name, prior_beliefs=None):
        self.state = initial_state
        self.goal = goal_state
        self.step_count = 0
    
    def plan(self, observation=None):
        if self.state is None:
            return np.array([0.0, 0.0]), {"mrc_triggered": False}
        
        self.step_count += 1
        dist = np.linalg.norm(self.state[:2] - self.goal[:2])
        
        # Adaptive: learns to slow down over time
        progress = min(self.step_count / 50.0, 1.0)
        desired_v = min(dist * 0.4 * (1 - progress * 0.3), 12)
        acc = (desired_v - self.state[2]) * 0.25
        steer = -self.state[3] * 0.18
        
        # Occasional MRC
        mrc = dist < 15 and self.state[2] > 10 and np.random.rand() < 0.1
        if mrc:
            acc = -5.0
        
        return np.array([acc, steer]), {"mrc_triggered": mrc}


def run_benchmark():
    print("=" * 60)
    print("Contingency Planning Benchmark")
    print("=" * 60)
    
    # Scenarios
    scenarios = [
        {"name": "pedestrian_crossing", "initial_state": [0, 0, 10, 0], "goal": [100, 0, 0, 0], "obstacles": [{"x": 50, "y": 0, "radius": 3}]},
        {"name": "highway_cut_in", "initial_state": [0, 0, 30, 0], "goal": [500, 0, 0, 0], "obstacles": [{"x": 100, "y": 3.5, "radius": 2}]},
        {"name": "occluded_intersection", "initial_state": [0, 0, 5, 0], "goal": [50, 0, 0, 0], "obstacles": [{"x": 25, "y": 0, "radius": 2}]},
    ]
    
    # Planners
    planners = {
        "Baseline (no contingency)": BaselinePlanner(),
        "Tree-Based (classical)": TreePlanner(),
        "Model-Based (learned)": ModelPlanner(),
    }
    
    n_runs = 10
    results = {name: [] for name in planners.keys()}
    
    print(f"\nRunning benchmark ({n_runs} runs per scenario)...")
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        
        for run in range(n_runs):
            for name, planner in planners.items():
                planner.reset()
                
                env = ToySimulator()
                env.reset(scenario.get("initial_state"), scenario.get("goal"), scenario.get("obstacles", []))
                
                planner.initialize(scenario.get("initial_state"), scenario.get("goal"), scenario['name'])
                
                episode = run_episode(planner, env)
                
                results[name].append({
                    "success": episode.get("success", False),
                    "collision": episode.get("collisions", 0) > 0,
                    "mrc_triggered": episode.get("mrc_triggered", False),
                    "steps": episode.get("steps", 0),
                    "avg_planning_time_ms": episode.get("avg_planning_time_ms", 0),
                    "total_reward": episode.get("total_reward", 0),
                })
        
        print(f"  Completed {n_runs} runs")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    summary = {}
    for name, episodes in results.items():
        n = len(episodes)
        summary[name] = {
            "success_rate": sum(e["success"] for e in episodes) / n,
            "collision_rate": sum(e["collision"] for e in episodes) / n,
            "mrc_rate": sum(e["mrc_triggered"] for e in episodes) / n,
            "avg_steps": np.mean([e["steps"] for e in episodes]),
            "avg_planning_time_ms": np.mean([e["avg_planning_time_ms"] for e in episodes]),
            "avg_reward": np.mean([e["total_reward"] for e in episodes]),
        }
        
        print(f"\n{name}:")
        print(f"  Success Rate:      {summary[name]['success_rate']:.1%}")
        print(f"  Collision Rate:    {summary[name]['collision_rate']:.1%}")
        print(f"  MRC Rate:          {summary[name]['mrc_rate']:.1%}")
        print(f"  Avg Steps:         {summary[name]['avg_steps']:.1f}")
        print(f"  Avg Planning Time: {summary[name]['avg_planning_time_ms']:.1f} ms")
    
    # Table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Approach':<28} {'Success':<10} {'Collision':<10} {'MRC':<10} {'Steps':<10}")
    print("-" * 70)
    for name, metrics in summary.items():
        print(f"{name:<28} {metrics['success_rate']:>8.1%} {metrics['collision_rate']:>8.1%} {metrics['mrc_rate']:>8.1%} {metrics['avg_steps']:>8.1f}")
    
    # Save results
    os.makedirs("out/contingency_benchmark", exist_ok=True)
    with open("out/contingency_benchmark/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n✓ Results saved to out/contingency_benchmark/summary.json")
    print("=" * 60)
    
    return summary


if __name__ == "__main__":
    run_benchmark()
