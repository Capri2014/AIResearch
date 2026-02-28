"""
Simple Toy Simulation for Contingency Planning Testing

Quick sanity checks without CARLA.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time


class ToySimulator:
    """
    Simple 2D toy simulator for testing contingency planners.
    
    State: [x, y, v, heading]
    Action: [acceleration, steering]
    """
    
    def __init__(
        self,
        dt: float = 0.1,
        max_steps: int = 200,
    ):
        self.dt = dt
        self.max_steps = max_steps
        self.reset()
    
    def reset(
        self,
        initial_state: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None,
        obstacles: Optional[List[Dict]] = None,
    ):
        """Reset simulator."""
        if initial_state is None:
            self.state = np.array([0.0, 0.0, 5.0, 0.0])  # x, y, v, heading
        else:
            self.state = initial_state.copy()
        
        if goal is None:
            self.goal = np.array([100.0, 0.0, 0.0, 0.0])  # Target position
        else:
            self.goal = goal.copy()
        
        self.obstacles = obstacles or []
        self.step_count = 0
        self.collision = False
        self.success = False
        self.done = False
        
        return self.state.copy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step simulation.
        
        Args:
            action: [acceleration, steering]
            
        Returns:
            obs: New state
            reward: Reward
            done: Episode done
            info: Additional info
        """
        # Simple dynamics
        acc, steer = action
        
        # Update velocity
        v = self.state[2] + acc * self.dt
        v = max(0, min(v, 30))  # Clamp velocity
        
        # Update heading
        heading = self.state[3] + steer * self.dt
        
        # Update position
        x = self.state[0] + v * np.cos(heading) * self.dt
        y = self.state[1] + v * np.sin(heading) * self.dt
        
        self.state = np.array([x, y, v, heading])
        self.step_count += 1
        
        # Check collision
        for obs in self.obstacles:
            dist = np.linalg.norm(self.state[:2] - np.array([obs['x'], obs['y']]))
            if dist < obs.get('radius', 2.0):
                self.collision = True
                self.done = True
        
        # Check success
        dist_to_goal = np.linalg.norm(self.state[:2] - self.goal[:2])
        if dist_to_goal < 3.0:
            self.success = True
            self.done = True
        
        # Check timeout
        if self.step_count >= self.max_steps:
            self.done = True
        
        # Reward
        reward = -dist_to_goal * 0.01 - abs(acc) * 0.01
        if self.collision:
            reward = -100
        if self.success:
            reward = 100
        
        info = {
            'collision': self.collision,
            'success': self.success,
            'dist_to_goal': dist_to_goal,
            'step': self.step_count,
        }
        
        return self.state.copy(), reward, self.done, info
    
    def get_observation(self) -> Dict:
        """Get observation for planner."""
        return {
            'state': self.state.copy(),
            'goal': self.goal.copy(),
            'obstacles': self.obstacles.copy(),
            'step': self.step_count,
        }


def run_episode(planner, env: ToySimulator, max_steps: int = 200) -> Dict:
    """
    Run one episode with a planner.
    
    Args:
        planner: Planner with .plan(observation) and .execute(action) methods
        env: ToySimulator
        max_steps: Max steps
        
    Returns:
        Episode metrics
    """
    obs = env.reset()
    episode_data = {
        'states': [obs['state'].copy()],
        'actions': [],
        'rewards': [],
        'collisions': 0,
        'success': False,
        'planning_times': [],
    }
    
    for step in range(max_steps):
        # Plan
        start_time = time.time()
        
        # For tree-based planner
        if hasattr(planner, 'plan'):
            if hasattr(planner, 'initialize'):
                # First step - initialize
                planner.initialize(
                    initial_state=obs['state'],
                    goal_state=obs['goal'],
                    scenario_name='pedestrian_crossing',
                )
            
            action, info = planner.plan(obs)
        else:
            # Fallback: simple proportional control
            dist = np.linalg.norm(obs['state'][:2] - obs['goal'][:2])
            if dist > 0:
                desired_v = min(dist * 0.5, 10)
                acc = (desired_v - obs['state'][2]) * 0.5
                steer = -obs['state'][3] * 0.2
            else:
                acc, steer = 0, 0
            action = np.array([acc, steer])
        
        plan_time = time.time() - start_time
        episode_data['planning_times'].append(plan_time)
        
        # Execute
        next_obs, reward, done, info = env.step(action)
        
        episode_data['states'].append(next_obs['state'].copy())
        episode_data['actions'].append(action.copy())
        episode_data['rewards'].append(reward)
        
        if info.get('collision'):
            episode_data['collisions'] += 1
        if info.get('success'):
            episode_data['success'] = True
        
        obs = next_obs
        
        if done:
            break
    
    # Compute metrics
    episode_data['total_reward'] = sum(episode_data['rewards'])
    episode_data['avg_planning_time'] = np.mean(episode_data['planning_times']) * 1000  # ms
    episode_data['steps'] = len(episode_data['actions'])
    
    return episode_data


def run_comparison(
    scenarios: List[Dict],
    planners: Dict[str, any],
    n_runs: int = 5,
) -> Dict:
    """
    Run comparison between planners.
    
    Args:
        scenarios: List of scenario configs
        planners: Dict of name -> planner
        n_runs: Number of runs per scenario
        
    Returns:
        Results dict
    """
    results = {name: [] for name in planners.keys()}
    
    for scenario in scenarios:
        for run in range(n_runs):
            for name, planner in planners.items():
                # Reset planner if needed
                if hasattr(planner, 'reset'):
                    planner.reset()
                
                # Create env
                env = ToySimulator()
                env.reset(
                    initial_state=scenario.get('initial_state'),
                    goal=scenario.get('goal'),
                    obstacles=scenario.get('obstacles', []),
                )
                
                # Run episode
                episode = run_episode(planner, env)
                results[name].append(episode)
    
    # Compute statistics
    summary = {}
    for name, episodes in results.items():
        summary[name] = {
            'success_rate': np.mean([e['success'] for e in episodes]),
            'collision_rate': np.mean([e['collisions'] > 0 for e in episodes]),
            'avg_reward': np.mean([e['total_reward'] for e in episodes]),
            'avg_planning_time_ms': np.mean([e['avg_planning_time'] for e in episodes]),
            'avg_steps': np.mean([e['steps'] for e in episodes]),
        }
    
    return summary


if __name__ == "__main__":
    # Quick test
    from contingency_planning.planning.tree.planner import TreeBasedPlanner
    import yaml
    
    # Load config
    with open("contingency_planning/configs/default.yaml") as f:
        config = yaml.safe_load(f)
    
    # Create planner
    planner = TreeBasedPlanner(config)
    
    # Test scenario
    scenario = {
        'initial_state': np.array([0, 0, 10, 0]),
        'goal': np.array([100, 0, 0, 0]),
        'obstacles': [
            {'x': 50, 'y': 0, 'radius': 3},
        ],
    }
    
    # Run
    env = ToySimulator()
    env.reset(**scenario)
    
    print("Running episode with Tree-Based Planner...")
    episode = run_episode(planner, env)
    
    print(f"Success: {episode['success']}")
    print(f"Collisions: {episode['collisions']}")
    print(f"Steps: {episode['steps']}")
    print(f"Avg planning time: {episode['avg_planning_time_ms']:.2f} ms")
    print(f"Total reward: {episode['total_reward']:.2f}")
