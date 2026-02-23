"""
Multi-Scenario RL Environment with Domain Randomization.

Handles different weather/lighting conditions for robust autonomous driving:
- clear: Standard conditions
- cloudy: Reduced lighting, softer shadows
- night: Low visibility, artificial lighting
- rain: Reduced friction, visibility, wet surfaces
- fog: Reduced visibility, distance-dependent

Supports:
- Domain randomization for sim-to-real transfer
- Curriculum learning (easy → hard scenarios)
- Scenario-specific reward shaping
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from enum import Enum
import random


class ScenarioType(Enum):
    """Weather/lighting conditions."""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    NIGHT = "night"
    RAIN = "rain"
    FOG = "fog"


# Scenario difficulty (for curriculum learning)
SCENARIO_DIFFICULTY = {
    ScenarioType.CLEAR: 1.0,
    ScenarioType.CLOUDY: 1.2,
    ScenarioType.NIGHT: 1.5,
    ScenarioType.RAIN: 1.8,
    ScenarioType.FOG: 2.0,
}

# Scenario-specific parameters
SCENARIO_PARAMS = {
    ScenarioType.CLEAR: {
        "visibility": 100.0,
        "friction": 1.0,
        "noise_std": 0.0,
        "light_factor": 1.0,
    },
    ScenarioType.CLOUDY: {
        "visibility": 80.0,
        "friction": 0.95,
        "noise_std": 0.05,
        "light_factor": 0.8,
    },
    ScenarioType.NIGHT: {
        "visibility": 50.0,
        "friction": 0.9,
        "noise_std": 0.15,
        "light_factor": 0.3,
    },
    ScenarioType.RAIN: {
        "visibility": 40.0,
        "friction": 0.7,
        "noise_std": 0.2,
        "light_factor": 0.6,
    },
    ScenarioType.FOG: {
        "visibility": 30.0,
        "friction": 0.85,
        "noise_std": 0.25,
        "light_factor": 0.5,
    },
}


class MultiScenarioWaypointEnv:
    """
    Multi-scenario environment with domain randomization.
    
    Extends WaypointEnv with:
    - Multiple weather/lighting conditions
    - Domain randomization for robust policies
    - Curriculum learning support
    """
    
    def __init__(
        self,
        horizon: int = 20,
        dt: float = 0.1,
        max_speed: float = 2.0,
        goal_threshold: float = 0.5,
        base_noise_std: float = 0.0,
        scenario: Optional[ScenarioType] = None,
        enable_domain_randomization: bool = True,
        curriculum_level: float = 1.0,  # 0.0 = easy only, 1.0 = all scenarios
        randomize_every_episode: bool = True,
    ):
        self.horizon = horizon
        self.dt = dt
        self.max_speed = max_speed
        self.goal_threshold = goal_threshold
        self.base_noise_std = base_noise_std
        self.enable_domain_randomization = enable_domain_randomization
        self.curriculum_level = curriculum_level
        self.randomize_every_episode = randomize_every_episode
        
        self.state_dim = 6 + len(ScenarioType)  # state + scenario encoding
        self.action_dim = 2
        
        # Current scenario
        self.scenario = scenario or ScenarioType.CLEAR
        self.scenario_params = SCENARIO_PARAMS[self.scenario].copy()
        
        # Environment state
        self.state = None
        self.goal = None
        self.step_count = 0
        self.max_steps = 100
        
        # Available scenarios based on curriculum
        self._update_available_scenarios()
        
    def _update_available_scenarios(self):
        """Update available scenarios based on curriculum level."""
        # Sort scenarios by difficulty
        sorted_scenarios = sorted(
            SCENARIO_DIFFICULTY.items(), 
            key=lambda x: x[1]
        )
        
        # Select scenarios up to curriculum level
        max_difficulty = self.curriculum_level * 2.0  # Max is 2.0
        self.available_scenarios = [
            s for s, diff in sorted_scenarios 
            if diff <= max_difficulty
        ]
        
        if not self.available_scenarios:
            self.available_scenarios = [ScenarioType.CLEAR]
    
    def _sample_scenario(self) -> ScenarioType:
        """Sample a scenario based on curriculum and randomization."""
        if not self.randomize_every_episode:
            return self.scenario
            
        if self.enable_domain_randomization and len(self.available_scenarios) > 1:
            # Sample from available scenarios (could weight by difficulty)
            return random.choice(self.available_scenarios)
        else:
            # Default to easiest scenario
            return ScenarioType.CLEAR
    
    def set_scenario(self, scenario: ScenarioType):
        """Manually set the scenario (for evaluation)."""
        self.scenario = scenario
        self.scenario_params = SCENARIO_PARAMS[scenario].copy()
    
    def set_curriculum_level(self, level: float):
        """Set curriculum learning level (0.0 = easy only, 1.0 = all)."""
        self.curriculum_level = np.clip(level, 0.0, 1.0)
        self._update_available_scenarios()
    
    def reset(self) -> np.ndarray:
        """Reset environment with scenario sampling."""
        # Sample scenario
        self.scenario = self._sample_scenario()
        self.scenario_params = SCENARIO_PARAMS[self.scenario].copy()
        
        # Apply domain randomization to parameters
        if self.enable_domain_randomization:
            self._randomize_params()
        
        # Random start position
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        
        # Random initial velocity
        vx = np.random.uniform(-0.5, 0.5)
        vy = np.random.uniform(-0.5, 0.5)
        
        # Random goal (away from start)
        goal_x = np.random.uniform(-5, 5)
        goal_y = np.random.uniform(-5, 5)
        while np.linalg.norm([goal_x - x, goal_y - y]) < 3:
            goal_x = np.random.uniform(-5, 5)
            goal_y = np.random.uniform(-5, 5)
        
        # Scale threshold by scenario difficulty
        effective_threshold = self.goal_threshold * SCENARIO_DIFFICULTY[self.scenario]
        
        self.state = np.array([x, y, vx, vy, goal_x, goal_y], dtype=np.float32)
        self.goal = np.array([goal_x, goal_y])
        self.step_count = 0
        self.effective_threshold = effective_threshold
        
        return self._get_obs()
    
    def _randomize_params(self):
        """Apply domain randomization to scenario parameters."""
        params = self.scenario_params
        
        # Randomize visibility (±20%)
        params["visibility"] *= np.random.uniform(0.8, 1.2)
        
        # Randomize friction (±10%)
        params["friction"] *= np.random.uniform(0.9, 1.1)
        
        # Randomize noise (±50% of base)
        if self.base_noise_std > 0:
            params["noise_std"] = self.base_noise_std * np.random.uniform(0.5, 1.5)
        
        # Randomize light factor (±20%)
        params["light_factor"] *= np.random.uniform(0.8, 1.2)
    
    def _get_obs(self) -> np.ndarray:
        """Get observation with scenario encoding."""
        # Base observation
        base_obs = self.state.copy()
        
        # Scenario one-hot encoding (use enum index)
        scenario_encoding = np.zeros(len(ScenarioType), dtype=np.float32)
        scenario_idx = list(ScenarioType).index(self.scenario)
        scenario_encoding[scenario_idx] = 1.0
        
        # Combine
        return np.concatenate([base_obs, scenario_encoding])
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action with scenario-specific reward shaping."""
        x, y, vx, vy, goal_x, goal_y = self.state
        self.step_count += 1
        
        params = self.scenario_params
        difficulty = SCENARIO_DIFFICULTY[self.scenario]
        
        # Simulate dynamics with scenario-specific effects
        total_reward = 0.0
        for i in range(self.horizon):
            target_x = x + waypoints[i, 0]
            target_y = y + waypoints[i, 1]
            
            dx = target_x - x
            dy = target_y - y
            
            # Apply friction effect
            friction = params["friction"]
            vx = np.clip(vx + self.dt * dx / self.dt * friction, -self.max_speed, self.max_speed)
            vy = np.clip(vy + self.dt * dy / self.dt * friction, -self.max_speed, self.max_speed)
            
            # Updateself.max_speed, position
            x = x + self.dt * vx
            y = y + self.dt * vy
            
            # Apply observation noise
            noise_std = params.get("noise_std", self.base_noise_std)
            if noise_std > 0:
                x += np.random.normal(0, noise_std)
                y += np.random.normal(0, noise_std)
            
            # Distance to goal
            dist = np.linalg.norm([x - goal_x, y - goal_y])
            
            # Base reward: negative distance
            reward_i = -dist * self.dt
            
            # Scenario-specific bonuses/penalties
            # Night: penalty for fast movement (safety)
            if self.scenario == ScenarioType.NIGHT:
                speed = np.linalg.norm([vx, vy])
                reward_i -= 0.1 * speed
            
            # Rain: penalty for high speed (slip risk)
            elif self.scenario == ScenarioType.RAIN:
                speed = np.linalg.norm([vx, vy])
                reward_i -= 0.15 * speed
            
            # Fog: penalty for distance traveled (prefer shorter paths)
            elif self.scenario == ScenarioType.FOG:
                reward_i -= 0.05 * self.dt
            
            # Goal reached bonus (scaled by difficulty)
            if dist < self.effective_threshold:
                reward_i += 10.0 * difficulty
            
            total_reward += reward_i
        
        # Update state
        self.state = np.array([x, y, vx, vy, goal_x, goal_y], dtype=np.float32)
        
        # Check termination
        dist = np.linalg.norm([x - goal_x, y - goal_y])
        done = dist < self.effective_threshold or self.step_count >= self.max_steps
        
        info = {
            "distance": dist,
            "steps": self.step_count,
            "goal_reached": dist < self.effective_threshold,
            "scenario": self.scenario.value,
            "difficulty": difficulty,
            "visibility": params["visibility"],
            "friction": params["friction"],
        }
        
        return self._get_obs(), total_reward, done, info
    
    def get_sft_waypoints(self, num_waypoints: int = None) -> np.ndarray:
        """Get SFT baseline waypoints."""
        if num_waypoints is None:
            num_waypoints = self.horizon
            
        x, y, _, _, goal_x, goal_y = self.state
        
        waypoints = []
        for i in range(num_waypoints):
            t = (i + 1) / num_waypoints
            wp_x = (goal_x - x) * t
            wp_y = (goal_y - y) * t
            waypoints.append([wp_x, wp_y])
        
        return np.array(waypoints, dtype=np.float32)
    
    def get_scenario_info(self) -> Dict[str, Any]:
        """Get current scenario information."""
        return {
            "scenario": self.scenario.value,
            "difficulty": SCENARIO_DIFFICULTY[self.scenario],
            **self.scenario_params,
        }


def make_multi_scenario_env(
    horizon: int = 20,
    scenario: str = "clear",
    enable_domain_randomization: bool = True,
    curriculum_level: float = 1.0,
    **kwargs
) -> MultiScenarioWaypointEnv:
    """Factory function for multi-scenario environment."""
    scenario_type = ScenarioType(scenario) if scenario else None
    return MultiScenarioWaypointEnv(
        horizon=horizon,
        scenario=scenario_type,
        enable_domain_randomization=enable_domain_randomization,
        curriculum_level=curriculum_level,
        **kwargs
    )


class CurriculumScheduler:
    """
    Curriculum learning scheduler.
    
    Gradually increases scenario difficulty over training.
    """
    
    def __init__(
        self,
        initial_level: float = 0.2,
        final_level: float = 1.0,
        total_steps: int = 100000,
        warmup_steps: int = 5000,
    ):
        self.initial_level = initial_level
        self.final_level = final_level
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
    
    def get_level(self, step: int) -> float:
        """Get curriculum level for given training step."""
        if step < self.warmup_steps:
            return self.initial_level
        
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = np.clip(progress, 0.0, 1.0)
        
        # Cosine annealing
        level = self.initial_level + (self.final_level - self.initial_level) * (
            0.5 * (1 + np.cos(np.pi * progress))
        )
        
        return level
