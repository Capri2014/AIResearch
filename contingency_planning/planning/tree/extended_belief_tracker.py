"""
Extended Belief Tracker with EKF and TTC

Extended Kalman Filter for continuous state estimation and
Time-to-Collision (TTC) computation for real-time safety.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class ObstacleState:
    """State of a detected obstacle."""
    position: np.ndarray  # [x, y]
    velocity: np.ndarray  # [vx, vy]
    acceleration: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None
    timestamp: float = 0.0


class ExtendedBeliefTracker:
    """
    Extended Kalman Filter for continuous belief tracking.
    
    Tracks obstacle states with uncertainty and computes
    Time-to-Collision for safety-critical decisions.
    """
    
    def __init__(
        self,
        state_dim: int = 4,  # [x, y, vx, vy]
        process_noise: float = 0.1,
        observation_noise: float = 0.5,
    ):
        self.state_dim = state_dim
        
        # State: [x, y, vx, vy]
        self.state = np.zeros(state_dim)
        
        # State covariance P
        self.P = np.eye(state_dim) * 1.0
        
        # Process noise Q
        self.Q = np.eye(state_dim) * process_noise
        
        # Observation noise R
        self.R = np.eye(2) * observation_noise  # Only observe position [x, y]
        
        # Last update time
        self.last_time = 0.0
        
        # History for TTC
        self.position_history: List[Tuple[float, np.ndarray]] = []
        self.max_history = 50
        
        # Hypothesis beliefs (for branching)
        self.hypothesis_belief: Dict[str, float] = {}
    
    def initialize(self, position: np.ndarray, velocity: np.ndarray = None, timestamp: float = 0.0):
        """Initialize state with first observation."""
        if velocity is None:
            velocity = np.zeros(2)
        
        self.state = np.array([position[0], position[1], velocity[0], velocity[1]])
        self.P = np.eye(self.state_dim) * 1.0
        self.last_time = timestamp
        self.position_history = [(timestamp, position.copy())]
    
    def predict(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step: propagate state forward.
        
        Args:
            dt: Time step
            
        Returns:
            Predicted state and covariance
        """
        # State transition matrix (constant velocity model)
        F = np.eye(self.state_dim)
        F[0, 2] = dt  # x += vx * dt
        F[1, 3] = dt  # y += vy * dt
        
        # Predict state
        self.state = F @ self.state
        
        # Predict covariance: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q * dt
        
        return self.state.copy(), self.P.copy()
    
    def update(self, observation: np.ndarray, timestamp: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step: incorporate new observation.
        
        Args:
            observation: Observed position [x, y]
            timestamp: Observation timestamp
            
        Returns:
            Updated state and covariance
        """
        # Time difference
        dt = timestamp - self.last_time
        if dt > 0:
            self.predict(dt)
        
        # Observation matrix (we observe position, not velocity)
        H = np.zeros((2, self.state_dim))
        H[0, 0] = 1  # observe x
        H[1, 1] = 1  # observe y
        
        # Innovation: y - H @ x
        innovation = observation - H @ self.state
        
        # Innovation covariance: S = H * P * H^T + R
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain: K = P * H^T * S^-1
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ innovation
        
        # Update covariance: P = (I - K*H) * P
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P
        
        self.last_time = timestamp
        
        # Store position history for TTC
        self.position_history.append((timestamp, observation.copy()))
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
        
        return self.state.copy(), self.P.copy()
    
    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.state[:2].copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.state[2:].copy()
    
    def get_uncertainty(self) -> float:
        """Get position uncertainty (trace of position covariance)."""
        return np.sqrt(self.P[0, 0] + self.P[1, 1])
    
    def compute_ttc(
        self,
        ego_position: np.ndarray,
        ego_velocity: np.ndarray,
        safety_margin: float = 1.5,
    ) -> float:
        """
        Compute Time-to-Collision.
        
        Args:
            ego_position: Ego vehicle position [x, y]
            ego_velocity: Ego vehicle velocity [vx, vy]
            safety_margin: Additional safety margin (meters)
            
        Returns:
            TTC in seconds, or float('inf') if no collision course
        """
        # Relative position
        rel_pos = self.get_position() - ego_position
        
        # Relative velocity
        rel_vel = self.get_velocity() - ego_velocity
        
        # Distance
        distance = np.linalg.norm(rel_pos)
        
        # Relative speed toward obstacle
        speed_toward = np.dot(rel_vel, rel_pos) / (distance + 1e-6)
        
        # If moving away or stationary, no collision
        if speed_toward >= 0:
            return float('inf')
        
        # Time to reach obstacle
        effective_distance = distance - safety_margin
        if effective_distance <= 0:
            return 0.0  # Already colliding
        
        ttc = effective_distance / (-speed_toward)
        
        # Clamp to reasonable range
        return max(0.0, min(ttc, 10.0))
    
    def compute_ttc_2d(
        self,
        ego_position: np.ndarray,
        ego_velocity: np.ndarray,
        ego_acceleration: np.ndarray = None,
        safety_time: float = 2.0,
    ) -> Dict[str, float]:
        """
        Compute 2D TTC with longitudinal and lateral components.
        
        Args:
            ego_position: Ego position [x, y]
            ego_velocity: Ego velocity [vx, vy]
            ego_acceleration: Ego acceleration [ax, ay] (optional)
            safety_time: Time buffer for lateral
            
        Returns:
            Dictionary with 'longitudinal_ttc', 'lateral_ttc', 'combined_ttc'
        """
        rel_pos = self.get_position() - ego_position
        rel_vel = self.get_velocity() - ego_velocity
        
        # Longitudinal (along obstacle velocity direction)
        obs_dir = self.get_velocity()
        obs_dir_norm = np.linalg.norm(obs_dir)
        
        if obs_dir_norm > 0.1:
            obs_unit = obs_dir / obs_dir_norm
            long_pos = np.dot(rel_pos, obs_unit)
            long_vel = np.dot(rel_vel, obs_unit)
            
            if long_vel < 0:
                long_ttc = max(0, long_pos / -long_vel)
            else:
                long_ttc = float('inf')
        else:
            long_ttc = float('inf')
        
        # Lateral (perpendicular to obstacle direction)
        if obs_dir_norm > 0.1:
            lat_dir = np.array([-obs_unit[1], obs_unit[0]])
            lat_pos = np.dot(rel_pos, lat_dir)
            lat_vel = np.dot(rel_vel, lat_dir)
            
            # Time to cross lateral clearance
            lateral_clearance = 2.0  # vehicle width + margin
            if abs(lat_vel) > 0.01:
                lat_ttc = max(0, (abs(lat_pos) - lateral_clearance) / abs(lat_vel))
            else:
                lat_ttc = float('inf') if abs(lat_pos) > lateral_clearance else 0
        else:
            lat_ttc = float('inf')
        
        # Combined (minimum)
        combined_ttc = min(long_ttc, lat_ttc)
        
        return {
            'longitudinal_ttc': long_ttc,
            'lateral_ttc': lat_ttc,
            'combined_ttc': combined_ttc,
        }
    
    def predict_collision_point(
        self,
        ego_position: np.ndarray,
        ego_velocity: np.ndarray,
        time_horizon: float = 3.0,
        dt: float = 0.1,
    ) -> Optional[Tuple[float, np.ndarray]]:
        """
        Predict where and when collision occurs (if at all).
        
        Args:
            ego_position: Ego position
            ego_velocity: Ego velocity
            time_horizon: Lookahead time
            dt: Time step
            
        Returns:
            (time, position) of predicted collision, or None
        """
        # Sample future positions
        for t in np.arange(0, time_horizon, dt):
            # Predict obstacle position
            obs_pos = self.state[:2] + self.state[2:] * t
            ego_pos = ego_position + ego_velocity * t
            
            # Check distance
            dist = np.linalg.norm(obs_pos - ego_pos)
            if dist < 1.5:  # collision threshold
                return t, (obs_pos + ego_pos) / 2
        
        return None
    
    def get_hypothesis_belief(self) -> Dict[str, float]:
        """Get belief over discrete hypotheses."""
        return self.hypothesis_belief.copy()
    
    def set_hypothesis_belief(self, hypothesis: str, belief: float):
        """Set belief for a specific hypothesis."""
        self.hypothesis_belief[hypothesis] = belief
    
    def get_risk_score(self, ego_position: np.ndarray, ego_velocity: np.ndarray) -> float:
        """
        Compute risk score [0, 1] based on TTC and uncertainty.
        
        Args:
            ego_position: Ego position
            ego_velocity: Ego velocity
            
        Returns:
            Risk score (0 = safe, 1 = collision imminent)
        """
        ttc = self.compute_ttc(ego_position, ego_velocity)
        uncertainty = self.get_uncertainty()
        
        if ttc == float('inf'):
            return 0.0
        
        # Risk from TTC
        if ttc < 0.5:
            ttc_risk = 1.0
        elif ttc < 2.0:
            ttc_risk = 1.0 - (ttc - 0.5) / 1.5
        else:
            ttc_risk = 0.0
        
        # Risk from uncertainty
        uncertainty_risk = min(1.0, uncertainty / 5.0)
        
        # Combined risk
        return min(1.0, ttc_risk * 0.8 + uncertainty_risk * 0.2)


class MultiObstacleTracker:
    """
    Tracks multiple obstacles simultaneously.
    """
    
    def __init__(self, process_noise: float = 0.1, observation_noise: float = 0.5):
        self.trackers: Dict[int, ExtendedBeliefTracker] = {}
        self.next_id = 0
        self.process_noise = process_noise
        self.observation_noise = observation_noise
    
    def add_obstacle(
        self,
        obstacle_id: int,
        position: np.ndarray,
        velocity: np.ndarray = None,
        timestamp: float = 0.0,
    ) -> int:
        """Add a new obstacle to track."""
        tracker = ExtendedBeliefTracker(
            process_noise=self.process_noise,
            observation_noise=self.observation_noise,
        )
        tracker.initialize(position, velocity, timestamp)
        
        if obstacle_id is None:
            obstacle_id = self.next_id
            self.next_id += 1
        
        self.trackers[obstacle_id] = tracker
        return obstacle_id
    
    def update(
        self,
        obstacle_id: int,
        observation: np.ndarray,
        timestamp: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update tracked obstacle."""
        if obstacle_id not in self.trackers:
            return self.add_obstacle(obstacle_id, observation, timestamp=timestamp)
        
        return self.trackers[obstacle_id].update(observation, timestamp)
    
    def predict_all(self, dt: float):
        """Predict all obstacle states."""
        for tracker in self.trackers.values():
            tracker.predict(dt)
    
    def get_closest_obstacle(self, ego_position: np.ndarray) -> Tuple[Optional[int], float]:
        """Get ID and distance of closest obstacle."""
        min_dist = float('inf')
        closest_id = None
        
        for oid, tracker in self.trackers.items():
            dist = np.linalg.norm(tracker.get_position() - ego_position)
            if dist < min_dist:
                min_dist = dist
                closest_id = oid
        
        return closest_id, min_dist
    
    def get_all_ttc(self, ego_position: np.ndarray, ego_velocity: np.ndarray) -> Dict[int, float]:
        """Get TTC for all tracked obstacles."""
        return {
            oid: tracker.compute_ttc(ego_position, ego_velocity)
            for oid, tracker in self.trackers.items()
        }
    
    def remove_obstacle(self, obstacle_id: int):
        """Remove obstacle from tracking."""
        if obstacle_id in self.trackers:
            del self.trackers[obstacle_id]


def create_tracker_for_scenario(scenario_name: str) -> ExtendedBeliefTracker:
    """
    Create a configured tracker for a scenario type.
    """
    tracker = ExtendedBeliefTracker(
        process_noise=0.1,
        observation_noise=0.3,
    )
    
    if scenario_name == "pedestrian_crossing":
        # Higher uncertainty for pedestrians
        tracker.Q[:2, :2] *= 2.0
    elif scenario_name == "highway_cut_in":
        # Lower lateral uncertainty
        tracker.Q[1, 1] *= 0.5
    
    return tracker
