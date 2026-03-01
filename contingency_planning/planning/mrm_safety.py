"""
MRM (Minimal Risk Maneuver) System and Safety Supervisor

MRM: Pre-computed or generated fallback trajectories when no safe plan exists
Safety Supervisor: Independent verification of final output

These are production-critical safety components.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class MRMType(Enum):
    """Types of Minimal Risk Maneuvers"""
    BRAKE_TO_STOP = "brake_to_stop"
    STOP_IN_LANE = "stop_in_lane"
    STOP_ON_SHOULDER = "stop_on_shoulder"
    EMERGENCY_STOP = "emergency_stop"
    CREEP_FORWARD = "creep_forward"


@dataclass
class MRMConfig:
    """Configuration for MRM system"""
    # Stop positions
    stop_position: float = 0.0  # s offset from current
    shoulder_distance: float = 3.5  # meters to shoulder
    
    # Timing
    brake_deceleration: float = -3.0  # m/s^2
    emergency_deceleration: float = -6.0  # m/s^2
    
    # Creep
    creep_speed: float = 2.0  # m/s
    creep_distance: float = 5.0  # meters to creep
    
    # Confidence threshold
    min_safe_probability: float = 0.8


@dataclass
class SafetyConfig:
    """Configuration for Safety Supervisor"""
    # Hard margins (cannot be violated)
    min_boundary_margin: float = 0.3  # meters
    min_keepout_margin: float = 0.5  # meters
    min_collision_distance: float = 2.0  # meters
    
    # Buffer scaling
    buffer_scale_factor: float = 1.2  # Scale conservative in supervisor
    
    # Override options
    allow_override: bool = True


class MRMSystem:
    """
    Minimal Risk Maneuver (MRM) System
    
    Generates safe fallback trajectories when no feasible candidate exists.
    """
    
    def __init__(self, config: Optional[MRMConfig] = None):
        self.config = config or MRMConfig()
        
    def generate_mrm(self,
                   mrm_type: MRMType,
                   current_state: Dict) -> List[Tuple[float, float]]:
        """
        Generate MRM trajectory.
        
        Args:
            mrm_type: Type of MRM to generate
            current_state: {'s', 'l', 'v', 'heading'}
            
        Returns:
            List of (s, l) waypoints
        """
        s = current_state.get('s', 0)
        l = current_state.get('l', 0)
        v = current_state.get('v', 10)
        
        if mrm_type == MRMType.BRAKE_TO_STOP:
            return self._brake_to_stop(s, l, v)
        elif mrm_type == MRMType.STOP_IN_LANE:
            return self._stop_in_lane(s, l, v)
        elif mrm_type == MRMType.STOP_ON_SHOULDER:
            return self._stop_on_shoulder(s, l, v)
        elif mrm_type == MRMType.EMERGENCY_STOP:
            return self._emergency_stop(s, l, v)
        elif mrm_type == MRMType.CREEP_FORWARD:
            return self._creep_forward(s, l)
        else:
            return self._brake_to_stop(s, l, v)
    
    def select_mrm(self,
                   current_state: Dict,
                   available_space: Dict) -> MRMType:
        """
        Select appropriate MRM based on situation.
        
        Args:
            current_state: Current ego state
            available_space: {'has_shoulder': bool, 'free_distance': float}
            
        Returns:
            Selected MRM type
        """
        # Check if shoulder is available
        if available_space.get('has_shoulder', False):
            if available_space.get('free_distance', 0) > 10:
                return MRMType.STOP_ON_SHOULDER
        
        # Check if there's space to stop in lane
        if available_space.get('free_distance', 0) > 20:
            return MRMType.STOP_IN_LANE
        
        # Otherwise brake in place
        return MRMType.BRAKE_TO_STOP
    
    def _brake_to_stop(self, s: float, l: float, v: float) -> List[Tuple[float, float]]:
        """Generate brake-to-stop trajectory in current lane"""
        trajectory = []
        dt = 0.1
        current_v = v
        
        # Time to stop at deceleration
        t_stop = abs(v / self.config.brake_deceleration) if v > 0 else 0
        steps = int(t_stop / dt) + 20  # Extra steps for safety
        
        for i in range(min(steps, 100)):
            t = i * dt
            # Kinematic: s = s0 + v*t + 0.5*a*t^2
            current_v = max(v + self.config.brake_deceleration * t, 0)
            ds = current_v * dt
            s += ds
            trajectory.append((s, l))
            
            if current_v < 0.1:
                break
        
        return trajectory
    
    def _stop_in_lane(self, s: float, l: float, v: float) -> List[Tuple[float, float]]:
        """Generate smooth stop in lane"""
        trajectory = []
        dt = 0.1
        
        # Target: stop after ~50m
        target_s = s + 50
        
        steps = 100
        for i in range(steps):
            t = i * dt
            # Smooth deceleration
            progress = t / (steps * dt)
            decel = self.config.brake_deceleration * (1 - progress * 0.5)
            
            current_v = max(v + decel * t, 0)
            s += current_v * dt
            trajectory.append((s, l))
            
            if current_v < 0.1:
                break
        
        return trajectory
    
    def _stop_on_shoulder(self, s: float, l: float, v: float) -> List[Tuple[float, float]]:
        """Generate trajectory to stop on shoulder"""
        trajectory = []
        dt = 0.1
        
        # First: move to shoulder
        target_l = l - self.config.shoulder_distance
        
        # Move laterally
        for i in range(30):
            t = i * dt
            progress = t / (30 * dt)
            # Smooth lateral
            current_l = l + (target_l - l) * progress
            s += v * 0.8 * dt  # Slow down while moving
            trajectory.append((s, current_l))
        
        # Then brake
        brake_traj = self._brake_to_stop(s, target_l, v * 0.8)
        trajectory.extend(brake_traj)
        
        return trajectory
    
    def _emergency_stop(self, s: float, l: float, v: float) -> List[Tuple[float, float]]:
        """Generate emergency stop with max deceleration"""
        trajectory = []
        dt = 0.1
        current_v = v
        
        steps = int(abs(v / self.config.emergency_deceleration) / dt) + 10
        
        for i in range(min(steps, 50)):
            t = i * dt
            current_v = max(v + self.config.emergency_deceleration * t, 0)
            s += current_v * dt
            trajectory.append((s, l))
            
            if current_v < 0.1:
                break
        
        return trajectory
    
    def _creep_forward(self, s: float, l: float) -> List[Tuple[float, float]]:
        """Generate creep forward trajectory"""
        trajectory = []
        dt = 0.1
        
        for i in range(50):
            t = i * dt
            s += self.config.creep_speed * dt
            trajectory.append((s, l))
        
        return trajectory


class SafetySupervisor:
    """
    Independent safety verification of planner output.
    
    Runs in parallel with planner to verify:
    - Hard margin compliance
    - Collision avoidance
    - Kinematic feasibility
    """
    
    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self.mrm_system = MRMSystem()
    
    def verify(self,
              trajectory: List[Tuple[float, float]],
              current_state: Dict,
              obstacles: List[Dict],
              boundaries: Dict) -> Tuple[bool, str, Optional[List[Tuple[float, float]]]]:
        """
        Verify trajectory safety.
        
        Args:
            trajectory: Planned trajectory
            current_state: Current ego state
            obstacles: List of obstacle states
            boundaries: {'left': float, 'right': float}
            
        Returns:
            (is_safe, reason, fallback_trajectory)
        """
        # 1. Check boundary margins
        boundary_ok, reason = self._check_boundaries(trajectory, boundaries)
        if not boundary_ok:
            return False, reason, None
        
        # 2. Check collisions
        collision, reason = self._check_collisions(trajectory, obstacles)
        if collision:
            return False, reason, None
        
        # 3. Check kinematic feasibility
        kinem_ok, reason = self._check_kinematics(trajectory)
        if not kinem_ok:
            return False, reason, None
        
        return True, "safe", None
    
    def _check_boundaries(self,
                         trajectory: List[Tuple[float, float]],
                         boundaries: Dict) -> Tuple[bool, str]:
        """Check lateral boundary margins"""
        left = boundaries.get('left', 10)
        right = boundaries.get('right', -10)
        
        margin = self.config.min_boundary_margin * self.config.buffer_scale_factor
        
        for s, l in trajectory:
            if l > left - margin:
                return False, f"violates left boundary: l={l:.2f} > {left - margin:.2f}"
            if l < right + margin:
                return False, f"violates right boundary: l={l:.2f} < {right + margin:.2f}"
        
        return True, ""
    
    def _check_collisions(self,
                         trajectory: List[Tuple[float, float]],
                         obstacles: List[Dict]) -> Tuple[bool, str]:
        """Check collision with obstacles"""
        collision_dist = self.config.min_collision_distance * self.config.buffer_scale_factor
        
        for s, l in trajectory:
            for obs in obstacles:
                o_s = obs.get('s', 0)
                o_l = obs.get('l', 0)
                
                dist = np.sqrt((s - o_s)**2 + (l - o_l)**2)
                
                if dist < collision_dist:
                    return True, f"collision with obstacle at s={o_s:.1f}, l={o_l:.1f}"
        
        return False, ""
    
    def _check_kinematics(self, 
                         trajectory: List[Tuple[float, float]]) -> Tuple[bool, str]:
        """Check kinematic feasibility"""
        if len(trajectory) < 2:
            return True, ""
        
        max_accel = 5.0  # m/s^2
        max_jerk = 3.0  # m/s^3
        
        dt = 0.1
        prev_v = 0
        
        for i in range(1, len(trajectory)):
            s1, l1 = trajectory[i-1]
            s2, l2 = trajectory[i]
            
            # Velocity
            dx = s2 - s1
            dy = l2 - l1
            v = np.sqrt(dx**2 + dy**2) / dt
            
            # Acceleration
            a = (v - prev_v) / dt
            
            if abs(a) > max_accel:
                return False, f"acceleration too high: {abs(a):.2f} > {max_accel}"
            
            prev_v = v
        
        return True, ""
    
    def get_safe_fallback(self,
                         current_state: Dict,
                         reason: str) -> List[Tuple[float, float]]:
        """Generate safe fallback when verification fails"""
        # Determine available space
        available_space = {
            'has_shoulder': True,  # Would come from perception
            'free_distance': 30   # Would come from map/perception
        }
        
        # Select MRM
        mrm_type = self.mrm_system.select_mrm(current_state, available_space)
        
        # Generate
        return self.mrm_system.generate_mrm(mrm_type, current_state)


def create_mrm_system() -> MRMSystem:
    """Factory function"""
    config = MRMConfig(
        brake_deceleration=-3.0,
        emergency_deceleration=-6.0,
        creep_speed=2.0,
        min_safe_probability=0.8
    )
    return MRMSystem(config)


def create_safety_supervisor() -> SafetySupervisor:
    """Factory function"""
    config = SafetyConfig(
        min_boundary_margin=0.3,
        min_keepout_margin=0.5,
        min_collision_distance=2.0,
        buffer_scale_factor=1.2
    )
    return SafetySupervisor(config)


if __name__ == "__main__":
    # Test MRM system
    mrm = create_mrm_system()
    state = {'s': 0, 'l': 0, 'v': 15}
    
    mrm_traj = mrm.generate_mrm(MRMType.BRAKE_TO_STOP, state)
    print(f"MRM trajectory: {len(mrm_traj)} points")
    
    # Test safety supervisor
    supervisor = create_safety_supervisor()
    trajectory = [(i*0.5, 0) for i in range(100)]
    obstacles = [{'s': 40, 'l': 0}]
    boundaries = {'left': 3.5, 'right': -3.5}
    
    is_safe, reason, fallback = supervisor.verify(trajectory, state, obstacles, boundaries)
    print(f"Safe: {is_safe}, Reason: {reason}")
    
    # Test with collision
    obstacles = [{'s': 5, 'l': 0}]
    is_safe, reason, fallback = supervisor.verify(trajectory, state, obstacles, boundaries)
    print(f"With collision: Safe: {is_safe}, Reason: {reason}")
