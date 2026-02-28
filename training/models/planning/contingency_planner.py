"""
Contingency Planning Module for Autonomous Driving

Implements fallback behaviors when primary planning fails:
- Fallback waypoint prediction
- Behavior tree structure for fail-safe behaviors  
- Graceful degradation handling
- Failure mode detection and response

Based on survey: docs/surveys/2026-02-21-contingency-planning.md
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from enum import Enum


class FailureMode(Enum):
    """Detected failure modes that trigger contingency behaviors."""
    NONE = "none"
    COLLISION_IMMINENT = "collision_imminent"
    OFF_ROAD = "off_road"
    TIMEOUT = "timeout"
    DEGRADED_SENSOR = "degraded_sensor"
    UNCERTAINTY_HIGH = "uncertainty_high"


@dataclass
class ContingencyConfig:
    """Configuration for contingency planning behaviors."""
    # Fallback waypoint settings
    num_fallback_waypoints: int = 5
    fallback_horizon: float = 10.0  # meters
    
    # Behavior tree settings
    max_tree_depth: int = 4
    enable_graceful_degradation: bool = True
    
    # Failure detection thresholds
    collision_threshold: float = 2.0  # meters
    off_road_threshold: float = 1.5  # meters from road center
    uncertainty_threshold: float = 0.7
    
    # Degradation levels
    sensor_degradation_threshold: float = 0.5


class ContingencyNetwork(nn.Module):
    """
    Contingency Network for autonomous driving.
    
    Learns to predict fallback trajectories when primary plan becomes invalid.
    Based on "Contingency Networks" from the survey.
    """
    
    def __init__(
        self,
        state_dim: int = 256,
        waypoint_dim: int = 3,
        hidden_dim: int = 512,
        num_contingencies: int = 3,
    ):
        super().__init__()
        
        self.num_contingencies = num_contingencies
        self.state_dim = state_dim
        self.waypoint_dim = waypoint_dim
        
        # Encode current state and primary plan
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Primary plan encoder
        self.primary_plan_encoder = nn.Sequential(
            nn.Linear(10 * waypoint_dim, hidden_dim),  # 10 waypoints
            nn.ReLU(),
        )
        
        # Contingency prediction heads
        self.contingency_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 10 * waypoint_dim),  # 10 waypoints per contingency
            )
            for _ in range(num_contingencies)
        ])
        
        # Failure mode classifier
        self.failure_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(FailureMode)),
            nn.Softmax(dim=-1),
        )
        
        # Confidence predictor for contingency selection
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_contingencies),
        )
    
    def forward(
        self,
        state: torch.Tensor,
        primary_waypoints: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            state: [B, state_dim] Current state encoding
            primary_waypoints: [B, 10, waypoint_dim] Primary planned waypoints
            
        Returns:
            contingency_waypoints: List of [B, 10, waypoint_dim] fallback plans
            failure_probs: [B, len(FailureMode)] Failure mode probabilities
            confidence: [B, num_contingencies] Confidence for each contingency
        """
        # Encode inputs
        state_enc = self.state_encoder(state)
        primary_enc = self.primary_plan_encoder(
            primary_waypoints.view(primary_waypoints.size(0), -1)
        )
        combined = torch.cat([state_enc, primary_enc], dim=-1)
        
        # Predict contingencies
        contingency_waypoints = [
            head(combined).view(-1, 10, self.waypoint_dim)
            for head in self.contingency_heads
        ]
        
        # Predict failure mode
        failure_probs = self.failure_classifier(combined)
        
        # Predict confidence
        confidence = self.confidence_predictor(combined)
        
        return contingency_waypoints, failure_probs, confidence
    
    def select_contingency(
        self,
        contingency_waypoints: List[torch.Tensor],
        failure_probs: torch.Tensor,
        confidence: torch.Tensor,
        current_failure_mode: FailureMode,
    ) -> torch.Tensor:
        """
        Select the best contingency based on failure mode and confidence.
        
        Args:
            contingency_waypoints: List of predicted fallback plans
            failure_probs: Predicted failure probabilities
            confidence: Confidence scores for each contingency
            current_failure_mode: Detected current failure mode
            
        Returns:
            selected_waypoints: [B, 10, waypoint_dim] Selected contingency plan
        """
        # Use failure mode to weight contingency selection
        failure_weights = failure_probs[:, :len(FailureMode)-1].sum(dim=-1)  # Exclude NONE
        
        # Combine confidence with failure probability
        selection_score = confidence * failure_weights.unsqueeze(-1)
        
        # Select best contingency
        best_idx = selection_score.argmax(dim=-1)
        
        selected = []
        for i, cw in enumerate(contingency_waypoints):
            mask = (best_idx == i).unsqueeze(-1).unsqueeze(-1)
            selected.append(cw * mask.float())
        
        # Stack and sum (only one is non-zero per batch element)
        selected_waypoints = torch.stack(selected).sum(dim=0)
        
        return selected_waypoints


class BehaviorTreeNode(nn.Module):
    """Base class for behavior tree nodes."""
    
    def __init__(self, name: str):
        self.name = name
        self.parent: Optional[BehaviorTreeNode] = None
        self.children: List[BehaviorTreeNode] = []
    
    def add_child(self, child: "BehaviorTreeNode"):
        child.parent = self
        self.children.append(child)
    
    def execute(self, state: Dict) -> bool:
        """Execute this node. Returns success/failure."""
        raise NotImplementedError


class SequenceNode(BehaviorTreeNode):
    """Behavior Tree Sequence node: succeeds if all children succeed."""
    
    def __init__(self, name: str):
        super().__init__(name)
    
    def execute(self, state: Dict) -> bool:
        for child in self.children:
            if not child.execute(state):
                return False
        return True


class SelectorNode(BehaviorTreeNode):
    """Behavior Tree Selector node: succeeds if any child succeeds."""
    
    def __init__(self, name: str):
        super().__init__(name)
    
    def execute(self, state: Dict) -> bool:
        for child in self.children:
            if child.execute(state):
                return True
        return False


class ConditionNode(BehaviorTreeNode):
    """Behavior Tree Condition node: checks a condition."""
    
    def __init__(self, name: str, condition_fn):
        super().__init__(name)
        self.condition_fn = condition_fn
    
    def execute(self, state: Dict) -> bool:
        return self.condition_fn(state)


class ActionNode(BehaviorTreeNode):
    """Behavior Tree Action node: executes an action."""
    
    def __init__(self, name: str, action_fn):
        super().__init__(name)
        self.action_fn = action_fn
    
    def execute(self, state: Dict) -> bool:
        return self.action_fn(state)


def create_safe_driving_tree() -> SelectorNode:
    """
    Create a behavior tree for safe driving with contingency behaviors.
    
    Structure:
    - Selector (main fallback chain)
    │   - Sequence (emergency stop)
    │   │   - Condition: collision imminent
    │   │   - Action: emergency brake
    │   ├── Sequence (conservative fallback)
    │   │   - Condition: uncertainty high OR off-road risk
    │   │   - Action: reduce speed, increase following distance
    │   │   - Action: use backup waypoints
    │   └── Sequence (normal driving)
    │       - Condition: all clear
    │       - Action: execute primary plan
    """
    root = SelectorNode("root")
    
    # Emergency stop subtree
    emergency_seq = SequenceNode("emergency")
    emergency_seq.add_child(ConditionNode(
        "collision_check",
        lambda s: s.get("failure_mode") == FailureMode.COLLISION_IMMINENT
    ))
    emergency_seq.add_child(ActionNode(
        "emergency_brake",
        lambda s: s.get("executed_action", "none") == "emergency_brake"
    ))
    root.add_child(emergency_seq)
    
    # Conservative fallback subtree
    conservative_seq = SequenceNode("conservative")
    conservative_seq.add_child(ConditionNode(
        "degraded_check",
        lambda s: s.get("failure_mode") in [
            FailureMode.OFF_ROAD,
            FailureMode.UNCERTAINTY_HIGH,
            FailureMode.TIMEOUT
        ]
    ))
    conservative_seq.add_child(ActionNode(
        "reduce_speed",
        lambda s: s.get("speed_reduction", 0) > 0
    ))
    conservative_seq.add_child(ActionNode(
        "use_fallback",
        lambda s: s.get("used_fallback", False)
    ))
    root.add_child(conservative_seq)
    
    # Normal driving subtree
    normal_seq = SequenceNode("normal")
    normal_seq.add_child(ConditionNode(
        "all_clear",
        lambda s: s.get("failure_mode") == FailureMode.NONE
    ))
    normal_seq.add_child(ActionNode(
        "execute_primary",
        lambda s: s.get("executed_action", "none") == "primary"
    ))
    root.add_child(normal_seq)
    
    return root


class GracefulDegradationController(nn.Module):
    """
    Handles graceful degradation when sensor degradation is detected.
    
    Based on survey: "Reduce capabilities safely when degradation detected"
    """
    
    def __init__(self, config: ContingencyConfig):
        super().__init__()
        self.config = config
        
        # Degradation level encoder
        self.degradation_encoder = nn.Sequential(
            nn.Linear(4, 64),  # 4 sensor modalities
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        
        # Capability reduction predictor
        self.capability_predictor = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 5),  # 5 capability dimensions
            nn.Sigmoid(),  # 0 = full capability, 1 = disabled
        )
        
        # Speed reduction factor
        self.speed_factor = 1.0
        
        # Following distance multiplier
        self.following_distance_multiplier = 1.0
    
    def forward(self, sensor_health: torch.Tensor) -> Dict[str, float]:
        """
        Args:
            sensor_health: [B, 4] Health scores for camera, lidar, radar, GPS (0-1)
            
        Returns:
            Dictionary of degradation parameters
        """
        enc = self.degradation_encoder(sensor_health)
        capability_reduction = self.capability_predictor(enc)
        
        # Calculate degraded parameters
        avg_health = sensor_health.mean(dim=-1)
        
        # Speed reduction based on lowest sensor health
        min_health = sensor_health.min(dim=-1)[0]
        speed_factor = torch.clamp(min_health * 2, 0.3, 1.0)
        
        # Following distance increase
        following_mult = 1.0 + (1.0 - min_health) * 2
        
        return {
            "speed_factor": speed_factor.item(),
            "following_distance_multiplier": following_mult.item(),
            "capability_reduction": capability_reduction.mean().item(),
        }
    
    def apply_degradation(
        self,
        waypoints: torch.Tensor,
        params: Dict[str, float],
    ) -> torch.Tensor:
        """Apply degradation to waypoints (reduce speed, increase spacing)."""
        # Reduce speed in waypoints (last dimension is typically speed/time)
        degraded = waypoints.clone()
        degraded[:, :, -1] *= params["speed_factor"]
        
        return degraded


class ContingencyPlanner(nn.Module):
    """
    Full contingency planner combining:
    - ContingencyNetwork for fallback prediction
    - BehaviorTree for rule-based safety
    - GracefulDegradationController for sensor failures
    """
    
    def __init__(
        self,
        state_dim: int = 256,
        waypoint_dim: int = 3,
        hidden_dim: int = 512,
        num_contingencies: int = 3,
    ):
        super().__init__()
        
        self.contingency_net = ContingencyNetwork(
            state_dim=state_dim,
            waypoint_dim=waypoint_dim,
            hidden_dim=hidden_dim,
            num_contingencies=num_contingencies,
        )
        
        self.config = ContingencyConfig()
        self.behavior_tree = None  # Built at runtime with current state
        self.degradation_controller = GracefulDegradationController(self.config)
    
    def forward(
        self,
        state: torch.Tensor,
        primary_waypoints: torch.Tensor,
        sensor_health: Optional[torch.Tensor] = None,
        failure_mode: Optional[FailureMode] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Main forward pass for contingency planning.
        
        Args:
            state: [B, state_dim] Current state encoding
            primary_waypoints: [B, 10, waypoint_dim] Primary planned waypoints
            sensor_health: [B, 4] Optional sensor health scores
            failure_mode: Optional explicit failure mode
            
        Returns:
            final_waypoints: [B, 10, waypoint_dim] Waypoints to execute
            info: Dict with contingency info, failure mode, etc.
        """
        # Get contingencies from network
        contingency_waypoints, failure_probs, confidence = self.contingency_net(
            state, primary_waypoints
        )
        
        # Determine failure mode
        if failure_mode is None:
            failure_mode_idx = failure_probs[:, :-1].argmax(dim=-1)  # Exclude NONE
            failure_mode = FailureMode(failure_mode_idx.item())
        
        # Handle graceful degradation
        if sensor_health is not None:
            degradation_params = self.degradation_controller(sensor_health)
            primary_waypoints = self.degradation_controller.apply_degradation(
                primary_waypoints, degradation_params
            )
        else:
            degradation_params = {}
        
        # Select contingency
        selected_contingency = self.contingency_net.select_contingency(
            contingency_waypoints, failure_probs, confidence, failure_mode
        )
        
        # Choose between primary and contingency based on failure mode
        if failure_mode == FailureMode.NONE:
            final_waypoints = primary_waypoints
        else:
            final_waypoints = selected_contingency
        
        info = {
            "failure_mode": failure_mode,
            "failure_probs": failure_probs,
            "confidence": confidence,
            "contingency_waypoints": contingency_waypoints,
            "selected_contingency": selected_contingency,
            "degradation_params": degradation_params,
        }
        
        return final_waypoints, info


def create_contingency_planner(
    state_dim: int = 256,
    waypoint_dim: int = 3,
    hidden_dim: int = 512,
) -> ContingencyPlanner:
    """Factory function to create a contingency planner."""
    return ContingencyPlanner(
        state_dim=state_dim,
        waypoint_dim=waypoint_dim,
        hidden_dim=hidden_dim,
    )


if __name__ == "__main__":
    # Simple test
    planner = create_contingency_planner()
    
    # Dummy inputs
    batch_size = 4
    state = torch.randn(batch_size, 256)
    primary_waypoints = torch.randn(batch_size, 10, 3)
    sensor_health = torch.rand(batch_size, 4)  # camera, lidar, radar, GPS
    
    final_waypoints, info = planner(state, primary_waypoints, sensor_health)
    
    print(f"Input state shape: {state.shape}")
    print(f"Primary waypoints shape: {primary_waypoints.shape}")
    print(f"Final waypoints shape: {final_waypoints.shape}")
    print(f"Failure mode: {info['failure_mode']}")
    print(f"Speed factor: {info['degradation_params'].get('speed_factor', 1.0):.2f}")
    
    # Test behavior tree
    tree = create_safe_driving_tree()
    test_state = {
        "failure_mode": FailureMode.NONE,
        "executed_action": "none",
        "speed_reduction": 0,
        "used_fallback": False,
    }
    result = tree.execute(test_state)
    print(f"\nBehavior tree test (normal): {result}")
    
    test_state["failure_mode"] = FailureMode.COLLISION_IMMINENT
    result = tree.execute(test_state)
    print(f"Behavior tree test (collision): {result}")
