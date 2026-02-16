# CoT Data Generation for Autonomous Driving

## Overview

Chain of Thought (CoT) reasoning traces for driving decisions can significantly improve model interpretability and performance. This document outlines a simple framework for generating reasoning traces and integrating them into our training pipeline.

---

## Why CoT for Autonomous Driving?

| Benefit | Description |
|---------|-------------|
| **Interpretability** | Understand why the model made a decision |
| **Better RL** | Reasoning traces provide structured reward signals |
| **Debugging** | Trace back failures to specific reasoning steps |
| **SFT Improvement** | Learn from expert reasoning patterns |

---

## CoT Trace Format for Driving

```
Input: Camera images, BEV features, History trajectory
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Perception Analysis                              │
│  "I see a car 20m ahead, lane markings, pedestrian cross-│
│   ing 50m ahead on the left"                              │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Risk Assessment                                  │
│  "Pedestrian has high uncertainty (partially occluded),    │
│   car ahead is stable but requires monitoring"           │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Trajectory Planning                              │
│  "Plan smooth lane-follow with slight deceleration,       │
│   maintain 1.5m left offset from center line"             │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Confidence Evaluation                            │
│  "High confidence in lane detection, medium confidence    │
│   in pedestrian intent, overall confidence: 0.85"        │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
Output: Waypoints + Reasoning trace
```

---

## Simple CoT Generation Pipeline

### 1. Rule-Based Traces (Heuristic Planner)

```python
# training/rl/cot_generator.py

class DrivingCoTGenerator:
    """
    Generate reasoning traces from heuristic driving rules.
    Used as teacher for CoT SFT training.
    """
    
    def generate_trace(self, state: DrivingState) -> CoTTrace:
        """
        Generate a reasoning trace for the given driving state.
        
        Args:
            state: Perception output (detections, lane markings, etc.)
            
        Returns:
            CoTTrace with reasoning steps + expert waypoints
        """
        trace = CoTTrace()
        
        # Step 1: Perception Summary
        trace.perception = self._summarize_perception(state)
        
        # Step 2: Scene Understanding  
        trace.scene_understanding = self._understand_scene(state)
        
        # Step 3: Behavior Prediction
        trace.behavior_prediction = self._predict_behaviors(state)
        
        # Step 4: Trajectory Decision
        trace.trajectory_decision = self._plan_trajectory(state)
        
        # Step 5: Confidence
        trace.confidence = self._evaluate_confidence(state)
        
        return trace
    
    def _summarize_perception(self, state: DrivingState) -> str:
        """Summarize what the model perceives."""
        n_vehicles = len(state.vehicles)
        n_pedestrians = len(state.pedestrians)
        lanes_detected = state.lane_lines.confidence > 0.5
        
        return (
            f"I detect {n_vehicles} vehicles and {n_pedestrians} pedestrians. "
            f"Lane detection confidence is {'high' if lanes_detected else 'low'}. "
            f"The ego vehicle is currently traveling at {state.ego_speed:.1f} m/s."
        )
    
    def _understand_scene(self, state: DrivingState) -> str:
        """Interpret the driving scene context."""
        context = []
        
        # Check for intersections
        if state.is_intersection:
            context.append("Approaching an intersection")
            
        # Check for pedestrians
        if state.pedestrians:
            nearby_ped = [p for p in state.pedestrians if p.distance < 30]
            if nearby_ped:
                context.append(f"Tracking {len(nearby_ped)} nearby pedestrians")
                
        # Check leading vehicle
        if state.leading_vehicle:
            context.append(f"Leading vehicle {state.leading_vehicle.distance:.1f}m ahead")
            
        return ". ".join(context) if context else "Open road, no immediate hazards"
    
    def _predict_behaviors(self, state: DrivingState) -> str:
        """Predict behaviors of other agents."""
        predictions = []
        
        for vehicle in state.vehicles[:3]:  # Top 3 vehicles
            if vehicle.distance < 50:
                if vehicle.velocity > state.ego_speed + 2:
                    predictions.append(f"Vehicle ahead is accelerating (+{vehicle.velocity - state.ego_speed:.1f} m/s)")
                elif vehicle.velocity < state.ego_speed - 2:
                    predictions.append(f"Vehicle ahead is slowing down ({vehicle.velocity:.1f} m/s)")
                else:
                    predictions.append(f"Vehicle ahead maintaining speed ({vehicle.velocity:.1f} m/s)")
        
        return " | ".join(predictions) if predictions else "No relevant agent behaviors predicted"
    
    def _plan_trajectory(self, state: DrivingState) -> str:
        """Generate trajectory reasoning."""
        actions = []
        
        # Speed adjustment
        if state.leading_vehicle and state.leading_vehicle.distance < 20:
            actions.append("Slowing down to maintain safe following distance")
        elif state.ego_speed < 5:
            actions.append("Accelerating to desired cruising speed")
        else:
            actions.append("Maintaining current speed")
        
        # Lane centering
        actions.append(f"Center lane positioning with {state.lateral_offset:.2f}m offset adjustment")
        
        # Turn preparation
        if state.upcoming_turn_ahead and state.upcoming_turn_ahead.distance < 50:
            turn_dir = "left" if state.upcoming_turn_ahead.direction == "left" else "right"
            actions.append(f"Preparing for {turn_dir} turn in {state.upcoming_turn_ahead.distance:.0f}m")
        
        return ". ".join(actions)
    
    def _evaluate_confidence(self, state: DrivingState) -> str:
        """Evaluate overall decision confidence."""
        confidence_factors = []
        
        if state.lane_lines.confidence > 0.8:
            confidence_factors.append("high lane detection confidence")
        else:
            confidence_factors.append("uncertain lane detection")
            
        if len(state.vehicles) < 5:
            confidence_factors.append("low traffic density")
        else:
            confidence_factors.append("complex traffic scene")
            
        if not state.pedestrians:
            confidence_factors.append("no pedestrians detected")
        else:
            confidence_factors.append("pedestrians require attention")
        
        return f"Confidence based on: {', '.join(confidence_factors)}"
```

---

## CoT Data Format (JSON)

```json
{
  "episode_id": "waymo_0001234",
  "timestamp": 1.5,
  "cot_trace": {
    "perception": "I detect 2 vehicles and 0 pedestrians. Lane detection confidence is high. The ego vehicle is currently traveling at 8.5 m/s.",
    "scene_understanding": "Approaching an intersection. Leading vehicle 15.2m ahead maintaining speed.",
    "behavior_prediction": "Vehicle ahead maintaining speed (8.2 m/s)",
    "trajectory_decision": "Slowing down to maintain safe following distance. Center lane positioning with -0.12m offset adjustment. Preparing for left turn in 40m.",
    "confidence": "Confidence based on: high lane detection confidence, low traffic density, no pedestrians detected"
  },
  "expert_waypoints": [
    {"t": 0.0, "x": 0.0, "y": 0.0},
    {"t": 0.5, "x": 4.2, "y": 0.08},
    {"t": 1.0, "x": 8.5, "y": 0.15},
    {"t": 1.5, "x": 12.7, "y": 0.18}
  ],
  "action": {
    "steering": -0.02,
    "throttle": 0.3,
    "brake": 0.0
  }
}
```

---

## Training with CoT Traces

### SFT with Reasoning

```python
# training/sft/train_cot_reasoning.py

class CoTReasoningTrainer:
    """
    Train waypoint predictor with CoT reasoning traces.
    """
    
    def compute_loss(
        self, 
        model_output: ModelOutput, 
        target: Waypoints,
        cot_trace: CoTTrace
    ) -> Tuple[Loss, Dict]:
        """
        Compute combined loss: waypoint MSE + CoT consistency.
        """
        # Waypoint prediction loss
        waypoint_loss = F.mse_loss(
            model_output.waypoints,
            target.waypoints
        )
        
        # CoT reasoning loss (optional: use LLM to score reasoning)
        cot_loss = self._compute_cot_loss(
            model_output.cot_generated,
            cot_trace
        )
        
        # Total loss
        total_loss = waypoint_loss + 0.1 * cot_loss
        
        return total_loss, {
            "waypoint_loss": waypoint_loss.item(),
            "cot_loss": cot_loss.item()
        }
```

---

## Comparison with Known Papers

| Paper | Approach | Our Difference |
|-------|----------|----------------|
| **VAD** | Vectorized planning with simple queries | We add explicit reasoning traces |
| **ADAPT** | Transformer with natural language | We use structured CoT format |
| **UniAD** | Task-query formulation | We focus on waypoint BC + CoT |
| **COOPER** | Joint perception-planning | We use CoT as training signal |

---

## Next Steps

1. **Generate 1000 CoT traces** from Waymo data using rule-based generator
2. **Train SFT model** with CoT conditioning
3. **Evaluate** if CoT improves waypoint prediction
4. **Extend** to multi-modal CoT (text + visual explanations)

---

# Kimi RL Survey (Moonshot AI)

## Overview

Kimi is Moonshot AI's LLM series. The Kimi RL paper introduces **LengthControl**, a novel approach to controlling response length in RLHF.

---

## TL;DR

- **LengthControl**: Reward shaping technique to guide model toward desired response lengths
- **Benefits**: More controllable output length without quality degradation
- **Application**: Useful for driving models that need consistent planning horizons

---

## LengthControl Mechanism

```
Standard RLHF:    Policy → Reward Model → Update
                  (may produce any length)

With LengthControl:
┌─────────────────────────────────────────────────────────────┐
│                    LengthControl                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Policy Output → Length Penalty → Modified Reward         │
│       │              (length - target)^2                   │
│       ▼                                                     │
│  RL Update with length-regularized reward                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Results

| Metric | Standard RLHF | Kimi RL + LengthControl |
|--------|---------------|--------------------------|
| Response length | Uncontrolled | Precisely controlled |
| Quality (QA) | Baseline | +2-3% improvement |
| Consistency | Variable | More consistent |
| Training stability | Standard | Improved |

---

## Relevance to Our Pipeline

| Our Use Case | Kimi RL Application |
|--------------|--------------------|
| **Waypoint prediction** | Control trajectory length/horizon |
| **RL refinement** | Shape exploration toward appropriate horizons |
| **SFT training** | Consistency in planning depth |

---

## Implementation Notes

1. **Length penalty**: Add `(length - target)^2` to reward function
2. **Adaptive target**: Adjust target length based on scene complexity
3. **Curriculum**: Start with loose length constraints, tighten over time

---

## References

- Kimi AI: https://kimi.ai/
- LengthControl mechanism described in Moonshot AI technical reports
