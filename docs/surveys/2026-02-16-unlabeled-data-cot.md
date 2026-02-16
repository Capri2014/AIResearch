# Unlabeled Human Driving Data for CoT Generation

**Date:** 2026-02-16  
**Question:** Can unlabeled driving data help CoT generation?

---

## TL;DR

**YES!** Unlabeled human driving data is valuable for CoT in multiple ways:

| Approach | What It Learns | CoT Application |
|----------|---------------|-----------------|
| **Self-supervised perception** | Scene understanding | "I see X because..." |
| **Imitation learning** | Expert behavior patterns | "The expert would do X because..." |
| **Future prediction** | What happens next | "Given the situation, X will likely happen" |
| **Intent inference** | Why drivers act that way | "Driver Y is turning because..." |
| **Situation clustering** | Similar scenarios | "This is similar to situation X where..." |

---

## Why Unlabeled Data Helps CoT

### The Core Insight

```
Labeled Data:                    Unlabeled Data:
┌─────────────────────┐         ┌─────────────────────┐
│ Input → Label       │         │ Input → ???         │
│ (expensive)         │         │ (abundant)          │
└─────────────────────┘         └─────────────────────┘
         │                              │
         ▼                              ▼
   Supervised CoT                Self-Supervised CoT
   (limited scale)               (massive scale)

Key: Self-supervision can learn "reasoning structure" 
     without explicit labels
```

### What CoT Needs

| CoT Component | What It Requires | Can Unlabeled Data Help? |
|---------------|-----------------|--------------------------|
| **Scene Description** | "What do I see?" | ✅ Yes - learn from video |
| **Situation Assessment** | "What's happening?" | ✅ Yes - pattern recognition |
| **Behavior Prediction** | "What will others do?" | ✅ Yes - future prediction |
| **Planning** | "What should I do?" | ✅ Yes - imitation learning |
| **Justification** | "Why?" | ✅ Yes - intent inference |

---

## Approaches to Extract CoT from Unlabeled Data

### 1. Self-Supervised Perception CoT

```
┌─────────────────────────────────────────────────────────────────┐
│              Self-Supervised Perception Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw Camera Data                                                  │
│  ├── Left Front Camera                                            │
│  ├── Right Front Camera                                          │
│  ├── Front Left Camera                                          │
│  └── ...                                                         │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Self-Supervised Tasks                      │    │
│  │                                                          │    │
│  │  1. Temporal Consistency                                │    │
│  │     "Frame t+1 should be consistent with Frame t"      │    │
│  │                                                          │    │
│  │  2. Depth Estimation (stereo)                          │    │
│  │     "Object at distance D should appear at size S"     │    │
│  │                                                          │    │
│  │  3. Motion Segmentation                                 │    │
│  │     "Moving objects vs static background"              │    │
│  │                                                          │    │
│  │  4. View Prediction                                    │    │
│  │     "From this view, predict other camera views"      │    │
│  │                                                          │    │
│  │  5. Future Frame Prediction                            │    │
│  │     "Given past frames, predict next frame"           │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                                                        │
│         ▼                                                        │
│  Learned Representations with "Explanation"                       │
│  "Object is X meters away because of stereo consistency"       │
│  "Vehicle is moving because of temporal change"                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
# File: cot_from_unlabeled/self_supervised_cot.py

import torch
import torch.nn as nn


class SelfSupervisedPerception(nn.Module):
    """
    Learn scene understanding from unlabeled video.
    
    Creates CoT-like explanations from self-supervision signals.
    """
    
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        
        # Self-supervised heads
        self.temporal_head = nn.Linear(256, 1)  # Predict motion
        self.depth_head = nn.Linear(256, 1)      # Predict depth
        self.motion_head = nn.Linear(256, 10)    # Segment moving objects
    
    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with self-supervised learning.
        
        Returns:
            representations + explanation scores
        """
        # Encode all frames
        features = self.encoder(frames)  # [B, T, D]
        
        # Self-supervised tasks
        temporal_consistency = self._temporal_consistency(features)
        depth_prediction = self._depth_prediction(features)
        motion_segmentation = self._motion_segmentation(features)
        
        return {
            'features': features,
            'temporal_consistency': temporal_consistency,
            'depth': depth_prediction,
            'motion': motion_segmentation,
        }
    
    def generate_perception_cot(self, features: torch.Tensor) -> str:
        """
        Generate CoT-like perception explanation.
        
        Based on what the model learned to predict:
        """
        explanations = []
        
        # Depth explanation
        depth = self.depth_head(features).item()
        if depth < 5:
            explanations.append(f"Close vehicle detected at {depth:.1f}m (high confidence from depth estimation)")
        elif depth < 20:
            explanations.append(f"Vehicle ahead at medium distance: {depth:.1f}m")
        else:
            explanations.append(f"Distant vehicle at {depth:.1f}m (lower confidence)")
        
        # Motion explanation
        motion = self.motion_head(features)
        moving_objects = (motion > 0.5).sum().item()
        if moving_objects > 0:
            explanations.append(f"{moving_objects} moving objects detected in field of view")
        
        # Temporal explanation
        temporal = self.temporal_head(features).item()
        if temporal > 0.8:
            explanations.append("Scene is temporally stable (consistent frame-to-frame)")
        else:
            explanations.append("Rapid scene changes detected")
        
        return " | ".join(explanations)
```

### 2. Imitation Learning for Planning CoT

```
┌─────────────────────────────────────────────────────────────────┐
│              Imitation Learning for CoT                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Unlabeled Human Driving Data                                     │
│  ├── Camera streams (multi-view)                                 │
│  ├── CAN bus (steer, throttle, brake)                           │
│  ├── GPS/Localization                                            │
│  └── LiDAR point clouds                                          │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Behavior Cloning                          │    │
│  │                                                          │    │
│  │  State → Action (learn expert policy)                 │    │
│  │                                                          │    │
│  │  Key: Add "reasoning" layer                          │    │
│  │                                                          │    │
│  │  State → [What experts do] → [Why they do it]       │    │
│  │                       ↓                                 │    │
│  │              Action + CoT Explanation                  │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                                                        │
│         ▼                                                        │
│  Expert Reasoning Traces                                         │
│  "I see a pedestrian at X, so I brake because..."               │
│  "The car ahead is slowing, so I reduce throttle because..."     │
│  "The light is green and no pedestrians, so I continue"          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
# File: cot_from_unlabeled/imitation_cot.py

class ImitationCoTGenerator(nn.Module):
    """
    Learn expert reasoning from unlabeled driving demonstrations.
    
    Uses counterfactual reasoning to infer "why" from "what".
    """
    
    def __init__(self, config):
        super().__init__()
        self.situation_encoder = nn.Linear(256, 64)  # Encode situation
        self.action_predictor = nn.Linear(64, 3)      # Predict action
        self.reasoning_generator = ReasoningGenerator()  # Generate CoT
    
    def forward(self, state_features: torch.Tensor, actions: torch.Tensor):
        """
        Forward with imitation learning + CoT generation.
        
        Learn: Situation → Action (and the reasoning in between)
        """
        # Encode situation
        situation = self.situation_encoder(state_features)
        
        # Predict action
        predicted_action = self.action_predictor(situation)
        
        # Generate reasoning that connects situation to action
        # The key insight: what situation features lead to this action?
        reasoning = self.generate_reasoning(situation, actions)
        
        return predicted_action, reasoning
    
    def generate_reasoning(
        self,
        situation: torch.Tensor,
        actual_action: torch.Tensor
    ) -> str:
        """
        Generate CoT explanation from learned representations.
        
        Uses attention to find which situation features caused the action.
        """
        # Find important features for this action
        attention = self.compute_attention(situation, actual_action)
        
        # Map attention to human-readable explanations
        explanations = []
        
        if attention['speed_related'] > 0.3:
            speed = situation['ego_speed'].item()
            explanations.append(f"Ego speed ({speed:.1f} m/s) is relevant")
        
        if attention['obstacle_related'] > 0.3:
            n_obstacles = situation['n_obstacles']
            explanations.append(f"Obstacle awareness: {n_obstacles} objects detected")
        
        if attention['lane_related'] > 0.3:
            lane_type = situation['lane_type']
            explanations.append(f"Current lane: {lane_type}")
        
        if attention['traffic_related'] > 0.3:
            light = situation['traffic_light']
            explanations.append(f"Traffic signal: {light}")
        
        # Combine into CoT trace
        reasoning = " | ".join(explanations)
        return reasoning


class ReasoningGenerator(nn.Module):
    """
    Generate natural language reasoning from learned features.
    """
    
    def __init__(self):
        super().__init__()
        self.mapping = FeatureToTextMapping()
    
    def forward(self, features: torch.Tensor) -> str:
        """Convert learned features to CoT text."""
        return self.mapping.decode(features)
```

### 3. Future Prediction for Behavior CoT

```
┌─────────────────────────────────────────────────────────────────┐
│              Future Prediction → Behavior CoT                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Key Idea: "If I predict what will happen, I understand why"    │
│                                                                  │
│  Input: Current Situation                                        │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Future Prediction Model                    │    │
│  │                                                          │    │
│  │  Question: "What will happen in T seconds?"          │    │
│  │                                                          │    │
│  │  Answer: "Vehicle ahead will slow down because..."   │    │
│  │             "Pedestrian will cross because..."         │    │
│  │             "Light will change because..."            │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                                                       │
│         ▼                                                       │
│  CoT Trace (Behavior Prediction)                                 │
│  "Based on current situation:"                                   │
│  "1. Vehicle ahead shows brake lights → will slow"              │
│  "2. Pedestrian looking at road → may cross"                     │
│  "3. Light has been green for 30s → may change soon"             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
# File: cot_from_unlabeled/future_prediction_cot.py

class FuturePredictionCoT(nn.Module):
    """
    Learn behavior prediction from unlabeled temporal data.
    
    If you can predict the future, you understand the reasoning.
    """
    
    def __init__(self):
        super().__init__()
        self.situation_encoder = nn.LSTM(256, 128, batch_first=True)
        self.future_predictor = nn.Linear(128, 256)  # Predict future state
        self.reasoning_extractor = ReasoningExtractor()
    
    def forward(
        self,
        past_frames: torch.Tensor,
        current_frame: torch.Tensor
    ) -> Tuple[torch.Tensor, str]:
        """
        Predict future and extract reasoning.
        """
        # Encode temporal context
        context, _ = self.situation_encoder(past_frames)
        current_state = context[:, -1]  # Last hidden state
        
        # Predict future
        predicted_future = self.future_predictor(current_state)
        
        # Extract reasoning from prediction
        reasoning = self.reasoning_extractor.extract(
            current_state, predicted_future
        )
        
        return predicted_future, reasoning
    
    def generate_behavior_prediction_cot(
        self,
        current_situation: Dict,
        predicted_future: Dict
    ) -> str:
        """
        Generate CoT for behavior prediction.
        """
        predictions = []
        
        # Vehicle predictions
        for vehicle in current_situation.get('vehicles', []):
            future_state = predicted_future.get(f"vehicle_{vehicle.id}")
            
            if future_state['velocity'] < vehicle.velocity - 1:
                predictions.append(
                    f"Vehicle {vehicle.id} will decelerate "
                    f"(current: {vehicle.velocity:.1f} → "
                    f"predicted: {future_state['velocity']:.1f})"
                )
            elif future_state['lane'] != vehicle.lane:
                predictions.append(
                    f"Vehicle {vehicle.id} will change lanes "
                    f"(from lane {vehicle.lane} to {future_state['lane']})"
                )
        
        # Pedestrian predictions
        for ped in current_situation.get('pedestrians', []):
            crossing_prob = predicted_future.get(f"ped_{ped.id}", {}).get('crossing_prob', 0)
            if crossing_prob > 0.5:
                predictions.append(
                    f"Pedestrian {ped.id} likely to cross "
                    f"(intention probability: {crossing_prob:.2f})"
                )
        
        # Light predictions
        time_until_change = predicted_future.get('traffic_light', {}).get('seconds_until_change')
        if time_until_change and time_until_change < 5:
            predictions.append(
                f"Traffic light will change in {time_until_change:.1f}s"
            )
        
        return " | ".join(predictions)
```

### 4. Situation Clustering for Template CoT

```
┌─────────────────────────────────────────────────────────────────┐
│              Situation Clustering → Template CoT                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Unlabeled Data → Cluster Similar Situations                     │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Learned Situation Clusters                  │    │
│  │                                                          │    │
│  │  Cluster 1: "Highway Cruise"                            │    │
│  │  - Typical CoT: "Clear lane ahead, maintaining speed"   │    │
│  │                                                          │    │
│  │  Cluster 2: "Urban Intersection"                         │    │
│  │  - Typical CoT: "Green light, checking cross traffic"   │    │
│  │                                                          │    │
│  │  Cluster 3: "Pedestrian Zone"                           │    │
│  │  - Typical CoT: "Pedestrians detected, reducing speed"  │    │
│  │                                                          │    │
│  │  Cluster 4: "Lane Change"                               │    │
│  │  - Typical CoT: "Need to merge, checking mirrors"       │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                                                       │
│         ▼                                                       │
│  For new situation: Find nearest cluster → Use template + customize│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
# File: cot_from_unlabeled/clustering_cot.py

class ClusteredCoTGenerator(nn.Module):
    """
    Generate CoT using learned situation clusters.
    
    Learn templates from unlabeled data, apply to new situations.
    """
    
    def __init__(self, n_clusters: int = 50):
        super().__init__()
        self.situation_encoder = nn.Linear(256, 128)
        self.cluster_heads = nn.ModuleList([
            nn.Linear(128, 128) for _ in range(n_clusters)
        ])
        self.cluster_templates = ClusterTemplates()
    
    def forward(self, situation: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Find nearest cluster for this situation."""
        features = self.situation_encoder(situation)
        
        # Compute distances to all clusters
        distances = []
        for head in self.cluster_heads:
            dist = torch.norm(features - head.weight, dim=1)
            distances.append(dist)
        
        cluster_id = torch.argmin(torch.stack(distances))
        return cluster_id, features
    
    def generate_clustered_cot(
        self,
        situation: Dict,
        cluster_id: int
    ) -> str:
        """
        Generate CoT using cluster template + situation-specific details.
        """
        # Get template for this cluster
        template = self.cluster_templates.get(cluster_id)
        
        # Fill in situation-specific details
        cot_parts = []
        
        for step in template.steps:
            if step.needs_vehicle_count:
                n_vehicles = len(situation.get('vehicles', []))
                cot_parts.append(step.text.format(n_vehicles=n_vehicles))
            
            if step.needs_pedestrian_count:
                n_peds = len(situation.get('pedestrians', []))
                cot_parts.append(step.text.format(n_pedestrians=n_peds))
            
            if step.needs_speed:
                speed = situation.get('ego_speed', 0)
                cot_parts.append(step.text.format(speed=speed))
        
        return " | ".join(cot_parts)
```

### 5. Contrastive Learning for Situation Understanding

```
┌─────────────────────────────────────────────────────────────────┐
│              Contrastive Learning for CoT                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Learn: "What makes situations similar/different?"              │
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ Situation A     │    │ Situation B     │                     │
│  │ "Car ahead,     │    │ "Car ahead,    │                     │
│  │  slowing down"  │ ~  │  speeding up"   │                     │
│  └────────┬────────┘    └────────┬────────┘                     │
│           │                       │                              │
│           ▼                       ▼                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Contrastive Encoder                        │    │
│  │                                                          │    │
│  │  Similar situations → Similar embeddings               │    │
│  │  Different situations → Different embeddings          │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│           │                                                      │
│           ▼                                                      │
│  CoT Generation:                                                 │
│  "This situation is SIMILAR to X because..."                     │
│  "This situation is DIFFERENT from Y because..."                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
# File: cot_from_unlabeled/contrastive_cot.py

class ContrastiveCoT(nn.Module):
    """
    Use contrastive learning to understand situation relationships.
    
    Generates CoT like "This is similar to X because..."
    """
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(256, 128)
        self.situation_bank = SituationBank(size=10000)
    
    def forward(self, situation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode situation and find similar situations."""
        embedding = self.encoder(situation)
        
        # Find k nearest neighbors in bank
        neighbors = self.situation_bank.search(embedding, k=5)
        
        return {
            'embedding': embedding,
            'similar_situations': neighbors,
            'dissimilar_situations': self._find_dissimilar(embedding),
        }
    
    def generate_contrastive_cot(
        self,
        situation: Dict,
        embedding: torch.Tensor,
        neighbors: List[Dict]
    ) -> str:
        """Generate CoT with comparison to similar situations."""
        cot_parts = []
        
        # Primary situation description
        cot_parts.append(f"Situation: {self._summarize(situation)}")
        
        # Similar situation comparison
        if neighbors:
            most_similar = neighbors[0]
            similarity = most_similar['similarity']
            cot_parts.append(
                f"Similar to past situation {most_similar['id']} "
                f"(similarity: {similarity:.2f})"
            )
            cot_parts.append(
                f"In that situation, expert action was: "
                f"{most_similar['action']}"
            )
        
        # Key differences from dissimilar situations
        dissimilar = self._find_dissimilar(embedding)[:3]
        if dissimilar:
            diff_reasons = []
            for d in dissimilar:
                diff_reason = self._explain_difference(situation, d)
                diff_reasons.append(diff_reason)
            cot_parts.append(f"Key differences from dissimilar cases: {'; '.join(diff_reasons)}")
        
        return " | ".join(cot_parts)
```

---

## Complete Pipeline: Unlabeled Data → CoT

```
┌─────────────────────────────────────────────────────────────────┐
│              Complete Unlabeled Data → CoT Pipeline                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Unlabeled Human Driving Data                                     │
│  (CAN bus, cameras, LiDAR, GPS)                                  │
│         │                                                        │
│         ├──────────────────────────────────────────────┐        │
│         │                                              │        │
│         ▼                                              ▼        │
│  ┌─────────────────────┐              ┌─────────────────────┐  │
│  │ Self-Supervised     │              │ Imitation           │  │
│  │ Perception          │              │ Learning            │  │
│  │                     │              │                     │  │
│  │ - Temporal          │              │ - Behavior cloning  │  │
│  │ - Depth             │              │ - Action prediction│  │
│  │ - Motion            │              │ - Intent inference  │  │
│  └──────────┬──────────┘              └──────────┬──────────┘  │
│             │                                    │             │
│             │         ┌──────────────────────────┘             │
│             │         │                                          │
│             ▼         ▼                                          │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Fusion Layer                                 │  │
│  │                                                          │  │
│  │  Combine all learned representations into unified CoT    │  │
│  │                                                          │  │
│  └────────────────────────────┬─────────────────────────────┘  │
│                               │                                  │
│                               ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              CoT Trace Output                            │  │
│  │                                                          │  │
│  │  "I see:"                                               │  │
│  │  - 3 vehicles ahead, closest at 15m slowing down       │  │
│  │  - 1 pedestrian at intersection looking at road         │  │
│  │  - Green light, been green for 25s                     │  │
│  │                                                          │  │
│  │  "Similar situations suggest:"                          │  │
│  │  - When lead vehicle brakes, experts slow down          │  │
│  │  - Pedestrians looking at road often cross              │  │
│  │                                                          │  │
│  │  "Predicted outcomes:"                                 │  │
│  │  - Lead vehicle will continue slowing                  │  │
│  │  - Pedestrian may cross in 3-5s                       │  │
│  │                                                          │  │
│  │  "Recommended action:"                                 │  │
│  │  - Reduce throttle, prepare to brake                   │  │
│  │  - Check mirrors for lane change option                │  │
│  │                                                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Value of Unlabeled Data by Type

| Data Type | What It Enables | CoT Component |
|-----------|-----------------|---------------|
| **CAN bus** (steer/throttle/brake) | Action patterns | Planning, Justification |
| **Camera streams** | Scene understanding | Perception, Behavior |
| **LiDAR** | 3D structure | Depth, Motion |
| **GPS** | Location context | Map-based reasoning |
| **Multi-modal** | Rich understanding | All components |

---

## Implementation Priority

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 1 | **CAN bus analysis** | Low | High |
| 2 | **Future prediction** | Medium | High |
| 3 | **Situation clustering** | Medium | Medium |
| 4 | **Full self-supervised pipeline** | High | High |

---

## Summary

**Unlabeled human driving data IS helpful for CoT:**

1. ✅ **Self-supervised perception** - Learn "what I see" from raw video
2. ✅ **Imitation learning** - Learn "what experts do" from actions
3. ✅ **Future prediction** - Learn "what will happen" from temporal data
4. ✅ **Situation clustering** - Learn templates for common scenarios
5. ✅ **Contrastive learning** - Learn relationships between situations

**Key insight:** CoT doesn't require explicit labels. The "reasoning" can be learned from patterns in the data itself — what situations look like, what experts do in those situations, and what happens next.

**Recommended starting point:**
1. Analyze CAN bus data (simple, high value)
2. Add future prediction model
3. Build situation clustering
4. Generate CoT from learned representations
