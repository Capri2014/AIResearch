# Real Dataset Training with Self-Generated CoT Labels

**Date:** 2026-02-16  
**Question:** Can we use real driving data with our own CoT labels?

---

## TL;DR

**YES!** We can use real driving datasets and generate CoT labels ourselves.

| Dataset | Availability | CoT-ability | Effort |
|---------|--------------|-------------|--------|
| **Waymo** | Requires download | ✅ High | Medium |
| **Alpamayo-R1** | HuggingFace available | ✅ Medium | Low |
| **nuScenes** | Public download | ✅ High | Medium |
| **BDD100K** | Public download | ✅ High | Medium |

---

## Option 1: Waymo Data + Self-Generated CoT

### Data Format Expected

```
data/waymo/
├── episode_0001/
│   ├── camera_front/          # Images
│   │   ├── 00001.jpg
│   │   └── ...
│   ├── lidar/                # Optional
│   ├── poses.json            # Ego poses
│   └── frames.json           # Extracted frames
│
├── episode_0002/
└── ...
```

### Our CoT Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│              Waymo Data + Self-Generated CoT                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw Waymo Data                                                  │
│  ├── Camera images (front, side, rear)                         │
│  ├── LiDAR point clouds                                          │
│  ├── 3D bounding boxes                                          │
│  └── ego_pose (position, heading, speed)                        │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              CoT Trace Generator                        │    │
│  │                                                          │    │
│  │  1. Perception: "What do I see?"                       │    │
│  │     - Count vehicles, pedestrians, lanes               │    │
│  │     - Detect traffic lights, signs                     │    │
│  │                                                          │    │
│  │  2. Prediction: "What will happen?"                    │    │
│  │     - Project future positions of agents                │    │
│  │     - Detect potential conflicts                         │    │
│  │                                                          │    │
│  │  3. Planning: "What should I do?"                       │    │
│  │     - Extract expert trajectory (from Waymo labels)      │    │
│  │     - Generate reasoning for each action                 │    │
│  │                                                          │    │
│  │  4. Justification: "Why this action?"                   │    │
│  │     - Safety rationale                                   │    │
│  │     - Comfort rationale                                  │    │
│  │     - Efficiency rationale                               │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                                                        │
│         ▼                                                        │
│  CoT Trace Output (per frame)                                    │
│  {                                                               │
│    "perception": "I see 3 vehicles ahead, 1 pedestrian...",   │
│    "prediction": "Lead vehicle will slow in 2s...",            │
│    "planning": "Maintain lane, reduce speed by 10%",            │
│    "justification": "Safe following distance maintained..."     │
│  }                                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
# File: training/sft/generate_cot_from_waymo.py

from typing import Dict, List
import json


class WaymoCoTGenerator:
    """
    Generate CoT traces from Waymo data.
    
    Uses Waymo's perception labels + ego trajectory to generate
    reasoning traces.
    """
    
    def __init__(self, config):
        self.config = config
    
    def generate_cot_for_frame(
        self,
        camera_image,
        lidar_points,
        box_3d_labels,
        ego_pose,
        expert_trajectory
    ) -> Dict:
        """Generate CoT trace for a single frame."""
        
        # 1. Perception: What do I see?
        perception = self._generate_perception(
            box_3d_labels, ego_pose
        )
        
        # 2. Prediction: What will happen?
        prediction = self._generate_prediction(
            box_3d_labels, ego_pose
        )
        
        # 3. Planning: What should I do?
        planning = self._generate_planning(
            ego_pose, expert_trajectory
        )
        
        # 4. Justification: Why?
        justification = self._generate_justification(
            perception, prediction, planning
        )
        
        return {
            "perception": perception,
            "prediction": prediction,
            "planning": planning,
            "justification": justification,
        }
    
    def _generate_perception(self, boxes, ego_pose) -> str:
        """Generate perception description."""
        n_vehicles = len([b for b in boxes if b.label_type == "vehicle"])
        n_pedestrians = len([b for b in boxes if b.label_type == "pedestrian"])
        
        # Distance to nearest objects
        lead_dist = self._find_lead_vehicle_distance(boxes, ego_pose)
        crosswalk_dist = self._find_nearest_crosswalk(boxes, ego_pose)
        
        return (
            f"Ego vehicle traveling at {ego_pose.speed:.1f} m/s. "
            f"Surrounding: {n_vehicles} vehicles, {n_pedestrians} pedestrians. "
            f"Lead vehicle at {lead_dist:.1f}m ahead. "
            f"Nearest crosswalk at {crosswalk_dist:.1f}m. "
            f"Traffic light: {self._get_traffic_light_state(boxes)}."
        )
    
    def _generate_prediction(self, boxes, ego_pose) -> str:
        """Generate prediction description."""
        predictions = []
        
        # Predict lead vehicle trajectory
        lead = self._find_lead_vehicle(boxes, ego_pose)
        if lead:
            predictions.append(
                f"Lead vehicle likely to maintain speed or slow slightly "
                f"(current: {lead.velocity:.1f} m/s)."
            )
        
        # Predict pedestrian intent
        peds = [b for b in boxes if b.label_type == "pedestrian"]
        for ped in peds[:2]:  # Top 2
            if ped.intent in ["walking", "crossing"]:
                predictions.append(
                    f"Pedestrian at ({ped.x:.1f}, {ped.y:.1f}) "
                    f"intent: {ped.intent}."
                )
        
        return " | ".join(predictions) if predictions else "No critical predictions."
    
    def _generate_planning(self, ego_pose, trajectory) -> str:
        """Generate planning description from expert trajectory."""
        # Extract next waypoint from expert trajectory
        next_wp = trajectory[0]  # First future waypoint
        
        # Compute control needed
        steer_angle = self._compute_steering(ego_pose, next_wp)
        accel = self._compute_acceleration(ego_pose, next_wp)
        
        if abs(steer_angle) > 0.1:
            direction = "left" if steer_angle < 0 else "right"
            action = f"Steer {direction} by {abs(steer_angle):.2f} rad"
        else:
            action = "Maintain steering"
        
        if accel > 0.1:
            action += f", accelerate by {accel:.2f} m/s²"
        elif accel < -0.1:
            action += f", brake by {abs(accel):.2f} m/s²"
        else:
            action += ", maintain speed"
        
        return action
    
    def _generate_justification(self, perception, prediction, planning) -> str:
        """Generate justification."""
        return (
            f"Planning decision based on current perception: {perception[:50]}... "
            f"Predicted changes: {prediction[:50]}... "
            f"Action: {planning} balances safety and efficiency."
        )


# Usage
def process_waymo_episodes(episode_dir: str, output_dir: str):
    """Process Waymo episodes and generate CoT traces."""
    generator = WaymoCoTGenerator()
    
    for episode_path in Path(episode_dir).iterdir():
        # Load Waymo data
        boxes = load_3d_boxes(episode_path / "labels.tfrecord")
        ego_pose = load_ego_pose(episode_path / "poses.json")
        camera = load_images(episode_path / "camera_front")
        trajectory = load_expert_trajectory(episode_path)
        
        # Generate CoT for each frame
        for frame_idx in range(len(camera)):
            cot = generator.generate_cot_for_frame(
                camera[frame_idx],
                None,  # LiDAR optional
                boxes[frame_idx],
                ego_pose[frame_idx],
                trajectory[frame_idx]
            )
            
            # Save to JSONL
            output_path = Path(output_dir) / "cot_traces.jsonl"
            with open(output_path, 'a') as f:
                f.write(json.dumps(cot) + "\n")
```

---

## Option 2: Alpamayo-R1 (NVIDIA) - Easiest Start

### Why Alpamayo-R1?

| Aspect | Details |
|--------|---------|
| **Availability** | HuggingFace: `nvidia/Alpamayo-R1-10B` |
| **Format** | VLA model - already outputs actions |
| **蒸馏 Potential** | Community distillation repo exists |
| **CoT Integration** | Model already has reasoning capabilities |

### Quick Start with Alpamayo

```python
# File: training/sft/alpamayo_integration.py

from transformers import AutoModelForVisionLanguage2, AutoProcessor


class AlpamayoCoTExtractor:
    """
    Extract reasoning traces from Alpamayo-R1 model.
    
    Use the model's internal reasoning to generate CoT traces.
    """
    
    def __init__(self, model_name: str = "nvidia/Alpamayo-R1-10B"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVisionLanguage2.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
    
    def extract_reasoning(
        self,
        images: List[torch.Tensor],
        query: str = "Describe the driving scene and plan your next action."
    ) -> Dict[str, str]:
        """Extract reasoning trace from Alpamayo."""
        
        # Process inputs
        inputs = self.processor(
            images=images,
            text=query,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)
        
        # Get model output with reasoning
        outputs = self.model(**inputs, output_attentions=True)
        
        # Extract reasoning from hidden states or attention
        reasoning = self._extract_reasoning_from_output(outputs)
        
        return {
            "reasoning": reasoning,
            "action": self._extract_action(outputs),
        }
    
    def generate_cot_for_training(
        self,
        images: torch.Tensor,
        state: torch.Tensor
    ) -> Dict:
        """Generate training-compatible CoT."""
        query = (
            "You are driving. Analyze the scene:\n"
            "1. What do you see?\n"
            "2. What will other vehicles do?\n"
            "3. What action will you take?\n"
            "4. Why this action?"
        )
        
        result = self.extract_reasoning([images], query)
        
        return {
            "perception": result["reasoning"]["perception"],
            "prediction": result["reasoning"]["prediction"],
            "planning": result["reasoning"]["planning"],
            "justification": result["reasoning"]["justification"],
            "action": result["action"],
        }
```

### Alpamayo Distilled Version (Smaller)

For local training without GPU requirements:

```python
# Use distilled version from: https://github.com/mu-hashmi/alpamayo-r1-distilled

from alpamayo_distilled import AlpamayoDistilled

model = AlpamayoDistilled("alpamayo-distilled-2B")
outputs = model(images, state)
# Get reasoning + actions for training
```

---

## Option 3: nuScenes (Public Dataset)

### nuScenes Advantages

| Aspect | nuScenes |
|--------|----------|
| **Size** | 1.4M labeled objects, 40K keyframes |
| **Download** | Free (requires registration) |
| **Format** | Well-documented, easy to parse |
| **Annotations** | 3D boxes, semantic map, ego pose |

### nuScenes + CoT Pipeline

```python
# nuScenes data format is clean and well-documented

from nuscenes.nuscenes import NuScenes


class NuScenesCoTGenerator:
    """Generate CoT traces from nuScenes data."""
    
    def __init__(self, data_root: str, version: str = "v1.0-trainval"):
        self.nusc = NuScenes(version=version, dataroot=data_root)
    
    def generate_cot_for_sample(self, sample_token: str) -> Dict:
        """Generate CoT for a single nuScenes sample."""
        
        sample = self.nusc.get('sample', sample_token)
        
        # Get perception data
        perception = self._get_perception(sample)
        
        # Get prediction
        prediction = self._get_prediction(sample)
        
        # Get planning from sample_annotation
        planning = self._get_planning(sample)
        
        return {
            "perception": perception,
            "prediction": prediction,
            "planning": planning,
        }
```

---

## Comparison: Which Dataset to Use?

| Dataset | Pros | Cons | Best For |
|---------|------|------|----------|
| **Waymo** | Largest, highest quality | Requires download (TB scale) | Full pipeline |
| **Alpamayo-R1** | VLA model ready, reasoning built-in | Large model (10B) | Quick CoT extraction |
| **nuScenes** | Public, well-documented | Smaller than Waymo | Academic/research |
| **BDD100K** | Diverse, public | Less detailed annotations | Scale testing |

---

## Recommendation

### Phase 1: Start with Alpamayo (Easiest)
1. Download distilled Alpamayo model (smaller)
2. Generate CoT on sample images
3. Fine-tune our model on Alpamayo-generated CoT

### Phase 2: Add nuScenes (Complementary)
1. Download nuScenes (free)
2. Generate CoT using same pipeline
3. Combine with Alpamayo CoT for diversity

### Phase 3: Scale to Waymo (Full Production)
1. Download Waymo (TB scale)
2. Run full CoT generation pipeline
3. Large-scale training

---

## Implementation Roadmap

```
Week 1: Alpamayo Integration
├── Load Alpamayo-R1 distilled model
├── Generate CoT on sample images
├── Validate CoT quality
└── Create training dataset (1K samples)

Week 2: nuScenes Integration  
├── Download nuScenes dataset
├── Adapt CoT generator for nuScenes format
├── Generate CoT (5K samples)
└── Combine with Alpamayo data

Week 3: Waymo Integration (Optional)
├── Download Waymo subset (100 episodes)
├── Run full CoT pipeline
├── Large-scale training experiment
```

---

## Files Created

- `/data/.openclaw/workspace/AIResearch-repo/docs/datasets/real-data-cot-strategy.md` - This document

---

## Summary

**Yes, we can use real data with self-generated CoT labels!**

| Option | Effort | Time to Start |
|--------|--------|---------------|
| Alpamayo-R1 | Low | Today |
| nuScenes | Medium | 1 week |
| Waymo | High | 2-4 weeks |

**Recommended:** Start with Alpamayo distilled model for quick validation, then scale to nuScenes/Waymo for production.
