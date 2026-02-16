# GAIA-2: Video-Generative World Model for Autonomous Driving

**Source:** Wayve (https://wayve.ai/thinking/gaia-2/)  
**Technical Report:** https://arxiv.org/abs/2503.20523

---

## TL;DR (5 bullets)

- **Purpose-built world model** for autonomous driving with latent diffusion architecture
- **Multi-camera video generation** with spatiotemporal consistency across viewpoints
- **Rich conditioning**: ego-actions, weather, time-of-day, road semantics, agent behaviors
- **Geographic diversity**: trained on UK, US, Germany driving data
- **Safety-critical scenarios**: synthesizes rare events (cut-ins, emergency braking, OOD conditions)

---

## Problem Statement

| Challenge | GAIA-2's Solution |
|-----------|-------------------|
| Real-world data limited | Generate diverse synthetic scenarios |
| Rare safety events | Precisely control agent positions/behaviors |
| Geographic variation | Multi-country training (UK, US, Germany) |
| Multi-view consistency | Native multi-camera video generation |
| Long-tail coverage | OOD generation with explicit conditioning |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GAIA-2 Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                          │
│  Input: Multi-camera video (past)                         │
│           │                                              │
│           ▼                                              │
│  ┌─────────────────┐                                    │
│  │ Video Tokenizer │  Compresses pixels → latent space   │
│  │  (compresses)  │                                    │
│  └────────┬────────┘                                    │
│           │ latent representation                        │
│           ▼                                              │
│  ┌───────────────────────────────────────────────┐       │
│  │     Latent Diffusion World Model             │       │
│  │                                               │       │
│  │  Conditioning:                              │       │
│  │  • Ego-actions (speed, steering)           │       │
│  │  • Agent 3D bounding boxes                │       │
│  │  • Weather, time-of-day                   │       │
│  │  • Road semantics (lanes, signs, etc.)     │       │
│  │  • External embeddings (CLIP, driving)     │       │
│  └─────────────────┬───────────────────────────┘       │
│                    │                                   │
│                    ▼                                   │
│           Future latent states                          │
│                    │                                   │
│                    ▼                                   │
│  ┌───────────────────────────┐                        │
│  │   Video Decoder         │  Reconstruct pixels    │
│  └───────────────────────────┘                        │
│                                                          │
│  Output: Multi-camera video (future)                   │
│                                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Conditioning Mechanisms

| Category | Parameters | Example Values |
|----------|-----------|---------------|
| **Ego-vehicle** | speed, steering curvature | 0-30 m/s, -0.1 to 0.1 1/m |
| **Agents** | 3D bounding boxes | x, y, z, heading per agent |
| **Environment** | weather, time-of-day | clear/rain/fog, dawn/midday/night |
| **Road** | lanes, speed limits, crossings | # lanes, limit values, boolean flags |

---

## Key Capabilities

### 1. Diverse Video Generation

```
Geographic Coverage:
├── UK: Left-hand traffic, roundabouts
├── US: Right-hand, highway speeds
└── Germany: Autobahn, European signage

Environmental Variation:
├── Weather: Clear, rain, fog
├── Time: Dawn, midday, night
└── Scene: Urban, suburban, highway
```

### 2. Augment Real-World Scenarios

```
Real log: Single weather/lighting condition
         │
         ▼
GAIA-2: Same scenario, different conditions
├── Same agents, different weather
├── Same trajectory, different time-of-day
└── Same road, different traffic density
```

### 3. Action-Conditioned Generation

Specify an action → Generate valid contexts:
- **Braking**: Generate scenarios where braking is appropriate
- **Yielding**: Intersections, pedestrian crossings
- **U-turns**: Appropriate road configurations

### 4. Safety-Critical Scenarios

| Scenario Type | Description |
|--------------|-------------|
| Near-collisions | Pre-collision situations |
| Emergency maneuvers | Harsh braking, swerving |
| Agent OOD | Unexpected behaviors (drifting, obstructions) |

---

## GAIA-1 vs GAIA-2 Comparison

| Aspect | GAIA-1 | GAIA-2 |
|--------|--------|--------|
| Architecture | Autoregressive transformer | Latent diffusion + tokenizer |
| Multi-camera | Limited | Native support |
| Geography | Single region | UK, US, Germany |
| Conditioning | Basic | Rich (ego, agents, weather, road) |
| Controllability | Limited | Fine-grained control |

---

## Comparison with Other World Models

| Model | Approach | GAIA-2 Differences |
|-------|----------|-------------------|
| **DriveArena** | Sim-to-real + physics | GAIA-2: Pure video generation, no physics engine |
| **WorldDreamer** | Transformer-based | GAIA-2: Latent diffusion, driving-specific |
| **SceneDiffuser** | Diffusion (static scenes) | GAIA-2: Temporal consistency, multi-view |
| **Genie 3** | Action-conditional | GAIA-2: Richer conditioning, multi-agent |

**GAIA-2's uniqueness:**
1. Driving-specific conditioning (ego-actions, road semantics)
2. Native multi-camera consistency
3. Safety-critical scenario synthesis
4. Geographic diversity built-in

---

## Practical Applications

### For Training
- Augment rare scenarios (weather, lighting variations)
- Generate diverse agent behaviors
- Create balanced training sets

### For Evaluation
- Systematic coverage of edge cases
- OOD generalization testing
- Benchmark under controlled conditions

### For Data Collection
- Reduce real-world driving hours needed
- Fill gaps in scenario coverage
- Accelerate domain adaptation

---

## Relevance to Our Pipeline

| Our Component | GAIA-2 Integration |
|--------------|------------------|
| **SSL Pretrain** | Could use GAIA-2 generated videos as additional training data |
| **Waypoint BC** | Synthetic scenarios for diverse training conditions |
| **RL Refinement** | Generate safety-critical scenarios for policy training |
| **CARLA Eval** | Complementary: GAIA-2 → CARLA rollout comparison |

---

## Limitations & Open Questions

1. **Realism gap**: How well do synthetic scenarios transfer to real driving?
2. **Computation**: Latent diffusion inference speed vs real-time requirements
3. **Conditioning accuracy**: How precisely can we control agent behaviors?
4. **Scalability**: Training data requirements for multi-geography coverage

---

## Action Items for This Repo

1. **Evaluate GAIA-2** (if accessible) vs our current synthetic data pipeline
2. **Integration option**: Use GAIA-2 as data augmentation for BC training
3. **Comparison**: Run our policies on GAIA-2 generated scenarios vs CARLA

---

## References

- Wayve blog: https://wayve.ai/thinking/gaia-2/
- Technical report: https://arxiv.org/abs/2503.20523
- GAIA-1 (previous version): https://wayve.ai/thinking/scaling-gaia-1/

---

## Dataset QA for Synthetic Data Integration

Before integrating GAIA-2 generated data into our pipeline, implement simple QA checks:

### 1. Camera Coverage Check

```python
def check_camera_coverage(episode):
    """Verify all expected cameras are present."""
    expected_cameras = {"front", "left", "right", "rear"}
    actual_cameras = set(episode.cameras)
    missing = expected_cameras - actual_cameras
    return len(missing) == 0, list(missing)
```

### 2. Missing Frames Check

```python
def check_missing_frames(episode):
    """Detect gaps in frame sequences."""
    timestamps = [f.t for f in episode.frames]
    expected_dt = 0.1  # 10 Hz
    gaps = []
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]
        if abs(dt - expected_dt) > 0.02:  # 20ms tolerance
            gaps.append((i, dt))
    return len(gaps) == 0, gaps
```

### 3. Label Sanity Checks

| Check | Description | Tolerance |
|-------|-------------|-----------|
| Waypoint bounds | Waypoints within road boundaries | ±0.5m |
| Trajectory smoothness | No sudden jumps between waypoints | < 1.0m/step |
| Speed consistency | Reasonable vehicle speeds | 0-35 m/s |
| Heading continuity | Smooth heading changes | < 15°/step |

### 4. GAIA-2 Specific QA

```python
def check_gaia2_conditioning(episode):
    """Verify GAIA-2 conditioning signals are valid."""
    checks = []
    
    # Ego-action validity
    if episode.ego_actions:
        speeds = [a.speed for a in episode.ego_actions]
        if max(speeds) > 35:
            checks.append("speed_exceeds_limit")
    
    # Weather consistency
    if episode.weather:
        if episode.weather.visibility < 10:  # Too foggy
            checks.append("low_visibility")
    
    return len(checks) == 0, checks
```

### QA Summary Table

| Check | GAIA-2 Generated | Real Data | Action |
|-------|-----------------|-----------|--------|
| Camera coverage | ✅ Complete | ✅ Complete | Pass |
| Frame gaps | ✅ None | ⚠️ Possible | Pass / Flag |
| Waypoint bounds | ✅ Validated | ⚠️ Possible | Validate |
| Label sanity | ✅ Controllable | ⚠️ Variable | Validate |
| Geographic diversity | ✅ Configurable | ⚠️ Limited | GAIA-2 wins |

---

## Decision: GAIA-2 for Synthetic Data Augmentation?

### Pros ✅

1. **Unlimited diversity**: Generate any condition (weather, time, location)
2. **Safety-critical scenarios**: Rare events easily synthesized
3. **Geographic coverage**: UK, US, Germany in one model
4. **Multi-camera consistency**: Native support for our 4-camera setup
5. **Action-conditional**: Generate scenarios for specific maneuvers

### Cons ❌

1. **Realism gap**: Synthetic → real transfer unknown
2. **Access**: GAIA-2 not publicly available (proprietary)
3. **Computation**: Latent diffusion inference cost
4. **Label alignment**: Generated labels may differ from real data

### Recommendation

| Use Case | GAIA-2 Fit | Alternative |
|----------|------------|------------|
| BC training augmentation | ⚠️ Medium | Real Waymo preferred |
| RL safety scenarios | ✅ High | GAIA-2 ideal |
| Edge case testing | ✅ High | GAIA-2 ideal |
| Pre-training | ⚠️ Medium | Real data preferred |

**Decision:** Use GAIA-2 for **RL safety scenarios** and **edge case testing**. For BC pre-training, prefer real data due to distribution concerns.

---

## Implementation Roadmap

1. **Phase 1**: Add QA scripts for GAIA-2 data integration
2. **Phase 2**: Generate safety-critical scenarios for RL
3. **Phase 3**: Evaluate transfer from synthetic to real
4. **Phase 4**: Scale synthetic data augmentation if transfer proven

