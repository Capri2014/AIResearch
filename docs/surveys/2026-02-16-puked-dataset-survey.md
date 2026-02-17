# Puked Dataset Survey: Processing for CoT Generation

**Date:** 2026-02-16  
**Source:** https://puked.osglab.com/  
**Status:** Initial Survey

---

## TL;DR

| Aspect | Assessment |
|--------|------------|
| **Data Type** | Appears to be CAN bus + camera driving data |
| **Format** | Likely custom format (React SPA frontend) |
| **Value for CoT** | High - expert driving demonstrations |
| **Processing Needed** | Schema mapping, format conversion |

---

## What We Know

### 1. Site Structure

From analyzing the frontend:
- Single Page Application (React-based)
- Likely handles trip/driving session data
- Has statistics/analysis features (`StatsManager`)
- Manages trip data (`TripsSection`, `TripAnalysisView`)
- Has brand/version management (`BrandVersionManager`)

### 2. Likely Data Components

Based on typical telematics/ driving data platforms:

| Component | Description |
|-----------|-------------|
| **Trips** | Driving sessions (start → end) |
| **CAN Bus Data** | Steering, throttle, brake, speed |
| **GPS** | Location, trajectory |
| **Camera** | Front/rear camera footage |
| **Events** | Harsh braking, acceleration, cornering |
| **Statistics** | Trip metrics, driving scores |

### 3. What We'd Need to Process

```
┌─────────────────────────────────────────────────────────────────┐
│              Data Processing Pipeline for CoT                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw Data (from Puked)                                           │
│  ├── CAN bus (steer, throttle, brake, speed)                  │
│  ├── GPS (lat, lon, heading)                                   │
│  ├── Camera (optional)                                          │
│  └── Events (harsh maneuvers, etc.)                            │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Data Extraction                             │    │
│  │                                                          │    │
│  │  1. Download raw data (CSV/JSON/binary)                 │    │
│  │  2. Parse timestamps and sync CAN/GPS                  │    │
│  │  3. Extract vehicle state at each timestep              │    │
│  │  4. Generate perception-like features from GPS/camera   │    │
│  │                                                          │    │
│  └────────────────────────────┬────────────────────────────┘    │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              CoT-Compatible Format                      │    │
│  │                                                          │    │
│  │  State: {                                                 │    │
│  │    "ego_speed": float,                                   │    │
│  │    "ego_heading": float,                                 │    │
│  │    "ego_position": [lat, lon],                           │    │
│  │    "steering": float,                                    │    │
│  │    "throttle": float,                                    │    │
│  │    "brake": float                                        │    │
│  │  }                                                        │    │
│  │                                                          │    │
│  │  Action: {                                                │    │
│  │    "steer": float,                                       │    │
│  │    "throttle": float,                                    │    │
│  │    "brake": float                                        │    │
│  │  }                                                        │    │
│  │                                                          │    │
│  └────────────────────────────┬────────────────────────────┘    │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              CoT Generation                              │    │
│  │                                                          │    │
│  │  Using the unlabeled data:                               │    │
│  │  1. CAN → Expert action sequences                       │    │
│  │  2. GPS → Road context                                  │    │
│  │  3. Events → Situation markers                          │    │
│  │  4. Generate CoT from behavior patterns                 │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Processing Steps

### Step 1: Data Export

```python
# Step 1: Export data from Puked platform
# Assuming API access or manual export

def export_puked_data(user_id: str, date_range: Tuple[str, str]) -> Dict:
    """
    Export driving data from Puked.
    
    Expected format:
    {
        "trips": [
            {
                "trip_id": "xxx",
                "start_time": "2024-01-01T10:00:00Z",
                "end_time": "2024-01-01T10:30:00Z",
                "can_data": [...],
                "gps_data": [...],
                "events": [...]
            }
        ]
    }
    """
    # Implementation depends on Puked API availability
    pass
```

### Step 2: CAN Bus Processing

```python
# Step 2: Process CAN bus data into standard format

@dataclass
class CANFrame:
    timestamp: float
    steering: float      # Steering angle (-1 to 1 or degrees)
    throttle: float     # Throttle position (0 to 1)
    brake: float        # Brake pressure (0 to 1)
    speed: float        # Vehicle speed (m/s or km/h)
    rpm: float          # Engine RPM


def process_can_data(raw_can: List[Dict]) -> List[CANFrame]:
    """
    Convert raw CAN data to standard format.
    
    Puked likely stores:
    - Steering wheel angle
    - Accelerator pedal position
    - Brake pedal position
    - Vehicle speed
    - Engine RPM
    """
    frames = []
    
    for record in raw_can:
        frame = CANFrame(
            timestamp=record['timestamp'],
            steering=normalize_steering(record['steering']),
            throttle=normalize_throttle(record['throttle']),
            brake=normalize_brake(record['brake']),
            speed=convert_speed(record['speed']),
            rpm=record['rpm']
        )
        frames.append(frame)
    
    return frames
```

### Step 3: GPS to Scene Context

```python
# Step 3: Convert GPS to scene understanding

def gps_to_scene_context(gps_data: List[Dict]) -> Dict:
    """
    Extract scene context from GPS data.
    
    What we can infer:
    - Road type (highway vs urban)
    - Speed limits (from location)
    - Turning patterns
    - Stop locations
    """
    context = {
        "road_type": infer_road_type(gps_data),
        "speed_limit": get_speed_limit(gps_data),
        "turns": detect_turns(gps_data),
        "stops": detect_stops(gps_data),
        "avg_speed": calculate_avg_speed(gps_data),
    }
    
    return context


def infer_road_type(gps_data: List[Dict]) -> str:
    """Infer road type from GPS trajectory."""
    speeds = [d['speed'] for d in gps_data]
    avg_speed = np.mean(speeds)
    
    if avg_speed > 25:  # ~90 km/h
        return "highway"
    elif avg_speed > 15:  # ~54 km/h
        return "urban_arterial"
    else:
        return "local_road"
```

### Step 4: Generate CoT from Expert Behavior

```python
# Step 4: Generate CoT traces from expert driving data

class ExpertBehaviorCoT:
    """
    Generate CoT traces from CAN bus expert demonstrations.
    
    Key insight: If we know what the expert DID,
    we can infer WHY they did it from the situation.
    """
    
    def __init__(self, can_data: List[CANFrame], gps_data: List[Dict]):
        self.can_data = can_data
        self.gps_data = gps_data
        self.scene_context = gps_to_scene_context(gps_data)
    
    def generate_trace(self, frame_idx: int) -> str:
        """Generate CoT trace for a specific frame."""
        frame = self.can_data[frame_idx]
        context = self.scene_context
        
        # Analyze situation
        situation = self._analyze_situation(frame, context)
        
        # Analyze expert action
        action = self._analyze_action(frame)
        
        # Generate reasoning (why did expert do this?)
        reasoning = self._generate_reasoning(situation, action)
        
        return f"Situation: {situation} | Action: {action} | Reasoning: {reasoning}"
    
    def _analyze_situation(self, frame: CANFrame, context: Dict) -> str:
        """Analyze current driving situation."""
        parts = []
        
        # Speed situation
        if frame.speed > context['speed_limit'] * 1.1:
            parts.append(f"speeding ({frame.speed:.1f} > {context['speed_limit']})")
        elif frame.speed < 2:
            parts.append("stopped or crawling")
        else:
            parts.append(f"cruising at {frame.speed:.1f} m/s")
        
        # Turning situation
        if abs(frame.steering) > 0.3:
            direction = "left" if frame.steering < 0 else "right"
            parts.append(f"turning {direction}")
        
        # Acceleration situation
        if frame.throttle > 0.8:
            parts.append("accelerating hard")
        elif frame.brake > 0.2:
            parts.append("braking")
        
        return ", ".join(parts) if parts else "normal driving"
    
    def _analyze_action(self, frame: CANFrame) -> str:
        """Analyze expert's control action."""
        action_parts = []
        
        if frame.throttle > 0.5:
            action_parts.append(f"throttle: {frame.throttle:.2f}")
        if frame.brake > 0.1:
            action_parts.append(f"brake: {frame.brake:.2f}")
        if abs(frame.steering) > 0.1:
            action_parts.append(f"steer: {frame.steering:.2f}")
        
        return ", ".join(action_parts) if action_parts else "maintaining"
    
    def _generate_reasoning(self, situation: str, action: str) -> str:
        """Generate reasoning for why expert took this action."""
        reasoning_templates = {
            "speeding": [
                "adjusting speed for traffic flow",
                "preparing to change lanes",
                "responding to clearing road ahead",
            ],
            "stopped": [
                "waiting at traffic light",
                "stopped behind another vehicle",
                "yielding to pedestrian",
            ],
            "turning": [
                "following road curvature",
                "changing lanes",
                "negotiating intersection",
            ],
            "braking": [
                "maintaining safe following distance",
                "responding to traffic ahead slowing",
                "preparing for stop",
            ],
        }
        
        # Match situation to template
        for key, templates in reasoning_templates.items():
            if key in situation:
                return np.random.choice(templates)
        
        return "maintaining safe, efficient trajectory"
```

### Step 5: Complete Pipeline

```python
# Step 5: Complete data processing pipeline

class PukedDataProcessor:
    """
    Complete pipeline to convert Puked data to CoT-compatible format.
    """
    
    def __init__(self, config: "PipelineConfig"):
        self.config = config
        
        # Processing components
        self.can_processor = CANProcessor()
        self.gps_processor = GPSProcessor()
        self.cot_generator = ExpertBehaviorCoT()
    
    def process_trip(self, trip_data: Dict) -> List[Dict]:
        """
        Process a single trip into CoT-compatible format.
        
        Output format:
        {
            "frame_idx": int,
            "timestamp": float,
            "state": {
                "ego_speed": float,
                "ego_heading": float,
                "ego_position": [lat, lon],
            },
            "action": {
                "steer": float,
                "throttle": float,
                "brake": float,
            },
            "cot_trace": str,
        }
        """
        # 1. Process CAN data
        can_frames = self.can_processor.process(trip_data['can_data'])
        
        # 2. Process GPS data
        gps_records = self.gps_processor.process(trip_data['gps_data'])
        
        # 3. Sync CAN and GPS
        synced_data = self._sync_can_gps(can_frames, gps_records)
        
        # 4. Generate CoT for each frame
        cot_records = []
        for idx, record in enumerate(synced_data):
            cot_trace = self.cot_generator.generate_trace(idx)
            
            cot_records.append({
                "frame_idx": idx,
                "timestamp": record['timestamp'],
                "state": {
                    "ego_speed": record['speed'],
                    "ego_heading": record['heading'],
                    "ego_position": [record['lat'], record['lon']],
                },
                "action": {
                    "steer": record['steering'],
                    "throttle": record['throttle'],
                    "brake": record['brake'],
                },
                "cot_trace": cot_trace,
            })
        
        return cot_records
    
    def batch_process(self, trips: List[Dict]) -> Dataset:
        """Process multiple trips."""
        all_records = []
        
        for trip in trips:
            records = self.process_trip(trip)
            all_records.extend(records)
        
        return Dataset(all_records)
```

---

## Data Schema Mapping

### Puked → Our Schema

| Puked Field | Our Field | Notes |
|--------------|-----------|-------|
| `steering_wheel_angle` | `steer` | May need conversion |
| `accelerator_pedal` | `throttle` | Normalize 0-1 |
| `brake_pedal` | `brake` | Normalize 0-1 |
| `vehicle_speed` | `ego_speed` | May need unit conversion |
| `gps_latitude` | `ego_position[0]` | |
| `gps_longitude` | `ego_position[1]` | |
| `gps_heading` | `ego_heading` | |
| `timestamp` | `timestamp` | Sync all data to this |

---

## What We Need from Puked

### 1. Data Access

| Requirement | Description |
|-------------|-------------|
| **API Access** | REST API to download trip data |
| **Data Format** | JSON, CSV, or binary export |
| **Schema Documentation** | Field names and meanings |
| **Volume Limits** | How much data can we export? |

### 2. Technical Requirements

| Component | Need |
|-----------|------|
| **CAN Bus Fields** | Steering, throttle, brake, speed, RPM |
| **GPS Data** | Lat, lon, heading, accuracy |
| **Timestamps** | Synchronized across all sensors |
| **Sample Rate** | 10 Hz, 20 Hz, or 50 Hz? |
| **Data Size** | Per trip, per day, total available |

### 3. Questions to Ask

```
1. What is the exact data schema/format?
2. What sample rate is CAN data recorded at?
3. Is GPS data included? At what rate?
4. Are there camera recordings available?
5. How is data synchronized across sensors?
6. What is the total data volume available?
7. What is the format for bulk export?
8. Are there any usage restrictions?
```

---

## Value Assessment for CoT

### High Value Components

| Component | Why It's Valuable |
|-----------|-------------------|
| **CAN Bus** | Expert action sequences for imitation learning |
| **GPS Trajectory** | Scene context (road type, speed limits) |
| **Events** | Markers for interesting situations (harsh braking, etc.) |

### Medium Value Components

| Component | Why It's Valuable |
|-----------|-------------------|
| **Timestamps** | Required for temporal analysis |
| **Trip Metadata** | Weather, time of day context |

### Processing Priority

| Priority | Task | Effort | Value |
|----------|------|--------|-------|
| 1 | **CAN bus extraction** | Low | High |
| 2 | **GPS context** | Low | High |
| 3 | **CoT generation** | Medium | High |
| 4 | **Camera integration** | High | Medium |

---

## Summary

### What We'd Need to Do

1. **Get data format documentation** from Puked
2. **Export sample data** to understand actual schema
3. **Build data pipeline** to convert to our format
4. **Generate CoT traces** from expert behavior patterns
5. **Validate** with small sample before bulk processing

### Estimated Effort

| Phase | Duration |
|-------|----------|
| Data exploration & schema mapping | 1 week |
| Pipeline development | 2 weeks |
| CoT generation | 2 weeks |
| Validation & iteration | 1 week |
| **Total** | **~6 weeks** |

### Questions to Resolve

1. What's the exact data format/schema?
2. What data volume is available?
3. What are the usage terms?
4. Is camera data available?

---

## Files Created

- `/data/.openclaw/workspace/AIResearch-repo/docs/surveys/2026-02-16-puked-dataset-survey.md` - This document

## Next Steps

1. **Contact Puked** for data format documentation
2. **Request sample export** (1-2 trips) for schema validation
3. **Build prototype pipeline** with sample data
4. **Evaluate CoT quality** from expert behavior
5. **Scale up** if promising
