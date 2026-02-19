# DeepSeek Engram Memory MVP Roadmap

**Date:** 2026-02-18  
**Scenario Focus:** Long-horizon memory (e.g., 4-way stop intersection)

---

## The Problem: 4-Way Stop Sign

```
4-Way Stop Requires Memory:
==========================

     Vehicle A ──────┐
                    │
                    ▼
              ┌──────────┐
              │ 4-WAY    │  Queue: [A, B, C, D] (arrival order)
              │  STOP    │  Remember: who arrived when, who went
              └──────────┘
                    ▲
     Vehicle B ──────┘
                    │
     Vehicle C ──────┐
                    │
     Vehicle D ──────┘

Memory requirements:
- Remember arrival order
- Track who has proceeded
- Handle new arrivals
- Timeout/fallback behavior
```

---

## Memory Architecture Comparison

| Approach | Where It Fits | Pros | Cons |
|----------|--------------|------|------|
| Engram | Pre-train + SFT | Efficient O(1) | Complex |
| Transformer Memory | SFT + RL | Natural fit | O(N²) attention |
| External Memory | All stages | Explicit | Engineering overhead |
| N-gram Extension | Pre-train + SFT | Simple | Limited context |

---

## MVP: Memory-Augmented AR Decoder

```
┌─────────────────────────────────────────────┐
│    Memory-Augmented AR Decoder              │
├─────────────────────────────────────────────┤
│                                             │
│  ┌────────────┐                           │
│  │ Scene     │──→ Encoder → [z₀...zₜ]    │
│  │ Encoder   │                           │
│  └────────────┘                           │
│         │                                 │
│         ▼                                 │
│  ┌────────────┐                           │
│  │ Memory    │ ← Query: state + question  │
│  │ Module    │ → Response: relevant       │
│  └────────────┘                           │
│         │                                 │
│         ▼                                 │
│  ┌──────────────────────────┐             │
│  │ AR Decoder (generates   │             │
│  │ waypoints + reasoning)  │             │
│  └──────────────────────────┘             │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Implementation Stages

### Stage 1: Memory Module (Week 1-2)
```
Goal: Add explicit memory state to AR Decoder

├── MemoryState dataclass:
│   ├── queue_order: List[VehicleID]
│   ├── last_action: str
│   ├── timestamp: float
│   └── context: Dict[str, Any]
│
├── MemoryEncoder:
│   ├── encode(state: Dict) → embedding [D]
│   └── decode(embedding [D]) → state Dict
│
└── MemoryRetrieval:
    ├── add(memory: MemoryState)
    ├── retrieve(query: [D], k: int) → memories
    └── update(memory: MemoryState)
```

### Stage 2: Training Data (Week 2-3)
```
Goal: Create data with memory traces

Sources:
├── Synthetic:
│   ├── 4-way stop with queue tracking
│   ├── Roundabout navigation
│   └── Merge scenarios
└── Classical AD:
    ├── nuPlan memory traces
    ├── Waymo motion prediction
    └── CARLA scenario logs

Data format:
{
    "scene_id": "stop_4way_001",
    "memory_trace": [
        {"step": 0, "vehicles": [{"id": "A"}], "queue": ["A"], "action": "waiting"},
        {"step": 5, "vehicles": [{"id": "A"}, {"id": "B"}], "queue": ["A", "B"], "action": "A_proceeding"}
    ],
    "ground_truth_waypoints": [...]
}
```

### Stage 3: SFT with Memory (Week 3-4)
```
Goal: Fine-tune AR Decoder with memory conditioning

Training objective:
L = L_waypoint + λ₁ * L_memory_reconstruct + λ₂ * L_reasoning

- L_waypoint: Standard MSE loss
- L_memory_reconstruct: MSE predicted vs true memory
- L_reasoning: Generate "why I remember X"

Architecture:
Input: [features, memory_embedding, query] → Concat → MLP → AR Decoder
```

### Stage 4: CoT with Memory Reasoning (Week 4-5)
```
Goal: Generate CoT explaining memory usage

Prompt template:
"""
Current scene: 4-way stop
Vehicles: A (arrived 1st), B (arrived 2nd), D (me, 3rd)

Memory trace:
- Step 0: A arrived, waiting
- Step 10: B arrived, queue [A, B]
- Step 20: C arrived, queue [A, B, C]

Question: What should D do?
Reasoning:
1. Queue is [A, B, C]
2. A and B have proceeded
3. It's now D's turn
Answer: Proceed through intersection
"""
```

### Stage 5: RL with Memory Optimization (Week 5-6)
```
Goal: Learn when to use memory vs compute

Reward: R = R_safety + R_progress + R_memory_efficiency

Memory RL algorithm:
1. State: [features, memory_embedding]
2. Action: [use_memory: binary, query: optional]
3. Reward: R(s, action, s')

Option A: Memory as action space
Option B: Memory as part of state (simpler)
```

---

## Integration in Existing Pipeline

```
┌─────────────────────────────────────────────────────┐
│           Complete Training Pipeline               │
├─────────────────────────────────────────────────────┤
│                                                      │
│  PRE-TRAIN                                          │
│  ├── Contrastive learning (SSL encoder)             │
│  └── N-gram language model (Engram pre-train)     │
│                                                      │
│  ↓                                                   │
│                                                      │
│  SFT (Supervised Fine-Tuning)                     │
│  ├── Waypoint BC (current)                       │
│  └── + Memory-augmented SFT (Stage 3) ← NEW    │
│                                                      │
│  ↓                                                   │
│                                                      │
│  COT (Chain of Thought)                          │
│  ├── Reasoning generation (current)                 │
│  └── + Memory CoT (Stage 4) ← NEW               │
│                                                      │
│  ↓                                                   │
│                                                      │
│  RL (Reinforcement Learning)                      │
│  ├── GRPO / SAC / ResAD (current)               │
│  └── + Memory RL (Stage 5) ← NEW               │
│                                                      │
│  ↓                                                   │
│                                                      │
│  EVALUATION                                        │
│  ├── Toy waypoint env                             │
│  └── CARLA ScenarioRunner (memory scenarios)      │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Memory Scenarios for Evaluation

| Scenario | Memory Requirement | Success Criteria |
|----------|-------------------|------------------|
| 4-way stop | Queue order, timestamps | Correct turn-taking |
| Roundabout | Entry order, priority | Merge in order |
| Left turn | Oncoming traffic memory | Wait for gap |
| Parking lot | Spot availability memory | Remember spots |
| Emergency vehicle | Priority queue memory | Correct yielding |

---

## Code Structure

```
training/
├── models/
│   └── memory/
│       ├── __init__.py
│       ├── memory_state.py        # MemoryState
│       ├── memory_encoder.py     # State → embedding
│       ├── memory_retriever.py   # k-NN retrieval
│       └── memory_augmented_decoder.py
├── data/
│   └── memory/
│       ├── __init__.py
│       ├── memory_dataset.py
│       └── scenarios/
│           ├── stop_4way.py
│           ├── roundabout.py
│           └── merge.py
└── sft/
    └── train_memory_sft.py

scripts/
└── generate_memory_data.py     # Synthetic data
```

---

## MVP Checklist

### Week 1-2: Memory Module
- [ ] MemoryState dataclass
- [ ] MemoryEncoder (MLP)
- [ ] MemoryRetriever (k-NN)
- [ ] AR Decoder integration
- [ ] Unit tests

### Week 2-3: Training Data
- [ ] 4-way stop scenario generator
- [ ] Memory trace extraction from nuPlan
- [ ] Dataset format
- [ ] 1000 samples

### Week 3-4: Memory SFT
- [ ] Modify ARCoTDecoder for memory input
- [ ] Training script
- [ ] Baseline comparison
- [ ] ADE/FDE on memory scenarios

### Week 4-5: Memory CoT
- [ ] MemoryCoTDecoder class
- [ ] Reasoning generation
- [ ] Evaluation on 4-way stop
- [ ] Human evaluation

### Week 5-6: Memory RL
- [ ] Memory reward design
- [ ] GRPO/SAC modification
- [ ] Memory efficiency comparison
- [ ] Safety evaluation

---

## Research Questions

1. How long should memory persist?
2. What granularity of memory?
3. When to forget?
4. Memory interference?
5. Distributed vs. centralized?

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Memory not used | Auxiliary loss forcing usage |
| Memory hallucination | Ground truth supervision |
| Scalability | Engram-style sparse retrieval |
| Evaluation | Dedicated memory scenario suite |

---

## References

- DeepSeek Engram: N-gram + Hash lookup
- RETRO: Retrieval-augmented generation
- Memory Networks (Facebook 2015)
- Differentiable Neural Computers (DeepMind 2016)
- nuPlan: Memory traces in planning

---

*Target: Scenarios requiring explicit temporal memory, starting with 4-way stop intersections.*
