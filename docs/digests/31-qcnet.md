# QCNet: Query-Centric Multi-Agent Prediction

**Date:** 2026-03-03  
**Status:** Survey Complete

## Paper

- **Title:** QCNet: Query-Centric Multi-Agent Trajectory Prediction
- **Venue:** CVPR 2023
- **Paper:** https://arxiv.org/abs/2204.08129

## Core Idea

Query-centric architecture for scalable multi-agent prediction:
- **Dynamic queries**: One query set per target agent
- **Hierarchical attention**: Agent → Scene → Agent interaction
- **Scalable**: O(n) scaling with number of agents (not O(n²))
- **Any number of agents**: Handles variable agent counts

## Architecture

```
Input: Multi-agent scene
        ↓
Per-Agent Query Initialization
        ↓
Hierarchical Attention
├── Stage 1: Agent → Map (lane attention)
├── Stage 2: Agent → Agent (interaction)
└── Stage 3: Agent → Temporal (motion)
        ↓
Trajectory Decoder
        ↓
Output: Per-agent future trajectories
```

## Key Innovations

### 1. Query-Centric Design

- Each target agent gets its own query set
- Queries are dynamic (not static/pretrained)
- Query = agent's current state + learned intent

### 2. Hierarchical Attention

- **Agent-to-Map**: Which lanes/roads are relevant?
- **Agent-to-Agent**: How do other agents affect me?
- **Agent-to-Temporal**: How did I move historically?

### 3. Scalability

- Uses Deformable Attention (sparse, not dense)
- Scales linearly with agent count
- Can handle 100+ agents in real-time

## Why It Matters

1. **Scalability**: O(n) vs O(n²) - handles dense traffic
2. **Variable agents**: Works with any number of agents
3. **Efficiency**: Deformable attention = sparse computation

## Comparison

| Method | Scaling | Interaction | Use Case |
|--------|---------|-------------|----------|
| **QCNet** | O(n) | Hierarchical | Dense traffic |
| **MTR** | O(n²) | Full | Medium density |
| **MultiPath++** | O(n) | None | Simple scenes |
| **Scene Transformer** | O(n²) | Full | Complex scenes |

## Implementation Insights

```python
class QCNetLayer(nn.Module):
    def __init__(self):
        self.agent_map_attn = DeformableAttention()
        self.agent_agent_attn = DeformableAttention()
        self.temporal_attn = MultiHeadAttention()
    
    def forward(self, queries, map_features, agent_features, history):
        # Stage 1: Which map elements matter?
        queries = self.agent_map_attn(queries, map_features)
        
        # Stage 2: How do agents interact?
        queries = self.agent_agent_attn(queries, agent_features)
        
        # Stage 3: Temporal motion
        queries = self.temporal_attn(queries, history)
        
        return queries
```

## Results

- SOTA on Argoverse 2.0
- Strong on Waymo Open Motion Dataset
- Real-time inference with 50+ agents

## References

- Paper: https://arxiv.org/abs/2204.08129
