# MTR (Motion Transformer): Motion Forecasting via Motion Queries

**Date:** 2026-03-03  
**Status:** Survey Complete

## Paper

- **Title:** MTR: Motion Forecasting via Motion Queries
- **Venue:** 2022
- **Paper:** https://arxiv.org/abs/2209.13508

## Core Idea

Query-based motion forecasting with learned motion queries:
- **Motion queries**: K learnable queries that capture driving patterns
- **Query-to-scene attention**: Each query attends to scene context
- **Query-to-query attention**: Queries interact with each other for agent interaction
- **Static + dynamic queries**: Separate query types for different motion patterns

## Architecture

```
Input: Agent history + Map
        ↓
Scene Encoder (CNN/Transformer)
        ↓
Motion Queries (K learnable)
        ↓
Query Attention Layers
├── Query-to-Scene: Attend to map/agents
└── Query-to-Query: Model interactions
        ↓
Trajectory Decoder
        ↓
Output: K trajectories with confidences
```

## Key Innovations

### 1. Learned Motion Queries

- K learnable query vectors
- Each query = a "driving intention" (go straight, turn, lane change)
- Queries learn to represent different motion patterns

### 2. Two-Stage Attention

- **Stage 1**: Query → Scene (attend to map/lanes)
- **Stage 2**: Query → Query (model agent interactions)

### 3. Static + Dynamic Separation

- Static queries: Represent typical lane-following paths
- Dynamic queries: Represent lane changes, turns

## Why It Matters

1. **Query-based**: More expressive than anchors, more efficient than fully generative
2. **Interaction modeling**: Query-to-query attention captures multi-agent interactions
3. **SOTA**: Competitive with anchor-based methods

## Comparison

| Method | Query Type | Interaction | Efficiency |
|--------|-----------|-------------|------------|
| **MTR** | Learned | Query-to-Query | Medium |
| **MultiPath++** | Anchor | None | High |
| **QCNet** | Dynamic | Hierarchical | Medium |
| **Motion Query** | Learned | Scene-only | Medium |

## Implementation Insights

```python
class MotionQuery(nn.Module):
    def __init__(self, K=64, dim=256):
        # K motion queries, each represents a mode
        self.queries = nn.Parameter(torch.randn(K, dim))
    
    def forward(self, scene_features):
        # Query-to-Scene attention
        q2s_output = self.cross_attn(self.queries, scene_features)
        
        # Query-to-Query attention (agent interaction)
        q2q_output = self.self_attn(q2s_output)
        
        # Decode trajectory
        trajectory = self.decoder(q2q_output)
        return trajectory
```

## References

- Paper: https://arxiv.org/abs/2209.13508
