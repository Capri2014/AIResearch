# MultiPath++: Multi-Agent Trajectory Prediction

**Date:** 2026-03-03  
**Status:** Survey Complete

## Paper

- **Title:** MultiPath++: Efficient Multi-Agent Trajectory Prediction
- **Venue:** Waymo Research, 2021
- **Paper:** https://arxiv.org/abs/2110.04040

## Core Idea

Anchor-based multi-modal trajectory prediction:
- **K anchor trajectories** as candidate futures
- **Per-anchor confidence** scoring
- **Per-anchor occupancy** prediction
- **Ensemble** of all anchors for final output

## Architecture

```
Input: Agent history + Scene context
        ↓
Scene Encoder (CNN/Transformer)
        ↓
Anchor Decoder (×K modes)
├── Anchor 1: trajectory + confidence + occupancy
├── Anchor 2: trajectory + confidence + occupancy
└── Anchor K: trajectory + confidence + occupancy
        ↓
Output: K trajectories with probabilities
```

## Key Innovations

### 1. Anchor-Based Prediction

- Pre-defined or learned anchor trajectories
- Each anchor = candidate future path
- More efficient than generating from scratch

### 2. Multi-Modal Output

- K different future trajectories
- Confidence score per mode
- Occupancy prediction for each mode

### 3. Efficiency

- Single forward pass for all K modes
- No autoregressive decoding
- Real-time inference capable

## Why It Matters

1. **Multi-modality**: Captures diverse futures (turn left, go straight, turn right)
2. **Efficiency**: K modes in one pass vs. autoregressive
3. **SOTA**: Strong results on Waymo and Argoverse benchmarks

## Comparison with Other Methods

| Method | Multi-modality | Efficiency | Use |
|--------|---------------|------------|-----|
| **MultiPath++** | Anchor-based | High | K modes, one pass |
| **MTR** | Learned queries | Medium | Query-based |
| **Scene Transformer** | Query-based | Medium | Unified scene |
| **Wayformer** | Latent space | High | Factorized attention |

## Implementation Insights

### Anchor Types

```python
# Option 1: Pre-defined anchors (straight, turn left, turn right)
anchors = [
    [0, 0, 0, 1, 2, 3],  # straight
    [0, 0, 0.5, 1, 1.5, 2],  # turn left
    [0, 0, -0.5, 1, -1.5, -2],  # turn right
]

# Option 2: Learnable anchors
self.anchor_embeddings = nn.Embedding(K, horizon * 2)
```

### Training Loss

```python
# Classification loss (which anchor is best)
ce_loss = F.cross_entropy(anchor_logits, true_mode_idx)

# Regression loss (refine anchor)
reg_loss = F.mse_loss(pred_trajectory, gt_trajectory)

# Total
loss = ce_loss + reg_loss
```

## References

- Paper: https://arxiv.org/abs/2110.04040
- Code: https://github.com/waymo-research/waymo-open-dataset
