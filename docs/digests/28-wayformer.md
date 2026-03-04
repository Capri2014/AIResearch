# Wayformer: Motion Forecasting via Simple & Efficient Attention Networks

**Date:** 2026-03-03  
**Status:** Survey Complete

## Paper

- **Title:** Wayformer: Motion Forecasting via Simple & Efficient Attention Networks
- **Venue:** Waymo Research, NeurIPS 2022
- **Paper:** https://arxiv.org/abs/2211.17141
- **Website:** https://waymo.com/research/wayformer/

## Core Idea

Simple & efficient attention-based architecture for motion forecasting:
- **Homogeneous design**: Single attention-based encoder + decoder (no modality-specific modules)
- **Three fusion strategies**: Early, Late, Hierarchical
- **Efficiency techniques**: Factorized attention, Latent query attention

## Architecture

```
Input (Multi-modal)
├── Agent history (temporal)
├── Map/lane polylines (spatial)
└── Traffic lights (temporal)
        ↓
Scene Encoder (Attention-based)
├── Early Fusion: All modalities in one encoder
├── Late Fusion: Separate encoder per modality
└── Hierarchical Fusion: Combine both
        ↓
Decoder (Cross-attention)
        ↓
Output: Multi-modal trajectory prediction
```

## Key Innovations

### 1. Fusion Strategies

| Strategy | Description | Pros/Cons |
|----------|-------------|-----------|
| **Early Fusion** | All inputs processed together | ✅ Simple, modality-agnostic, SOTA |
| **Late Fusion** | Separate encoder per modality | ✅ Modality-specific, but complex |
| **Hierarchical** | Combines early + late | ⚡ Balance |

### 2. Efficiency Techniques

- **Factorized Attention**: Separate spatial + temporal attention (reduces O(n²))
- **Latent Query Attention**: Learnable latent queries compress scene

### 3. Results

- **SOTA on Waymo Open Motion Dataset (WOMD)**
- **SOTA on Argoverse leaderboard**
- Better quality/efficiency trade-off than complex modality-specific designs

## Why It Matters

1. **Simplicity**: One homogeneous architecture vs. complex hand-designed modules
2. **Efficiency**: Factorized/latent attention enable real-time inference
3. **Effectiveness**: Early fusion achieves SOTA despite simplicity

## Implementation Insights

### For Our Pipeline

```python
# Wayformer-style scene encoder
class WayformerEncoder(nn.Module):
    def __init__(self, fusion_type="early"):
        self.scene_encoder = TransformerEncoder(
            num_layers=6,
            dim=256,
            attention="factorized"  # or "latent"
        )
        self.fusion_type = fusion_type  # early/late/hierarchical
    
    def forward(self, agent_history, map_features, traffic):
        # Fuse multi-modal inputs
        if self.fusion_type == "early":
            x = torch.cat([agent_history, map_features, traffic], dim=1)
            return self.scene_encoder(x)
        # ... late/hierarchical variants
```

### Key Hyperparameters

- `dim`: 256 (feature dimension)
- `num_layers`: 6 (encoder layers)
- `num_queries`: 64 (output trajectories)
- `factorized`: Spatial + temporal separation

## Comparison with Scene Transformer

| Aspect | Scene Transformer | Wayformer |
|--------|-----------------|-----------|
| Attention | Full attention | Factorized/latent |
| Fusion | Agent + map queries | Early/late/hierarchical |
| Complexity | Higher | Lower |
| Efficiency | Slower | Faster (real-time) |
| Performance | Good | SOTA |

## References

- Paper: https://arxiv.org/abs/2211.17141
- Waymo Blog: https://waymo.com/research/wayformer/
