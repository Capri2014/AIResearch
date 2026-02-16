# Learning Notes: JEPA and Proposal Head Architecture

This document explains key concepts from the Drive-JEPA inspired implementation.

---

## 1. Contrastive vs JEPA Masked Prediction

### Contrastive Learning (Traditional)

```
Images: [frame_0, frame_1, frame_2, frame_3]
           │
           ▼
       Encoder
           │
           ▼
    Embeddings: [e0, e1, e2, e3]
           │
           ▼
       Loss: maximize similarity(e0, e1) for positive pairs
           │
           ▼
    Problem: Requires negative samples to avoid collapse
```

### JEPA Masked Prediction

```
Images: [frame_0, frame_1, frame_2, frame_3, frame_4]
           │
           ▼
       Encoder
           │
           ▼
    Embeddings: [e0, e1, e2, e3, e4]
           │
           ▼
       Mask 30% randomly:
           [e0, e1,  MASK  , e3, e4]
           │
           ▼
      Predictor (Transformer)
           │
           ▼
    Predicted embedding ≈ Target embedding
           │
           ▼
       MSE Loss only on masked positions
```

### Why JEPA is Better for Driving

| Aspect | Contrastive | JEPA Masked |
|--------|------------|-------------|
| Negatives needed | Yes (~100-1000) | No |
| Learning signal | "These are similar" | "What's missing?" |
| Planning alignment | Indirect | Direct (predict future) |
| Collapse risk | High (needs careful tuning) | Low (prediction forces meaning) |

---

## 2. Proposal Head Architecture

### Full Pipeline

```
Encoder Output: z (B, 128)
                 │
                 ▼
         ┌───────────────┐
         │ Shared Trunk │  ← One MLP processes z once
         └───────────────┘
                 │
         ┌───────┴───────┐
         ▼       ▼       ▼
    ┌───────┐ ┌───────┐ ┌───────┐
    │Decoder│ │Decoder│ │Decoder│  ← K=3 parallel decoders
    └───────┘ └───────┘ └───────┘
         │       │       │
         ▼       ▼       ▼
       prop1   prop2   prop3
       (B,H,2) (B,H,2) (B,H,2)
```

### Why Shared Trunk?

The trunk is a small MLP (128 → 256 → 256) that processes the encoder output once.

**Without shared trunk:**
- Each decoder needs its own MLP to process `z`
- 3× more parameters
- No feature sharing between proposals

**With shared trunk:**
- One MLP processes `z`, shared by all decoders
- More efficient (fewer parameters)
- All proposals benefit from the same transformed features

---

## 3. Training: Minimum-Over-Proposals Loss

The K proposals are trained together from scratch using a special loss function:

### Loss Computation

```
Target: T (B, H, 2) ← Ground truth waypoints

For each proposal k:
    pred_k = Decoder_k(trunk_out)  # (B, H, 2)
    loss_k = MSE(pred_k, T)        # Mean squared error

proposal_losses = [loss_0, loss_1, loss_2, ...]  # Shape: (B, K)

# Only the BEST proposal gets trained
best_loss, best_idx = proposal_losses.min(dim=1)  # (B,)
min_proposal_loss = best_loss.mean()  # Scalar loss for backprop
```

### Why This Works

Over time, different decoders specialize:

| Proposal | Becomes Specialist For |
|----------|----------------------|
| Decoder 1 | Straight highway driving |
| Decoder 2 | Sharp turns |
| Decoder 3 | Lane changes |
| ... | ... |

The **scorer** (optional) learns to predict which proposal is best:

```python
# Scorer takes [trunk_out, proposal] and outputs a score
score_k = Scorer(trunk_out, proposal_k)

# Train scorer with cross-entropy: best proposal should have highest score
scorer_loss = CrossEntropyLoss(scores, best_idx)
```

---

## 4. Jitter Metrics

Jitter measures how much predictions fluctuate between frames:

```python
def compute_jitter(predictions, window=1):
    """
    predictions: List of (H, 2) waypoint arrays
    window: Compare predictions N frames apart
    
    Returns mean L2 change between consecutive predictions.
    """
    deltas = []
    for i in range(len(predictions) - window):
        delta = ||predictions[i] - predictions[i+window]||_2
        deltas.append(delta.mean())
    
    return np.mean(deltas)  # Lower is better (more stable)
```

### Example

```
Frame 0 prediction:  [(0,0), (1,0), (2,0), ...]
Frame 1 prediction:  [(0.1,0), (1.1,0), (2.1,0), ...]
                    ↑
                    └─ Small jitter = stable

Frame 0 prediction:  [(0,0), (1,0), (2,0), ...]
Frame 1 prediction:  [(5,5), (6,5), (7,5), ...]
                    ↑
                    └─ Large jitter = unstable!
```

---

## 5. Summary: How Pieces Connect

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Images ──► Encoder ──► z ──► Shared Trunk ──►           │
│                                      │                      │
│                    ┌────────────────┼────────────────┐     │
│                    ▼                ▼                ▼     │
│              ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│              │Decoder 1│    │Decoder 2│    │Decoder K│  │
│              └─────────┘    └─────────┘    └─────────┘  │
│                    │                │                │     │
│                    ▼                ▼                ▼     │
│                  prop1            prop2           propK  │
│                    │                │                │     │
│                    └────────────────┼────────────────┘     │
│                                     ▼                      │
│                           Minimum-over-proposals            │
│                           Loss (only best proposal)        │
│                                     │                      │
│                                     ▼                      │
│                              ┌──────────┐                  │
│                              │ Scorer   │  (optional)      │
│                              │ (learns  │                  │
│                              │  which   │                  │
│                              │  is best)│                  │
│                              └──────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_proposals` | 5 | K proposals to generate |
| `hidden_dim` | 256 | Trunk/decoder hidden size |
| `proposal_loss_weight` | 1.0 | Weight for trajectory loss |
| `scoring_loss_weight` | 0.5 | Weight for scorer loss |
| `mask_ratio` | 0.3 | For JEPA: fraction of frames to mask |

---

## 7. References

- Drive-JEPA Paper: https://arxiv.org/abs/2601.22032
- V-JEPA: https://arxiv.org/abs/2305.15241
- Implementation: `training/sft/proposal_waypoint_head.py`
