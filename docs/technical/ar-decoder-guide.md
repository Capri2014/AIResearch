# ARDecoder & ARCoTDecoder: Comprehensive Documentation

**Date:** 2026-02-17  
**Status:** Complete Technical Documentation  
**Based on:** Q&A Session with Developer

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Training vs Inference](#3-training-vs-inference)
4. [Key Components Explained](#4-key-components-explained)
5. [Code Walkthrough](#5-code-walkthrough)
6. [FAQ: Common Questions](#6-faq-common-questions)
7. [Usage Examples](#7-usage-examples)

---

## 1. Overview

### What is ARDecoder?

**ARDecoder** is an **autoregressive decoder** that generates waypoints one at a time, each conditioned on previous predictions.

```
Input:  Features [B, feature_dim]  (from image encoder, state encoder, etc.)
Output: Waypoints [B, T, 3]         (x, y, heading for T timesteps)
```

### What is ARCoTDecoder?

**ARCoTDecoder** extends ARDecoder by adding **Chain-of-Thought (CoT) reasoning** conditioning.

```
Input:  Features [B, feature_dim] + CoT tokens [B, L]
Output: Waypoints [B, T, 3] + Explanation [B, cot_dim]
```

### Why Autoregressive?

| Approach | Pros | Cons |
|----------|------|-------|
| **Parallel** (non-AR) | Fast, all at once | Can't capture dependencies |
| **Autoregressive** | Captures sequential dependencies, variable length | Slower (one at a time) |

---

## 2. Architecture

### ARDecoder Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AR Decoder Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Features [B, 768]                                             │
│        ↓                                                        │
│  feature_proj (Linear 768→256)                                  │
│        ↓                                                        │
│  memory [B, 256]                                               │
│        ↓ unsqueeze(1)                                           │
│  memory [B, 1, 256] ← CROSS-ATTENTION KEY/VALUE               │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  START TOKEN: tgt = zeros [B, 1, 256]                    │  │
│  │                                                           │  │
│  │  FOR t in range(T):                                       │  │
│  │    1. Add positional encoding to tgt                       │  │
│  │    2. Self-attention: tgt attends to tgt                  │  │
│  │    3. Cross-attention: tgt attends to memory             │  │
│  │    4. FFN processing                                      │  │
│  │    5. Predict waypoint wpₜ                                │  │
│  │    6. Append to tgt for next iteration                    │  │
│  │                                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│        ↓                                                        │
│  Waypoints [B, T, 3]                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### ARCoTDecoder Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   ARCoTDecoder Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  Features       │    │  CoT Tokens    │                    │
│  │  [B, 768]      │    │  [B, L]        │                    │
│  └────────┬────────┘    └────────┬────────┘                    │
│           │                       │                              │
│           │                       ▼                              │
│           │              cot_embedding (Embedding)                │
│           │                       │                              │
│           │                       ▼                              │
│           │              cot_encoder (LSTM)                      │
│           │                       │                              │
│           │                       ▼                              │
│           │              cot_embedding [B, 256]                  │
│           │                       │                              │
│           └───────────┬───────────┘                              │
│                       ▼                                         │
│              ┌────────────────┐                                 │
│              │ Concatenate    │                                 │
│              │ [B, 768+256]  │                                 │
│              └───────┬────────┘                                 │
│                      ▼                                          │
│              feature_fusion (Linear)                            │
│                      ↓                                          │
│              fused [B, 256]                                     │
│                      ↓                                          │
│              ┌──────────────────────────────────────────────┐   │
│              │  AR Decoder (same as ARDecoder)              │   │
│              │  - Positional encoding                       │   │
│              │  - Transformer decoder layers (6×)            │   │
│              │  - Self-attention + Cross-attention          │   │
│              │  - FFN                                     │   │
│              └─────────────────────┬────────────────────────┘   │
│                                    ↓                            │
│              Waypoints [B, T, 3] + Explanation [B, cot_dim]   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Training vs Inference

### Training: Teacher Forcing with Causal Mask

During training, we use **teacher forcing** - we feed the correct waypoints and use a **causal mask** to prevent looking at future positions.

```python
# ARDecoder.forward() during training
if waypoints is not None:
    # 1. Embed waypoints
    tgt = self.wp_embedding(waypoints)  # [B, T, 3] → [B, T, hidden_dim]
    
    # 2. Add positional encoding
    tgt = self.pos_encoding(tgt)
    
    # 3. Apply causal mask (can't see future)
    tgt_mask = self.causal_mask[:T, :T]
    
    # 4. Pass through decoder layers
    for layer in self.decoder:
        tgt = layer(tgt, memory, tgt_mask)
    
    # 5. Predict
    waypoint_preds = self.waypoint_head(tgt)
```

### Causal Mask Visualization

For T=4 waypoints, the causal mask looks like:

```
                    To (column)
              0      1      2      3
          ┌──────┬──────┬──────┬──────┐
      0   │  ✓    │  ✗    │  ✗    │  ✗    │
          ├──────┼──────┼──────┼──────┤
F   1    │  ✓    │  ✓    │  ✗    │  ✗    │
r   2    │  ✓    │  ✓    │  ✓    │  ✗    │
o   3    │  ✓    │  ✓    │  ✓    │  ✓    │
m          └──────┴──────┴──────┴──────┘
(row)

Position t can only attend to positions 0...t
```

### Inference: Autoregressive Generation

During inference, there's no future to look at, so no mask is needed.

```python
# ARDecoder.generate() - inference mode
tgt = torch.zeros(B, 1, hidden_dim)  # Start token

for t in range(max_steps):
    # 1. Add positional encoding
    pos_embed = self.wp_embedding.embedding[t:t+1]
    tgt_with_pos = tgt + pos_embed
    
    # 2. Apply decoder layers (no mask needed!)
    for layer in self.decoder:
        tgt_with_pos = layer(tgt_with_pos, memory, tgt_mask=None)
    
    # 3. Predict waypoint
    wp = self.waypoint_head(tgt_with_pos[:, -1, :])
    all_waypoints.append(wp)
    
    # 4. Append to tgt for next iteration
    tgt = torch.cat([tgt, wp.unsqueeze(1)], dim=1)

waypoints = torch.cat(all_waypoints, dim=1)
```

### Comparison Table

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Input** | `waypoints` provided | No `waypoints` |
| **Mask** | Causal mask active | No mask needed |
| **Mode** | Parallel (teacher forcing) | Sequential (one at a time) |
| **Speed** | Fast (all at once) | Slower (T steps) |

---

## 4. Key Components Explained

### 4.1 Feature Projection (`feature_proj`)

```python
self.feature_proj = nn.Linear(feature_dim, hidden_dim)
```

**Purpose:** Change feature dimension to match decoder's hidden dimension.

```
Input:  features [B, 768]  (from SSL encoder)
Output: memory [B, 256]     (decoder hidden size)
```

### 4.2 Memory Block

**Purpose:** Conditioning signal - what the decoder attends to during generation.

```
memory [B, 1, hidden_dim] ← CROSS-ATTENTION KEY/VALUE

The decoder's cross-attention uses memory as the "what to attend to" signal.
```

### 4.3 Waypoint Embedding (`wp_embedding`)

**Purpose:** Project waypoints from `[x, y, heading]` to decoder hidden dimension, plus add positional information.

```python
# What wp_embedding does:
waypoints [B, T, 3]           # Raw waypoints
       ↓ wp_embedding.proj     # Linear: 3→256
       ↓
projected [B, T, 256]
       ↓ + positional encoding
       ↓
embedded [B, T, 256]
```

### 4.4 Positional Encoding

**Purpose:** Tell the model the position of each waypoint in the sequence.

Two types supported:

| Type | Description | When to Use |
|------|-------------|-------------|
| **Learned** | `nn.Embedding` for positions | Default, most flexible |
| **Sinusoidal** | Fixed sin/cos encoding | Simpler, no extra params |

### 4.5 Transformer Decoder Layers

Each layer has three sub-layers:

```python
class ARTransformerDecoderLayer(nn.Module):
    def forward(self, tgt, memory, tgt_mask=None):
        # 1. Self-attention: Waypoints attend to each other
        attn_out = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + attn_out)
        
        # 2. Cross-attention: Waypoints attend to memory (images)
        cross_out = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + cross_out)
        
        # 3. Feed-forward network
        ffn_out = F.relu(self.linear1(tgt))
        ffn_out = self.linear2(ffn_out)
        tgt = self.norm3(tgt + ffn_out)
        
        return tgt
```

**What each does:**

| Operation | Purpose |
|-----------|---------|
| **Self-attention** | Waypoint 3 can "see" waypoints 0, 1, 2 |
| **Cross-attention** | Each waypoint can "see" the image features |
| **FFN** | Non-linear transformation |

### 4.6 Causal Mask

**Purpose:** During training, prevent the model from "cheating" by looking at future waypoints.

```python
# Creates triangular mask
self.register_buffer(
    "causal_mask",
    torch.triu(torch.ones(max_len, max_len), diagonal=1).bool().T,
)
```

---

## 5. Code Walkthrough

### 5.1 ARDecoderConfig

```python
@dataclass
class ARDecoderConfig:
    # Feature dimensions
    feature_dim: int = 768   # Input feature dimension
    hidden_dim: int = 512    # Hidden dimension
    
    # Waypoint settings
    num_waypoints: int = 20
    waypoint_dim: int = 3    # x, y, heading
    
    # AR settings
    max_decode_steps: int = 20
    use_embedding: bool = True  # Use learned waypoint embeddings
    
    # Architecture
    num_layers: int = 6       # Number of decoder layers
    num_heads: int = 8       # Attention heads
    dropout: float = 0.1
```

### 5.2 ARDecoder Forward

```python
def forward(
    self,
    features: torch.Tensor,
    waypoints: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Forward pass for AR decoder.
    
    Training: Use teacher forcing (parallel decoding)
    Inference: Autoregressive generation
    
    Args:
        features: [B, feature_dim] conditioning features
        waypoints: [B, T, waypoint_dim] target waypoints (optional)
        
    Returns:
        - waypoints: [B, T, 3] predicted waypoints
        - embeddings: [B, T, hidden_dim] intermediate embeddings
    """
    B = features.size(0)
    T = self.config.num_waypoints
    
    # 1. Project features to hidden dim
    memory = self.feature_proj(features)  # [B, hidden_dim]
    memory = memory.unsqueeze(1)         # [B, 1, hidden_dim]
    
    if waypoints is not None:
        # === TRAINING MODE ===
        # Embed waypoints
        tgt = self.wp_embedding(waypoints)  # [B, T, hidden_dim]
        tgt = self.pos_encoding(tgt)
        
        # Apply causal mask
        tgt_mask = self.causal_mask[:T, :T]
        
        # Pass through decoder layers
        for layer in self.decoder:
            tgt = layer(tgt, memory, tgt_mask)
        
        # Predict waypoints
        waypoint_preds = self.waypoint_head(tgt)
        embeddings = tgt
        
    else:
        # === INFERENCE MODE ===
        waypoint_preds = self.generate(features)
        embeddings = None
    
    return {
        'waypoints': waypoint_preds,
        'embeddings': embeddings,
    }
```

### 5.3 ARDecoder Generate (Inference)

```python
def generate(
    self,
    features: torch.Tensor,
    max_steps: Optional[int] = None,
) -> torch.Tensor:
    """Autoregressive generation of waypoints."""
    B = features.size(0)
    max_steps = max_steps or self.config.num_waypoints
    
    # Project features
    memory = self.feature_proj(features)
    memory = memory.unsqueeze(1)
    
    # Initialize with start token
    tgt = torch.zeros(B, 1, self.config.hidden_dim, device=features.device)
    
    # Generate waypoints one by one
    all_waypoints = []
    
    for t in range(max_steps):
        # Add positional encoding
        pos_embed = self.wp_embedding.embedding[t:t+1]
        tgt_with_pos = tgt + pos_embed
        
        # Apply decoder layers
        for layer in self.decoder:
            tgt_with_pos = layer(tgt_with_pos, memory, tgt_mask=None)
        
        # Predict from last position
        last_hidden = tgt_with_pos[:, -1, :]
        wp = self.waypoint_head(last_hidden)
        all_waypoints.append(wp)
        
        # Append to sequence
        tgt = torch.cat([tgt, last_hidden.unsqueeze(1)], dim=1)
    
    return torch.cat(all_waypoints, dim=1)
```

### 5.4 ARCoTDecoder Forward

```python
def forward(
    self,
    features: torch.Tensor,
    cot_input: Optional[torch.Tensor] = None,
    cot_mask: Optional[torch.Tensor] = None,
    waypoints: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Forward pass for AR + CoT decoder.
    
    Args:
        features: [B, feature_dim] image features
        cot_input: [B, L] CoT token IDs (optional)
        cot_mask: [B, L] attention mask (optional)
        waypoints: [B, T, waypoint_dim] target waypoints (optional)
    """
    B = features.size(0)
    
    # === 1. ENCODE CoT ===
    if cot_input is not None:
        cot_input = self.cot_embedding(cot_input)  # [B, L] → [B, L, cot_dim]
        cot_out, (h_n, c_n) = self.cot_encoder(cot_input)
        cot_embedding = h_n[-1, :, :]               # [B, hidden_dim]
    else:
        cot_embedding = torch.zeros(B, self.config.hidden_dim, device=features.device)
    
    # === 2. FUSE Features + CoT ===
    fused = torch.cat([features, cot_embedding], dim=-1)
    fused = self.feature_fusion(fused)  # [B, hidden_dim]
    
    # === 3. AR DECODE ===
    if waypoints is not None:
        # Training mode
        ar_output = self.ar_decoder(fused, waypoints)
        waypoints_pred = ar_output['waypoints']
        embeddings = ar_output.get('embeddings')
    else:
        # Inference mode
        waypoints_pred = self.ar_decoder.generate(fused)
        embeddings = None
    
    # === 4. GENERATE EXPLANATION ===
    if self.config.include_explanation:
        explanation = self.explanation_head(cot_embedding)
    else:
        explanation = None
    
    return {
        'waypoints': waypoints_pred,
        'explanation': explanation,
        'cot_embedding': cot_embedding,
        'embeddings': embeddings,
    }
```

---

## 6. FAQ: Common Questions

### Q1: What does `B` mean?

**Answer:** `B` is **batch size** - the number of samples processed together.

```
B = 32  (32 driving scenarios in one forward pass)

images:     [B, C, H, W] = [32, 3, 224, 224]
waypoints:  [B, T, 3]    = [32, 10, 3]
```

### Q2: What does `memory` represent?

**Answer:** `memory` is the **conditioning signal** - what the decoder uses to generate waypoints. It contains:

| Component | What it provides |
|-----------|-----------------|
| Image features | What the car "sees" |
| CoT embedding | Reasoning context |
| State features | Speed, heading, etc. |

### Q3: Why stack multiple decoder layers?

**Answer:** Each layer refines the representations. Think of it like:

- **Layer 0**: "Basic understanding"
- **Layer 1-5**: "Refining and combining information"

```python
self.decoder = nn.ModuleList([layer] * 6)  # 6 identical layers
```

### Q4: What does `causal_mask` do?

**Answer:** During training, it prevents the model from "cheating" by looking at future waypoints.

```
Position 0: Can only see position 0
Position 1: Can see positions 0, 1
Position 2: Can see positions 0, 1, 2
```

### Q5: Why use `detach()` when appending to `tgt`?

```python
tgt = torch.cat([tgt, last_hidden.detach()], dim=1)
```

**Answer:** We don't want gradients flowing through previous hidden states. Only the current step's loss matters for backprop.

### Q6: What's the difference between ARDecoder and ARCoTDecoder?

| Aspect | ARDecoder | ARCoTDecoder |
|--------|-----------|--------------|
| **CoT** | ✗ No | ✓ Yes |
| **Conditioning** | Features only | Features + CoT |
| **Explanation head** | ✗ No | ✓ Yes |
| **Output** | Waypoints only | Waypoints + Explanation |

### Q7: When should I use ARCoTDecoder instead of ARDecoder?

**Use ARCoTDecoder when:**
- You have reasoning traces (CoT) available
- You want interpretable explanations
- Reasoning improves prediction quality

**Use ARDecoder when:**
- You don't have CoT data
- Speed is critical
- Simplicity is preferred

### Q8: Why start with a start token instead of empty sequence?

**Answer:** An empty sequence can't be processed by the decoder (no positions to attend to). The start token provides an initial state.

```python
# Wrong (empty sequence):
tgt = torch.zeros(B, 0, hidden_dim)  # Can't process!

# Correct (start token):
tgt = torch.zeros(B, 1, waypoint_dim)
tgt = self.wp_embedding(tgt)  # [B, 1, hidden_dim]
```

### Q9: What's the purpose of positional encoding?

**Answer:** Waypoints are just `[x, y, heading]` numbers - they don't have inherent order. Positional encoding tells the model "this is waypoint #3 out of #10".

### Q10: How does cross-attention use `memory`?

**Answer:** The decoder's cross-attention operation:

```
Query (Q): tgt - "What I'm generating"
Key (K):   memory - "What I know about the world"
Value (V): memory - "Content to attend to"

Output: tgt with information from memory incorporated
```

---

## 7. Usage Examples

### 7.1 Basic ARDecoder

```python
import torch
from training.sft.ar_decoder import ARDecoder, ARDecoderConfig

# Create model
config = ARDecoderConfig(
    feature_dim=768,
    hidden_dim=256,
    num_waypoints=10,
)
decoder = ARDecoder(config)

# Training mode
features = torch.randn(2, 768)      # [B, feature_dim]
target_waypoints = torch.randn(2, 10, 3)  # [B, T, waypoint_dim]

output = decoder(features, target_waypoints)
print(f"Predicted: {output['waypoints'].shape}")  # [2, 10, 3]

# Inference mode
with torch.no_grad():
    waypoints = decoder.generate(features)
print(f"Generated: {waypoints.shape}")  # [2, 10, 3]
```

### 7.2 ARCoTDecoder with CoT

```python
import torch
from training.sft.ar_decoder import ARCoTDecoder, ARCoTConfig

# Create model
config = ARCoTConfig(
    feature_dim=768,
    hidden_dim=256,
    cot_dim=256,
    num_waypoints=10,
    cot_encoder_type='lstm',
    include_explanation=True,
)
ar_cot = ARCoTDecoder(config)

# Inputs
features = torch.randn(2, 768)           # [B, feature_dim]
cot_tokens = torch.randint(0, 100, (2, 20))  # [B, L]
target_waypoints = torch.randn(2, 10, 3)      # [B, T, waypoint_dim]

# Training with CoT
output = ar_cot(features, cot_tokens, waypoints=target_waypoints)
print(f"Waypoints: {output['waypoints'].shape}")    # [2, 10, 3]
print(f"Explanation: {output['explanation'].shape}")  # [2, 256]

# Inference with CoT
with torch.no_grad():
    output = ar_cot.generate(features, cot_tokens)
print(f"Generated: {output['waypoints'].shape}")    # [2, 10, 3]
```

### 7.3 Integration with SFT Trainer

```python
from training.sft.train_waypoint_bc_cot import SFTCoTConfig, ARDecoderWrapper

# Create config with AR decoder
config = SFTCoTConfig(
    use_ar_decoder=True,
    ar_hidden_dim=256,
    ar_max_waypoints=10,
    cot_encoder_type='lstm',
)

# Use wrapper for seamless integration
wrapper = ARDecoderWrapper(config, use_cot=True)

# Forward pass
fused_features = torch.randn(2, 256)
cot_tokens = torch.randint(0, 100, (2, 20))
waypoints = torch.randn(2, 10, 3)

output = wrapper(fused_features, waypoints, cot_tokens)
print(f"Waypoints: {output['waypoints'].shape}")  # [2, 10, 3]
```

---

## Summary

| Component | Role |
|-----------|------|
| **ARDecoder** | Autoregressive waypoint generator |
| **ARCoTDecoder** | AR decoder + CoT conditioning |
| **feature_proj** | Dimension projection (feature_dim → hidden_dim) |
| **memory** | Conditioning signal for cross-attention |
| **wp_embedding** | Project waypoints + add positional encoding |
| **causal_mask** | Prevent looking at future positions (training only) |
| **Transformer layers** | Self-attention + cross-attention + FFN |

---

## References

- **Paper:** "Attention Is All You Need" (Vaswani et al., 2017)
- **Code:** `training/sft/ar_decoder.py`
- **Related:** `training/sft/train_waypoint_bc_cot.py`
