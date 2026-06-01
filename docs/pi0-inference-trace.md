# π0 Inference Pipeline — File Reading Order + Mental Model

> Target: 100 lines max, minimal runnable script  
> Repo assumed: https://github.com/Physical-Intelligence/openpi

---

## 📁 Expected File Structure

```
openpi/
├── pi0/
│   ├── __init__.py
│   ├── model/
│   │   ├── pi0_model.py       # ← START HERE: Main model class
│   │   ├── pi0_config.py    # ← Config dataclass
│   │   ├── action_head.py   # Flow matching head
│   │   └── encoders.py     # Vision + state encoders
│   ├── inference/
│   │   ├── inference.py    # ← Entry point
│   │   ├── runner.py       # Batch/streaming runner
│   │   └── client.py     # Robot client interface
│   └── data/
│       ├── dataset.py       # Episode format
│       ├── transforms.py  # Image/state preprocessing
│       └── collate.py    # Batching
├── scripts/
│   └── run_inference.py  # ← RUN THIS
├── configs/
│   ├── pi0_base.yaml
│   ├── ur5e_pickplace.yaml
│   └── ...
└── checkpoints/         # Download from HF
```

---

## 🔢 6-Step Inference Path

### Step 1: Config Loading
```
configs/**/*.yaml → Pi0Config (dataclass)
    ↓
Loads: model_path, obs_keys, chunk_length, action_dim, robot_config
```

### Step 2: Preprocessing
```
raw_image (H,W,3)        → encoder        → [B, Seq, D]
raw_state (q, dq)        → state_encoder → [B, D]
language "pick cup"       → text_tokenizer → [B, L]
```
**Key file:** `pi0/data/transforms.py`

### Step 3: Model Forward
```
Pi0Model.forward(image_tokens, text_tokens, state_tokens)
    ↓
VLM backbone (PaliGemma-style transformer)
    ↓
Action Expert output
```
**Key file:** `pi0/model/pi0_model.py`

### Step 4: Flow Matching Head
```
action_latent = flow_matching_head(hidden_states, conditioning)
    ↓
v_t = velocity_network(z_t, condition)
    ↓
z_{t+k} = z_t + v_t * dt   (4-10 ODE steps)
```
**Key file:** `pi0/model/action_head.py`

### Step 5: Action Chunk Output
```
z_chunk [B, chunk_len, action_dim] → decode → delta_q [B, chunk_len, joint_dim]
    ↓
flattened → send to robot controller
```
**Key file:** `pi0/model/pi0_model.py` (output property)

### Step 6: Client/Server
```
local:      pi0/inference/client.py → sends commands via socket/serial
server:     robot_control_node → PD loop → joint positions
```
**Key file:** `pi0/inference/client.py`

---

## 📖 Minimal Reading Order (Top-Down)

| Order | File | What You Learn |
|-------|------|--------------|
| 1 | `scripts/run_inference.py` | Entry point + CLI args |
| 2 | `pi0/inference/inference.py` | InferencePipeline class |
| 3 | `pi0/model/pi0_model.py` | Pi0Model + forward |
| 4 | `pi0/model/action_head.py` | Flow matching logic |
| 5 | `pi0/model/encoders.py` | Vision/state encoders |
| 6 | `pi0/data/transforms.py` | Input preprocessing |

---

## 🐍 Minimal Runnable Script (~80 lines)

Save as `scripts/pi0_minimal_inference.py`:

```python
#!/usr/bin/env python3
"""Minimal π0 inference — understand the data flow."""

import torch
from dataclasses import dataclass
from pathlib import Path

# ==== Step 1: Config (fake, just for mental model) ====
@dataclass
class Pi0Config:
    model_path: str = "checkpoints/pi0-ur5e-base"
    image_size: tuple = (224, 224)
    chunk_length: int = 7
    action_dim: int = 7  # 6 joints + gripper
    obs_keys: dict = None  # {"image": ..., "state": ..., "language": ...}

    def __post_init__(self):
        self.obs_keys = {"image": "image", "state": "state", "language": "language"}

# ==== Step 2: Mock Preprocessing ====
def preprocess_inputs(image_path, language, robot_state):
    """Mental model of preprocessing."""
    # IMAGE: Load + resize + normalize
    # image_tensor = load(image_path).resize(224,224).normalize()
    print(f"[2] IMAGE: loaded, shape would be [1, 3, 224, 224]")
    
    # LANGUAGE: Tokenize
    # text_tokens = tokenizer(language)
    print(f"[2] LANGUAGE: '{language}' → tokenized")
    
    # STATE: joint positions + velocities
    # state = torch.tensor(robot_state)  # [14] = [q(7) + dq(7)]
    print(f"[2] STATE: {robot_state} → [1, 14]")
    
    return {
        "image": torch.randn(1, 3, 224, 224),
        "language": language,
        "state": torch.randn(1, 14)
    }

# ==== Step 3: Model Forward (mock) ====
def mock_forward(inputs, config):
    """Mental model of forward pass."""
    B = inputs["image"].shape[0]
    print(f"[3] FORWARD:")
    print(f"     image tokens: [B, seq, D]  (VLM backbone)")
    print(f"     text tokens: [B, L]   (embedding)")
    print(f"     state tokens: [B, D]   (state encoder)")
    print(f"     Hidden state after fusion: [B, D]")
    
    # Output: action latent (before flow matching)
    action_logits = torch.randn(B, config.chunk_length, config.action_dim)
    print(f"     action logits: {action_logits.shape}")
    return action_logits

# ==== Step 4: Flow Matching Head (mock) ====
def flow_matching_decode(action_logits, config):
    """Mental model of flow matching → action chunk."""
    B, T, D = action_logits.shape
    print(f"[4] FLOW MATCHING:")
    print(f"     input: action_logits {action_logits.shape}")
    print(f"     ODE solver: 4-10 steps")
    print(f"     velocity_field(z, condition) → z_next")
    print(f"     output: latent chunk [B, T, D]")
    
    # Final: decode to joint deltas
    action_chunk = torch.randn(B, T, config.action_dim)
    print(f"     action_chunk: {action_chunk.shape}  ← Δq per timestep")
    return action_chunk

# ==== Step 5: Output + Robot Interface ====
def to_robot(action_chunk, config):
    """Mental model of robot interface."""
    print(f"[5] OUTPUT:")
    print(f"     action_chunk: {action_chunk.shape}")
    print(f"     flattened: {action_chunk.flatten(1,2).shape}  [B, T*action_dim]")
    print(f"     → PD controller @ 50Hz (20ms per chunk)")
    print(f"     → wait for execution → next observation")

# ==== MAIN ====
def main():
    config = Pi0Config()
    
    print("=" * 60)
    print("π0 INFERENCE PIPELINE (Mental Model)")
    print("=" * 60)
    
    # Inputs
    image_path = "demo/image.jpg"  
    language = "pick up the red cup"
    robot_state = [0.0] * 14  # q(7) + dq(7)
    
    # Pipeline
    inputs = preprocess_inputs(image_path, language, robot_state)
    action_logits = mock_forward(inputs, config)
    action_chunk = flow_matching_decode(action_logits, config)
    to_robot(action_chunk, config)
    
    print("=" * 60)
    print("✓ Mental model complete")
    print("  Next: Replace mocks with real code from OpenPI repo")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## 📊 Mental Model Summary (100 lines max)

| Stage | Input | Output | Key File |
|-------|-------|--------|---------|
| 1. Config | YAML | Pi0Config | `configs/*.yaml` |
| 2. Preprocess | Raw tensors | Encoded tokens | `data/transforms.py` |
| 3. VLM Backbone | Image+Text+State tokens | Hidden state | `model/pi0_model.py` |
| 4. Action Expert | Hidden state | Action logits | `model/action_head.py` |
| 5. Flow Match | Action logits | Action chunk | `model/action_head.py` |
| 6. Robot IF | Action chunk | Commands | `inference/client.py` |

---

## 🚀 Your Day 2 Tasks

| Task | Goal | Deliverable |
|------|------|------------|
| **Task 1** | Run inference | `scripts/pi0_minimal_inference.py` with REAL code |
| **Task 2** | Understand action rep | Comparison table (ACT vs Diff Pol vs π0 vs FAST) |
| **Task 3** | Fine-tune/data | LoRA adapter OR dataset format converter |

---

*Mental model complete. Replace mocks with real OpenPI code.*