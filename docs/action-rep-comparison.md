# Action Representation Comparison

> π0 Flow vs π0-FAST vs ACT vs Diffusion Policy

| Method | Type | Output Shape | Pros | Cons | Robotics Use |
|--------|------|-------------|------|------|--------------|
| **ACT** | Chunked BC | `[T, action_dim]` | Simple, stable | Weak gen | Pick-place |
| **Diffusion Policy** | Denoising trajectory | `[T, action_dim]` | Multimodal | Slow sampling | Complex manipulation |
| **π0 (Flow)** | Flow matching conditional | `[chunk,T,action_dim]` | Continuous control, high freq | Training complex | Dexterous manipulation |
| **π0-FAST** | Discrete token seq AR | `[Tvocab]` | AR infra friendly | Tokenization loss | Needs language foll |

---

## π0 Flow Matching — Key Insight

```
Diffusion:  z_t = α_t * x + σ_t * ε     (stochastic)
Flow:     dz/dt = v(z_t, condition)   (deterministic ODE)

Sampling:
- Diffusion: 50-100 denoise steps
- Flow:      4-10 ODE steps

For robotics: Flow is faster + maintains multimodality via OT coupling.
```

---

## Code Probe (mental)

```python
# Flow matching head in π0
velocity = velocity_net(hidden_state, conditioning)
z_next = z_current + velocity * dt  # 4-10 steps → action chunk
```

## Discrete Tokenization (π0-FAST)

```
Continuous:  a_t ∈ ℝ^d
     ↓ DCT/Frequency transform
Quantize:   a_t → discrete token index
     ↓
AR decode:  autoregressive next-token prediction
```

Tradeoff: 
- Training faster (standard AR loss)
- Inference slower (longer sequence)
- Language following better
- Action precision potentially lost

---

## Decision Guide

| Use Case | Recommended |
|----------|-------------|
| High-frequency dexterous (>30Hz) | π0 Flow |
| Needs LLM-style prompting | π0-FAST |
| Simple pick-place, fixed env | ACT / BC |
| Multimodal recovery | Diffusion Policy |
| Full autonomy, multiple stages | π0.5 pipeline |