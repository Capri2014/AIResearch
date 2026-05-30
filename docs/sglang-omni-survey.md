# SGLang Omni: Inference Framework for Multi-Stage Generative Models

A Comprehensive Technical Survey

---

## 1. Introduction & Core Motivation

### 1.1 What It Is and Why It Matters

**SGLang Omni** is a specialized inference framework extension of SGLang, designed specifically for **multi-stage generative models**. While the original SGLang excels at single-stage autoregressive decoding (standard LLM/VLM inference), SGLang Omni addresses a growing class of models whose decoding processes span multiple heterogeneous stages.

The key insight from the SGLang Omni team: Rather than classifying models by input/output modality (which is superficial), classify by **computational characteristics** — specifically, whether the decoding process is multi-stage.

### 1.2 The Problem It Solves

Traditional inference frameworks like SGLang main handle:

- Single-stage autoregressive decoding (standard LLM)
- Prefill/decode disaggregation
- Continuous batching
- KV cache management

**But they struggle with:**

- Multi-stage models where different stages have wildly different:
  - Compute intensity (compute-bound vs. memory-bound)
  - Latency requirements
  - Memory access patterns
  - Inter-stage dependency patterns

Examples of multi-stage models that motivated SGLang Omni:

| Model | Stages | Type |
|-------|--------|------|
| **Qwen3-Omni** | Thinker → Talker → MTP | Text + Speech |
| **FishAudio S2 Pro** | Slow AR → Fast AR | TTS |
| **Ming Omni** | AR backbone → Diffusion | Fully omni-modal |
| **LLaDA Uni** | AR → Diffusion | Fully omni-modal |

---

## 2. Understanding Omni Models

### 2.1 What Makes a Model "Omni"?

There are competing definitions:

| Definition | Examples | Input | Output |
|------------|----------|-------|--------|
| **VLM + Audio input** | MiMo Omni, Nemotron Omni | Audio, video, image, text | Text |
| **+ Audio output** | Qwen3-Omni, GPT-4o | Voice + text + video + image | Text + voice |
| **Fully omni-modal** | Ming Omni, LLaDA Uni | All modalities | All modalities |

**SGLang Omni's position:** Forget modality. Focus on the computation. If decoding is multi-stage → SGLang Omni target.

### 2.2 Qwen3-Omni: The Canonical Example

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                QWEN3-OMNI MULTI-STAGE DECODING                         │
├─────────────────────────────────────────────────────────────────────�───────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                      │
│  │ THINKER  │───▶│ TALKER   │───▶│   MTP    │                      │
│  │  (AR)    │    │  (AR)    │    │ (parallel│                      │
│  │          │    │          │    │ codec    │                      │
│  │ Generates│    │ Generates│    │complete.│                      │
│  │ text     │    │ 0th codec│    │         │                      │
│  │ tokens   │    │ token    │    │         │                      │
│  └──────────┘    └──────────┘    └──────────┘                      │
│       │               │               │                               │
│       ▼               ▼               ▼                               │
│   Standard         Light AR        Tiny prefill                      │
│   LLM decode      each step      + embedding                      │
│   loop                                               write-back                    │
│                                                                      │
│  Pipeline: Thinker ───async──▶ Talker ───sync──▶ MTP                │
│  (tokens + hidden     (tight feedback)                             │
│   states buffer)                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Decoding flow:**

1. **Thinker** autoregressively generates text tokens
2. **Talker** autoregressively generates the 0-th codec token for each timestep
3. **MTP (Multi-Token Prediction)** completes remaining codec tokens in parallel
4. Repeat until complete

### 2.3 Not Just Omni — TTS Models Too

Multi-stage decoding is ubiquitous in speech generation:

| Model | Stage 1 | Stage 2 | Type |
|--------|---------|---------|------|
| **FishAudio S2 Pro** | ~4B Slow AR (semantic tokens) | ~400M Fast AR (acoustic tokens) | Serial nesting |
| **Qwen3-TTS** | AR backbone | Codec completion | Serial |
| **Voxtral** | AR | Codec completion | Serial |
| **Higgs** | AR | Codec completion | Serial |

**Pattern:** Almost all audio-output models use AR backbone + codec completion → multi-stage by default.

---

## 3. Computational Characteristics

### 3.1 Three Distinct Execution Paradigms

| Stage | Compute Pattern | Bottleneck | Goal |
|-------|----------------|-----------|------|
| **Thinker (standard LLM)** | Standard AR decode | KV cache memory bandwidth | Maximize TPOT + throughput |
| **Talker (light AR)** | Light attention, short context | Neither compute nor memory | Minimize latency |
| **MTP (parallel completion)** | Tiny prefill, few tokens | Kernel launch overhead | Minimize sync latency |

### 3.2 Detailed Characterization

#### Thinker (Standard Autoregressive)

```
Compute:     Prefill = compute-bound
             Decode = memory-bound

Key operations:
- Large GEMMs (attention, FFN)
- Long KV cache reads
- Standard decode loop

Optimization focus:
- Prefill/decode disaggregation
- Chunked prefill
- Continuous batching
- Paged KV cache
```

#### Talker (Lightweight AR)

```
Compute:     NOT compute-bound
             NOT memory-bound
             
Key operations:
- Small backbone forward
- Minimal attention (short context)
- Input: Thinker embedding + MTP feedback

Challenge:
- GPU utilization naturally LOW
- Operations too light for compute saturation
- Kernel launch overhead dominates
```

#### MTP (Multi-Token Prediction)

```
Compute:     Tiny prefill
             ~few codec tokens per call
             
Key operations:
- Multi-head completion
- Embedding write-back to Talker

Challenge:
- Per-call compute too small
- Synchronization overhead dominant
- Single-step feedback dependency with Talker
```

### 3.3 Key Challenges

| Challenge | Description | Impact |
|-----------|-------------|--------|
| **Heterogeneity** | Three paradigms forced into one scheduler | Thinker throughput degraded by Talker's fine-grained ops |
| **Divergent deps** | Async (Thinker↔Talker) + Sync (Talker↔MTP) | Different communication mechanisms needed |
| **Memory contention** | Weights + KV cache + buffers + feedback | Memory allocation varies by step |
| **Kernel overhead** | Light ops → launch overhead dominates | Need CUDA Graph fusion |

---

## 4. System Architecture

### 4.1 Design Principles

1. **Scheduling decoupling** — Heterogeneous stages must run on separate schedulers
2. **Inter-stage streaming** — Low-overhead buffer with controlled slack for async stages
3. **Synchronous coupling** — Ultra-low latency for tight Talker↔MTP loop
4. **Cross-stage memory** — Unified memory allocation across stages
5. **CUDA Graph fusion** — Cover entire multi-stage decode with single kernel

### 4.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SGLANG OMNI ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ORCHESTRATOR LAYER                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │
│  │  │ Scheduler  │  │ Scheduler  │  │ Scheduler  │              │   │
│  │  │ (Thinker) │  │ (Talker)  │  │   (MTP)   │              │   │
│  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘              │   │
│  │        │             │             │                               │   │
│  └────────┼─────────────┼─────────────┼───────────────────────────┘   │
│           │             │             │                               │
│           ▼             ▼             ▼                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              STREAMING BUFFER LAYER                         │   │
│  │                                                              │   │
│  │  ┌─────────────────┐    ┌──────────────────┐              │   │
│  │  │ Async Buffer   │    │ Sync Coupling    │              │   │
│  │  │ (Thinker→Talker)   │ (Talker↔MTP)     │              │   │
│  │  │ - token queue │    │ - zero-copy     │              │   │
│  │  │ - buffer pool │    │ - event sync    │              │   │
│  │  └─────────────────┘    └──────────────────┘              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐
│  │                   EXECUTION LAYER                                │
│  │                                                              │
│  │  ┌──────────────────────────────────────────────────┐             │
│  │  │           UNIFIED CUDA GRAPH                          │             │
│  │  │  - Single graph for entire multi-stage pipeline     │             │
│  │  │  - Eliminates kernel launch overhead             │             │
│  │  │  - Full fusion across stages                    │             │
│  │  └──────────────────────────────────────────────────┘             │
│  │                                                              │
│  └─────────────────────────────────────────────────────────────────────┘
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐
│  │                   MEMORY LAYER                                    │
│  │                                                              │
│  │  Cross-stage allocator:                                          │
│  │  - Dynamic weight offloading                                      │
│  │  - Per-step KV cache management                                │
│  │  - Feedback buffer allocation                                 │
│  │  - Stage-specific memory pools                                 │
│  │                                                              │
│  └─────────────────────────────────────────────────────────────────────┘
│                                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Key Components

#### Multiple Schedulers

```python
# Each stage gets its own scheduler for heterogeneity
class OmniScheduler:
    def __init__(self, num_stages):
        self.schedulers = [
            ThinkerScheduler(),   # Standard LLM scheduler (from SGLang main)
            TalkerScheduler(),  # Light AR scheduler
            MTPScheduler()   # Parallel completion scheduler
        ]
    
    def schedule_async(self, stage_a, stage_b, buffer):
        # Async schedule: independent decode loops
        # Buffer with slack for rate mismatch
        return AsyncPipeline(stage_a, stage_b, buffer)
    
    def schedule_sync(self, stage_a, stage_b, event):
        # Sync schedule: tight coupling
        # Event-based synchronization
        # Single-step latency minimization
        return SyncPipeline(stage_a, stage_b, event)
```

#### Streaming Communication

```python
# Two communication patterns for two dependency types
class InterStageBuffer:
    def __init__(self, mode="async"):
        if mode == "async":
            # Async: Thinker → Talker
            # Low overhead, some slack allowed
            self.buffer = RingBuffer(capacity=1024)
        else:
            # Sync: Talker → MTP
            # Zero-copy, event-triggered
            self.coupling = ZeroCopyCoupling()
    
    def produce(self, data):
        # Producer writes to buffer
        self.buffer.push(data)
    
    def consume(self):
        # Consumer reads from buffer
        return self.buffer.pop()
```

#### Unified CUDA Graph

```python
# The killer feature: fuse entire multi-stage decode
@cuda_graph
def omni_decode_graph(thinker_input, talker_state, mtp_input):
    # All stages in single graph
    thinker_out = thinker(thinker_input)
    talker_out = talker(thinker_out, talker_state)
    mtp_out = mtp(talker_out, mtp_input)
    return mtp_out

# Captured once, replayed many times:
graph = cuda.graph.capture(omni_decode_graph)
while running:
    graph.replay()  # Kernel launch overhead eliminated
```

---

## 5. Comparison with Alternatives

### 5.1 Single-Stage vs. Multi-Stage

| Characteristic | Single-Stage | Multi-Stage |
|-------------|-------------|-------------|
| **Models** | MiMo Omni, Nemotron Omni, standard LLM | Qwen3-Omni, TTS models, Ming Omni |
| **SGLang fit** | SGLang main | SGLang Omni |
| **Scheduler** | Single | Multiple (decoupled) |
| **CUDA Graph** | Per-stage | Unified |
| **Communication** | KV cache only | Streaming buffers + sync coupling |

### 5.2 Where SGLang Main Excels

SGLang main is optimized for:

- Standard LLM/VLM autoregressive decoding
- Prefill/decode disaggregation
- Massive batch inference
- KV cache optimization

It doesn't need SGLang Omni for:

- MiMo Omni (VLM with text output) — single stage
- Nemotron Omni — single stage  
- Qwen ASR — single stage
- Standard image generators (Wan, Qwen-Image) — single denoising stage

### 5.3 SGLang Family Overview

| Framework | Target | Key Feature |
|-----------|--------|-------------|
| **SGLang main** | Single-stage AR decoding | Maximum throughput |
| **SGLang Omni** | Multi-stage models | Stage fusion + unified CUDA Graph |
| **SGLang Diffusion** | Diffusion models | High-quality image/video generation |

---

## 6. Practical Implications

### 6.1 When to Use SGLang Omni

| Scenario | Use | Reason |
|----------|-----|-------|
| Qwen3-Omni serving | SGLang Omni | Multi-stage decode |
| TTS inference (FishAudio, etc.) | SGLang Omni | Serial AR stages |
| Ming Omni deployment | SGLang Omni | Fully omni-modal pipeline |
| Standard VLM (MiMo, Nemotron) | SGLang main | Single stage |
| Image generation (Wan) | SGLang Diffusion | Single diffusion stage |

### 6.2 Performance Gains

From the SGLang team blog (Feb 2026):

- **25x inference performance** on NVIDIA GB300 NVL72
- Multi-stage models benefit from:
  - Reduced kernel overhead (~eliminated via CUDA Graph)
  - Better GPU utilization (stages fused)
  - Eliminated scheduler contention

### 6.3 Supported Models (as of May 2026)

| Model Type | Examples | Status |
|-----------|----------|--------|
| Omni (text + speech) | Qwen3-Omni | ✅ Supported |
| TTS | FishAudio S2 Pro, Qwen3-TTS | ✅ Supported |
| Fully omni-modal | Ming Omni, LLaDA Uni | ✅ Supported |
| VLM (text only) | MiMo Omni, Nemotron Omni | ❌ Use SGLang main |

---

## 7. Implementation Details

### 7.1 Stage Fusion

```python
# Conceptual: how stages fuse
class StageFusion:
    def __init__(self, stages):
        self.stages = stages
    
    def fuse_all(self):
        # Analyze data dependencies
        deps = analyze_dependencies(self.stages)
        
        # Group sync stages together
        sync_group = group_synced(deps)
        
        # Create unified CUDA Graph
        graph = cuda.graph.create(
            forward_all(sync_group)
        )
        
        return graph
```

### 7.2 Memory Management

```python
class OmniMemoryManager:
    def allocate_stage_memory(self, stage, config):
        if stage == "thinker":
            return MemoryPool(
                weights=config.thinker_weights,
                kv_cache=config.thinker_kv,
                prefix_cache=True
            )
        elif stage == "talker":
            return MemoryPool(
                weights=config.talker_weights,
                feedback_buffer=True,
                kv_cache="small"
            )
        else:  # mtp
            return MemoryPool(
                embeddings=True
            )
    
    def cross_stage_allocate(self, total_available):
        # Rebalance dynamically across stages
        return dynamic_rebalance(total_available)
```

### 7.3 Scheduling Strategy

```python
def omni_schedule(request):
    # Request starts at Thinker
    thinker_future = thinker_scheduler.submit(request)
    
    # Async: Thinker → Talker
    talker_buffer = async_buffer()
    
    # Wait for first token, then start Talker
    talker_result = talker_schedule.run_async(
        thinker_future.result(),
        buffer=talker_buffer
    )
    
    # Sync: Talker → MTP (tight loop)
    mtp_result = mtp_schedule.run_sync(
        talker_result.first_token(),
        event=sync_event
    )
    
    return mtp_result
```

---

## 8. Technical Deep Dive

### 8.1 Why Decoupling Is Necessary

If you force all three stages into a single SGLang main scheduler:

```
Problem Flow:
───────────
Thinker (large batch, compute-bound)
    ↓ degraded
Talker (fine-grained ops, needs low latency)
    ↓ blocked
MTP (kernel launch overhead dominant)
    ↓ waits

Result: 
- Thinker throughput killed by Talker's scheduling
- Talker latency inflated by Thinker's batch
- MTP starves waiting
```

**Solution:** Decoupled schedulers can progress at their own rates.

### 8.2 The Two Dependency Patterns

| Pattern | Relationship | Communication | Tolerance |
|---------|--------------|--------------|-----------|
| **Async** | Thinker → Talker | Token queue + hidden states | Some slack OK |
| **Sync** | Talker ↔ MTP | Immediate write-back | Single-step minimum |

### 8.3 CUDA Graph Benefits

Without CUDA Graph:
```python
for step in range(1000):
    thinker.forward()    # kernel launch
    talker.forward()    # kernel launch  
    mtp.forward()      # kernel launch
# 3000 kernel launches for 1000 steps!
```

With Unified CUDA Graph:
```python
graph = capture(omni_decode_loop)  # Capture once
for step in range(1000):
    graph.replay()      # Single launch per step
# 1000 graph replays (much cheaper)
```

---

## 9. Architecture Decision Matrix

| Feature | Design Choice | Rationale |
|---------|--------------|-----------|
| **Scheduler** | Multiple decoupled | Heterogeneous compute |
| **Communication** | Two modes (async + sync) | Divergent dependencies |
| **Memory** | Cross-stage allocation | Dynamic stage needs |
| **Execution** | Unified CUDA Graph | Minimize kernel overhead |
| **Focus** | Computation, not modality | Precise targeting |

---

## 10. Open Questions & Future Directions

### 10.1 Current Gaps

- Dynamic model switching between stages
- Better profiling for multi-stage bottlenecks
- Automated stage fusion detection

### 10.2 Future Roadmap

- Auto-detection of multi-stage structure from model config
- More sophisticated cross-stage memory rebalancing
- Integration with SGLang Diffusion for mixed pipelines
- Support for arbitrary stage graphs (not just linear)

### 10.3 Research Opportunities

- Model-agnostic multi-stage detection
- Automatic fusion strategy selection  
- Dynamic stage skipping (early exit)
- Hierarchical scheduling across clusters

---

## Quick Reference

| Item | Value |
|------|-------|
| **Framework** | SGLang Omni |
| **Parent** | SGLang (LMSYS) |
| **Target** | Multi-stage generative models |
| **Key innovation** | Stage fusion + unified CUDA Graph |
| **Representative models** | Qwen3-Omni, FishAudio S2 Pro, Ming Omni |
| **Performance** | Up to 25x on GB300 NVL72 |

---

## References

- SGLang GitHub: https://github.com/sgl-project/sglang
- SGLang Blog: https://lmsys.org/blog/
- SGLang Documentation: https://docs.sglang.io/
- Author: Chenyang Zhao (LinkedIn article, May 2026)

---

*Survey completed: May 2026*