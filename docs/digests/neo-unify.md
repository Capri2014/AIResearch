# NEO-unify: Native Unified Multimodal Model Architecture

**Date:** 2026-03-08  
**Status:** Survey Complete  
**Source:** SenseTime + Nanyang Technological University

---

## 1. Overview

**Paper:** NEO-unify (preview)  
**Authors:** SenseTime, Nanyang Technological University  
**Related:** NEO (Diao et al., ICLR 2026)  
**HuggingFace:** https://huggingface.co/blog/sensenova/neo-unify

---

## 2. Core Problem: Multimodal Architecture Dilemma

### Traditional Paradigm
```
Visual Encoder (VE) → Understanding
VAE → Generation
```
- VE handles perception/understanding
- VAE handles content generation
- Creates a gap between understanding and generation

### Recent Attempts
- **Shared encoder** approaches (compromise)
- New structural design trade-offs emerge

---

## 3. NEO-unify: Native Unified Architecture

### Core Innovation: No Encoder Design

**Key breakthrough:** Eliminates both Vision Encoder (VE) and VAE entirely!

```
Traditional: VE → VAE → Components
NEO-unify:   Pixels + Text → MoT → Understanding + Generation
```

### 3.1 Three Key Innovations

| Innovation | Description |
|------------|-------------|
| **No Encoder** | Direct pixel + text input, no pretrained visual encoder |
| **MoT Architecture** | Mixture-of-Transformer for unified visual-language processing |
| **Native End-to-End** | Single model for understanding + generation |

### 3.2 Architecture

```
┌─────────────────────────────────────────────────────┐
│              NEO-unify                              │
├─────────────────────────────────────────────────────┤
│  Input:                                             │
│  ├── Near-lossless pixel stream                     │
│  └── Text tokens                                    │
│                                                     │
│  Processing:                                        │
│  └── Mixture-of-Transformer (MoT)                  │
│       ├── Visual understanding branch                │
│       └── Visual generation branch                   │
│                                                     │
│  Output:                                           │
│  ├── Multimodal understanding                       │
│  └── High-fidelity image generation                │
└─────────────────────────────────────────────────────┘
```

---

## 4. Technical Details

### 4.1 Near-Lossless Visual Interface
- Direct pixel input with minimal compression
- Preserves fine-grained visual details
- Eliminates information bottleneck from VE

### 4.2 Mixture-of-Transformer (MoT)
- Unified architecture for vision + language
- Both understanding and generation in same framework
- Minimizes internal conflicts between modalities

### 4.3 Unified Learning Framework
- **Text**: Autoregressive cross-entropy
- **Vision**: Pixel stream matching
- Single training objective

---

## 5. Results

### 5.1 Image Reconstruction

| Model | PSNR | SSIM |
|-------|------|------|
| **NEO-unify (2B)** | 31.56 | 0.85 |
| Flux VAE | 32.65 | 0.91 |

**Note:** Achieved with **no pretrained VE or VAE**!

### 5.2 Image Editing

- **Benchmark:** ImgEdit
- **Score:** 3.32
- **Key:** Works even with understanding branch frozen

### 5.3 Data Efficiency

> "NEO-unify shows higher data training efficiency, achieving better performance with fewer training tokens compared to Bagel"

---

## 6. Key Insights

### 6.1 Encoder-Free Works!
- Native end-to-end models can learn rich semantic representations
- Even with frozen understanding branch, generation branch can extract fine details
- Challenges the assumption that pretrained encoders are necessary

### 6.2 MoT Synergy
- Understanding and generation improve together
- Minimal internal conflict
- Both branches benefit from shared backbone

### 6.3 Data Efficiency
- No encoder = no pretrained prior = more efficient learning
- Better scaling without encoder bottlenecks

---

## 7. Implications for Autonomous Driving

### 7.1 Vision-Language Models for Driving

| Traditional | NEO-unify Approach |
|-------------|-------------------|
| VE → BEV → Planning | Direct pixel → MoT → Planning |
| Pretrained encoder needed | End-to-end from scratch |
| Modality gap | Native unification |

### 7.2 Potential Applications

- **Unified perception**: No separate encoder for camera/lidar
- **End-to-end planning**: Direct pixel → action
- **World model**: Unified understanding + prediction
- **Efficient**: No VE bottleneck

### 7.3 Driving Pipeline Impact

```
Traditional:
  Camera → VE → BEV → Planning → Control

NEO-unify:
  Camera + Map → MoT → Planning + Generation → Control
                ↑
        (unified understanding + prediction)
```

---

## 8. Related Work

| Paper | Relation |
|-------|----------|
| **NEO** (ICLR 2026) | Foundation work on native end-to-end |
| **LLaVA** | LLM-based multimodal |
| **GPT-4V** | Vision-language |
| **World Models** | GAIA-1, DreamerV3 |

---

## 9. Survey Status

- [x] Core problem: multimodal architecture dilemma
- [x] NEO-unify architecture: No encoder + MoT
- [x] Technical details: near-lossless input, unified training
- [x] Results: competitive with VE-based methods
- [x] Implications for autonomous driving

---

## 10. References

1. NEO-unify: https://huggingface.co/blog/sensenova/neo-unify
2. SenseTime Blog: https://www.sensetime.com/en/news-detail/51170542
3. NEO (Diao et al., ICLR 2026)
