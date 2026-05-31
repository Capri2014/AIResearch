# π0 系列 3 天突击计划 — Day 1 架构梳理

**目标：** 产出一页架构图 + 完全理解 π0 系列的技术演进

---

## 一句话理解

> **π0 = VLM语义理解 + Flow Matching连续动作生成 → 可跨任务/跨embodiment fine-tune的robot foundation policy**

---

## π0 核心架构详图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           π0 FULL ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUTS                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  ┌───────────┐      │
│  │Images(I) │  │Language │  │Robot State      │  │ Actions  │      │
│  │(vision) │  │(task desc)│ │(proprioception)│  │ (target)│      │
│  │  h×w   │  │ token   │  │ joint angles, ee │  │  A:t   │      │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  └───┬────┘      │
│       │             │                  │                 │            │
│       ▼             ▼                  ▼                 ▼            │
│  ┌─────────────────────────────────────────────────────────────────────┐      │
│  │               VLM BACKBONE (PaliGemma-style)                       │      │
│  │  ┌──────────────────────────────────────────────────────┐         │      │
│  │  │  Vision Encoder ──▶ Visual Tokens                    │         │      │
│  │  │      ↓                                            │         │      │
│  │  │  Pretrained Language Model (Transformer)           │         │      │
│  │  │      ↓                                            │         │      │
│  │  │  Late Fusion: Visual + Text tokens in same space     │         │      │
│  │  └──────────────────────────────────────────────────────┘         │      │
│  │         Weights initialized from PaliGemma (internet-scale)    │      │
│  └──────────────────────────┬───────────────────────────────────────┘      │
│                             │                                             │
│                             ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │              ROBOT ACTION EXPERT                                 │      │
│  │  ┌─���───────────────────────────────────────────────────────┐           │      │
│  │  │  State Encoder: robot q, dq → state embeddings          │           │      │
│  │  │  Action Heads: per-dim output layers (flow matching)     │           │      │
│  │  │  Shared latent: conditioned on VLM output              │           │      │
│  │  └─────────────────────────────────────────────────────────┘           │      │
│  │  Purpose: Separate expert weights from VLM backbone                  │      │
│  └──────────────────────────┬───────────────────────────────────────────┘      │
│                             │                                             │
│                             ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐       │
│  │              CONDITIONAL FLOW MATCHING                            │       │
│  │                                                                │       │
│  │  p(a_t | visual context, robot state, action context)            │       │
│  │                                                                │       │
│  │  Flow matching (variant of diffusion):                        │       │
│  │  - Models velocity field v = dx/dt                            │       │
│  │  - OT (Optimal Transport) flow for clean trajectory            │       │
│  │  - NO iterative denoising (unlike DDPM)                       │       │
│  │  - Supports multimodal action distributions                    │       │
│  │                                                                │       │
│  └──────────────────────────┬─────────────────────────────────────────┘       │
│                             │                                             │
│                             ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │              ACTION CHUNKING (up to 50Hz)                       │       │
│  │  ┌─────────────────────────────────────────────────────┐                 │      │
│  │  │  Output: [Δq_1, Δq_2, ..., Δq_N] × action_dim     │                 │      │
│  │  │  Typically N=7-14 timesteps                      │                 │      │
│  │  │  Chunk passed to low-level PD controller       │                 │      │
│  │  └─────────────────────────────────────────────────────┘                 │      │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                      │
└─────────────────────��─────────────────────────────────────────────────────────
```

---

## 五问五答

### Q1: VLM backbone 为什么要用 pre-trained 而非从头训？

| 明智选择 | 说明 |
|----------|------|
| **Internet-scale语义** | ImageNet + COCO + 网页文本 pretrain，学到scene understanding, object recognition, spatial reasoning |
| **泛化基础** | zero-shot instruction following, novelty generalization |
| **效率** | 10k小时 robot data 不够 pretrain 大模型; 用 PaliGemma 3B/10B initialization |

### Q2: Robot state 怎么处理？

两种方案：
1. **Concatenate** → 直接拼接到 token sequence
2. **Dedicated encoder** → 小 MLP 编码后融入 hidden state

π0 论文用的是后者：**separate action expert** 处理 proprioception，确保不破坏 VLM 能力

### Q3: Flow Matching vs Diffusion 有什么区别？

```
┌────────────────────────────────────────────────────────────────────┐
│            DIFFUSION (DDPM/DDIM) vs FLOW MATCHING              │
├────────────────────────────────────────────────────────────────────┤
│ Feature           │ Diffusion          │ Flow Matching         │
├────────────────────────────────────────────────────────────────────┤
│ Training target  │ ε-prediction     │ velocity field (v)   │
│ Sampling        │ Multi-step iter  │ Few-step ODE solve   │
│ Speed         │ 50-100 steps   │ 4-10 steps         │
│ Multimodal     │ Good           │ Good (with OT)      │
│ Mathematics   │ Stochastic(DDPM)│ Deterministic(ODE)│
│ OptimalTransport│ No            │ Yes (often)        │
└────────────────────────────────────────────────────────────────────┘
```

**关键优势(flow matching for robotics):**
- 更快 (10x fewer steps)
- 更好地 multimodal modeling
- 无需 classifier-free guidance也能有 diversity

### Q4: 为什么不能一步一step输(action)，而要chunking?

**延迟问题:**
```
单步输出: 每步计算 → 等待执行 → 下一步 (50Hz = 20ms/step)
Chunk输出: 计算一次 → 执行N步 (throughput提升)

Example:
- 50Hz dexterous control (折叠衣物)
- 单步无法提前"看到"完整trajectory
- PD controller需要feedforward
```

### Q5: 为什么cross-embodiment？

| 多个robot | 数据scale | 好处 |
|-----------|----------|------|
| 7 robot configs | 10k+ hours | 泛化到新机器人 |
| 68 tasks | | 共享semantic知识 |
| single/dual-arm + mobile | | 互补skills |

---

## π0-FAST vs π0 (Flow) 对比

| 特性 | π0 (Flow) | π0-FAST |
|------|----------|----------|
| **Action表示** | 连续分布 | 离散token (vocab压缩) |
| **生成方式** | Flow Matching ODE | Autoregressive Transformer |
| **采样** | 4-10步 | N步 (同AR LLM) |
| **Dexterous control** | 50Hz OK | 可能降质 |
| **训练效率** | 中等 | 高(5x faster) |
| **推理速度** | 快(step少但每步重) | 慢(sequence长) |
| **语言following** | 中等 | 好 |
| **复现难度** | 中 | 中 |

**FAST核心思想:**
- Action trajectory → Discrete tokens via frequency-domain压缩(DCT)
- 解决高频 dexterous action 的 binning 问题
- 类似 audio codec思路

```
FAST Tokenizer:
┌───────────────────��─��───────────────────────┐
│  A_t = [a_0, a_1, ..., a_T]              │
│       ↓ DCT/frequency transform            │
│  A_freq = [freq_0, freq_K]               │
│       ↓ quantize (codebook lookup)        │
│  discrete_action_tokens ──▶ 送入AR transformer
└─────────────────────────────────────────────┘
```

---

## π0.5 关键增量

| 维度 | π0 | π0.5 |
|------|-----|-------|
| **数据mix** | 10k hr robot | +Web data +semantic tasks |
| **泛化** | Task-specific | Open-world (新房间/物体/语义) |
| **训练Recipe**| Pre-train→Post-train | Heterogeneous co-training |
| **能力** | Physical skills | +Long-horizon decomposition |

**π0.5 数据组成:**
```
Data Mixture:
├── Robot data (multiple robots, embodiments)
├── Web image/video data
├── High-level semantic prediction
├── Object detection (open vocabulary)
├── Subtask commands (grasp, place, push...)
└── Mobile manipulation data
```

---

## π0系列演进图

```
                        ┌─────────────────────┐
                        │   INTERNET SCALE    │
                        │      VISION+TEXT    │
                        │   (PaliGemma 3B)   │
                        └─────────┬───────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        π0                              │
│  1. VLM backbone (PaliGemma init)                        │
│  2. + Robot State/Action Expert                       │
│  3. + Conditional Flow Matching                       │
│  4. + Action Chunking (50Hz)                         │
│                                                         │
│  Output: Continuous action distribution              │
│  Best: Dexterous manipulation, physical skills       │
└──────────────────────┬─────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                       ▼
    ┌─────────────┐      ┌─────────────┐
    │  π0-FAST   │      │   π0.5    │
    │            │      │            │
    │Discrete   │      │+Heterogen.│
    │action    │      │co-training│
    │tokens    │      │           │
    │AR decode │      │+Web data  │
    │          │      │+Semantic  │
    │训练快    │      │Tasks     │
    │         │      │          │
    │语言fol.|      │Open-world│
    │      │      │generalize│
    └──────┘      └──────────┘
```

---

## Day 1 晚上产出表

| 版本 | 核心问题 | 技术方案 | 价值 | 风险 |
|------|----------|---------|------|------|
| **π0** | 如何把VLM变成robot policy | VLM + flow matching + action chunking | 连续action、dexterous control、50Hz | 训练复杂、需要大量robot数据 |
| **π0-FAST** | 如何接入AR VLA pipeline | 频率空间action token化→AR transformer | 训练快5x、infra统一 | action降质、vocab设计 |
| **π0.5** | 如何开放世界泛化 | 异构co-training + web数据 | 新场景、新任务、长程 | 复现难、数据配方复杂 |

---

## 实验任务

| Task | Robots | Success率 |
|------|--------|----------|
| 衣物折叠 | mobile manipulator | ~80% (需要fine-tune) |
| 餐桌清理 | single-arm | ~75% |
| 放入微波炉 | single-arm | ~85% |
| 鸡蛋装箱 | dual-arm | ~60% |
| 装盒子 | dual-arm | ~70% |
| 杂货装袋 | dual-arm | ~65% |

---

## 参考资料

| 资源 | 链接 |
|------|------|
| **π0 论文** | https://arxiv.org/abs/2410.24164 |
| **FAST 论文** | (搜索 Physical Intelligence FAST) |
| **π0.5 论文** | (搜索 Physical Intelligence π0.5) |
| **OpenPI代码** | https://github.com/Physical-Intelligence/openpi |
| **HF检查点** | https://huggingface.co/collections/physicalintelligence/pi0 |

---

## Day 2 预告

1. **OpenPI环境搭建** - pip install + download checkpoint
2. **Out-of-box inference** - 跑通一个demo
3. **代码结构剖析** - 找到关键模块
4. **最小实验** - 修改action chunk / inference config

---

*Day 1 完成 ✓* 

**下一步：** Commit并继续Day 2