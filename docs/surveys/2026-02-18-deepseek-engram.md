# DeepSeek Engram & Memory Research Survey

**Date:** 2026-02-18  
**Status:** Complete Survey  
**Author:** OpenClaw (Pipeline Agent)

---

## Research Background Overview

DeepSeek Engram is the culmination of prior research work. To understand Engram, we need to trace two research lineages:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DeepSeek Engram Research Lineage                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  【Memory Track】                    【N-gram Track】                   │
│  FFN = Key-Value Memory              Traditional N-gram LM              │
│       ↓                                    ↓                             │
│  Knowledge Neurons                    N-Grammer (Google 2022)           │
│  Product Key Memory (PKM)            Scaling Embedding (Google 2025)    │
│  Memory Layer (128B params)                  ↓                         │
│  RETRO (DeepMind 2022)                       ↓                         │
│  External Memory (Google 2023)                    ↓                     │
│       ↓                                            ↓                     │
│       └─────────────────────┬───────────────────┘                       │
│                           ↓                                             │
│                   DeepSeek Engram                                        │
│                   (N-gram Embedding + Hash Lookup)                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Memory Track: FFN as Key-Value Memory

### 1.1 Core Thesis: Transformer FFN Layer is Essentially a Key-Value Memory

This perspective was first introduced by the following papers:

| Year | Institution | Paper | Core Contribution |
|------|-------------|-------|-------------------|
| 2019 | Facebook | Language Models as Knowledge Bases | First proposed LM as knowledge base |
| 2020 | Google | T5 as Knowledge Base | Validated T5's knowledge storage |
| 2021 | - | Transformer FFN as KV Memory | FFN layer = memory neuron collection |
| 2022 | Microsoft | Knowledge Neurons | Located neurons storing specific knowledge |
| 2023 | Google | External Memory | External memory enhanced |

### 1.2 FFN = Memory Mathematical Proof

```
Transformer FFN Forward Pass Formula:

h_out = σ(W₂ · σ(W₁ · h_in + b₁) + b₂)

Where:
- Each row of W₁ can be viewed as a "Key"
- Each row of W₂ can be viewed as a "Value"
- Dot product of h_in with W₁ determines which "Keys" to activate
- Activated "Keys" corresponding "Values" are weighted-summed and output

This is completely equivalent to a Key-Value Memory read operation!
```

### 1.3 Key Paper Analysis

#### Facebook 2019: Language Models as Knowledge Bases

**Core Thesis:** Language models can store factual knowledge

```python
class KnowledgeBaseLM:
    """
    Framework viewing LM as a knowledge base
    
    Example:
    Input: "Paris is the capital of"
    Output: "France"
    
    This demonstrates LM stores "Paris → France" knowledge mapping internally
    """
    
    def extract_knowledge(self, prompt: str) -> str:
        # Use language model to predict masked token
        return self.fill_mask(prompt)
```

**Limitations:**
- Knowledge is implicitly stored, difficult to interpret and edit
- Knowledge capacity limited by model parameters
- Prone to hallucination

#### Microsoft 2022: Knowledge Neurons

**Core Thesis:** Locate FFN neurons storing specific knowledge

```python
class KnowledgeNeurons:
    """
    Locate knowledge neurons via integrated gradients
    
    Analysis steps:
    1. Define knowledge query: "Paris is the capital of [MASK]"
    2. Compute integrated gradients of output w.r.t. intermediate activations
    3. Identify neurons with highest contribution
    4. These neurons are "Knowledge Neurons" storing the knowledge
    """
    
    def find_knowledge_neurons(self, prompt: str, target: str):
        # Integrated gradient computation
        integrated_gradients = self.compute_ig(prompt, target)
        
        # Locate high-contribution neurons
        neurons = np.where(integrated_gradients > threshold)[0]
        
        return neurons
```

#### Meta 2022: Memory Layer (128B Parameters)

**Core Thesis:** Extend FFN layer to learnable Memory Layer

```python
class MemoryLayer(nn.Module):
    """
    Memory Layer Implementation
    
    Compared to standard FFN:
    - Key-Value pairs are explicitly learnable
    - Supports larger Memory capacity (128B parameter scale)
    - Can retrieve via fast lookup
    """
    
    def __init__(self, num_keys: int, key_dim: int, value_dim: int):
        # Learnable Keys and Values
        self.keys = nn.Parameter(torch.randn(num_keys, key_dim))
        self.values = nn.Parameter(torch.randn(num_keys, value_dim))
        
        # Top-k sparse retrieval
        self.top_k = 4  # Each Query activates only top-4 Keys
    
    def forward(self, query: torch.Tensor) -> torch.Tensor:
        # 1. Compute similarity between Query and all Keys
        scores = query @ self.keys.T  # [B, num_keys]
        
        # 2. Top-k sparsification
        topk_scores, topk_indices = scores.topk(self.top_k)  # [B, top_k]
        
        # 3. Softmax normalization
        weights = F.softmax(topk_scores, dim=-1)  # [B, top_k]
        
        # 4. Weighted-sum Values
        selected_values = self.values[topk_indices]  # [B, top_k, value_dim]
        output = (weights.unsqueeze(-1) * selected_values).sum(dim=1)
        
        return output
```

**Core Innovations:**
- Explicit Key-Value Memory structure
- Sparse activation (Top-k) ensures computational efficiency
- Scalable to ultra-large scale (128B parameters)

#### DeepMind 2022: RETRO (Retrieval-Enhanced Transformer)

```python
class RETRO(nn.Module):
    """
    RETRO: Retrieval-Enhanced Transformer
    
    Architecture:
    1. Encoder: Encode input as Query
    2. Retriever: Retrieve from external database
    3. Cross-Attention: Fuse retrieved results
    """
    
    def __init__(self, retro_config):
        self.encoder = TransformerEncoder(retro_config)
        self.retriever = BertRetriever(retro_config)
        self.decoder = TransformerDecoder(retro_config)
    
    def forward(self, input_ids: torch.Tensor):
        # Encode input
        query = self.encoder(input_ids)
        
        # Retrieve relevant documents
        retrieved = self.retriever.retrieve(query, top_k=2)
        
        # Fuse retrieved results
        output = self.decoder(query, retrieved=retrieved)
        
        return output
```

---

## 2. N-gram Track: From Statistical Language Models to Engram

### 2.1 Traditional N-gram Language Models

```python
# Traditional N-gram model
# Statistics P(wₙ | wₙ₋₁, wₙ₋₂, ..., wₙ₋ₙ₊₁)

class NGramLM:
    """
    Traditional N-gram language model
    
    Advantages:
    - Simple training: count word frequencies
    - Fast inference: lookup O(1)
    - Strong interpretability
    
    Disadvantages:
    - Sparsity problem (lots of zero probabilities)
    - Long-range dependency loss (N is limited)
    - Poor generalization
    """
    
    def __init__(self, n: int = 3):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = defaultdict(Counter)
    
    def train(self, corpus: List[str]):
        # Count N-gram frequencies
        for sentence in corpus:
            tokens = sentence.split()
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + n - 1])
                ngram = tokens[i + n - 1]
                self.ngram_counts[context][ngram] += 1
```

### 2.2 Google 2022: N-Grammer

```python
class NGrammer(nn.Module):
    """
    N-Grammer: Introduce N-gram ideas into Transformer
    
    Core thesis:
    1. Extract N-gram features from input sequences as explicit features
    2. Learn an embedding for each N-gram
    3. Integrate N-gram embedding into Transformer
    
    Formula:
    h_out = Transformer(h_in) + NgramEmbedding(ngrams)
    """
    
    def __init__(self, ngram_sizes: List[int], embed_dim: int):
        self.ngram_sizes = ngram_sizes
        
        # N-gram embedding table
        self.ngram_embeddings = nn.ModuleDict()
        for n in ngram_sizes:
            self.ngram_embeddings[str(n)] = nn.Embedding(
                num_embeddings=10**n,  # Large hash space
                embedding_dim=embed_dim
            )
```

**Key Insight:**
- N-gram embedding can be viewed as a "soft lookup"
- Hash maps variable-length N-gram to fixed-size embedding space
- More efficient than explicit N-gram statistics

### 2.3 Google 2025: Scaling Embedding Layer

```python
class ScaledEmbeddingLayer(nn.Module):
    """
    Scale embedding layer to support larger scale
    
    Technical Points:
    1. Product Quantization (PQ) compression
    2. Hierarchical indexing
    3. O(1) complexity lookup
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, num_subcodes: int = 8):
        # Product Quantization
        self.subcode_dim = embed_dim // num_subcodes
        self.num_subcodes = num_subcodes
        
        # Sub-codebooks (learnable)
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(vocab_size, self.subcode_dim))
            for _ in range(num_subcodes)
        ])
```

---

## 3. DeepSeek Engram: The Culmination

### 3.1 Core Thesis

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DeepSeek Engram Core Architecture                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: [B, L, D]                                                       │
│      ↓                                                                   │
│  ┌─────────────────┐  ┌───────────────────────────────────────┐         │
│  │ N-gram Hashing │→│  Hash Function                            │         │
│  │                │  │  h = Hash(ngram) mod M                   │         │
│  └─────────────────┘  └───────────────────────────────────────┘         │
│                              ↓                                         │
│  ┌───────────────────────────────────────────────────────────┐         │
│  │                    Memory Table                            │         │
│  │  [hash_0: value_0]                                       │         │
│  │  [hash_1: value_1]                                       │         │
│  │  ...                                                       │         │
│  │  [hash_M: value_M]                                       │         │
│  └───────────────────────────────────────────────────────────┘         │
│                              ↓                                         │
│  ┌───────────────────────────────────────────────────────────┐         │
│  │                    Gating Mechanism                          │         │
│  │  g = sigmoid(W · [h_in, h_memory])                        │         │
│  │  h_out = g * h_memory + (1-g) * h_ffn                    │         │
│  └───────────────────────────────────────────────────────────┘         │
│                              ↓                                         │
│  Output: [B, L, D]                                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Relationship with MoE

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MoE vs Engram: Comparison & Relationship             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┬──────────────────┬──────────────────────────────┐ │
│  │     Dimension     │       MoE         │         Engram              │ │
│  ├──────────────────┼──────────────────┼──────────────────────────────┤ │
│  │ Selection        │ Top-k Experts     │  Hash Lookup              │ │
│  │ Complexity       │ O(k * d²)        │  O(1)                     │ │
│  │ Parameters       │ Large (multiple Experts) │ Large (Memory Table)│ │
│  │ Sparsity         │ Expert sparse activation │ Memory Slot sparse update│ │
│  │ Output           │ Weighted sum of Expert outputs │ Memory lookup result│ │
│  │ Purpose          │ Improve capacity/diversity │ Efficient commonsense retrieval│ │
│  └──────────────────┴──────────────────┴──────────────────────────────┘ │
│                                                                          │
│  Shared Goal: Break "Impossible Triangle"                          │
│                                                                          │
│        Performance ←───────────────┐                                  │
│              ↑                    │                                  │
│              │                    ↓                                  │
│    Compute Efficiency ──────── Model Size                            │
│              │                    ↑                                  │
│              └───────────────→   │                                  │
│                                  │                                  │
│                    MoE + Engram = Best of Both Worlds             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Technical Details

### 4.1 N-gram Hashing

```python
class NgramHasher(nn.Module):
    """
    N-gram Hash Function
    
    Maps variable-length N-gram to fixed-size integer index
    
    Formula:
    h = (Σᵢ wordᵢ * Pᵢ) mod M
    
    Where Pᵢ is a random prime sequence
    """
    
    def __init__(self, max_n: int, hash_dim: int, modulus: int = 2**61 - 1):
        self.max_n = max_n
        self.hash_dim = hash_dim
        self.modulus = modulus
        
        # Random prime sequence
        self.primes = nn.Parameter(
            torch.randint(1, modulus, (max_n, hash_dim)),
            requires_grad=False
        )
```

### 4.2 Memory Table with Sparsity

```python
class EngramMemoryTable(nn.Module):
    """
    Engram Memory Table
    
    Memory Table supporting sparse updates
    
    Features:
    1. Large capacity (scalable to TB level)
    2. Sparse update (only update activated slots)
    3. O(1) complexity lookup
    """
    
    def __init__(self, num_slots: int, slot_dim: int):
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        
        # Memory slots (learnable)
        self.slots = nn.Parameter(torch.randn(num_slots, slot_dim))
```

### 4.3 Gating Mechanism

```python
class EngramGating(nn.Module):
    """
    Engram Gating Mechanism
    
    Decides when to use Memory, when to use FFN
    
    Formula:
    g = σ(W_g · [h_in, h_memory])
    
    Interpretation:
    - g ≈ 1: Prioritize Memory (remember existing knowledge)
    - g ≈ 0: Prioritize FFN (compute new knowledge)
    """
    
    def __init__(self, hidden_dim: int, memory_dim: int):
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + memory_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
```

---

## 5. Autonomous Driving Applications

### 5.1 Engram for Driving Scene Knowledge Retrieval

```python
class DrivingEngram:
    """
    Engram application for driving scenarios
    
    Knowledge types stored:
    1. Traffic rules (red light stop, green light go)
    2. Scene patterns (highway cruising, intersection left turn)
    3. Common sense reasoning (car ahead slowing → possible braking)
    """
    
    def plan_with_engram(
        self,
        perception: Dict[str, torch.Tensor],
        goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Driving planning using Engram
        
        Process:
        1. Perception → Extract current scene N-gram
        2. Engram → Retrieve relevant driving knowledge
        3. Fuse perception + knowledge → Plan trajectory
        """
        # 1. Perception encoding
        scene_features = self.encode_scene(perception)
        
        # 2. Engram knowledge retrieval
        driving_knowledge = self.engram.read(scene_ngrams)
        
        # 3. Fuse perception + knowledge
        fused = self.fuse_perception_knowledge(scene_features, driving_knowledge)
        
        # 4. Plan trajectory
        trajectory = self.planner(fused)
        
        return trajectory, driving_knowledge
```

### 5.2 Engram + AR Decoder Integration

```python
class EngramARDecoder(nn.Module):
    """
    Engram + AR Decoder Integration
    
    Advantages:
    1. Engram provides commonsense knowledge (fast retrieval)
    2. AR Decoder learns scene-specific behavior (training)
    3. Complement each other: General knowledge + Scene adaptation
    """
    
    def __init__(self, ar_config, engram_config):
        self.ar_decoder = ARDecoder(ar_config)
        self.engram = EngramLayer(**engram_config)
    
    def forward(self, features, input_ids=None):
        # 1. AR Decoder base prediction
        ar_waypoints = self.ar_decoder.generate(features)
        
        # 2. Engram knowledge retrieval
        knowledge = self.engram(features, input_ids)
        
        # 3. Knowledge-enhanced trajectory prediction
        enhanced = ar_waypoints + knowledge.unsqueeze(1)
        
        return enhanced
```

---

## 6. Research Lineage Summary

### 6.1 Five Core Insights

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Five Core Insights                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1️⃣  FFN Layer is Essentially a Key-Value Memory                      │
│     - Transformer FFN = soft lookup operation                           │
│     - W₁ = Keys, W₂ = Values                                            │
│     - Knowledge implicitly stored in weights                            │
│                                                                          │
│  2️⃣  Sparsity is the Key to Breaking "Impossible Triangle"            │
│     - Performance / Compute / Model Size cannot be optimized simultaneously│
│     - Sparse activation/update breaks this limitation                   │
│                                                                          │
│  3️⃣  Hash Lookup Enables O(1) Complexity N-gram Retrieval           │
│     - No "computation", use "lookup" to obtain commonsense              │
│     - Large-scale Memory can also be efficient                         │
│                                                                          │
│  4️⃣  Engram Lets Model "Remember" Commonsense Without "Computing"    │
│     - Commonsense knowledge retrieved via Memory                        │
│     - Professional knowledge learned via training                       │
│                                                                          │
│  5️⃣  DeepSeek Engram is the Culmination of Prior Research           │
│     - Combines Memory track's "explicit memory"                         │
│     - Combines N-gram track's "fast lookup"                            │
│     - Proposes complete Engram architecture                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Timeline

```
2019          2020          2021          2022          2023          2025          2026
  │              │              │              │              │              │              │
  ├──────────────┴──────────────┴──────────────┴──────────────┴──────────────┤
  │                         Memory Track                                 │
  │                                                                    │
  │  FB LM    Google T5    FFN=KV     Meta PKM    DeepMind    Google   │
  │  =KB      =KB         Memory     Memory     RETRO      External  │
  │              │              │              │              │              │
  │              └──────────────┴──────────────┴──────────────┴──────────────┤
  │                            N-gram Track                             │
  │                                                                    │
  │      N-gram      Google        Google                                │
  │      LM         N-Grammer    Scaling                               │
  │                  2022        2025                                  │
  │                                        │                            │
  │                                        │                            │
  │                                        ↓                            │
  │                           DeepSeek Engram                            │
  │                           (Culmination)                             │
  └─────────────────────────────────────────────────────────────────────┘
```

### 6.3 Future Directions

| Direction | Description | Potential |
|-----------|-------------|-----------|
| **Larger-scale Memory** | TB-level Memory Table | High |
| **Hierarchical Memory** | L1/L2/L3 Cache structure | High |
| **Editable Memory** | Dynamic modify/delete memories | Medium |
| **Memory Compression** | PQ, Huffman coding | Medium |
| **Multimodal Engram** | Unified Memory for image/audio/text | High |
| **Engram + MoE** | Combined usage | High |

---

## 7. References

| Paper | Year | Institution |
|-------|------|-------------|
| Language Models as Knowledge Bases | 2019 | Facebook |
| T5 as Knowledge Base | 2020 | Google |
| Transformer FFN as KV Memory | 2021 | - |
| Knowledge Neurons | 2022 | Microsoft |
| Product Key Memory (PKM) | 2019 | Facebook |
| Memory Layer | 2022 | Meta |
| RETRO | 2022 | DeepMind |
| External Memory | 2023 | Google |
| N-Grammer | 2022 | Google |
| Scaling Embedding Layer | 2025 | Google |
| DeepSeek Engram | 2026 | DeepSeek |

---

## 8. Code Reference

| Component | File | Location |
|-----------|------|----------|
| **Engram Layer** | `training/rl/engram.py` | Full implementation |
| **N-gram Hasher** | `training/rl/engram.py` | See class implementation |
| **Memory Table** | `training/rl/engram.py` | See class implementation |
| **Gating** | `training/rl/engram.py` | See class implementation |
| **Integration Example** | `training/rl/engram.py` | See class implementation |

---

*Document Updated: 2026-02-18*
*Alignment Verified: Bilibili video BV1x3zWB6EU6 - DeepSeek Engram 论文串讲*
