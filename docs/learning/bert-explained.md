# BERT Explained: A Beginner's Guide for Autonomous Driving

**Created:** 2026-02-16  
**Purpose:** Understanding BERT for CoT (Chain of Thought) reasoning in driving

---

## Table of Contents

1. [What is BERT?](#what-is-bert)
2. [Why BERT for CoT?](#why-bert-for-cot)
3. [How BERT Works Internally](#how-bert-works-internally)
4. [Tokenization: WordPiece](#tokenization-wordpiece)
5. [Position Embeddings](#position-embeddings)
6. [Multi-Head Self-Attention (The Magic)](#multi-head-self-attention-the-magic)
7. [Feed-Forward and Residual Connections](#feed-forward-and-residual-connections)
8. [The CLS Token](#the-cls-token)
9. [BERT vs LSTM: Comparison](#bert-vs-lstm-comparison)
10. [Why BERT is Powerful](#why-bert-is-powerful)
11. [BERT for Autonomous Driving CoT](#bert-for-autonomous-driving-cot)
12. [Quick Reference](#quick-reference)

---

## What is BERT?

**BERT** = **B**idirectional **E**ncoder **R**epresentations from **T**ransformers

Released by Google in 2018, BERT revolutionized NLP by understanding context bidirectionally.

### Key Facts

| Fact | Description |
|------|-------------|
| **Type** | Transformer-based neural network |
| **Direction** | Reads text both left-to-right AND right-to-left |
| **Pre-trained** | Trained on massive text corpus (Wikipedia + Books) |
| **Sizes** | BERT-base (110M params), BERT-large (340M params) |
| **Input** | Text sequences up to 512 tokens |
| **Output** | Contextual embeddings for each token |

---

## Why BERT for CoT?

### The Problem with Previous Models

```
Input: "The car parked on the street."

Previous models (LSTM):
- Read word by word, left to right
- "parked" only knows about words BEFORE it ("The", "car", "on", "the")
- Doesn't know what comes AFTER ("on the street")

BERT:
- Reads ALL words at once
- "parked" knows about EVERY word in the sentence
- Full context understanding!
```

### For Autonomous Driving CoT

```
Driving CoT Example:
"I see a car [SEP] The car is slowing down [SEP] I should brake"

With BERT:
- "car" in first sentence knows about "slowing down" and "brake"
- Understands the RELATIONSHIP between perception and action
- Better reasoning about the driving situation!
```

---

## How BERT Works Internally

### Overall Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                       BERT Pipeline                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. INPUT: "I see a car ahead"                                 │
│                           │                                      │
│                           ▼                                      │
│  2. TOKENIZATION: Split into word pieces                         │
│                    ["I", "see", "a", "car", "ahead"]           │
│                           │                                      │
│                           ▼                                      │
│  3. ADD SPECIAL TOKENS                                         │
│                    ["[CLS]", "I", "see", "a", "car", "ahead", "[SEP]"] │
│                           │                                      │
│                           ▼                                      │
│  4. CONVERT TO NUMBERS (Vocabulary lookup)                     │
│                    [101, 1045, 2156, 1037, 2482, 4228, 102]    │
│                           │                                      │
│                           ▼                                      │
│  5. ADD POSITION EMBEDDINGS                                     │
│                    Know WHERE each word is in sequence           │
│                           │                                      │
│                           ▼                                      │
│  6. BERT ENCODER (12 layers)                                   │
│                    Multi-head self-attention + Feed-forward     │
│                           │                                      │
│                           ▼                                      │
│  7. OUTPUT: Contextual embeddings for each token                │
│                    Each word now has FULL CONTEXT!              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tokenization: WordPiece

BERT uses **WordPiece** tokenization to split words into manageable pieces.

### How WordPiece Works

```
Input: "I see a car ahead"

Step 1: Split into words
        ["I", "see", "a", "car", "ahead"]

Step 2: Check vocabulary
        - Common words: keep as is
        - Rare words: split into subwords

Example: "ahead"
        └─ If "ahead" is rare, split:
        └─ "a" + "head"
        └─ If still rare:
        └─ "a" + "he" + "ad"

Result: ["I", "see", "a", "car", "a", "head"]
```

### Special Tokens

| Token | Meaning | Purpose |
|-------|---------|---------|
| `[CLS]` | Classification | Special first token for classification |
| `[SEP]` | Separator | Separates sentences or segments |
| `[PAD]` | Padding | Fills sequence to fixed length |
| `[UNK]` | Unknown | Word not in vocabulary |

```
Input: "I see a car" + "It's red"

Output: ["[CLS]", "I", "see", "a", "car", "[SEP]", "It", "'s", "red", "[SEP]"]
         ↑                                                      ↑
      Start                                                   End
```

### Vocabulary

```
BERT-base vocabulary: 30,522 tokens

Includes:
- Common English words: "the", "car", "see"
- Subword pieces: "##ing", "##ed", "##s"
- Special tokens: [CLS], [SEP], [PAD], [UNK]
```

---

## Position Embeddings

BERT needs to know **where** each word is in the sequence.

### Why Position Matters

```
"The car hit the bus" vs "The bus hit the car"
                     ↑
              Same words, different order, different meaning!

BERT needs position info to distinguish:
- Word at position 1
- Word at position 5
- etc.
```

### How Position Embeddings Work

```
Sequence: ["[CLS]", "I", "see", "a", "car", "[SEP]"]
Position:   0       1     3     4     5      7

Each position has a learned embedding vector (768 dimensions for BERT-base)

┌────────────────────────────────────────┐
│ Position 0: [CLS] → [0.1, 0.3, ...]  │
│ Position 1: "I"    → [0.5, 0.2, ...] │
│ Position 2: "see"  → [0.8, 0.1, ...]│
│ Position 3: "a"     → [0.2, 0.7, ...]│
│ Position 4: "car"   → [0.4, 0.4, ...]│
│ Position 5: [SEP]   → [0.9, 0.0, ...]│
└────────────────────────────────────────┘
```

### Combined Embedding

```
Final Input = Token Embedding + Position Embedding + Segment Embedding

Token Embedding:   "car" → [0.4, 0.4, ...]
Position Embedding: pos 4 → [0.1, 0.2, ...]
Segment Embedding: sentence 1 → [1, 0, ...]

Sum: [0.5, 0.6, ...] ← "car" at position 4 in sentence 1
```

---

## Multi-Head Self-Attention (The Magic)

This is the core innovation of BERT. Self-attention lets each word "attend to" all other words.

### What is Attention?

```
Attention = "How much should each word pay attention to every other word?"

Example: "The car parked on the street"

- "parked" should pay attention to "car" (it did the parking)
- "parked" should pay attention to "street" (where it parked)
- "The" is less relevant to "parked"
```

### Step 1: Create Q, K, V Vectors

For each word, create 3 vectors:

```
Word: "parked"

Query (Q): "What am I looking for?" → [0.2, 0.5, ...]
Key (K):    "What do I offer?"      → [0.3, 0.1, ...]
Value (V):  "My actual content"   → [0.4, 0.6, ...]
```

### Step 2: Calculate Attention Scores

```
For "parked", calculate attention to EVERY word:

      "The"    → Score: 0.1  (low attention)
      "car"    → Score: 0.8  (high attention!)
      "parked" → Score: 0.5  (self-attention)
      "on"     → Score: 0.3
      "the"    → Score: 0.1
      "street" → Score: 0.7  (high attention!)

Score = softmax(Q · K / √d)
```

### Step 3: Weighted Sum

```
New "parked" = 0.1×"The" + 0.8×"car" + 0.5×"parked" + ... + 0.7×"street"
```

**Result:** "parked" now contains information from ALL words, weighted by relevance!

### Multi-Head = Multiple Perspectives

```
BERT has 12 attention heads (BERT-base)

Each head learns DIFFERENT relationships:

Head 1: Subject-Verb
         "car" → "parked" (car did the parking)

Head 2: Location
         "parked" → "street" (where it happened)

Head 3: Article-Noun
         "the" → "street" (which street)

All heads combined = Rich understanding!
```

### Visualization

```
Input: "I see a car ahead"

Attention Pattern (simplified):

                    I     see    a     car    ahead
                  ┌─────┬─────┬─────┬─────┬─────┐
              I   │ ███ │  ░  │  ░  │  ░  │  ░  │
                  ├─────┼─────┼─────┼─────┼─────┤
            see   │  ░  │ ███ │  ░  │  ░  │ ███ │
                  ├─────┼─────┼─────┼─────┼─────┤
              a   │  ░  │  ░  │ ███ │  ░  │  ░  │
                  ├─────┼─────┼─────┼─────┼─────┤
            car   │  ░  │  ░  │  ░  │ ███ │ ███ │
                  ├─────┼─────┼─────┼─────┼─────┤
          ahead   │  ░  │ ███ │  ░  │ ███ │ ███ │
                  └─────┴─────┴─────┴─────┴─────┘
                  
Legend: █ = high attention, ░ = low attention
```

---

## Feed-Forward and Residual Connections

### Transformer Block Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                  Single Transformer Block                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input                                                           │
│     │                                                           │
│     ▼                                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Multi-Head Self-Attention                   │   │
│  │                                                         │   │
│  │   - Each word attends to all others                    │   │
│  │   - Creates context-aware representations               │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│                    Add & LayerNorm                              │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Feed-Forward Neural Network                 │   │
│  │                                                         │   │
│  │   - 2-layer neural network                             │   │
│  │   - ReLU activation                                    │   │
│  │   - Expands and contracts dimension                    │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│                    Add & LayerNorm                              │
│                           │                                     │
│                           ▼                                     │
│  Output (passed to next block)                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Residual Connections

```
Why Residual Connections?
- Prevents vanishing gradient
- Allows gradient to flow directly
- Makes training easier

Input: X
 │
 ├──────────────────────┐
 │                      ▼
 │              ┌─────────────┐
 │              │   Layer      │
 │              │   (e.g.,     │
 │              │   Attention) │
 │              └──────┬──────┘
 │                     │
 └──────────────┬──────┘
                 │
                 ▼
        Output = Input + Layer(Input)

Example:
Input = [0.1, 0.2, 0.3]
Layer(Input) = [0.5, 0.6, 0.7]
Output = [0.1+0.5, 0.2+0.6, 0.3+0.7] = [0.6, 0.8, 1.0]
```

### Complete BERT Architecture

```
BERT-base: 12 layers, 768 hidden size, 12 attention heads

Layer 1: Input → Block 1 → Output
Layer 2: Output → Block 2 → Output
...
Layer 12: Output → Block 12 → Final Output

Each Block:
    Input
       │
       ▼
┌─────────────────────────┐
│  Multi-Head Attention   │
│  (12 heads)             │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Add & LayerNorm        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Feed-Forward Network   │
│  (3072 → 768 → 3072)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Add & LayerNorm        │
└───────────┬─────────────┘
            │
            ▼
      Next Layer Input
```

---

## The CLS Token

### What is CLS?

```
CLS = Classification token

Special token inserted at the START of every input sequence:

Input:  ["[CLS]", "I", "see", "a", "car", "[SEP]"]
           ↑
        CLS token
```

### Why CLS?

```
After 12 layers of BERT, the CLS token has:

1. Processed through attention with ALL other tokens
2. Absorbed information from the ENTIRE sentence
3. Become a SUMMARY of the whole input

Use cases:
- Sentence classification
- Question answering
- Sentence similarity
```

### CLS in Our CoT Model

```
Input CoT: "I see a car [SEP] The car is slowing down [SEP] I should brake"
            ↑
           CLS

After BERT: CLS embedding contains summary of:
- Perception: "I see a car"
- Prediction: "The car is slowing down"
- Action: "I should brake"

Used for: Predicting the next action!
```

---

## BERT vs LSTM: Comparison

| Aspect | LSTM | BERT |
|--------|------|------|
| **Architecture** | Recurrent (sequential) | Transformer (parallel) |
| **Direction** | Left-to-right | Bidirectional |
| **Context** | Limited to previous words | Full context |
| **Pre-training** | Needs task-specific data | Pre-trained on massive corpus |
| **Attention** | No native attention | Multi-head self-attention |
| **Speed** | Slower (sequential) | Faster (parallel) |
| **Parameters** | Fewer | More (110M+) |

### Visual Comparison

```
LSTM:                    BERT:
"I" ──► "see" ──► "a" ──► "car"
 ↑          │          │          │
 │          │          │          │
 │          ▼          │          │
 │     "parked" ◄─────┘          │
 │              │                 │
 │              ▼                 │
 │        "on the street" ──────┘
 │
 ▼
Output

                                          "I"
                                          │
                                          ▼
"I" ───► "see" ───► "a" ───► "car" ───► "parked" ───► "on" ───► "the" ───► "street"
  │       │        │        │         │          │          │          │           │
  │       │        │        │         │          │          │          │           │
  └───────┴────────┴────────┴─────────┴──────────┴──────────┴──────────┴───────────┘
                                         All connected!
                                        (bidirectional)
```

### For Driving CoT

```
Example: "The car [MASK] on the street"

LSTM:
- "car" only knows words BEFORE it ("The")
- Harder to predict what happened

BERT:
- "car" knows BOTH "The" (before) AND "on the street" (after)
- Can predict: "was parked", "drove", "stopped"
```

---

## Why BERT is Powerful

### 1. Bidirectional Context

```
"I went to the bank to withdraw money"

BERT understands: "bank" = financial institution (context from "money")

"I sat by the river bank"

BERT understands: "bank" = river edge (context from "river")
```

### 2. Deep Understanding

```
12 layers = 12 levels of abstraction

Layer 1: Surface-level ("car" is a word)
Layer 6: Syntactic ("car" is a noun)
Layer 12: Semantic ("car" is a vehicle, needs road, has wheels, etc.)
```

### 3. Transfer Learning

```
Instead of training from scratch:

1. Download BERT (pre-trained on 16GB text)
2. Fine-tune on YOUR task (CoT reasoning)
3. Benefits from massive pre-training!

vs.

1. Train LSTM from scratch on YOUR data
2. No transfer from general language knowledge
```

### 4. Attention Visualization

```
Can see what BERT "looks at" for each prediction:

For "I should brake":
- High attention: "car", "slowing", "ahead"
- Low attention: "I", "see", "a"

Shows BERT is REASONING about the driving situation!
```

---

## BERT for Autonomous Driving CoT

### Our Use Case

```
CoT Input:
"I see a car 50m ahead [SEP] The car is braking [SEP] I should brake too"

Goal: Predict next waypoints based on reasoning

BERT's role:
1. Encode the entire CoT text
2. "brake" understands context from "car ahead" and "braking"
3. CLS token summarizes: "Obstacle detected, need to brake"
4. Use CLS for waypoint prediction
```

### Architecture in Our Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    BERT for Driving CoT                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: "I see a car 50m ahead [SEP] The car is braking [SEP] I should brake" │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              BERT Encoder (bert-base-uncased)           │    │
│  │                                                         │    │
│  │   - 12 transformer layers                               │    │
│  │   - 768 hidden dimensions                               │    │
│  │   - 12 attention heads                                  │    │
│  │                                                         │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Output: CLS embedding (768-dim)           │    │
│  │                      ↓                                  │    │
│  │         ┌────────────────────────────┐                │    │
│  │         │   Fusion + Decoder         │                │    │
│  │         │   (predict waypoints)     │                │    │
│  │         └────────────────────────────┘                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│                    Waypoint Prediction                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Benefits for Driving

| BERT Feature | Driving Application |
|--------------|-------------------|
| **Context** | "car" in "parking car" vs "red car" |
| **Bidirectional** | "slowing" understands both perception and action |
| **Pre-trained** | Already knows driving terms |
| **Attention** | See what reasoning steps are important |
| **Transfer** | General language → driving reasoning |

---

## Quick Reference

### Key Concepts

| Term | Definition |
|------|------------|
| **Tokenization** | Splitting text into word pieces |
| **Embedding** | Converting tokens to numbers |
| **Attention** | How words relate to each other |
| **Self-Attention** | Each word attends to all other words |
| **Multi-Head** | Multiple attention patterns |
| **Residual** | Adding input to output |
| **LayerNorm** | Normalizing layer outputs |
| **CLS** | Classification token for summaries |

### BERT-base Specifications

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| Parameters | 110M |
| Vocabulary | 30,522 tokens |
| Max sequence | 512 tokens |

### Common Commands

```python
# Using BERT in our code
from transformers import BertModel, BertTokenizer

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize text
inputs = tokenizer("I see a car", return_tensors='pt')

# Forward pass
outputs = model(**inputs)
embeddings = outputs.last_hidden_state  # [1, seq_len, 768]
cls_embedding = embeddings[:, 0, :]  # [1, 768]
```

---

## Summary

### Why BERT for CoT?

1. **Full Context**: Each word understands all others
2. **Pre-trained**: Knows English and driving terms
3. **Attention**: See what reasoning steps matter
4. **Easy**: Just download and fine-tune

### Key Takeaways

```
BERT = Tokenization + Position Embeddings + 
       Multi-Head Self-Attention × 12 +
       Feed-Forward + Residual Connections

Magic = Self-attention lets words "talk" to each other
```

---

## Further Reading

| Resource | Link |
|----------|------|
| Original BERT Paper | https://arxiv.org/abs/1810.04805 |
| BERT Explained (Jay Alammar) | https://jalammar.github.io/illustrated-bert/ |
| HuggingFace BERT | https://huggingface.co/transformers/model_doc/bert.html |

---

*Document created for learning purposes. Understanding BERT helps implement CoT reasoning for autonomous driving.*
