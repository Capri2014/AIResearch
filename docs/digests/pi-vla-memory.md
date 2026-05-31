# Pi Robotics: VLAs with Long and Short-Term Memory

**Date:** 2026-03-08  
**Status:** Survey Complete  
**Source:** Physical Intelligence (π) - https://www.pi.website/research/memory

---

## 1. Overview

**Paper:** VLAs with Long and Short-Term Memory  
**Authors:** Marcel Torne, Karl Pertsch, Homer Walke, Kyle Vedder, Suraj Nair, Brian Ichter, Allen Ren, Haohuan Wang, Jiaming Tang, Kyle Stachowicz, Karan Dhabalia, Michael Equi, Quan Vuong, Jost Tobias Springenberg, Sergey Levine, Chelsea Finn, Danny Driess  
**Published:** March 3, 2026  
**Venue:** Physical Intelligence

---

## 2. Core Problem

Current robotic foundation models (VLAs) can perform individual skills like:
- "wash the frying pan"
- "fold the laundry"
- "make a peanut butter sandwich"

**The Bottleneck:** As individual skills become robust, the bottleneck shifts to **how the robot deploys skills to solve complex tasks** — not the skills themselves.

**Requirements for complex multi-stage behaviors:**
1. Maintain coherent narrative of task progress
2. Remember object locations even when out of view
3. Recall what worked/didn't work in the past
4. **This requires both long-term and short-term memory**

---

## 3. Key Challenges

### 3.1 Context Length Problem
- Keeping full history in context over minutes/hours is infeasible
- Raw images in context impractical due to real-time control requirements

### 3.2 Causal Confusion
- Poorly designed memory can **harm** performance
- Memory exacerbates spurious correlations in imitation learning
- "Causal confusion" phenomenon (see: arXiv:1905.11979)

### 3.3 Why Memory is Hard
- Even capable learning systems typically demonstrated on short tasks (few minutes)
- Designing effective memory architectures is challenging

---

## 4. Solution: Multi-scale Embodied Memory (MEM)

### 4.1 Architecture Overview

MEM provides VLAs with memory for **long-horizon tasks** (up to 15 minutes):

```
┌─────────────────────────────────────────────────────────┐
│                    VLA + MEM                             │
├─────────────────────────────────────────────────────────┤
│  Short-Term Memory:                                     │
│  - Raw observations (video frames)                      │
│  - Efficient video encoder                              │
│  - Recent events in detail                              │
├─────────────────────────────────────────────────────────┤
│  Long-Term Memory:                                      │
│  - Abstract concepts in natural language               │
│  - Task progress narrative                              │
│  - Object locations, past experiences                   │
├─────────────────────────────────────────────────────────┤
│  Reasoning Mechanism:                                   │
│  - What to remember                                    │
│  - What to forget                                      │
│  - High-level subtask selection                        │
│  - In-context adaptation                               │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Two Key Components

| Component | Description | Purpose |
|-----------|-------------|---------|
| **Video Encoder** | Efficient short-horizon frame-based history | Short-term memory |
| **Language-based Memory** | Natural language abstraction | Long-term context |

### 4.3 What MEM Enables

- **15-minute tasks**: Solve tasks requiring up to 15 minutes of memory
- **Partial observability**: Handle objects not in current view
- **In-context adaptation**: Correct mistakes through adaptation
- **Multi-scale reasoning**: Both short-term details + long-term abstractions

---

## 5. Multimodal Memory Design

### 5.1 Short-Term Memory
- Maintained as **raw observations** (recent events in detail)
- Efficient video encoding for real-time control
- Frames capture immediate context

### 5.2 Long-Term Memory
- Stored as **abstract concepts in natural language**
- Examples from π0.6 with MEM:
  ```
  "I moved to the stove"
  "I picked up the lid from the pot"
  "I placed the lid on the countertop to the left of the stove"
  "I am currently holding the pot and moved to the sink"
  ```

### 5.3 Active Memory Management
- Model actively chooses **what to remember**
- Model actively chooses **how to remember**
- Reasoning mechanism also selects **high-level subtasks**
- "Reasoning about what to do AND what to remember"

---

## 6. Results & Capabilities

### 6.1 Demonstrations
- Clean up entire kitchen
- Cook a grilled cheese sandwich from scratch
- 15-minute manipulation tasks

### 6.2 Key Capabilities
| Capability | Description |
|------------|-------------|
| **Long-horizon** | Up to 15 minutes of memory |
| **Partial observability** | Remember objects not in view |
| **In-context adaptation** | Correct mistakes without retraining |
| **Multi-task** | Diverse robot and non-robot data |

---

## 7. Relation to Driving

This work is highly relevant to autonomous driving:

### 7.1 Similar Challenges
- Long-horizon planning (multi-minute missions)
- Memory of past events (traffic, road conditions)
- Partial observability (occlusions, limited sensor range)
- In-context adaptation (handle edge cases)

### 7.2 Potential Applications

| Driving Scenario | MEM Solution |
|-----------------|--------------|
| Navigate complex route | Remember sub-goals, completed segments |
| Remember parked locations | Long-term memory in language |
| Handle occlusions | Short-term video history |
| Learn from mistakes | In-context adaptation |

### 7.3 Architecture Inspiration

```
Driving VLA + MEM
├── Short-term: Recent frames, sensor history
├── Long-term: Route progress, map context, past experiences
└── Reasoning: What to remember, next subtask
```

---

## 8. Related Pi Robotics Papers

| Paper | Key Idea |
|-------|----------|
| **FAST** | Efficient Robot Action Tokenization |
| **Hirobot** | Teaching Robots to Listen and Think Harder |
| **Human to Robot** | Emergence of Human to Robot Transfer in VLAs |
| **Knowledge Insulation** | VLAs that Train Fast, Run Fast, Generalize |
| **Real-Time Chunking** | Real-Time Action Chunking with Large Models |

---

## 9. References

1. Physical Intelligence Memory Paper: https://www.pi.website/research/memory
2. Causal Confusion in Imitation Learning: https://arxiv.org/abs/1905.11979
3. π0.5 (previous VLA work)
4. Open X-Embodiment (robotics foundation models)

---

## 10. Implementation Ideas

### For Driving

```python
class EmbodiedMemory:
    def __init__(self):
        self.short_term = VideoMemoryEncoder()  # Recent frames
        self.long_term = LanguageMemory()      # Abstract context
        self.reasoning = ReasoningModule()      # What to remember
    
    def update(self, observation, task_context):
        # Short-term: Encode current frame
        short_mem = self.short_term.encode(observation)
        
        # Reasoning: What matters?
        memory_decision = self.reasoning.decide(
            observation, self.long_term, task_context
        )
        
        # Long-term: Abstract to language
        if memory_decision.should_remember:
            abstract = self.reasoning.abstract(observation)
            self.long_term.add(abstract)
        
        return short_mem, self.long_term.get_context()
    
    def get_action(self, observation, task):
        short = self.short_term.encode(observation)
        long = self.long_term.get_context()
        return self.vla.act(observation, short, long, task)
```

---

## Survey Status

- [x] Problem statement
- [x] MEM architecture
- [x] Short-term + Long-term memory
- [x] Reasoning mechanism
- [x] Driving applications
- [x] Implementation ideas
