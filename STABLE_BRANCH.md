# Stable Improvements Branch (2026-02)

This branch contains proven/improved components for the autonomous driving pipeline.

## What's Included

### 1. Surveys & Documentation
- **VLA + World Model Survey** (`docs/surveys/2026-02-16-vla-world-model-2025-survey.md`) - 50+ papers
- **GigaBrain VLA+RL Survey** (`docs/surveys/2026-02-16-gigabrain-vla-rl.md`)
- **CoT Survey** (`docs/surveys/2026-02-15-cot-survey.md`)
- **Horizon Robotics Survey** (`docs/surveys/2026-02-16-horizon-robotics.md`)
- **BERT Explained** (`docs/papers/bert-explained.md`)
- **Complete Pipeline** (`docs/pipeline/complete-pipeline.md`)

### 2. SFT Training (Tested)
- **CoT Training** (`training/sft/train_waypoint_bc_cot.py`) - BERT-based reasoning
- **LLM CoT Generator** (`training/sft/llm_cot_generator.py`) - Generate reasoning traces
- **Waypoint BC** (`training/sft/train_bc.py`, `dataloader_waypoint_bc.py`)

### 3. RL Training (Tested)
- **PPO Waypoint Delta** (`training/rl/train_ppo_waypoint_delta.py`) - RL refinement
- **Compare SFT vs RL** (`training/rl/compare_sft_vs_rl.py`) - Evaluation
- **Toy Waypoint Env** (`training/rl/toy_waypoint_env.py`) - Testing environment

### 4. Data Pipeline
- **Unified Dataset** (`training/data/unified_dataset.py`) - SSL + waypoint data
- **Quickstart** (`training/data/quickstart.py`)

## Validation Results (Toy Environment)

```
ADE: 14.12m (SFT) → 13.70m (RL) [+3%]
FDE: 41.92m (SFT) → 41.16m (RL) [+2%]
Success: 0% (SFT) → 0% (RL) [+0%]
```

Both SFT and RL working. RL shows consistent +2-3% improvement over SFT.

## What's NOT Included (Experimental)

The following are in the experimental branch (`feature/daily-2026-02-16-rebase`):
- VLA Planner (`training/models/vla_planner.py`)
- World Model (`training/models/world_model.py`)
- Safety Layer (`training/models/safety_layer.py`)
- Unified Trainer (`training/unified/trainer.py`)
- Language Conditioning (`training/sft/language_conditioning.py`)

## Next Steps

1. Merge this branch
2. Integrate with real driving data (Waymo, nuScenes)
3. Add GPU training
4. Revisit experimental features after stable pipeline is working

---

**Branch:** `feature/stable-improvements-2026-02`
**Last Updated:** 2026-02-16
