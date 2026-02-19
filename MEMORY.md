# MEMORY.md - Long-Term Memory

Last updated: 2026-02-15

## Key Learnings

### Architecture Patterns

**Residual Delta Learning for RL after SFT**
- Keep SFT model fixed, train only a small delta head
- More sample-efficient, safer, modular
- Pattern: `final_waypoints = sft_waypoints + delta_head(z)`

**Evaluation-First Design**
- Add ADE/FDE metrics during training, not after
- Enables checkpoint selection based on quality metrics
- Critical for autonomous driving where precision matters

### Tools & Infrastructure

**MiniMax-M2.5 as Default Model**
- Primary model for most tasks
- Fallbacks: Claude Sonnet 4.5, GPT-5.2, Gemini 3 Flash, Grok 4.1

**Agent Configuration**
- `main` and `survey` agents available for interactive spawning
- `pipeline` agent handles cron-scheduled daily PR tasks

### Project Structure

**Driving-First Pipeline**
- Waymo episodes → SSL pretrain → Waypoint BC → RL refinement → CARLA ScenarioRunner eval
- Evaluation at multiple stages: SSL contrastive, waypoint BC (ADE/FDE), CARLA metrics

## Preferences

- Prefer PR-first workflow for roadmap/maintain tasks
- Use GitHub SSH for repo operations
- Default model: MiniMax-M2.5
- PPO improvements: stable RL advances (clipping, value centering, GAE), model-based RL integration (GAIA-2 style), LoRA for RL heads
- CoT data generation: explore reasoning trace synthesis for driving decisions
- RL algorithm upgrade: plan GRPO migration after PPO baseline is stable
- DeepSeek pipeline: survey pre-train → reasoning SFT → RL → SFT/RLHF for future upgrades

### Implemented RL Algorithms

**ResAD (Residual with Attention and Dynamics)** - 2026-02-18
- Location: `training/rl/resad.py`, `training/rl/resad_train.py`
- Architecture: UncertaintyHead + ResADResidualHead + InertialReferenceTransform
- Formula: Δ_norm = (y - ŷ) / σ
- Loss: NLL + MSE + KL regularization
- Use case: Residual correction after frozen SFT model
- Branch: `feature/grpo-implementation`

### Memory Research: DeepSeek Engram

**DeepSeek Engram Survey** - 2026-02-18
- Location: `docs/surveys/2026-02-18-deepseek-engram.md`
- Research lineages:
  - Memory 支线: FFN=KV Memory → Knowledge Neurons → RETRO → Memory Layer
  - N-gram 支线: Traditional N-gram → N-Grammer → Scaling Embedding
- Five core insights:
  1. FFN is Key-Value Memory
  2. Sparsity breaks "Impossible Triangle" (Performance/Compute/Model Size)
  3. Hash table enables O(1) N-gram retrieval
  4. Engram allows "remembering" commonsense without "computing"
  5. DeepSeek Engram is culmination of prior research
- MoE vs Engram relationship documented
- Autonomous driving application: Engram + AR Decoder integration
- Branch: `feature/deepseek-engram-survey`

## Contacts & Context

- User: Qi (timezone: America/New_York, often schedules in America/Los_Angeles for CL cadence)
- Research tasks: Use survey agent by default
