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

## Contacts & Context

- User: Qi (timezone: America/New_York, often schedules in America/Los_Angeles for CL cadence)
- Research tasks: Use survey agent by default
