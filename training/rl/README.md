# RL (reinforcement learning) — skeleton

RL is used to optimize task reward + constraints beyond imitation.

## Variants to consider

### Offline RL (from logs)
- Pros: no simulator interaction required; safer.
- Cons: algorithmic complexity; distributional shift; need well-logged rewards/costs.

### Online RL in simulation (e.g., PPO/SAC)
- Pros: direct reward optimization; can improve beyond demonstrations.
- Cons: requires a stable sim environment + careful safety constraints.

### Preference optimization / RLHF-style (trajectory preferences)
- Learn a reward model from comparisons, then optimize policy.

## What this repo provides now
- An **environment interface contract** (so we can swap CARLA/MuJoCo/toy envs)
- A **PPO training stub** to show wiring (not a complete implementation)

Once we choose the first runnable sim loop, we can implement one RL path fully.
