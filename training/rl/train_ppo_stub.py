"""PPO training stub.

This file exists to show intended wiring for online RL.

Real PPO requires:
- policy + value networks
- advantage estimation (GAE)
- minibatch updates with clipping

Implement once we decide the first simulator environment.
"""

from __future__ import annotations

from training.rl.env_interface import ToyEnv


def main() -> None:
    env = ToyEnv()
    obs = env.reset()

    # Placeholder random policy
    done = False
    total_reward = 0.0
    while not done:
        action = {"a": 0.0, "kappa": 0.0}
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"[rl/ppo] finished stub rollout. total_reward={total_reward}")


if __name__ == "__main__":
    main()
