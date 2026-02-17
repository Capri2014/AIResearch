# DreamerV3 World Model: Learned Simulator for Autonomous Driving

**Survey:** DreamerV3 (Hafner et al., Nature 2025) + Driving World Model Applications  
**Date:** 2026-02-17  
**Context:** Ashok's "video + action → next video" simulator claim  

---

## TL;DR

DreamerV3 demonstrates that a **learned world model** can serve as a simulator: given current observation and action, predict next observation. This matches Ashok's core claim for autonomous driving. Key results:

- **Unified framework**: Works across Atari, DMLab, continuous control with **fixed hyperparameters**
- **Latent dynamics**: RSSM model learns compact representation of scene dynamics
- **Imagined rollouts**: Policy trained entirely in latent space without real environment interaction
- **Scalability**: Performance improves with larger models and more training

**For AIResearch**: Implement a minimal driving-specific world model following DreamerV3 architecture. Use nuScenes data for training. Enable synthetic scenario generation for regression testing and adversarial injection.

---

## Model Objective & Architecture

### Core Idea: Learn P(next_obs | obs, action)

```
┌─────────────────────────────────────────────────────────────────┐
│              DreamerV3: Learned Simulator Framework             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Real world interaction:                                        │
│   obs_t → encode → z_t → dynamics(z_t, a_t) → z_{t+1} → decode → obs_{t+1}    │
│                                                                  │
│   Imagined rollout (no real env):                               │
│   z_T → actor(z_T) → a_T → dynamics → z_{T+1} → ... → trajectory    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### World Model (RSSM - Recurrent State-Space Model)

**Objective**: Maximize evidence lower bound (ELBO) on observation likelihood

```
L_worldmodel = E_q[ log p(x_t | z_t) ] - KL( q(z_t | h_t, x_t) || p(z_t | h_t) )
```

| Component | Equation | Purpose |
|-----------|----------|---------|
| **Posterior** | q(z_t \| h_t, x_t) | Encode observation → latent (uses true frame) |
| **Prior** | p(z_t \| h_t) | Predict latent from dynamics only |
| **Dynamics** | h_t = f(h_{t-1}, a_{t-1}, z_{t-1}) | RNN updates hidden state |
| **Reconstruction** | p(x_t \| z_t) | Decode latent → observation |
| **Reward predictor** | p(r_t \| z_t) | Learn reward from latent |

**Key insight**: Latent `z_t` is categorical (128 buckets per dimension) for stable discrete representation.

### Rollout Mechanism

```
Training (teacher forcing):
obs_t → encode → z_t → dynamics(h_{t-1}, a_{t-1}, z_{t-1}) → h_t → predict z'_t, r_t, x'_t
                                                                ↑
                                                          Use predicted z'_t

Inference (autoregressive):
z_t → actor(z_t) → a_t → dynamics(h_t, a_t, z_t) → h_{t+1}, z_{t+1} → repeat
```

---

## Action-Conditioned Video Generation Requirements

### For Driving: "Video + Action → Next Video"

| Requirement | DreamerV3 Approach | Driving Adaptation |
|-------------|-------------------|---------------------|
| **Temporal consistency** | LSTM dynamics over `h_t` | Same mechanism |
| **Multi-camera input** | Single-frame encoder | Encode all views, concatenate or cross-attend |
| **Ego-motion conditioning** | Actions include velocity | CAN bus data (speed, steering) as actions |
| **Long-horizon prediction** | Imagined rollouts to 100+ steps | Same, but need error mitigation |
| **Multi-view consistency** | Not natively supported | Add cross-view attention mechanism |

### Minimal Driving World Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DrivingWorldModel(nn.Module):
    """
    Minimal world model: P(next_video | current_video, actions)
    Following DreamerV3 RSSM architecture, adapted for multi-view driving.
    """
    def __init__(self, obs_shape=(3, 224, 224), action_dim=3, 
                 hidden_dim=512, latent_dim=32, latent_classes=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.latent_classes = latent_classes
        
        # Encoder: observation → hidden
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, hidden_dim),
            nn.ReLU(),
        )
        
        # Recurrent dynamics: (hidden, action, latent) → next hidden
        self.dynamics_rnn = nn.GRUCell(hidden_dim + action_dim + latent_dim * latent_classes, 
                                        hidden_dim)
        
        # Prior network: hidden → latent logits
        self.prior_net = nn.Linear(hidden_dim, latent_dim * latent_classes)
        
        # Posterior network: hidden + encoded obs → latent logits
        self.post_net = nn.Linear(hidden_dim * 2, latent_dim * latent_classes)
        
        # Decoder: latent → observation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * latent_classes, 256 * 6 * 6),
            nn.Unflatten(1, (256, 6, 6)),
            nn.ConvTranspose2d(256, 128, 4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2),
        )
        
        # Reward predictor
        self.reward_head = nn.Linear(latent_dim * latent_classes, 1)
    
    def forward(self, obs, action, hidden=None, use_posterior=True):
        """
        obs: [B, C, H, W]
        action: [B, action_dim] (steer, throttle, brake)
        hidden: [B, hidden_dim], optional
        """
        batch_size = obs.shape[0]
        
        # Encode observation
        obs_feat = self.encoder(obs)  # [B, hidden_dim]
        
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_dim, device=obs.device)
        
        # Posterior: use true observation
        post_feat = torch.cat([hidden, obs_feat], dim=-1)
        post_logits = self.post_net(post_feat)  # [B, latent_dim * latent_classes]
        post_z = self.sample_latent(post_logits)  # [B, latent_dim]
        
        # Prior: without observation
        prior_logits = self.prior_net(hidden)
        prior_z = self.sample_latent(prior_logits)
        
        # Choose z based on training vs inference
        z = post_z if use_posterior else prior_z
        
        # Dynamics update
        z_flat = post_z.flatten(start_dim=1)  # [B, latent_dim * latent_classes]
        dynamics_in = torch.cat([hidden, action, z_flat], dim=-1)
        hidden = self.dynamics_rnn(dynamics_in, hidden)
        
        # Decode next observation
        recon = self.decoder(z_flat)
        
        # Predict reward
        reward = self.reward_head(z_flat)
        
        return recon, reward, hidden, prior_logits, post_logits
    
    def sample_latent(self, logits):
        """Sample categorical latent with gumbel-softmax during training."""
        # logits: [B, latent_dim * latent_classes]
        # Return: [B, latent_dim] (argmax for inference, gumbel-softmax for training)
        if self.training:
            # Gumbel-softmax for differentiable sampling
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8))
            soft_z = F.softmax((logits + gumbel) / 0.1, dim=-1)
            z = soft_z.view(soft_z.shape[0], self.latent_dim, self.latent_classes)
            z = z.sum(dim=-1)  # [B, latent_dim]
        else:
            # Argmax for inference
            z = logits.view(logits.shape[0], self.latent_dim, self.latent_classes)
            z = z.argmax(dim=-1)  # [B, latent_dim]
        return z
    
    def imagine_rollout(self, start_obs, actions, horizon=50):
        """
        Generate imagined trajectory without real environment.
        start_obs: [B, C, H, W]
        actions: [horizon, B, action_dim]
        """
        batch_size = start_obs.shape[0]
        hidden = None
        traj_rewards = []
        
        for t in range(horizon):
            _, reward, hidden, _, _ = self.forward(
                start_obs if t == 0 else None,  # Only use true obs at t=0
                actions[t],
                hidden,
                use_posterior=(t == 0)  # Use posterior only at start
            )
            traj_rewards.append(reward)
        
        return torch.stack(traj_rewards)  # [horizon, B, 1]


class Actor(nn.Module):
    """Policy that operates on latent states."""
    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, z):
        return self.network(z)


class Critic(nn.Module):
    """Value function on latent states."""
    def __init__(self, latent_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, z):
        return self.network(z)
```

---

## Regression Testing & Adversarial Injection

### The Simulator as Test Harness

```
┌─────────────────────────────────────────────────────────────────┐
│              Learned Simulator Testing Pipeline                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Train world model on real driving data (nuScenes/Waymo)   │
│                         ↓                                        │
│  2. Freeze model → use as fixed simulator                       │
│                         ↓                                        │
│  3. Generate synthetic episodes:                                 │
│     - Baseline scenarios (training distribution)                │
│     - Edge cases (adversarial perturbations)                    │
│                         ↓                                        │
│  4. Run planner/simulator through synthetic episodes           │
│                         ↓                                        │
│  5. Measure: collision rate, violation rate, scenario coverage   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Adversarial Injection Strategies

| Strategy | Implementation | Use Case |
|----------|---------------|----------|
| **Latent perturbation** | `z' = z + ε, ε ~ N(0, σ)` | Generate variant scenarios |
| **Action noise** | `a' = clip(a + δ, a_min, a_max)` | Test control robustness |
| **Initial state sampling** | Sample z from distribution tails | Find failure modes |
| **Scenario composition** | Combine latent representations | Complex multi-agent scenes |

### Regression Testing Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Collision rate** | % episodes with collision | < 0.1% |
| **ADE/FDE** | Average/Final Displacement Error for waypoints | < 0.5m / < 1.0m |
| **Scenario coverage** | % unique scenario types covered | > 90% |
| **Adversarial success rate** | % adversarial episodes completed safely | > 80% |

### Adversarial Testing Code Stub

```python
def generate_adversarial_episodes(simulator, planner, n_episodes=1000, epsilons=[0.01, 0.05, 0.1]):
    """
    Generate adversarial scenarios via latent space perturbations.
    """
    results = {}
    
    for eps in epsilons:
        failures = 0
        
        for _ in range(n_episodes):
            # Sample initial state from training distribution
            init_obs = sample_nuscenes_batch(batch_size=1)
            init_action = sample_actions(batch_size=1)
            
            # Perturb latent
            z = simulator.encode(init_obs)
            z_perturbed = z + torch.randn_like(z) * eps
            
            # Roll out trajectory using planner
            traj = simulator.rollout_from_latent(z_perturbed, horizon=50)
            
            # Check for collision/violation
            if detect_collision(traj) or detect_violation(traj):
                failures += 1
        
        results[f"eps_{eps}"] = {
            "failure_rate": failures / n_episodes,
            "n_failures": failures,
            "n_total": n_episodes
        }
    
    return results


def regression_test_suite(simulator, planner, test_scenarios):
    """
    Run standardized regression tests.
    """
    results = {}
    
    for name, scenario in test_scenarios.items():
        # Generate episode
        traj = simulator.rollout(scenario.init_obs, scenario.actions)
        
        # Evaluate
        results[name] = {
            "collision": detect_collision(traj),
            "violation": detect_violation(traj),
            "ade": compute_ade(traj, scenario.ground_truth),
            "fde": compute_fde(traj, scenario.ground_truth),
        }
    
    # Aggregate
    aggregate = {
        "collision_rate": sum(1 for r in results.values() if r["collision"]) / len(results),
        "violation_rate": sum(1 for r in results.values() if r["violation"]) / len(results),
        "mean_ade": sum(r["ade"] for r in results.values()) / len(results),
        "mean_fde": sum(r["fde"] for r in results.values()) / len(results),
    }
    
    return results, aggregate
```

---

## Action Items for AIResearch

### Phase 1: Minimal V1 (2-3 weeks)

| Task | Deliverable | Dependencies |
|------|-------------|--------------|
| Implement `DrivingWorldModel` class | PyTorch module matching RSSM architecture | None |
| Training loop on nuScenes | Script training world model on 10K frames | nuScenes API |
| Baseline metrics | ADE/FDE on validation set | None |
| Evaluation harness | ADE/FDE, collision detection code | nuScenes devkit |

### Phase 2: Multi-View Extension (2 weeks)

| Task | Deliverable | Dependencies |
|------|-------------|--------------|
| Multi-camera encoder | Encode front/left/right views jointly | Phase 1 |
| Cross-view attention | Consistent multi-view predictions | Phase 1 |
| Extended nuScenes training | Train on full multi-view data | nuScenes 6-camera data |

### Phase 3: Simulator Integration (3 weeks)

| Task | Deliverable | Dependencies |
|------|-------------|--------------|
| Adversarial injection framework | Latent perturbation generation | Phase 1 |
| CARLA integration | Sim-to-sim transfer test | CARLA simulator |
| Regression test suite | 100+ standardized test scenarios | Phase 1 + 2 |

### Success Criteria

| Metric | Target (V1) | Target (V2) |
|--------|-------------|-------------|
| ADE @ 1s | < 1.0m | < 0.5m |
| FDE @ 3s | < 2.0m | < 1.0m |
| Adversarial success rate | > 70% | > 85% |
| Inference speed | > 10 FPS | > 30 FPS |

---

## Citations & Links

### Primary Paper

**DreamerV3 (Nature 2025)**:
- Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2025). Mastering diverse control tasks through world models. *Nature*.
- arXiv: https://arxiv.org/abs/2301.04104 (original ICLR 2023 version)
- Code: https://github.com/danijar/dreamerv3

### Related World Model Work

**DreamerV2 (ICLR 2021)**:
- Hafner, D., et al. (2021). Mastering Atari with discrete world models. *ICLR*.
- https://arxiv.org/abs/2010.02193

**World Models (NeurIPS 2018)**:
- Ha, D., & Schmidhuber, J. (2018). World models. *NeurIPS*.
- https://arxiv.org/abs/1803.10122

**GAIA-1 (Wayve)**:
- Generative world model for autonomous driving (mentioned in Wayve research)
- https://wayve.ai/thinking/gaia1/

### Datasets

| Dataset | URL | Notes |
|---------|-----|-------|
| nuScenes | https://www.nuscenes.org/download | Multi-view, 1.4M samples |
| Waymo Open | https://waymo.com/open/ | High-quality, 200M samples |
| Argoverse | https://www.argoverse.org/ | Driving-specific |

### Code Resources

| Resource | URL | Purpose |
|----------|-----|---------|
| DreamerV3 official | https://github.com/danijar/dreamerv3 | Reference implementation |
| DreamerV3 minset | https://github.com/NM512/dreamerv3-miniset | Simplified version |
| JAX version | https://github.com/danijar/dreamerv3 | Original framework |

---

## Summary

**What**: DreamerV3 provides a unified framework for learning world models that function as simulators—predicting next observations given current state and actions.

**Why it matters**: Matches Ashok's "video + action → next video" claim. Enables:
- Policy training without real-world interaction (imagined rollouts)
- Synthetic scenario generation for regression testing
- Adversarial injection via latent space perturbations

**Key components**:
1. **RSSM dynamics model**: Learns compact latent representation of scene dynamics
2. **Latent rollouts**: Autoregressive prediction of future states
3. **Fixed hyperparameters**: Works across domains without tuning

**For AIResearch**:
- Implement minimal `DrivingWorldModel` following DreamerV3 (2-3 weeks)
- Train on nuScenes with MSE reconstruction + KL loss
- Enable adversarial testing via latent perturbations
- Integrate with CARLA for sim-to-sim validation

**Gaps to address**: Error accumulation in long rollouts, multi-view consistency, real-time inference constraints.

---

*PR: Survey PR #4: DreamerV3 World Model Digest*  
*Summary: DreamerV3 (Nature 2025) demonstrates that learned world models can function as simulators—P(next_obs | obs, action)—matching Ashok's core claim. Architecture: RSSM with categorical latents, recurrent dynamics, MSE reconstruction + KL loss. Key results: unified framework works across Atari/DMLab/continuous control with fixed hyperparameters. For AIResearch: Implement minimal DrivingWorldModel (RSSM encoder-dynamics-decoder) trained on nuScenes. Enable regression testing via imagined rollouts and adversarial injection via latent perturbations (ε ~ N(0, σ)). Deliverables: V1 model in 2-3 weeks, multi-view extension in 2 weeks, CARLA integration in 3 weeks. Targets: ADE < 1.0m, adversarial success > 70%. Citations: DreamerV3 (arXiv:2301.04104), DreamerV2 (arXiv:2010.02193), nuScenes (nuscenes.org).*
