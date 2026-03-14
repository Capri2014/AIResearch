# Pipeline PR #2: SSL-to-Waypoint BC Transfer Learning

## Summary
Add SSL pretrained model loader and integration with waypoint BC pipeline, enabling transfer learning from self-supervised pretrained encoders to the driving policy.

## Changes

### New File: `training/sft/ssl_pretrained_loader.py`
- **SSLConfig**: Configuration dataclass for SSL pretrained models
- **SSLEncoder**: ResNet-based encoder wrapper with projection head
- **JEPAEncoder**: Joint Embedding Predictive Architecture encoder with predictor
- **SSLFeatureExtractor**: Feature extraction utility class
- **load_ssl_pretrained()**: Load SSL checkpoint and return model
- **BCWithSSLEncoder**: Waypoint BC model with SSL pretrained encoder initialization
- **create_bc_with_ssl_pretrained()**: Factory function to create BC model with SSL encoder

### Updated: `sim/driving/carla_srunner/policy_wrapper.py`
- Added SSL_PRETRAINED_AVAILABLE flag
- Added **SSLWaypointPolicyWrapper**: CARLA-integrable policy using SSL pretrained encoder
- Supports loading from SSL checkpoints (JEPA, contrastive, temporal_contrastive)
- Seamlessly integrates with existing waypoints_to_control() method

## Architecture
```
Waymo Episodes → SSL Pretrain (JEPA/Contrastive)
                        ↓
              SSL Encoder (frozen)
                        ↓
              BCWithSSLEncoder → waypoints → CARLA
```

## Usage

### Python API
```python
from training.sft.ssl_pretrained_loader import (
    load_ssl_pretrained,
    create_bc_with_ssl_pretrained,
)

# Load SSL pretrained checkpoint
ssl_model = load_ssl_pretrained("checkpoints/ssl_jepa.pt", model_type="jepa")

# Create BC model with SSL encoder
bc_model = create_bc_with_ssl_pretrained(
    checkpoint_path="checkpoints/ssl_jepa.pt",
    num_waypoints=8
)
```

### CARLA Integration
```python
from sim.driving.carla_srunner.policy_wrapper import SSLWaypointPolicyWrapper

policy = SSLWaypointPolicyWrapper(
    ssl_checkpoint=Path("checkpoints/ssl_jepa.pt"),
    model_type="jepa",
    num_waypoints=8
)
policy.initialize()
waypoints = policy.predict(images)
control = policy.waypoints_to_control(waypoints)
```

## Testing
- ✓ ssl_pretrained_loader smoke test
- ✓ policy_wrapper import test

## Notes
- Falls back to simple CNN encoder if torchvision unavailable
- Supports multiple SSL model types: jepa, contrastive, temporal_contrastive
- Encoder weights are frozen by default for transfer learning
