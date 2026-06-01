#!/usr/bin/env python3
"""
Minimal π0/π0.5 Inference Trace — understand the data flow.
Replace mocks with real code from OpenPI repo when cloned.

Usage: python scripts/pi0_minimal_inference.py
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# ==============================================================================
# Step 1: Config (from YAML or hardcoded for mental model)
# ==============================================================================
@dataclass
class Pi0Config:
    """π0 model configuration."""
    # Model
    model_name: str = "pi0-ur5e-base"
    vlm_backbone: str = "paligemma-3b"  # or "Qwen2-VL"
    
    # Observation config
    image_size: tuple = (224, 224)
    image_keys: List[str] = field(default_factory=lambda: ["front", "wrist"])
    state_dim: int = 14  # 7 joints + 7 velocities
    
    # Action config
    action_dim: int = 7  # 6 joints + gripper
    chunk_length: int = 7  # timesteps per forward pass
    
    # Control
    control_frequency_hz: float = 50.0  # 20ms per step
    
    # Robot config
    robot_type: str = "ur5e"
    
    def obs_keys(self) -> Dict[str, str]:
        return {"image": self.image_keys, "state": "state", "language": "language"}


@dataclass  
class Pi05Config(Pi0Config):
    """π0.5 config adds semantic heads."""
    # Extra heads
    semantic_head: bool = True
    subtask_prediction: bool = True
    

# ==============================================================================
# Step 2: Preprocessing — INPUT FORMATS
# ==============================================================================
class Observation:
    """Single observation at timestep t."""
    def __init__(
        self,
        images: Dict[str, torch.Tensor],  # {"front": [3,H,W], "wrist": [3,H,W]}
        state: torch.Tensor,           # [14] = q(7) + dq(7)
        language: str = "",         # "pick up the cup"
    ):
        self.images = images
        self.state = state        # [14]
        self.language = language
        self.timestamp = None
    
    def shapes(self) -> str:
        imgs = ", ".join([f"{k}: {v.shape}" for k, v in self.images.items()])
        return f"Observer({{images: {imgs}, state: {self.state.shape}, lang: len={len(self.language)}}})"


class Episode:
    """Full episode = list of observations + actions."""
    def __init__(
        self,
        observations: List[Observation],
        actions: torch.Tensor,  # [T, action_dim]
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.observations = observations
        self.actions = actions  # [T, action_dim]
        self.metadata = metadata or {}
    
    def __len__(self):
        return len(self.observations)


# PREPROCESSING STEPS:
def preprocess_observation(obs: Observation, config: Pi0Config) -> Dict[str, torch.Tensor]:
    """
    Transform raw observation → model input tensors.
    
    Returns:
        {
            "image_tokens": [B, seq, D],     # from vision encoder
            "text_ids": [B, L],           # token IDs
            "state_tokens": [B, D],         # from state encoder  
        }
    """
    # IMAGE: stack all cameras → normalize
    # (Real impl: resize, normalize ImageNet, stack)
    image_tensor = torch.stack([obs.images[k] for k in config.image_keys], dim=0)  # [N_cams, 3, H, W]
    
    # LANGUAGE: tokenize
    # (Real impl: tokenizer(obs.language))
    text_ids = torch.randint(0, 32000, (32,))  # [L]
    
    # STATE: flatten q + dq
    state = obs.state  # [14]
    
    return {
        "image": image_tensor,  
        "text": text_ids,
        "state": state,
    }


# ==============================================================================
# Step 3: Model Forward (VLM Backbone)
# ==============================================================================
class Pi0Model(torch.nn.Module):
    """π0 model wrapper."""
    
    def __init__(self, config: Pi0Config):
        super().__init__()
        self.config = config
        
        # NOTE: In real impl, these are loaded from checkpoint
        # self.vlm = load_vlm(config.vlm_backbone)
        # self.vision_encoder = VisionEncoder()
        # self.state_encoder = StateEncoder(state_dim=config.state_dim)
        # self.action_expert = ActionExpert(action_dim=config.action_dim)
        # self.flow_matching_head = FlowMatchingHead()
    
    def encode_inputs(self, preprocessed: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Step 3a: Encode all modalities."""
        # Real impl:
        # image_features = self.vision_encoder(preprocessed["image"])
        # text_features = self.vlm.embed_tokens(preprocessed["text"])
        # state_features = self.state_encoder(preprocessed["state"])
        
        return {
            "image_features": torch.randn(1, 196, 1024),   # [B, seq, D]
            "text_features": torch.randn(1, 32, 1024),  # [B, L, D]
            "state_features": torch.randn(1, 1024),     # [B, D]
        }
    
    def forward(self, preprocessed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Full forward pass.
        
        Returns:
            action_logits: [B, chunk_length, action_dim]
        """
        encoded = self.encode_inputs(preprocessed)
        
        # FUSE: Late fusion (concat image + text + state)
        # hidden = fuse(encoded["image_features"], encoded["text_features"], encoded["state_features"])
        
        hidden = torch.randn(1, self.config.chunk_length, 1024)  # [B, T, D]
        
        # ACTION EXPERT: predict action latents
        # action_logits = self.action_expert(hidden)
        action_logits = torch.randn(1, self.config.chunk_length, self.config.action_dim)
        
        return action_logits


# ==============================================================================
# Step 4: Flow Matching Action Head
# ==============================================================================
class FlowMatchingHead(torch.nn.Module):
    """
    Flow matching for continuous action distribution.
    
    Key idea: Model velocity field v(z_t), integrate via ODE.
    
    p(a|cond) = argmax_z p(z|a)  where dz/dt = v(z, cond)
    
    vs Diffusion: No stochastic noise schedule, OT flow.
    """
    
    def __init__(self, action_dim: int = 7, hidden_dim: int = 1024):
        super().__init__()
        # Velocity network: z_t + condition → v_t
        self.velocity_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, action_dim),
        )
        self.num_ode_steps = 7  # Typical: 4-10
    
    def forward(self, hidden: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Single forward = one flow matching step.
        Real inference: Run ODE solver for num_steps.
        """
        # v_t = velocity_net(z_t, cond)
        velocity = self.velocity_net(torch.cat([hidden, cond], dim=-1))
        return velocity  # [B, action_dim]
    
    def decode(self, action_logits: torch.Tensor) -> torch.Tensor:
        """
        Decode action logits → action chunk.
        
        Real impl:
            z = action_logits
            for i in range(num_ode_steps):
                v = velocity_net(z, cond)
                z = z + v * dt
            return z
        """
        # Simplified: direct decode (real impl uses ODE)
        return action_logits  # Already [B, T, action_dim]


# ==============================================================================
# Step 5: Action Chunk Output
# ==============================================================================
def decode_action(action_logits: torch.Tensor, config: Pi0Config) -> torch.Tensor:
    """
    Action logits → robot command chunk.
    
    Returns:
        action_chunk: [chunk_length, action_dim]
            - timestep 0: Δq[0:6] + gripper[6]
            - ...
            - timestep T: Δq[0:6] + gripper[6]
    """
    # Real impl applies flow matching ODE
    return action_logits[0]  # [chunk_length, action_dim]


def format_for_robot(action_chunk: torch.Tensor, config: Pi0Config) -> List[float]:
    """Format action chunk for robot controller."""
    flat = action_chunk.flatten().tolist()
    # real impl: scale to joint limits, send to robot
    return flat


# ==============================================================================
# Step 6: Robot Client Interface  
# ==============================================================================
class RobotClient:
    """Interface to robot controller."""
    
    def __init__(self, robot_type: str = "ur5e", ip: str = "192.168.1.100"):
        self.robot_type = robot_type
        self.ip = ip
        self.connected = False
        
    def connect(self):
        # Real impl: socket / ROS / Flexiv RSDK
        print(f"[CLIENT] Connecting to {self.robot_type}@{self.ip}...")
        self.connected = True
        
    def send_action(self, action_chunk: List[float]):
        """Send action chunk to robot."""
        # Real impl: UDP packet + wait for execution
        print(f"[CLIENT] Sent {len(action_chunk)} values, waiting for execution...")
        
    def get_observation(self) -> Observation:
        """Get latest observation from robot."""
        # Real impl: subscribe to sensor topics
        return Observation(
            images={"front": torch.randn(3, 224, 224)},
            state=torch.randn(14),
            language=""  # filled by human or HLM
        )


# ==============================================================================
# MAIN: Run the Pipeline
# ==============================================================================
def run_inference_example():
    """Mental model run — prints shapes at each stage."""
    config = Pi0Config()
    
    print("=" * 70)
    print("π0 INFERENCE TRACE")
    print("=" * 70)
    
    # ===== Stage 0: Init =====
    print("\n[0] INIT")
    print(f"    config: {config.model_name}")
    print(f"    image_keys: {config.image_keys}")  
    print(f"    state_dim: {config.state_dim}")
    print(f"    action_dim: {config.action_dim}")
    print(f"    chunk_length: {config.chunk_length}")
    print(f"    control_freq: {config.control_frequency_hz}Hz")
    
    # ===== Stage 1: Mock observation =====
    print("\n[1] OBSERVATION (mock)")
    mock_obs = Observation(
        images={
            "front": torch.randn(3, 480, 640),   # raw camera
            "wrist": torch.randn(3, 224, 224),
        },
        state=torch.randn(14),  # q + dq
        language="pick up the red cup and place it in the box",
    )
    print(f"    observation: {mock_obs.shapes()}")
    print(f"    language: '{mock_obs.language[:40]}...'")
    
    # ===== Stage 2: Preprocess =====
    print("\n[2] PREPROCESS")
    preprocessed = preprocess_observation(mock_obs, config)
    print(f"    image: {preprocessed['image'].shape} (stacked, not yet normalized)")
    print(f"    text: {preprocessed['text'].shape}")
    print(f"    state: {preprocessed['state'].shape}")
    print(f"    → real impl: resize, normalize, tokenize, clamp")
    
    # ===== Stage 3: Model forward =====
    print("\n[3] MODEL FORWARD") 
    model = Pi0Model(config)
    
    encoded = model.encode_inputs(preprocessed)
    print(f"    image_features: {encoded['image_features'].shape}")
    print(f"    text_features: {encoded['text_features'].shape}")
    print(f"    state_features: {encoded['state_features'].shape}")
    print(f"    → fusion (late) → hidden state")
    
    action_logits = model(preprocessed)
    print(f"    action_logits (before flow decode): {action_logits.shape}")
    
    # ===== Stage 4: Flow matching =====
    print("\n[4] FLOW MATCHING DECODE")
    flow_head = FlowMatchingHead(config.action_dim)
    action_chunk_flow = flow_head.decode(action_logits)
    print(f"    input shape: {action_logits.shape}")
    print(f"    ODE steps: {flow_head.num_ode_steps}")
    print(f"    output: action_chunk {action_chunk_flow.shape}")
    print(f"    → real impl: z = z + v(z,cond)*dt for 7 steps")
    
    # ===== Stage 5: Action output =====
    print("\n[5] ACTION OUTPUT")
    action_chunk = decode_action(action_logits, config)
    print(f"    action_chunk: {action_chunk.shape}")  
    print(f"    flattened: {action_chunk.flatten().shape}")
    print(f"    values preview: {action_chunk[0][:3].tolist()}... (Δq for joints)")
    
    robot_cmd = format_for_robot(action_chunk, config)
    print(f"    → robot command: {len(robot_cmd)} floats, ready to send")
    
    # ===== Stage 6: Robot interface =====
    print("\n[6] ROBOT CLIENT")
    robot = RobotClient(config.robot_type)
    # robot.connect()  # Uncomment to connect real robot
    print(f"    connected: {robot.connected} (use robot.connect() in real)")
    robot.send_action(robot_cmd[:14])  # Just first timestep
    
    # ===== Done =====
    print("\n" + "=" * 70)
    print("✓ COMPLETE — 6 stages traced")
    print("  Next: Replace mocks with real code from OpenPI repo")
    print("=" * 70)


if __name__ == "__main__":
    run_inference_example()