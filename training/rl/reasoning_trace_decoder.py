"""
Reasoning Trace Decoder for Interpretable Driving Decisions.

This module generates reasoning traces that explain why the model predicts
specific waypoints. It complements the latent dynamics model by providing
human-interpretable explanations for driving decisions.

Architecture:
    - SceneEncoder: Encodes visual context (roads, obstacles, traffic)
    - ReasoningDecoder: Generates reasoning tokens explaining decisions
    - WaypointReasoningModel: Combined model for prediction + explanation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


class ReasoningToken:
    """Reasoning token types for driving decisions."""
    # Scene understanding
    OBSERVATION = "observe"      # What the model observes
    PREDICTION = "predict"       # What it predicts will happen
    PLANNING = "plan"            # What action it plans
    CAUTION = "caution"          # Potential risk factor
    CONFIDENCE = "confidence"    # How confident the decision is
    
    # Action types
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    STEER_LEFT = "steer_left"
    STEER_RIGHT = "steer_right"
    MAINTAIN = "maintain"
    
    # Context
    LANE_KEEP = "lane_keep"
    LANE_CHANGE = "lane_change"
    FOLLOW = "follow"
    STOP = "stop"
    YIELD = "yield"


@dataclass
class ReasoningTrace:
    """Structured reasoning trace for a driving decision."""
    observation: str      # What was observed
    prediction: str       # What was predicted  
    plan: str             # What action was planned
    confidence: float     # Confidence level [0, 1]
    risk_factor: float    # Risk assessment [0, 1]
    tokens: List[str]     # Reasoning tokens


class SceneEncoder(nn.Module):
    """Encodes scene context for reasoning generation."""
    
    def __init__(
        self,
        state_dim: int = 6,
        hidden_dim: int = 128,
        num_reasoning_tokens: int = 32
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_reasoning_tokens = num_reasoning_tokens
        
        # Encode current state (position, velocity, goal)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Encode trajectory history
        self.traj_encoder = nn.Sequential(
            nn.Linear(20 * 2, hidden_dim),  # 20 timesteps x 2 (x, y)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Scene context projection
        self.scene_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Reasoning token embeddings
        self.reasoning_token_embedding = nn.Embedding(
            num_reasoning_tokens, hidden_dim
        )
        
    def forward(
        self, 
        state: torch.Tensor, 
        trajectory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, state_dim) - [x, y, vx, vy, goal_x, goal_y]
            trajectory: (batch, 20, 2) - historical trajectory
            
        Returns:
            scene_features: (batch, hidden_dim)
            token_embeddings: (batch, num_tokens, hidden_dim)
        """
        # Encode current state
        state_features = self.state_encoder(state)  # (batch, hidden_dim)
        
        # Encode trajectory if provided
        if trajectory is not None:
            traj_flat = trajectory.reshape(trajectory.shape[0], -1)
            traj_features = self.traj_encoder(traj_flat)
            # Combine state and trajectory
            scene_features = torch.cat([state_features, traj_features], dim=-1)
        else:
            scene_features = state_features
            
        scene_features = self.scene_proj(scene_features)
        
        # Generate reasoning token embeddings
        token_ids = torch.arange(
            self.num_reasoning_tokens, 
            device=state.device
        ).unsqueeze(0).expand(state.shape[0], -1)
        token_embeddings = self.reasoning_token_embedding(token_ids)
        
        return scene_features, token_embeddings


class ReasoningDecoder(nn.Module):
    """Generates reasoning traces explaining waypoint predictions."""
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_reasoning_tokens: int = 32,
        vocab_size: int = 50  # Number of reasoning token types
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_reasoning_tokens = num_reasoning_tokens
        self.vocab_size = vocab_size
        
        # Cross-attention between scene and reasoning tokens
        self.attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=4,
            batch_first=True
        )
        
        # Reasoning token prediction
        self.reasoning_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Risk factor prediction
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        scene_features: torch.Tensor,
        token_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            scene_features: (batch, hidden_dim)
            token_embeddings: (batch, num_tokens, hidden_dim)
            
        Returns:
            reasoning_logits: (batch, num_tokens, vocab_size)
            confidence: (batch, 1)
            risk_factor: (batch, 1)
        """
        # Add scene as key/value for attention
        scene = scene_features.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Cross-attention: query=tokens, key/value=scene
        attended, _ = self.attention(
            token_embeddings, scene, scene
        )  # (batch, num_tokens, hidden_dim)
        
        # Predict reasoning tokens
        reasoning_logits = self.reasoning_head(attended)
        
        # Predict confidence and risk
        # Use mean of attended tokens
        context = attended.mean(dim=1)  # (batch, hidden_dim)
        confidence = self.confidence_head(context)
        risk_factor = self.risk_head(context)
        
        return reasoning_logits, confidence, risk_factor


class WaypointReasoningModel(nn.Module):
    """
    Combined model for waypoint prediction with reasoning traces.
    
    Architecture:
        1. SceneEncoder: Encode state and trajectory
        2. WaypointHead: Predict waypoints
        3. ReasoningDecoder: Generate reasoning traces
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        hidden_dim: int = 128,
        horizon: int = 20,
        num_reasoning_tokens: int = 32,
        vocab_size: int = 50
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        
        # Scene encoder
        self.scene_encoder = SceneEncoder(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_reasoning_tokens=num_reasoning_tokens
        )
        
        # Waypoint prediction head
        self.waypoint_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * 2)  # (x, y) for each timestep
        )
        
        # Reasoning decoder
        self.reasoning_decoder = ReasoningDecoder(
            hidden_dim=hidden_dim,
            num_reasoning_tokens=num_reasoning_tokens,
            vocab_size=vocab_size
        )
        
        # Initialize vocabulary mapping
        self._init_reasoning_vocab()
        
    def _init_reasoning_vocab(self):
        """Initialize reasoning vocabulary."""
        self.reasoning_vocab = {
            0: ReasoningToken.OBSERVATION,
            1: ReasoningToken.PREDICTION,
            2: ReasoningToken.PLANNING,
            3: ReasoningToken.CAUTION,
            4: ReasoningToken.CONFIDENCE,
            5: ReasoningToken.ACCELERATE,
            6: ReasoningToken.DECELERATE,
            7: ReasoningToken.STEER_LEFT,
            8: ReasoningToken.STEER_RIGHT,
            9: ReasoningToken.MAINTAIN,
            10: ReasoningToken.LANE_KEEP,
            11: ReasoningToken.LANE_CHANGE,
            12: ReasoningToken.FOLLOW,
            13: ReasoningToken.STOP,
            14: ReasoningToken.YIELD,
        }
        
    def forward(
        self,
        state: torch.Tensor,
        trajectory: Optional[torch.Tensor] = None,
        return_reasoning: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: (batch, state_dim) - [x, y, vx, vy, goal_x, goal_y]
            trajectory: (batch, 20, 2) - historical trajectory
            return_reasoning: Whether to generate reasoning traces
            
        Returns:
            Dictionary with:
                - waypoints: (batch, horizon, 2)
                - reasoning_logits: (batch, num_tokens, vocab_size)
                - confidence: (batch, 1)
                - risk_factor: (batch, 1)
        """
        # Encode scene
        scene_features, token_embeddings = self.scene_encoder(
            state, trajectory
        )
        
        # Predict waypoints
        waypoint_flat = self.waypoint_head(scene_features)
        waypoints = waypoint_flat.reshape(-1, self.horizon, 2)
        
        if not return_reasoning:
            return {"waypoints": waypoints}
            
        # Generate reasoning
        reasoning_logits, confidence, risk_factor = self.reasoning_decoder(
            scene_features, token_embeddings
        )
        
        return {
            "waypoints": waypoints,
            "reasoning_logits": reasoning_logits,
            "confidence": confidence,
            "risk_factor": risk_factor
        }
    
    def decode_reasoning(
        self,
        reasoning_logits: torch.Tensor
    ) -> List[ReasoningTrace]:
        """
        Decode reasoning logits into human-readable traces.
        
        Args:
            reasoning_logits: (batch, num_tokens, vocab_size)
            
        Returns:
            List of ReasoningTrace objects
        """
        # Get argmax tokens
        token_ids = reasoning_logits.argmax(dim=-1)  # (batch, num_tokens)
        
        traces = []
        batch_size = token_ids.shape[0]
        
        for i in range(batch_size):
            tokens = [
                self.reasoning_vocab.get(
                    token_ids[i, j].item(), 
                    "unknown"
                )
                for j in range(token_ids.shape[1])
            ]
            
            # Extract key information
            observation = "road clear" if ReasoningToken.CAUTION not in tokens else "potential obstacle ahead"
            prediction = "continue path" 
            plan = self._extract_plan(tokens)
            confidence = reasoning_logits[i, 0, :].softmax(-1).max().item()
            risk_factor = 0.1  # Placeholder
            
            traces.append(ReasoningTrace(
                observation=observation,
                prediction=prediction,
                plan=plan,
                confidence=confidence,
                risk_factor=risk_factor,
                tokens=tokens
            ))
            
        return traces
    
    def _extract_plan(self, tokens: List[str]) -> str:
        """Extract plan from reasoning tokens."""
        action_tokens = {
            ReasoningToken.ACCELERATE,
            ReasoningToken.DECELERATE,
            ReasoningToken.STEER_LEFT,
            ReasoningToken.STEER_RIGHT,
            ReasoningToken.MAINTAIN,
            ReasoningToken.STOP,
            ReasoningToken.YIELD
        }
        
        for token in tokens:
            if token in action_tokens:
                return token
        return ReasoningToken.MAINTAIN


class ReasoningLoss(nn.Module):
    """Loss function for reasoning trace training."""
    
    def __init__(
        self,
        waypoint_weight: float = 1.0,
        confidence_weight: float = 0.1,
        risk_weight: float = 0.1,
        reasoning_weight: float = 0.05
    ):
        super().__init__()
        self.waypoint_weight = waypoint_weight
        self.confidence_weight = confidence_weight
        self.risk_weight = risk_weight
        self.reasoning_weight = reasoning_weight
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Model output
            targets: Ground truth
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Waypoint loss (MSE)
        if "waypoints" in predictions and "waypoints" in targets:
            losses["waypoint"] = F.mse_loss(
                predictions["waypoints"],
                targets["waypoints"]
            )
            
        # Confidence loss (BCE - encourage high confidence for correct predictions)
        if "confidence" in predictions and "confidence_target" in targets:
            losses["confidence"] = F.binary_cross_entropy(
                predictions["confidence"],
                targets["confidence_target"]
            )
            
        # Risk loss (BCE - encourage accurate risk assessment)
        if "risk_factor" in predictions and "risk_target" in targets:
            losses["risk"] = F.binary_cross_entropy(
                predictions["risk_factor"],
                targets["risk_target"]
            )
            
        # Reasoning loss (Cross-entropy)
        if "reasoning_logits" in predictions and "reasoning_tokens" in targets:
            reasoning_logits = predictions["reasoning_logits"]
            reasoning_tokens = targets["reasoning_tokens"]
            
            # Reshape for cross-entropy
            batch_size, num_tokens, vocab_size = reasoning_logits.shape
            logits_flat = reasoning_logits.reshape(-1, vocab_size)
            tokens_flat = reasoning_tokens.reshape(-1)
            
            losses["reasoning"] = F.cross_entropy(
                logits_flat, tokens_flat
            )
            
        # Total loss
        total = sum(
            weight * losses[key] 
            for key, weight in [
                ("waypoint", self.waypoint_weight),
                ("confidence", self.confidence_weight),
                ("risk", self.risk_weight),
                ("reasoning", self.reasoning_weight)
            ]
            if key in losses
        )
        losses["total"] = total
        
        return losses


def create_reasoning_waypoint_model(
    state_dim: int = 6,
    horizon: int = 20,
    hidden_dim: int = 128
) -> WaypointReasoningModel:
    """Factory function to create a reasoning waypoint model."""
    return WaypointReasoningModel(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        horizon=horizon
    )


# Smoke test
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create model
    model = WaypointReasoningModel(
        state_dim=6,
        hidden_dim=64,
        horizon=20,
        num_reasoning_tokens=16,
        vocab_size=20
    )
    
    # Dummy input
    batch_size = 4
    state = torch.randn(batch_size, 6)
    trajectory = torch.randn(batch_size, 20, 2)
    
    # Forward pass
    outputs = model(state, trajectory, return_reasoning=True)
    
    print("=== Reasoning Waypoint Model Smoke Test ===")
    print(f"Waypoints shape: {outputs['waypoints'].shape}")
    print(f"Reasoning logits shape: {outputs['reasoning_logits'].shape}")
    print(f"Confidence shape: {outputs['confidence'].shape}")
    print(f"Risk factor shape: {outputs['risk_factor'].shape}")
    
    # Decode reasoning
    traces = model.decode_reasoning(outputs["reasoning_logits"])
    print(f"\nGenerated {len(traces)} reasoning traces")
    for i, trace in enumerate(traces[:2]):
        print(f"  Trace {i}: confidence={trace.confidence:.3f}, plan={trace.plan}")
    
    # Loss test
    criterion = ReasoningLoss()
    targets = {
        "waypoints": torch.randn(batch_size, 20, 2),
        "confidence_target": torch.rand(batch_size, 1),
        "risk_target": torch.rand(batch_size, 1),
        "reasoning_tokens": torch.randint(0, 20, (batch_size, 16))
    }
    
    losses = criterion(outputs, targets)
    print(f"\nLosses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\n✓ Smoke test passed!")
