"""Proposal-centric waypoint head with K-proposals and scoring.

This module implements an alternative waypoint prediction head:
- K proposals: predict K candidate trajectory sets
- Scorer: learn which proposal is most likely correct
- Flag: `--use-proposal-head` to enable

Usage
-----
python -m training.sft.train_waypoint_bc_torch_v0 \
  --use-proposal-head \
  --num-proposals 5 \
  --proposal-scoring

Output structure:
- proposals: (B, K, H, 2) - K candidate waypoint sequences
- scores: (B, K) - probability/score for each proposal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProposalHeadConfig:
    """Configuration for proposal-centric waypoint head."""
    # Proposals
    num_proposals: int = 5  # K proposals
    horizon_steps: int = 20
    
    # Scoring
    use_scoring: bool = True  # Learn proposal scores
    
    # Architecture
    hidden_dim: int = 256
    proposal_depth: int = 3
    
    # Training
    proposal_loss_weight: float = 1.0
    scoring_loss_weight: float = 0.5


class ProposalWaypointHead(nn.Module):
    """K-proposals + scorer head for waypoint prediction.
    
    Architecture:
    - Encoder embedding -> shared trunk
    - K parallel proposal decoders
    - Optional scorer to weight proposals
    """
    
    def __init__(
        self,
        in_dim: int,
        horizon_steps: int = 20,
        num_proposals: int = 5,
        hidden_dim: int = 256,
        depth: int = 3,
        use_scoring: bool = True,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.horizon_steps = horizon_steps
        self.num_proposals = num_proposals
        self.use_scoring = use_scoring
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Per-proposal decoders
        self.proposal_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, horizon_steps * 2),  # (x, y) per timestep
            )
            for _ in range(num_proposals)
        ])
        
        # Scorer network (optional)
        if use_scoring:
            self.scorer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict K proposals with optional scores.
        
        Args:
            z: (B, D) encoder embedding
        
        Returns:
            proposals: (B, K, H, 2) waypoint predictions
            scores: (B, K) proposal scores (logits, before softmax)
        """
        B = z.shape[0]
        
        # Shared trunk
        trunk_out = self.trunk(z)  # (B, D_hidden)
        
        # Generate K proposals
        proposals = []
        for decoder in self.proposal_decoders:
            proposal = decoder(trunk_out)  # (B, H*2)
            proposal = proposal.view(B, self.horizon_steps, 2)
            proposals.append(proposal)
        
        proposals = torch.stack(proposals, dim=1)  # (B, K, H, 2)
        
        # Compute scores if enabled
        if self.use_scoring:
            # Concatenate trunk output with each proposal
            trunk_expanded = trunk_out.unsqueeze(1).expand(-1, self.num_proposals, -1)  # (B, K, D)
            proposal_flat = proposals.view(B, self.num_proposals, -1)  # (B, K, H*2)
            combined = torch.cat([trunk_expanded, proposal_flat], dim=-1)  # (B, K, D + H*2)
            
            score_logits = self.scorer(combined).squeeze(-1)  # (B, K)
        else:
            # Uniform scores
            score_logits = torch.zeros(B, self.num_proposals, device=z.device)
        
        return proposals, score_logits
    
    def get_best_proposal(
        self,
        proposals: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """Select the highest-scoring proposal."""
        probs = F.softmax(scores, dim=-1)  # (B, K)
        best_idx = probs.argmax(dim=-1)  # (B,)
        
        # Gather best proposals
        best = proposals[torch.arange(proposals.shape[0]), best_idx]  # (B, H, 2)
        return best


class ProposalLoss:
    """Loss for proposal-centric waypoint prediction.
    
    Uses minimum-over-proposals loss: only penalize the best proposal.
    Optionally add scoring supervision.
    """
    
    def __init__(self, config: ProposalHeadConfig | None = None):
        self.config = config or ProposalHeadConfig()
    
    def compute_loss(
        self,
        proposals: torch.Tensor,  # (B, K, H, 2)
        scores: torch.Tensor,  # (B, K)
        target: torch.Tensor,  # (B, H, 2)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute proposal-centric loss.
        
        Args:
            proposals: (B, K, H, 2) K candidate trajectories
            scores: (B, K) proposal scores (logits)
            target: (B, H, 2) ground truth waypoints
        
        Returns:
            loss: Total loss
            info: Dict with loss components
        """
        B, K, H, _ = proposals.shape
        
        # Minimum-over-proposals loss: compute loss for each proposal, take minimum
        proposal_losses = []
        for k in range(K):
            loss_k = F.mse_loss(proposals[:, k], target, reduction="none")  # (B, H, 2)
            loss_k = loss_k.mean(dim=(1, 2))  # (B,)
            proposal_losses.append(loss_k)
        
        proposal_losses = torch.stack(proposal_losses, dim=1)  # (B, K)
        
        # Best proposal is one with minimum loss
        best_loss, best_idx = proposal_losses.min(dim=1)  # (B,), (B,)
        
        # Mean over batch
        min_proposal_loss = best_loss.mean()
        
        info = {
            "proposal_min_loss": min_proposal_loss.item(),
        }
        
        # Total loss
        loss = min_proposal_loss * self.config.proposal_loss_weight
        
        # Optional: scoring loss (encourage correct proposal to have high score)
        if self.config.use_scoring:
            # Target: best proposal should have highest score
            # Use cross-entropy between scores and proposal quality
            with torch.no_grad():
                proposal_qualities = -proposal_losses  # Higher is better
                proposal_qualities = F.softmax(proposal_qualities, dim=1)
            
            scoring_loss = F.cross_entropy(scores, best_idx)
            loss = loss + scoring_loss * self.config.scoring_loss_weight
            info["proposal_scoring_loss"] = scoring_loss.item()
        
        info["total_loss"] = loss.item()
        
        return loss, info


def demo():
    """Demo of proposal head."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--proposals", type=int, default=5)
    args = parser.parse_args()
    
    # Config
    config = ProposalHeadConfig(
        num_proposals=args.proposals,
        horizon_steps=args.horizon,
        use_scoring=True,
    )
    
    # Create head
    head = ProposalWaypointHead(
        in_dim=128,
        horizon_steps=args.horizon,
        num_proposals=args.proposals,
        use_scoring=True,
    )
    
    # Dummy embedding
    z = torch.randn(args.batch_size, 128)
    
    # Forward
    proposals, scores = head(z)
    print(f"Proposals shape: {proposals.shape}")  # (B, K, H, 2)
    print(f"Scores shape: {scores.shape}")  # (B, K)
    
    # Loss
    target = torch.randn(args.batch_size, args.horizon, 2)
    loss_fn = ProposalLoss(config)
    loss, info = loss_fn.compute_loss(proposals, scores, target)
    print(f"Loss: {loss.item():.4f}")
    print(f"Info: {info}")
    
    # Best proposal
    best = head.get_best_proposal(proposals, scores)
    print(f"Best proposal shape: {best.shape}")


if __name__ == "__main__":
    demo()
