"""
Contingency Network

Neural network for learning-based contingency planning.
Predicts branching trajectories given discrete uncertainties.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional


class ContingencyNetwork(nn.Module):
    """
    Neural network that predicts contingency plans.
    
    Architecture:
    - Encoder: processes state + context
    - Branching heads: predicts trajectory for each contingency
    - Uncertainty head: predicts probability distribution
    
    Based on hybrid approach: learning for prediction + classical for safety
    """
    
    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 512,
        action_dim: int = 2,
        horizon: int = 20,
        n_contingencies: int = 4,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_contingencies = n_contingencies
        
        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Branching heads (one per contingency)
        self.branch_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, horizon * action_dim),
            )
            for _ in range(n_contingencies)
        ])
        
        # Uncertainty predictor (belief over contingencies)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_contingencies),
            nn.Softmax(dim=-1)
        )
        
        # Safety critic (evaluates safety of plans)
        self.safety_critic = nn.Sequential(
            nn.Linear(hidden_dim + horizon * action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Safety score 0-1
        )
        
        # Context encoder (for conditioning on contingency type)
        self.context_encoder = nn.Sequential(
            nn.Embedding(n_contingencies, 64),
            nn.Linear(64, 64),
        )
    
    def forward(
        self, 
        state: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: [B, state_dim] State encoding
            context: [B, n_contingencies] Optional context (e.g., which contingencies to plan for)
            
        Returns:
            plans: List of [B, horizon, action_dim] trajectories for each contingency
            uncertainty: [B, n_contingencies] Probability distribution
        """
        batch_size = state.size(0)
        
        # Encode state
        enc = self.encoder(state)
        
        # Generate branching plans
        plans = []
        for head in self.branch_heads:
            plan = head(enc).view(batch_size, self.horizon, self.action_dim)
            plans.append(plan)
        
        # Predict uncertainty
        uncertainty = self.uncertainty_head(enc)
        
        return plans, uncertainty
    
    def forward_with_context(
        self,
        state: torch.Tensor,
        contingency_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass conditioned on specific contingency.
        
        Args:
            state: [B, state_dim]
            contingency_idx: [B] Which contingency to plan for
            
        Returns:
            plan: [B, horizon, action_dim]
            safety_score: [B, 1]
        """
        enc = self.encoder(state)
        
        # Encode contingency context
        cont_emb = self.context_encoder(contingency_idx)  # [B, 64]
        enc_cont = torch.cat([enc, cont_emb], dim=-1)
        
        # Generate plan
        plan = self.branch_heads[contingency_idx[0]](enc).view(-1, self.horizon, self.action_dim)
        
        # Evaluate safety
        plan_flat = plan.view(plan.size(0), -1)
        safety_score = self.safety_critic(torch.cat([enc, plan_flat], dim=-1))
        
        return plan, safety_score
    
    def get_plan_for_hypothesis(
        self,
        state: torch.Tensor,
        hypothesis_idx: int
    ) -> torch.Tensor:
        """Get plan for specific hypothesis."""
        plans, _ = self.forward(state)
        return plans[hypothesis_idx]
    
    def compute_loss(
        self,
        state: torch.Tensor,
        expert_plans: List[torch.Tensor],
        uncertainty_target: torch.Tensor,
        safety_labels: Optional[torch.Tensor] = None,
        lambda_plan: float = 1.0,
        lambda_uncertainty: float = 0.5,
        lambda_safety: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss.
        
        Args:
            state: [B, state_dim]
            expert_plans: List of [B, horizon, action_dim] expert trajectories
            uncertainty_target: [B, n_contingencies] Target distribution
            safety_labels: [B] 1 = safe, 0 = unsafe (optional)
            
        Returns:
            loss: Total loss
            metrics: Dictionary of loss components
        """
        plans, uncertainty = self.forward(state)
        
        # Plan reconstruction loss
        plan_loss = 0
        for i, expert_plan in enumerate(expert_plans):
            if i < len(plans):
                plan_loss = plan_loss + nn.functional.mse_loss(plans[i], expert_plan)
        plan_loss = plan_loss / len(plans) if plans else 0
        
        # Uncertainty loss (KL divergence)
        uncertainty_loss = nn.functional.kl_div(
            uncertainty.log(), 
            uncertainty_target, 
            reduction='batchmean'
        )
        
        # Safety loss (if labels provided)
        safety_loss = 0
        if safety_labels is not None:
            for plan in plans:
                plan_flat = plan.view(plan.size(0), -1)
                enc = self.encoder(state)
                safety_pred = self.safety_critic(torch.cat([enc, plan_flat], dim=-1)).squeeze(-1)
                safety_loss = safety_loss + nn.functional.binary_cross_entropy(
                    safety_pred, safety_labels.float()
                )
            safety_loss = safety_loss / len(plans)
        
        # Total loss
        total_loss = (
            lambda_plan * plan_loss + 
            lambda_uncertainty * uncertainty_loss + 
            lambda_safety * safety_loss
        )
        
        metrics = {
            "total_loss": total_loss.item(),
            "plan_loss": plan_loss.item() if isinstance(plan_loss, torch.Tensor) else plan_loss,
            "uncertainty_loss": uncertainty_loss.item(),
            "safety_loss": safety_loss.item() if safety_labels is not None else 0,
        }
        
        return total_loss, metrics


class SafetyFilter(nn.Module):
    """
    Safety filter using Control Barrier Functions (CBF).
    
    Applies minimally invasive correction to ensure safety.
    """
    
    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 2,
        safety_margin: float = 1.5,
        barrier_gain: float = 2.0,
    ):
        super().__init__()
        
        self.safety_margin = safety_margin
        self.barrier_gain = barrier_gain
        
        # Learnable safety margin adjuster
        self.margin_network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output: 0.5-1.5 multiplier
        )
        
        # Learnable barrier gain
        self.gain_network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output: 1.0-3.0
        )
    
    def forward(
        self,
        nominal_plan: torch.Tensor,
        state: torch.Tensor,
        obstacles: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply safety filter to nominal plan.
        
        Args:
            nominal_plan: [B, horizon, action_dim] Unfiltered plan
            state: [B, state_dim] Current state
            obstacles: [B, N, 3] Obstacle positions (optional)
            
        Returns:
            safe_plan: [B, horizon, action_dim] Safety-corrected plan
            safety_scores: [B, horizon] Safety score per timestep
        """
        batch_size = nominal_plan.size(0)
        horizon = nominal_plan.size(1)
        
        # Get adaptive safety parameters
        margin_mult = self.margin_network(state)  # [B, 1]
        gain_mult = self.gain_network(state)  # [B, 1]
        
        margin = self.safety_margin * (0.5 + margin_mult)  # Range: 0.75-2.25
        gain = self.barrier_gain * (1.0 + gain_mult)  # Range: 2.0-6.0
        
        # Simple safety correction (reduce speed near obstacles)
        safe_plan = nominal_plan.clone()
        safety_scores = torch.ones(batch_size, horizon, device=nominal_plan.device)
        
        # If obstacles provided, apply stronger correction
        if obstacles is not None:
            # Simple distance-based correction
            for t in range(horizon):
                # Approximate position at time t (simplified)
                # In practice, would integrate dynamics
                dist_to_obstacles = torch.cdist(
                    state[:, :2].unsqueeze(1), 
                    obstacles[:, :, :2]
                )  # [B, N]
                
                min_dist = dist_to_obstacles.min(dim=-1)[0]  # [B]
                
                # If close to obstacle, reduce acceleration
                close_mask = min_dist < margin * 2
                safe_plan[close_mask, t, 0] = torch.clamp(
                    safe_plan[close_mask, t, 0],
                    -6.0,  # Max braking
                    0.0   # No acceleration
                )
                
                # Safety score
                safety_scores[:, t] = torch.clamp(min_dist / margin, 0, 1)
        
        return safe_plan, safety_scores
    
    def compute_barrier_loss(
        self,
        plan: torch.Tensor,
        state: torch.Tensor,
        safety_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute barrier-based safety loss for training.
        
        Args:
            plan: [B, horizon, action_dim]
            state: [B, state_dim]
            safety_target: [B] 1 = safe, 0 = unsafe
            
        Returns:
            barrier_loss
        """
        plan_flat = plan.view(plan.size(0), -1)
        enc = self.encoder(state)
        
        # Predict safety
        safety_pred = self.safety_critic(torch.cat([enc, plan_flat], dim=-1)).squeeze(-1)
        
        # Barrier loss: penalize unsafe predictions
        barrier_loss = nn.functional.binary_cross_entropy(
            safety_pred, safety_target.float()
        )
        
        return barrier_loss


class ModelBasedPlanner(nn.Module):
    """
    Full model-based contingency planner.
    
    Combines:
    - ContingencyNetwork for trajectory prediction
    - SafetyFilter for safety guarantees
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        model_config = config.get("model", {})
        safety_config = config.get("safety_filter", {})
        
        self.contingency_net = ContingencyNetwork(
            state_dim=model_config.get("state_dim", 256),
            hidden_dim=model_config.get("hidden_dim", 512),
            action_dim=model_config.get("action_dim", 2),
            horizon=model_config.get("horizon", 20),
            n_contingencies=model_config.get("n_contingencies", 4),
        )
        
        self.safety_filter = SafetyFilter(
            state_dim=model_config.get("state_dim", 256),
            action_dim=model_config.get("action_dim", 2),
            safety_margin=safety_config.get("safety_margin", 1.5),
            barrier_gain=safety_config.get("barrier_gain", 2.0),
        )
        
        self.enable_safety_filter = safety_config.get("enable", True)
    
    def forward(
        self,
        state: torch.Tensor,
        obstacles: Optional[torch.Tensor] = None,
        apply_safety: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            state: [B, state_dim]
            obstacles: [B, N, 3] (optional)
            apply_safety: Whether to apply safety filter
            
        Returns:
            selected_plan: [B, horizon, action_dim] Selected safe plan
            uncertainty: [B, n_contingencies] Contingency probabilities
            all_plans: List of all branch plans
        """
        # Get contingency plans
        plans, uncertainty = self.contingency_net(state)
        
        # Select best plan based on uncertainty (expected utility)
        # E[plan] = sum(p_i * plan_i)
        selected_plan = torch.zeros(
            state.size(0), 
            self.contingency_net.horizon, 
            self.contingency_net.action_dim,
            device=state.device
        )
        
        for i, plan in enumerate(plans):
            selected_plan = selected_plan + uncertainty[:, i:i+1, None] * plan
        
        # Apply safety filter
        if apply_safety and self.enable_safety_filter:
            selected_plan, safety_scores = self.safety_filter(
                selected_plan, state, obstacles
            )
        
        return selected_plan, uncertainty, plans
    
    def plan_for_contingency(
        self,
        state: torch.Tensor,
        contingency_idx: int,
        obstacles: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Plan for specific contingency.
        
        Args:
            state: [B, state_dim]
            contingency_idx: Hypothesis index
            obstacles: [B, N, 3] (optional)
            
        Returns:
            plan: [B, horizon, action_dim]
            safety_score: [B]
        """
        plan = self.contingency_net.get_plan_for_hypothesis(state, contingency_idx)
        
        if self.enable_safety_filter and obstacles is not None:
            plan, safety_scores = self.safety_filter(plan, state, obstacles)
            safety_score = safety_scores.mean(dim=-1)
        else:
            safety_score = torch.ones(state.size(0), device=state.device)
        
        return plan, safety_score
