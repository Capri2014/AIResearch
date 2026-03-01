"""
Risk Aggregator with CVaR (Conditional Value at Risk)

CVaR is more robust than worst-case:
- α=0.2 means we care about the worst 20% of scenarios
- Less sensitive to outliers than pure max
- Switches to worst-case for low-confidence corridors

Reference: CVaR aggregation from the production planner survey
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class RiskMetric(Enum):
    """Risk aggregation methods"""
    WORST_CASE = "worst_case"
    CVAR_ALPHA = "cvar"  # Conditional Value at Risk
    MEAN = "mean"
    CVAR_WORST_CASE = "cvar_worst_case"  # Switch to worst-case for low confidence


@dataclass
class RiskConfig:
    """Configuration for risk aggregation"""
    # CVaR parameters
    alpha: float = 0.2  # Tail fraction to consider
    
    # Confidence threshold for switching to worst-case
    confidence_threshold: float = 0.5
    
    # Component weights
    weight_collision: float = 100.0
    weight_boundary: float = 10.0
    weight_keepout: float = 20.0
    weight_violation: float = 50.0
    
    # Margins
    boundary_margin_threshold: float = 0.3
    keepout_margin_threshold: float = 0.5


@dataclass
class ScenarioEvaluation:
    """Evaluation result for one scenario"""
    scenario_id: int
    collision: bool
    min_boundary_margin: float
    min_keepout_margin: float
    progress: float
    comfort: float
    
    # Aggregated metrics
    cost: float = 0.0
    risk_score: float = 0.0


class RiskAggregator:
    """
    Computes risk using CVaR aggregation.
    
    CVaR_α = E[loss | loss > VaR_α]
    
    Where VaR_α is the (1-α) quantile of losses.
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
    
    def compute_cvar(self, losses: np.ndarray, alpha: float = None) -> float:
        """
        Compute CVaR (Conditional Value at Risk).
        
        CVaR = Expected loss given loss is in the worst (1-α) percentile.
        """
        alpha = alpha or self.config.alpha
        
        if len(losses) == 0:
            return 0.0
        
        # Sort losses
        sorted_losses = np.sort(losses)
        
        # Find threshold index
        k = int(np.ceil((1 - alpha) * len(sorted_losses)))
        k = min(k, len(sorted_losses) - 1)
        
        # VaR (value at risk)
        var = sorted_losses[k]
        
        # CVaR = mean of losses above VaR
        cvar = np.mean(sorted_losses[k:])
        
        return cvar
    
    def compute_worst_case(self, losses: np.ndarray) -> float:
        """Compute worst-case loss"""
        if len(losses) == 0:
            return 0.0
        return np.max(losses)
    
    def compute_mean(self, losses: np.ndarray) -> float:
        """Compute mean loss"""
        if len(losses) == 0:
            return 0.0
        return np.mean(losses)
    
    def compute_scenario_cost(self, eval: ScenarioEvaluation) -> float:
        """Compute cost for a single scenario"""
        cost = 0.0
        
        # Collision penalty
        if eval.collision:
            cost += self.config.weight_collision
        
        # Boundary margin penalty
        if eval.min_boundary_margin < self.config.boundary_margin_threshold:
            margin_penalty = self.config.boundary_margin_threshold - eval.min_boundary_margin
            cost += self.config.weight_boundary * margin_penalty
        
        # Keepout margin penalty
        if eval.min_keepout_margin < self.config.keepout_margin_threshold:
            margin_penalty = self.config.keepout_margin_threshold - eval.min_keepout_margin
            cost += self.config.weight_keepout * margin_penalty
        
        # Progress reward (negative = cost)
        cost += -0.1 * eval.progress
        
        # Comfort cost
        cost += self.config.weight_comfort * (1.0 - eval.comfort)
        
        return cost
    
    def aggregate_risk(self, 
                      evaluations: List[ScenarioEvaluation],
                      corridor_confidence: float = 1.0,
                      metric: RiskMetric = RiskMetric.CVAR_ALPHA) -> float:
        """
        Aggregate risk across all scenarios using specified method.
        
        Args:
            evaluations: List of scenario evaluations
            corridor_confidence: Confidence in corridor validity (0-1)
            metric: Risk aggregation method
            
        Returns:
            Aggregated risk score
        """
        if not evaluations:
            return float('inf')
        
        # Compute cost for each scenario
        costs = np.array([self.compute_scenario_cost(e) for e in evaluations])
        
        # Choose aggregation method
        if metric == RiskMetric.CVAR_ALPHA:
            # CVaR: robust to outliers, cares about tail risk
            risk = self.compute_cvar(costs)
            
            # If corridor confidence is low, weight worst-case more
            if corridor_confidence < self.config.confidence_threshold:
                worst = self.compute_worst_case(costs)
                # Blend based on confidence
                confidence_factor = corridor_confidence / self.config.confidence_threshold
                risk = risk * confidence_factor + worst * (1 - confidence_factor)
                
        elif metric == RiskMetric.WORST_CASE:
            risk = self.compute_worst_case(costs)
            
        elif metric == RiskMetric.MEAN:
            risk = self.compute_mean(costs)
            
        elif metric == RiskMetric.CVAR_WORST_CASE:
            # CVaR with fallback to worst-case
            if corridor_confidence >= self.config.confidence_threshold:
                risk = self.compute_cvar(costs)
            else:
                risk = self.compute_worst_case(costs)
        else:
            risk = self.compute_mean(costs)
        
        return risk
    
    def compute_safe_probability(self, evaluations: List[ScenarioEvaluation]) -> float:
        """
        Compute probability of success across scenarios.
        
        P(safe) = (# safe scenarios) / (# total scenarios)
        """
        if not evaluations:
            return 0.0
        
        safe_count = sum(1 for e in evaluations if not e.collision)
        return safe_count / len(evaluations)
    
    def compute_expected_cost(self, evaluations: List[ScenarioEvaluation]) -> float:
        """Compute expected cost (mean)"""
        costs = [self.compute_scenario_cost(e) for e in evaluations]
        return np.mean(costs) if costs else 0.0
    
    def get_risk_breakdown(self, evaluations: List[ScenarioEvaluation]) -> Dict:
        """
        Get detailed risk breakdown for debugging.
        """
        if not evaluations:
            return {}
        
        costs = np.array([self.compute_scenario_cost(e) for e in evaluations])
        
        return {
            'num_scenarios': len(evaluations),
            'num_collisions': sum(1 for e in evaluations if e.collision),
            'safe_probability': self.compute_safe_probability(evaluations),
            'mean_cost': self.compute_mean(costs),
            'worst_cost': self.compute_worst_case(costs),
            'cvar_20': self.compute_cvar(costs, alpha=0.2),
            'cvar_10': self.compute_cvar(costs, alpha=0.1),
            'min_boundary_margin': min(e.min_boundary_margin for e in evaluations),
            'min_keepout_margin': min(e.min_keepout_margin for e in evaluations),
        }


class ScenarioEvaluator:
    """
    Evaluates a trajectory candidate against multiple scenarios.
    
    Scenarios include different predictions of other agents.
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.risk_aggregator = RiskAggregator(config)
    
    def evaluate_candidate(self,
                         trajectory: List[Tuple[float, float]],
                         scenarios: List[Dict],
                         corridor_confidence: float = 1.0) -> Dict:
        """
        Evaluate a trajectory across multiple scenarios.
        
        Args:
            trajectory: List of (s, l) positions
            scenarios: List of scenario dicts with agent predictions
            corridor_confidence: Confidence in corridor validity
            
        Returns:
            {'risk': float, 'is_safe': bool, 'breakdown': dict}
        """
        evaluations = []
        
        for i, scenario in enumerate(scenarios):
            eval = self._evaluate_single_scenario(trajectory, scenario, i)
            evaluations.append(eval)
        
        # Aggregate risk
        risk = self.risk_aggregator.aggregate_risk(
            evaluations, 
            corridor_confidence,
            RiskMetric.CVAR_ALPHA
        )
        
        # Determine if safe
        safe_prob = self.risk_aggregator.compute_safe_probability(evaluations)
        is_safe = safe_prob > 0.8  # 80% threshold
        
        # Get breakdown
        breakdown = self.risk_aggregator.get_risk_breakdown(evaluations)
        
        return {
            'risk': risk,
            'is_safe': is_safe,
            'safe_probability': safe_prob,
            'breakdown': breakdown,
            'num_scenarios': len(scenarios)
        }
    
    def _evaluate_single_scenario(self,
                                  trajectory: List[Tuple[float, float]],
                                  scenario: Dict,
                                  scenario_id: int) -> ScenarioEvaluation:
        """Evaluate trajectory in one scenario"""
        # Check collision
        collision = False
        min_dist = float('inf')
        
        agents = scenario.get('agents', [])
        
        for t, (s, l) in enumerate(trajectory):
            for agent in agents:
                a_s, a_l = agent.get('s', 0), agent.get('l', 0)
                
                # Simple distance check
                dist = np.sqrt((s - a_s)**2 + (l - a_l)**2)
                min_dist = min(min_dist, dist)
                
                if dist < 3.0:  # Collision threshold
                    collision = True
        
        # Boundary and keepout margins (simplified)
        l_values = [p[1] for p in trajectory]
        min_boundary = min(abs(l) for l in l_values)
        min_keepout = min_boundary  # Simplified
        
        # Progress
        progress = trajectory[-1][0] if trajectory else 0
        
        # Comfort (lateral motion)
        l_range = max(l_values) - min(l_values)
        comfort = max(0, 1 - l_range / 10)
        
        return ScenarioEvaluation(
            scenario_id=scenario_id,
            collision=collision,
            min_boundary_margin=min_boundary,
            min_keepout_margin=min_keepout,
            progress=progress,
            comfort=comfort
        )


def create_risk_aggregator() -> RiskAggregator:
    """Factory function"""
    config = RiskConfig(
        alpha=0.2,
        confidence_threshold=0.5,
        weight_collision=100.0,
        weight_boundary=10.0,
        weight_keepout=20.0,
        weight_violation=50.0
    )
    return RiskAggregator(config)


if __name__ == "__main__":
    # Test risk aggregator
    aggregator = create_risk_aggregator()
    
    # Create test evaluations
    evaluations = [
        ScenarioEvaluation(0, False, 1.0, 1.0, 50, 0.9),
        ScenarioEvaluation(1, False, 0.8, 0.8, 45, 0.85),
        ScenarioEvaluation(2, False, 0.5, 0.5, 40, 0.8),
        ScenarioEvaluation(3, True, 0.1, 0.1, 10, 0.5),  # Collision
        ScenarioEvaluation(4, False, 0.3, 0.3, 35, 0.7),
    ]
    
    # Compute risk
    risk = aggregator.aggregate_risk(evaluations, corridor_confidence=0.9)
    print(f"CVaR Risk: {risk:.2f}")
    
    # Breakdown
    breakdown = aggregator.get_risk_breakdown(evaluations)
    print(f"Breakdown: {breakdown}")
    
    # With low confidence
    risk_low = aggregator.aggregate_risk(evaluations, corridor_confidence=0.3)
    print(f"CVaR Risk (low confidence): {risk_low:.2f}")
