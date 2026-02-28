"""
Belief Tracker

Tracks belief over discrete hypotheses using Bayesian updates.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Observation:
    """Represents an observation for belief update."""
    type: str          # e.g., "pedestrian_detected", "vehicle_speed"
    value: any         # Actual observation value
    timestamp: float


class BeliefTracker:
    """
    Tracks belief state over discrete contingencies.
    
    Uses Bayesian inference: P(h | o) ∝ P(o | h) * P(h)
    """
    
    def __init__(self, hypotheses: List[str]):
        """
        Initialize belief tracker.
        
        Args:
            hypotheses: List of hypothesis names
        """
        self.hypotheses = hypotheses
        self.n_hypotheses = len(hypotheses)
        
        # Initialize uniform prior
        self.belief = {h: 1.0 / self.n_hypotheses for h in hypotheses}
        self.prior_belief = self.belief.copy()
        
        # Observation history
        self.observations: List[Observation] = []
        
        # Likelihood models for each observation type
        self.likelihood_models: Dict[str, Dict[str, callable]] = {}
    
    def register_likelihood_model(
        self, 
        observation_type: str, 
        hypothesis: str,
        likelihood_fn: callable
    ):
        """
        Register a likelihood function P(observation | hypothesis).
        
        Args:
            observation_type: Type of observation (e.g., "pedestrian_motion")
            hypothesis: Hypothesis name
            likelihood_fn: Function that computes P(obs | hypothesis)
                          Should take observation value and return probability
        """
        if observation_type not in self.likelihood_models:
            self.likelihood_models[observation_type] = {}
        
        self.likelihood_models[observation_type][hypothesis] = likelihood_fn
    
    def update(self, observation: Observation) -> Dict[str, float]:
        """
        Update belief given new observation.
        
        Args:
            observation: New observation
            
        Returns:
            Updated belief dictionary
        """
        self.observations.append(observation)
        
        # Compute likelihoods for each hypothesis
        likelihoods = {}
        for h in self.hypotheses:
            if (observation.type in self.likelihood_models and 
                h in self.likelihood_models[observation.type]):
                
                # Use registered likelihood model
                likelihood_fn = self.likelihood_models[observation.type][h]
                try:
                    likelihoods[h] = likelihood_fn(observation.value)
                except:
                    likelihoods[h] = 0.01  # Default if computation fails
            else:
                # Default likelihood (slight preference for current belief)
                likelihoods[h] = 0.5
        
        # Bayesian update: posterior ∝ likelihood * prior
        posterior = {}
        for h in self.hypotheses:
            prior = self.belief.get(h, 1.0 / self.n_hypotheses)
            likelihood = likelihoods.get(h, 0.01)
            posterior[h] = likelihood * prior
        
        # Normalize
        total = sum(posterior.values())
        if total > 1e-10:
            self.belief = {h: p / total for h, p in posterior.items()}
        else:
            # Reset to uniform if numerical issues
            self.belief = {h: 1.0 / self.n_hypotheses for h in self.hypotheses}
        
        return self.belief.copy()
    
    def batch_update(self, observations: List[Observation]) -> Dict[str, float]:
        """Update belief with multiple observations."""
        for obs in observations:
            self.update(obs)
        return self.belief.copy()
    
    def reset(self):
        """Reset belief to prior."""
        self.belief = self.prior_belief.copy()
        self.observations.clear()
    
    def get_belief(self) -> Dict[str, float]:
        """Get current belief state."""
        return self.belief.copy()
    
    def get_most_likely(self) -> str:
        """Get most likely hypothesis."""
        return max(self.belief.keys(), key=lambda h: self.belief[h])
    
    def is_resolved(self, threshold: float = 0.9) -> bool:
        """Check if belief is resolved (one hypothesis > threshold)."""
        return any(b > threshold for b in self.belief.values())
    
    def get_resolved_hypothesis(self, threshold: float = 0.9) -> Optional[str]:
        """Get resolved hypothesis if any."""
        for h, b in self.belief.items():
            if b > threshold:
                return h
        return None
    
    def get_entropy(self) -> float:
        """Get entropy of belief distribution."""
        entropy = 0.0
        for b in self.belief.values():
            if b > 1e-10:
                entropy -= b * np.log(b)
        return entropy
    
    def __repr__(self):
        return f"BeliefTracker({self.belief})"


# === Pre-defined likelihood models for common scenarios ===

def create_pedestrian_likelihood_models() -> Dict[str, Dict[str, callable]]:
    """
    Create likelihood models for pedestrian crossing scenario.
    """
    return {
        "pedestrian_motion": {
            "pedestrian_yields": lambda v: 0.9 if v < 0.5 else 0.1,  # Stopped = yields
            "pedestrian_crosses": lambda v: 0.8 if v > 1.0 else 0.2,  # Moving = crossing
        },
        "pedestrian_position": {
            "pedestrian_yields": lambda d: 0.8 if d > 5.0 else 0.2,  # Far = yields
            "pedestrian_crosses": lambda d: 0.7 if d < 3.0 else 0.3,  # Close = crossing
        }
    }


def create_vehicle_likelihood_models() -> Dict[str, Dict[str, callable]]:
    """
    Create likelihood models for highway cut-in scenario.
    """
    return {
        "lateral_velocity": {
            "vehicle_maintains_lane": lambda v: 0.85 if abs(v) < 0.1 else 0.15,
            "vehicle_cuts_in": lambda v: 0.7 if v > 0.3 else 0.3,
        },
        "distance": {
            "vehicle_maintains_lane": lambda d: 0.8 if d > 5.0 else 0.2,
            "vehicle_cuts_in": lambda d: 0.75 if d < 3.0 else 0.25,
        }
    }


def create_sensor_likelihood_models() -> Dict[str, Dict[str, callable]]:
    """
    Create likelihood models for sensor degradation scenario.
    """
    return {
        "detection_confidence": {
            "nominal": lambda c: 0.9 if c > 0.8 else 0.1,
            "degraded": lambda c: 0.8 if c < 0.5 else 0.2,
        },
        "latency": {
            "nominal": lambda l: 0.9 if l < 0.05 else 0.1,
            "degraded": lambda l: 0.7 if l > 0.2 else 0.3,
        }
    }


def create_belief_tracker_for_scenario(scenario_name: str) -> BeliefTracker:
    """
    Create a pre-configured belief tracker for a scenario.
    """
    if scenario_name == "pedestrian_crossing":
        tracker = BeliefTracker(["pedestrian_yields", "pedestrian_crosses"])
        models = create_pedestrian_likelihood_models()
    elif scenario_name == "highway_cut_in":
        tracker = BeliefTracker(["vehicle_maintains_lane", "vehicle_cuts_in"])
        models = create_vehicle_likelihood_models()
    elif scenario_name == "sensor_degradation":
        tracker = BeliefTracker(["nominal", "degraded"])
        models = create_sensor_likelihood_models()
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    # Register models
    for obs_type, hypotheses in models.items():
        for hypothesis, likelihood_fn in hypotheses.items():
            tracker.register_likelihood_model(obs_type, hypothesis, likelihood_fn)
    
    return tracker
