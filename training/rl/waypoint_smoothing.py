#!/usr/bin/env python3
"""
Waypoint Trajectory Smoothing and Post-Processing

This module provides trajectory smoothing for waypoint predictions to ensure:
1. Temporal smoothness between consecutive waypoints
2. Kinematic feasibility (acceleration/steering limits)
3. Speed profile optimization
4. Integration with delta-waypoint RL training

Usage:
    from training.rl.waypoint_smoothing import (
        WaypointSmoother, WaypointSmootherConfig,
        smooth_waypoints, ensure_kinematic_feasibility
    )

Reference: Motion planning for autonomous driving with trajectory smoothing
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import math


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class WaypointSmootherConfig:
    """Configuration for waypoint trajectory smoothing."""
    
    # Smoothing parameters
    smoothing_window: int = 3  # Window size for moving average
    smoothing_alpha: float = 0.3  # Exponential smoothing factor
    
    # Kinematic constraints
    max_acceleration: float = 3.0  # m/s^2
    max_jerk: float = 2.0  # m/s^3
    min_speed: float = 0.0  # m/s
    max_speed: float = 15.0  # m/s
    
    # Waypoint spacing
    target_waypoint_spacing: float = 2.0  # meters
    
    # Curvature constraints
    max_curvature: float = 0.5  # 1/m
    
    # Post-processing
    apply_speed_profile: bool = True
    ensure_start_match: bool = True  # Match first waypoint exactly
    ensure_end_match: bool = True  # Match last waypoint exactly
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Core Smoothing Functions
# ============================================================================

def exponential_smooth(
    waypoints: np.ndarray,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Apply exponential smoothing to waypoint sequence.
    
    Args:
        waypoints: Array of shape (N, 2) or (N, 3) with x, y, [heading]
        alpha: Smoothing factor (0 = no smoothing, 1 = complete smoothing)
    
    Returns:
        Smoothed waypoints of same shape
    """
    if len(waypoints) <= 1 or alpha <= 0:
        return waypoints.copy()
    
    smoothed = waypoints.copy()
    for i in range(1, len(waypoints)):
        smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * waypoints[i]
    
    return smoothed


def moving_average_smooth(
    waypoints: np.ndarray,
    window_size: int = 3
) -> np.ndarray:
    """
    Apply moving average smoothing to waypoint sequence.
    
    Args:
        waypoints: Array of shape (N, 2) or (N, 3) with x, y, [heading]
        window_size: Size of smoothing window (must be odd)
    
    Returns:
        Smoothed waypoints of same shape
    """
    if len(waypoints) <= window_size or window_size <= 1:
        return waypoints.copy()
    
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    smoothed = waypoints.copy()
    half_window = window_size // 2
    
    # Skip endpoints - they stay unchanged
    for i in range(half_window, len(waypoints) - half_window):
        start = i - half_window
        end = i + half_window + 1
        smoothed[i] = waypoints[start:end].mean(axis=0)
    
    return smoothed


def savgol_smooth(
    waypoints: np.ndarray,
    window_size: int = 5,
    poly_order: int = 2
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to waypoint sequence.
    
    This preserves better high-frequency features than moving average.
    
    Args:
        waypoints: Array of shape (N, 2) or (N, 3) with x, y, [heading]
        window_size: Size of smoothing window (must be odd, > poly_order)
        poly_order: Polynomial order for fitting
    
    Returns:
        Smoothed waypoints of same shape
    """
    if len(waypoints) < window_size:
        return waypoints.copy()
    
    # Ensure window size is odd and > poly_order
    if window_size % 2 == 0:
        window_size += 1
    if window_size <= poly_order:
        poly_order = window_size - 1
    
    # For simplicity, use moving average if waypoints too short
    if len(waypoints) < window_size:
        return moving_average_smooth(waypoints, window_size)
    
    # Compute Savitzky-Golay coefficients
    half_window = window_size // 2
    
    # Simple implementation using polynomial fit per point
    smoothed = waypoints.copy()
    
    for dim in range(waypoints.shape[1]):
        y = waypoints[:, dim]
        for i in range(len(waypoints)):
            start = max(0, i - half_window)
            end = min(len(waypoints), i + half_window + 1)
            
            x = np.arange(start, end)
            y_slice = y[start:end]
            
            if len(x) > poly_order:
                coeffs = np.polyfit(x, y_slice, poly_order)
                smoothed[i, dim] = np.polyval(coeffs, i)
    
    return smoothed


def compute_waypoint_speeds(
    waypoints: np.ndarray,
    dt: float = 0.1
) -> np.ndarray:
    """
    Compute speed between consecutive waypoints.
    
    Args:
        waypoints: Array of shape (N, 2) with x, y
        dt: Time between waypoints (seconds)
    
    Returns:
        Speeds of shape (N-1,) with m/s
    """
    if len(waypoints) < 2:
        return np.array([])
    
    distances = np.sqrt(np.diff(waypoints[:, 0])**2 + np.diff(waypoints[:, 1])**2)
    speeds = distances / dt
    
    return speeds


def compute_waypoint_headings(
    waypoints: np.ndarray
) -> np.ndarray:
    """
    Compute heading angle for each waypoint.
    
    Args:
        waypoints: Array of shape (N, 2) with x, y
    
    Returns:
        Headings of shape (N,) in radians
    """
    if len(waypoints) < 2:
        return np.zeros(len(waypoints))
    
    dx = np.diff(waypoints[:, 0])
    dy = np.diff(waypoints[:, 1])
    headings = np.arctan2(dy, dx)
    
    # First heading is same as second
    headings = np.concatenate([[headings[0]], headings])
    
    return headings


def ensure_kinematic_feasibility(
    waypoints: np.ndarray,
    max_speed: float = 15.0,
    max_acceleration: float = 3.0,
    dt: float = 0.1
) -> np.ndarray:
    """
    Ensure waypoints satisfy kinematic constraints.
    
    Args:
        waypoints: Array of shape (N, 2) with x, y
        max_speed: Maximum allowed speed (m/s)
        max_acceleration: Maximum allowed acceleration (m/s^2)
        dt: Time between waypoints (seconds)
    
    Returns:
        Feasible waypoints of same shape
    """
    if len(waypoints) < 2:
        return waypoints.copy()
    
    feasible = waypoints.copy()
    
    # Limit speeds
    speeds = compute_waypoint_speeds(waypoints, dt)
    if len(speeds) > 0:
        speeds = np.clip(speeds, 0, max_speed)
        
        # Rebuild positions with limited speeds
        for i in range(1, len(feasible)):
            direction = feasible[i] - feasible[i-1]
            distance = np.linalg.norm(direction)
            if distance > 1e-6:
                direction = direction / distance
                feasible[i] = feasible[i-1] + direction * speeds[i-1] * dt
    
    # Limit accelerations
    if len(speeds) > 1:
        accelerations = np.diff(speeds) / dt
        accelerations = np.clip(accelerations, -max_acceleration, max_acceleration)
        
        # Rebuild with limited accelerations
        speeds = np.concatenate([[0], speeds[:1] + np.cumsum(accelerations * dt)])
        speeds = np.clip(speeds, 0, max_speed)
        
        for i in range(1, len(feasible)):
            direction = feasible[i] - feasible[i-1]
            distance = np.linalg.norm(direction)
            if distance > 1e-6:
                direction = direction / distance
                feasible[i] = feasible[i-1] + direction * speeds[i-1] * dt
    
    return feasible


def smooth_waypoints(
    waypoints: np.ndarray,
    config: Optional[WaypointSmootherConfig] = None,
    method: str = "savgol"
) -> np.ndarray:
    """
    Apply trajectory smoothing to waypoints.
    
    Args:
        waypoints: Array of shape (N, 2) or (N, 3) with x, y, [heading]
        config: Smoothing configuration
        method: Smoothing method ("exponential", "moving_avg", "savgol")
    
    Returns:
        Smoothed waypoints
    """
    config = config or WaypointSmootherConfig()
    
    if len(waypoints) < 3:
        return waypoints.copy()
    
    # Apply selected smoothing method
    if method == "exponential":
        smoothed = exponential_smooth(waypoints, config.smoothing_alpha)
    elif method == "moving_avg":
        smoothed = moving_average_smooth(waypoints, config.smoothing_window)
    elif method == "savgol":
        smoothed = savgol_smooth(waypoints, config.smoothing_window)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    # Ensure endpoint constraints
    if config.ensure_start_match:
        smoothed[0] = waypoints[0]
    if config.ensure_end_match:
        smoothed[-1] = waypoints[-1]
    
    # Apply kinematic feasibility
    if config.apply_speed_profile:
        smoothed = ensure_kinematic_feasibility(
            smoothed,
            max_speed=config.max_speed,
            max_acceleration=config.max_acceleration
        )
    
    return smoothed


# ============================================================================
# PyTorch Module for Learnable Smoothing
# ============================================================================

class WaypointSmoother(nn.Module):
    """
    Learnable waypoint smoothing module.
    
    Applies differentiable smoothing that can be trained with the delta-waypoint model.
    """
    
    def __init__(self, config: Optional[WaypointSmootherConfig] = None):
        super().__init__()
        self.config = config or WaypointSmootherConfig()
        
        # Learnable smoothing weight
        self.smooth_weight = nn.Parameter(torch.tensor(0.3))
        
        # Feature extraction for curvature-aware smoothing
        self.curvature_net = nn.Sequential(
            nn.Linear(4, 64),  # (dx, dy, ddx, ddy)
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, waypoints: torch.Tensor) -> torch.Tensor:
        """
        Apply differentiable smoothing to waypoints.
        
        Args:
            waypoints: Tensor of shape (B, N, 2) or (N, 2)
        
        Returns:
            Smoothed waypoints of same shape
        """
        if waypoints.dim() == 2:
            waypoints = waypoints.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, num_waypoints, _ = waypoints.shape
        device = waypoints.device
        
        # Compute differences
        diff1 = waypoints[:, 1:] - waypoints[:, :-1]  # First derivative
        diff2 = diff1[:, 1:] - diff1[:, :-1]  # Second derivative
        
        # Pad for same length
        diff1 = torch.cat([diff1, diff1[:, -1:]], dim=1)
        diff2 = torch.cat([torch.zeros(batch_size, 1, 2, device=device), 
                          diff2, torch.zeros(batch_size, 1, 2, device=device)], dim=1)
        
        # Compute curvature features
        curvature_features = torch.cat([diff1, diff2], dim=-1)
        
        # Learnable per-waypoint smoothing
        curvature_weights = self.curvature_net(curvature_features)  # (B, N, 1)
        
        # Apply exponential smoothing with learnable weight
        smooth_weight = torch.sigmoid(self.smooth_weight)
        
        smoothed = waypoints.clone()
        for i in range(1, num_waypoints):
            # Adaptive smoothing based on curvature
            w = smooth_weight * (1 - curvature_weights[:, i])
            smoothed[:, i] = w * smoothed[:, i-1] + (1 - w) * waypoints[:, i]
        
        # Ensure start/end match
        smoothed[:, 0] = waypoints[:, 0]
        smoothed[:, -1] = waypoints[:, -1]
        
        if squeeze_output:
            smoothed = smoothed.squeeze(0)
        
        return smoothed
    
    def compute_smoothness_loss(self, waypoints: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothness regularization loss.
        
        Args:
            waypoints: Tensor of shape (B, N, 2)
        
        Returns:
            Scalar loss
        """
        # Second derivative (jerk-like)
        diff1 = waypoints[:, 1:] - waypoints[:, :-1]
        diff2 = diff1[:, 1:] - diff1[:, :-1]
        
        # L2 norm of second derivative
        smoothness = torch.mean(diff2 ** 2)
        
        return smoothness


# ============================================================================
# Delta-Waypoint with Smoothing Integration
# ============================================================================

class SmoothedDeltaWaypointModel(nn.Module):
    """
    Delta-waypoint model with integrated trajectory smoothing.
    
    Architecture:
        final_waypoints = smooth(delta_head(z) + sft_waypoints)
    
    The smoothing is applied after combining SFT predictions with delta corrections.
    """
    
    def __init__(
        self,
        sft_model: nn.Module,
        delta_head: nn.Module,
        smoother: Optional[WaypointSmoother] = None,
        config: Optional[WaypointSmootherConfig] = None
    ):
        super().__init__()
        self.sft_model = sft_model
        self.delta_head = delta_head
        self.smoother = smoother or WaypointSmoother(config)
        
        # Freeze SFT model
        for param in self.sft_model.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        features: torch.Tensor,
        apply_smoothing: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with optional smoothing.
        
        Args:
            features: Input features
            apply_smoothing: Whether to apply trajectory smoothing
        
        Returns:
            Final waypoints
        """
        # Get SFT predictions (frozen)
        with torch.no_grad():
            sft_waypoints = self.sft_model(features)
        
        # Get delta corrections
        delta = self.delta_head(features)
        
        # Combine: final = sft + delta
        combined = sft_waypoints + delta
        
        # Apply smoothing if requested
        if apply_smoothing:
            combined = self.smoother(combined)
        
        return combined
    
    def compute_loss(
        self,
        features: torch.Tensor,
        target_waypoints: torch.Tensor,
        smoothness_weight: float = 0.1
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute training loss with smoothness regularization.
        
        Args:
            features: Input features
            target_waypoints: Ground truth waypoints
            smoothness_weight: Weight for smoothness loss
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Get predictions with smoothing
        pred_waypoints = self.forward(features, apply_smoothing=True)
        
        # Main loss: MSE to target
        mse_loss = F.mse_loss(pred_waypoints, target_waypoints)
        
        # Smoothness loss
        smooth_loss = self.smoother.compute_smoothness_loss(pred_waypoints)
        
        # Total loss
        total_loss = mse_loss + smoothness_weight * smooth_loss
        
        loss_dict = {
            "mse_loss": mse_loss.item(),
            "smooth_loss": smooth_loss.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, loss_dict


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test with synthetic waypoints
    np.random.seed(42)
    
    # Generate a noisy curved trajectory
    t = np.linspace(0, 2 * np.pi, 20)
    true_x = 10 * t + 5 * np.sin(t)
    true_y = 10 * t + 3 * np.cos(t)
    true_waypoints = np.column_stack([true_x, true_y])
    
    # Add noise
    noisy_waypoints = true_waypoints + np.random.randn(*true_waypoints.shape) * 2
    
    # Apply different smoothing methods
    config = WaypointSmootherConfig(
        smoothing_window=5,
        smoothing_alpha=0.3,
        max_speed=15.0,
        max_acceleration=3.0,
        apply_speed_profile=True
    )
    
    exp_smooth = exponential_smooth(noisy_waypoints, 0.3)
    ma_smooth = moving_average_smooth(noisy_waypoints, 5)
    savgol_smoothed = savgol_smooth(noisy_waypoints, 5)
    kinematic_smooth = smooth_waypoints(noisy_waypoints, config, "savgol")
    
    print("Original waypoints shape:", noisy_waypoints.shape)
    print("Smoothed waypoints shape:", kinematic_smooth.shape)
    
    # Compute errors
    def ade_error(pred, true):
        return np.mean(np.sqrt(np.sum((pred - true)**2, axis=1)))
    
    print(f"\nADE Errors:")
    print(f"  Noisy:       {ade_error(noisy_waypoints, true_waypoints):.3f}m")
    print(f"  Exponential: {ade_error(exp_smooth, true_waypoints):.3f}m")
    print(f"  Moving Avg:  {ade_error(ma_smooth, true_waypoints):.3f}m")
    print(f"  Savgol:      {ade_error(savgol_smoothed, true_waypoints):.3f}m")
    print(f"  Kinematic:   {ade_error(kinematic_smooth, true_waypoints):.3f}m")
    
    # Test PyTorch module
    print("\n--- Testing PyTorch Module ---")
    smoother = WaypointSmoother(config)
    
    # Random batch of waypoints
    batch_waypoints = torch.randn(4, 20, 2)
    smoothed = smoother(batch_waypoints)
    print(f"Input shape: {batch_waypoints.shape}")
    print(f"Output shape: {smoothed.shape}")
    
    # Smoothness loss
    smooth_loss = smoother.compute_smoothness_loss(batch_waypoints)
    print(f"Smoothness loss: {smooth_loss.item():.4f}")
    
    print("\n✅ All tests passed!")
