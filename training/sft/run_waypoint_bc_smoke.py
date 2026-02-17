"""Smoke test for waypoint BC with metrics."""

import sys
import numpy as np
sys.path.insert(0, "/data/.openclaw/workspace/AIResearch-repo")

from training.sft.train_waypoint_bc_with_metrics import (
    compute_ade,
    compute_fde,
    compute_metrics,
    WaypointBCModel,
    BCConfig,
    train_waypoint_bc,
)

def test_metrics():
    """Test ADE/FDE computation."""
    pred = np.array([[0, 0], [1, 1], [2, 2]])
    gt = np.array([[0, 0], [1, 0], [2, 0]])
    
    ade = compute_ade(pred, gt)
    fde = compute_fde(pred, gt)
    
    print(f"ADE: {ade:.4f} (expected: 1.0)")
    print(f"FDE: {fde:.4f} (expected: 2.0)")
    
    assert abs(ade - 1.0) < 0.01, f"ADE mismatch: {ade}"
    assert abs(fde - 2.0) < 0.01, f"FDE mismatch: {fde}"
    
    # Test batch metrics
    preds = np.array([[[0, 0], [1, 1]], [[0, 0], [1, 1]]])
    gts = np.array([[[0, 0], [1, 0]], [[0, 0], [1, 0]]])
    
    m = compute_metrics(preds, gts, prefix="test_")
    print(f"Batch ADE: {m['test_ade']:.4f}")
    print(f"Batch FDE: {m['test_fde']:.4f}")

def test_model():
    """Test model forward pass."""
    model = WaypointBCModel(num_waypoints=4, latent_dim=512)
    
    import torch
    z = torch.randn(2, 512)  # Batch of 2
    waypoints = model(z)
    
    print(f"Input shape: {z.shape}")
    print(f"Output shape: {waypoints.shape}")
    
    assert waypoints.shape == (2, 4, 2), f"Shape mismatch: {waypoints.shape}"

def test_training():
    """Test full training loop with small config."""
    import torch
    from torch.utils.data import TensorDataset
    
    config = BCConfig(
        n_samples=100,
        num_waypoints=4,
        epochs=5,
        batch_size=16,
    )
    
    # Create tiny synthetic dataset
    latents = torch.randn(100, 512)
    waypoints = torch.randn(100, 4, 2)
    dataset = TensorDataset(latents, waypoints)
    
    model, metrics = train_waypoint_bc(
        config=config,
        train_dataset=dataset,
        eval_dataset=dataset,  # Use same for quick test
    )
    
    print(f"Final metrics: {metrics}")

if __name__ == "__main__":
    print("=== Testing Metrics ===")
    test_metrics()
    print("\n=== Testing Model ===")
    test_model()
    print("\n=== Testing Training ===")
    test_training()
    print("\n✓ All smoke tests passed!")
