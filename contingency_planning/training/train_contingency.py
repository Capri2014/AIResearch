"""
Training Loop for Contingency Network

Train neural contingency planning model.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
import os
from pathlib import Path


class ContingencyDataset(Dataset):
    """Dataset for contingency planning training."""
    
    def __init__(
        self,
        data_path: str,
        n_contingencies: int = 4,
        horizon: int = 20,
    ):
        self.data_path = data_path
        self.n_contingencies = n_contingencies
        self.horizon = horizon
        
        # Load data
        self.states = []
        self.plans = []
        self.uncertainties = []
        self.safety_labels = []
        
        if os.path.exists(data_path):
            self._load_data()
        else:
            # Generate synthetic data for now
            self._generate_synthetic()
    
    def _load_data(self):
        """Load data from file."""
        data = np.load(self.data_path, allow_pickle=True).item()
        self.states = data.get('states', [])
        self.plans = data.get('plans', [])
        self.uncertainties = data.get('uncertainties', [])
        self.safety_labels = data.get('safety_labels', [])
    
    def _generate_synthetic(self):
        """Generate synthetic training data."""
        n_samples = 1000
        
        # Random states
        self.states = [np.random.randn(256).astype(np.float32) for _ in range(n_samples)]
        
        # Random plans for each contingency
        self.plans = [
            [np.random.randn(self.horizon, 2).astype(np.float32) * 0.5 
             for _ in range(self.n_contingencies)]
            for _ in range(n_samples)
        ]
        
        # Random uncertainty targets (one-hot-ish)
        self.uncertainties = [
            np.random.dirichlet(np.ones(self.n_contingencies))
            for _ in range(n_samples)
        ]
        
        # Safety labels (1=safe, 0=unsafe)
        self.safety_labels = np.random.randint(0, 2, n_samples).tolist()
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return {
            'state': torch.from_numpy(self.states[idx]),
            'plans': [torch.from_numpy(p) for p in self.plans[idx]],
            'uncertainty': torch.from_numpy(self.uncertainties[idx]),
            'safety_label': torch.tensor(self.safety_labels[idx], dtype=torch.float32),
        }


def train_contingency_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Dict = None,
    device: str = "cuda",
) -> Dict:
    """
    Train contingency network.
    
    Args:
        model: ContingencyNetwork
        train_loader: Training data
        val_loader: Validation data
        config: Training config
        device: Device
        
    Returns:
        Training history
    """
    if config is None:
        config = {
            'lr': 1e-4,
            'epochs': 100,
            'lambda_plan': 1.0,
            'lambda_uncertainty': 0.5,
            'lambda_safety': 1.0,
        }
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_plan_loss': [],
        'train_uncertainty_loss': [],
        'train_safety_loss': [],
    }
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        epoch_loss = 0
        epoch_plan_loss = 0
        epoch_uncertainty_loss = 0
        epoch_safety_loss = 0
        
        for batch in train_loader:
            states = batch['state'].to(device)
            plans = [p.to(device) for p in batch['plans']]
            uncertainties = batch['uncertainty'].to(device)
            safety_labels = batch['safety_label'].to(device)
            
            # Forward
            loss, metrics = model.compute_loss(
                states,
                plans,
                uncertainties,
                safety_labels,
                lambda_plan=config['lambda_plan'],
                lambda_uncertainty=config['lambda_uncertainty'],
                lambda_safety=config['lambda_safety'],
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += metrics['total_loss']
            epoch_plan_loss += metrics.get('plan_loss', 0)
            epoch_uncertainty_loss += metrics.get('uncertainty_loss', 0)
            epoch_safety_loss += metrics.get('safety_loss', 0)
        
        n_batches = len(train_loader)
        history['train_loss'].append(epoch_loss / n_batches)
        history['train_plan_loss'].append(epoch_plan_loss / n_batches)
        history['train_uncertainty_loss'].append(epoch_uncertainty_loss / n_batches)
        history['train_safety_loss'].append(epoch_safety_loss / n_batches)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    states = batch['state'].to(device)
                    plans = [p.to(device) for p in batch['plans']]
                    uncertainties = batch['uncertainty'].to(device)
                    safety_labels = batch['safety_label'].to(device)
                    
                    loss, _ = model.compute_loss(
                        states, plans, uncertainties, safety_labels,
                        lambda_plan=config['lambda_plan'],
                        lambda_uncertainty=config['lambda_uncertainty'],
                        lambda_safety=config['lambda_safety'],
                    )
                    val_loss += loss.item()
            
            history['val_loss'].append(val_loss / len(val_loader))
            
            # Learning rate scheduling
            scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}")
            print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
            if val_loader:
                print(f"  Val Loss: {history['val_loss'][-1]:.4f}")
    
    return history


def create_model(config: Dict) -> nn.Module:
    """Create model from config."""
    from .contingency_network import ContingencyNetwork, ModelBasedPlanner
    
    model_config = config.get('model', {})
    
    if config.get('use_safety_filter', True):
        return ModelBasedPlanner(config)
    else:
        return ContingencyNetwork(
            state_dim=model_config.get('state_dim', 256),
            hidden_dim=model_config.get('hidden_dim', 512),
            action_dim=model_config.get('action_dim', 2),
            horizon=model_config.get('horizon', 20),
            n_contingencies=model_config.get('n_contingencies', 4),
        )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Dict,
    path: str,
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
) -> Tuple[int, Dict]:
    """Load training checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['history']


if __name__ == "__main__":
    import yaml
    
    # Load config
    config_path = "contingency_planning/configs/default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data
    train_dataset = ContingencyDataset(
        "data/contingency_train.npy",
        n_contingencies=4,
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Train
    history = train_contingency_model(model, train_loader, config=config['model'])
    
    # Save
    os.makedirs("out/contingency_models", exist_ok=True)
    save_checkpoint(model, None, 100, history, "out/contingency_models/model.pt")
