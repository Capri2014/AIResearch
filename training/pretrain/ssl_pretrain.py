"""
SSL Pretrain for Driving Pipeline

Self-supervised pretraining on Waymo episodes before waypoint BC.
Uses contrastive learning and temporal prediction objectives.

Pipeline: Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
import argparse
import json
import os
from pathlib import Path


class TemporalAugmentation:
    """Temporal augmentations for driving sequences."""

    @staticmethod
    def temporal_crop(waypoints: torch.Tensor, min_len: int = 5) -> torch.Tensor:
        """Randomly crop temporal sequence."""
        T = waypoints.shape[0]
        if T <= min_len:
            return waypoints
        start = np.random.randint(0, T - min_len)
        end = np.random.randint(start + min_len, T + 1)
        return waypoints[start:end]

    @staticmethod
    def temporal_mask(waypoints: torch.Tensor, mask_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly mask waypoints for masked prediction."""
        T = waypoints.shape[0]
        mask = torch.rand(T) < mask_ratio
        masked_waypoints = waypoints.clone()
        masked_waypoints[mask] = 0
        return masked_waypoints, mask

    @staticmethod
    def add_noise(waypoints: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
        """Add Gaussian noise to waypoints."""
        noise = torch.randn_like(waypoints) * noise_std
        return waypoints + noise


class DrivingEncoder(nn.Module):
    """Encoder for driving state sequences."""

    def __init__(
        self,
        input_dim: int = 2,  # (x, y) waypoint coordinates
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # GRU encoder for temporal sequences
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) waypoint sequence
        Returns:
            (B, embedding_dim) embedding
        """
        # GRU encoding
        output, hidden = self.gru(x)

        # Concatenate forward and backward final hidden states
        # hidden: (num_layers * 2, B, hidden_dim)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=-1)

        # Project to embedding space
        embedding = self.projection(combined)
        return embedding


class ContrastiveLoss(nn.Module):
    """NT-Xent contrastive loss for driving embeddings."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: (B, D) embeddings from view i
            z_j: (B, D) embeddings from view j
        Returns:
            Scalar loss
        """
        B = z_i.shape[0]

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        # Concatenate all embeddings
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)

        # Create positive mask (i with j, j with i)
        sim_ij = torch.diag(sim_matrix, B)
        sim_ji = torch.diag(sim_matrix, -B)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        # Create negative mask (all except diagonal)
        mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)
        negatives = sim_matrix[mask].reshape(2 * B, -1)

        # InfoNCE loss
        loss = -torch.logsumexp(positives - negatives.max(dim=-1, keepdim=True)[0], dim=-1)
        return loss.mean()


class MaskedPredictionLoss(nn.Module):
    """Loss for masked waypoint prediction."""

    def __init__(self, predictor_hidden: int = 64):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(2, predictor_hidden),
            nn.ReLU(),
            nn.Linear(predictor_hidden, 2),
        )

    def forward(self, masked_input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            masked_input: (B, T, 2) masked waypoints
            target: (B, T, 2) original waypoints
            mask: (B, T) boolean mask indicating masked positions
        Returns:
            Scalar loss
        """
        # Predict masked positions
        pred = self.predictor(masked_input)

        # Compute MSE only on masked positions
        mse = F.mse_loss(pred[mask], target[mask], reduction='mean')
        return mse


class FuturePredictionLoss(nn.Module):
    """Predict future waypoints from current state."""

    def __init__(self, embedding_dim: int = 64, prediction_horizon: int = 10):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        output_dim = prediction_horizon * 2  # 2 coords per waypoint
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_dim),
        )

    def forward(self, encoder_output: torch.Tensor, future_waypoints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: (B, embedding_dim) current state encoding
            future_waypoints: (B, horizon, 2) future waypoints to predict
        Returns:
            Scalar loss
        """
        B = encoder_output.shape[0]
        horizon = future_waypoints.shape[1]
        if horizon != self.prediction_horizon:
            # Resize if needed
            future_flat = F.interpolate(
                future_waypoints.permute(0, 2, 1),  # (B, 2, horizon)
                size=self.prediction_horizon,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1).reshape(B, -1)
        else:
            future_flat = future_waypoints.reshape(B, -1)
        pred = self.predictor(encoder_output)
        return F.mse_loss(pred, future_flat)


class SSLPretrainModel(nn.Module):
    """SSL pretrained encoder with multiple objectives."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.encoder = DrivingEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.contrastive_loss = ContrastiveLoss(temperature)
        self.masked_loss = MaskedPredictionLoss()
        self.future_loss = FuturePredictionLoss(embedding_dim, prediction_horizon=10)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """
        Forward pass.

        Args:
            x: (B, T, 2) waypoint sequence
            return_embedding: If True, return embedding only
        """
        if return_embedding:
            return self.encoder(x)

        # Create two views for contrastive learning
        x_i = x + torch.randn_like(x) * 0.1  # View 1: slight noise
        x_j = TemporalAugmentation.add_noise(x, noise_std=0.1)  # View 2: more noise

        z_i = self.encoder(x_i)
        z_j = self.encoder(x_j)

        return z_i, z_j

    def compute_loss(
        self,
        x: torch.Tensor,
        future_waypoints: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute SSL losses.

        Args:
            x: (B, T, 2) current waypoint sequence
            future_waypoints: (B, 20, 2) future waypoints (optional)

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Contrastive loss from two augmented views
        x_i = x + torch.randn_like(x) * 0.05
        x_j = TemporalAugmentation.add_noise(x, noise_std=0.1)

        z_i = self.encoder(x_i)
        z_j = self.encoder(x_j)
        losses['contrastive'] = self.contrastive_loss(z_i, z_j)

        # Masked prediction loss (simplified)
        T = x.shape[1]
        masked_x, mask = TemporalAugmentation.temporal_mask(x)
        # Use the masked_x directly as prediction target is x
        # For simplicity, we just use MSE between original and noisy
        losses['masked'] = F.mse_loss(x_i, x)

        # Future prediction loss
        if future_waypoints is not None:
            encoder_out = self.encoder(x)
            losses['future'] = self.future_loss(encoder_out, future_waypoints)

        # Total loss
        losses['total'] = sum(losses.values())

        return losses

    def predictor_from_embedding(self, embedding: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Predict masked waypoints from embedding."""
        # Simple linear projection for masked prediction
        B, T = mask.shape
        pred = self.masked_loss.predictor(embedding.unsqueeze(1).expand(-1, T, -1))
        return pred.reshape(B, T, -1)


class WaypointSequenceDataset(Dataset):
    """Dataset for SSL pretrain on waypoint sequences."""

    def __init__(
        self,
        num_episodes: int = 100,
        sequence_length: int = 30,
        prediction_horizon: int = 20,
        generate_synthetic: bool = True,
    ):
        self.num_episodes = num_episodes
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        if generate_synthetic:
            self.data = self._generate_synthetic_episodes()
        else:
            self.data = self._load_from_files()

    def _generate_synthetic_episodes(self) -> list:
        """Generate synthetic driving episodes."""
        episodes = []
        for _ in range(self.num_episodes):
            # Generate realistic driving trajectory
            T = self.sequence_length + self.prediction_horizon

            # Start position
            x, y = 0.0, 0.0
            vx, vy = 5.0, 0.0  # 5 m/s forward

            waypoints = []
            for t in range(T):
                waypoints.append([x, y])

                # Simple kinematic model with some noise
                noise = np.random.randn(2) * 0.2
                x += vx * 0.1 + noise[0]
                y += vy * 0.1 + noise[1]

                # Add some turning
                if t > T // 3 and t < 2 * T // 3:
                    angle = np.random.randn() * 0.1
                    vx_new = vx * np.cos(angle) - vy * np.sin(angle)
                    vy_new = vx * np.sin(angle) + vy * np.cos(angle)
                    vx, vy = vx_new, vy_new

            waypoints = np.array(waypoints, dtype=np.float32)
            episodes.append(waypoints)

        return episodes

    def _load_from_files(self) -> list:
        """Load episodes from files (placeholder)."""
        # TODO: Load from Waymo npz files
        return self._generate_synthetic_episodes()

    def __len__(self) -> int:
        # Ensure we don't go past the end of episodes
        max_start = self.sequence_length + self.prediction_horizon
        valid_starts = self.sequence_length // 5
        return max(0, len(self.data) * valid_starts)

    def __getitem__(self, idx: int) -> dict:
        episode_idx = idx // (self.sequence_length // 5)
        start_idx = (idx % (self.sequence_length // 5)) * 5

        episode = self.data[episode_idx]
        
        # Ensure we have enough data
        end_idx = start_idx + self.sequence_length + self.prediction_horizon
        if end_idx > len(episode):
            # Pad with zeros if needed
            padding = torch.zeros(end_idx - len(episode), 2)
            episode_padded = torch.from_numpy(episode).float()
            episode_padded = torch.cat([episode_padded, padding], dim=0)
        else:
            episode_padded = torch.from_numpy(episode).float()
        
        current_seq = episode_padded[start_idx:start_idx + self.sequence_length]
        future_seq = episode_padded[start_idx + self.sequence_length:start_idx + self.sequence_length + self.prediction_horizon]

        return {
            'current': current_seq,
            'future': future_seq,
        }


def train_ssl_pretrain(
    model: SSLPretrainModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    log_interval: int = 10,
) -> dict:
    """Train SSL pretrain model."""
    model = model.to(device)
    model.train()

    history = {'total': [], 'contrastive': [], 'masked': [], 'future': []}

    for epoch in range(num_epochs):
        epoch_losses = {k: 0.0 for k in history.keys()}

        for batch_idx, batch in enumerate(dataloader):
            current = batch['current'].to(device)
            future = batch['future'].to(device)

            optimizer.zero_grad()

            # Compute losses
            losses = model.compute_loss(current, future)

            # Backward
            losses['total'].backward()
            optimizer.step()

            # Log
            for k, v in losses.items():
                if k in epoch_losses:
                    epoch_losses[k] += v.item()

        # Average losses
        num_batches = len(dataloader)
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            history[k].append(epoch_losses[k])

        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            for k, v in epoch_losses.items():
                print(f"  {k}: {v:.4f}")

    return history


def evaluate_embeddings(model: SSLPretrainModel, dataloader: DataLoader, device: str = 'cpu') -> dict:
    """Evaluate learned embeddings."""
    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            current = batch['current'].to(device)
            emb = model.encoder(current)
            embeddings.append(emb.cpu())

    embeddings = torch.cat(embeddings, dim=0)

    # Compute embedding statistics
    stats = {
        'mean': embeddings.mean(dim=0).numpy().tolist(),
        'std': embeddings.std(dim=0).numpy().tolist(),
        'norm_mean': embeddings.norm(dim=-1).mean().item(),
    }

    return stats


def save_checkpoint(
    model: SSLPretrainModel,
    optimizer: torch.optim.Optimizer,
    history: dict,
    path: str,
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def main():
    parser = argparse.ArgumentParser(description='SSL Pretrain for Driving Pipeline')
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--sequence-length', type=int, default=30, help='Sequence length')
    parser.add_argument('--prediction-horizon', type=int, default=20, help='Prediction horizon')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--embedding-dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--num-layers', type=int, default=2, help='GRU layers')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.1, help='Contrastive temperature')
    parser.add_argument('--output-dir', type=str, default='out/ssl_pretrain', help='Output directory')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create dataset and dataloader
    dataset = WaypointSequenceDataset(
        num_episodes=args.num_episodes,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")

    # Create model
    model = SSLPretrainModel(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        temperature=args.temperature,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    print("Starting SSL pretraining...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    history = train_ssl_pretrain(
        model,
        dataloader,
        optimizer,
        num_epochs=args.num_epochs,
        device=device,
        log_interval=args.log_interval,
    )

    # Evaluate embeddings
    print("Evaluating embeddings...")
    eval_stats = evaluate_embeddings(model, dataloader, device=device)
    print(f"Embedding norm mean: {eval_stats['norm_mean']:.4f}")

    # Save checkpoint
    checkpoint_path = os.path.join(args.output_dir, 'ssl_pretrain_checkpoint.pt')
    save_checkpoint(model, optimizer, history, checkpoint_path)

    # Save metrics
    metrics = {
        'final_total_loss': history['total'][-1],
        'final_contrastive_loss': history['contrastive'][-1],
        'final_masked_loss': history['masked'][-1],
        'final_future_loss': history['future'][-1] if 'future' in history else None,
        'embedding_stats': eval_stats,
        'config': vars(args),
    }

    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print("SSL pretraining complete!")


if __name__ == '__main__':
    main()
