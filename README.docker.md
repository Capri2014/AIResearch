# Docker Training Environment for E2EAD Pipeline

This document describes how to set up and use the Docker-based training environment for the E2EAD (End-to-End Autonomous Driving) pipeline.

## Overview

The Docker environment provides a reproducible, GPU-accelerated training setup with:

- **CUDA 12.1** with PyTorch for GPU-accelerated training
- **CARLA 0.9.15** simulator client for closed-loop evaluation
- **All dependencies** from `requirements.txt` pre-installed
- **Non-root user** for security
- **Multi-stage build** for optimized image size
- **Volume mounts** for data persistence and code development

## Prerequisites

### System Requirements

- Docker Engine 24.0+
- Docker Compose v2.20+
- NVIDIA Driver 525+ (for CUDA 12.1)
- NVIDIA Container Toolkit
- 16GB+ RAM (32GB+ recommended)
- 50GB+ disk space

### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify installation
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.1-devel-ubuntu22.04 \
    nvidia-smi
```

## Quick Start

### 1. Build the Docker Image

```bash
# Build the training image
docker build -f Dockerfile.training -t e2ead-training:latest .

# Or use Docker Compose to build
docker compose build training
```

### 2. Start the Training Container

```bash
# Using Docker Compose (recommended)
docker compose up -d training

# Attach to the container
docker exec -it e2ead-training bash

# Or run a command directly
docker run --rm --runtime=nvidia --gpus all \
    -v $(pwd):/data/.openclaw/workspace \
    e2ead-training:latest \
    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### 3. Run Training

```bash
# Inside the container
python -m training.sft.train_waypoint --config configs/sft_waypoint_bc.yaml

# With Weights & Biases logging
python -m training.sft.train_waypoint \
    --config configs/sft_waypoint_bc.yaml \
    --wandb_project e2ead-training \
    --wandb_entity your-entity
```

## Docker Compose Services

| Service | Description | Command |
|---------|-------------|---------|
| `training` | Main training environment | `docker compose up -d training` |
| `carla` | CARLA simulator (optional) | `docker compose --profile simulation up -d carla` |
| `evaluation` | Closed-loop evaluation | `docker compose --profile evaluation up -d evaluation` |
| `jupyter` | Jupyter Lab notebook | `docker compose --profile notebook up -d jupyter` |

### Start Specific Services

```bash
# Training only
docker compose up training

# Training + CARLA simulation
docker compose --profile simulation up

# Training + Jupyter Lab
docker compose --profile notebook up

# All services
docker compose --profile simulation --profile notebook up
```

## Volume Mounts

The following directories are mounted into the container:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./` | `/data/.openclaw/workspace` | Project source code |
| `./data` | `/data/.openclaw/workspace/data` | Training data |
| `./out` | `/data/.openclaw/workspace/outputs` | Output logs and artifacts |
| `./model_checkpoints` | `/data/.openclaw/workspace/model_checkpoints` | Model checkpoints |
| `./wandb_logs` | `/home/trainer/wandb_logs` | Weights & Biases logs |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WANDB_API_KEY` | Weights & Biases API key | - |
| `WANDB_PROJECT` | W&B project name | `e2ead-training` |
| `WANDB_ENTITY` | W&B entity (team) | - |
| `HF_TOKEN` | HuggingFace access token | - |
| `TRAIN_DATA_DIR` | Training data directory | `/data/.openclaw/workspace/data/training` |
| `OUTPUT_DIR` | Output directory | `/data/.openclaw/workspace/outputs` |
| `CHECKPOINT_DIR` | Checkpoint directory | `/data/.openclaw/workspace/model_checkpoints` |

### Setting Environment Variables

```bash
# Using .env file
cp .env.example .env
# Edit .env with your values

# Or export in shell
export WANDB_API_KEY=your-api-key
export HF_TOKEN=your-hf-token
```

## CARLA Simulation

### Starting CARLA Server

```bash
# Using Docker Compose
docker compose --profile simulation up -d carla

# CARLA will be available at: 127.0.0.1:2000
```

### Running Closed-Loop Evaluation

```bash
# Start CARLA first
docker compose --profile simulation up -d carla

# Wait for CARLA to be healthy
docker compose ps

# Run evaluation
docker compose --profile evaluation up evaluation
```

### Manual CARLA Connection

```python
import carla

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)
world = client.get_world()
```

## Jupyter Lab

### Access Jupyter Lab

```bash
# Start Jupyter service
docker compose --profile notebook up -d jupyter

# Access at http://localhost:8888
# Token: e2ead-secure-token (or set JUPYTER_TOKEN)
```

### Running Notebooks

```python
# Example notebook cell
import torch
from training.sft.waypoint_model import WaypointPolicy

model = WaypointPolicy.load_from_checkpoint("/data/.openclaw/workspace/model_checkpoints/latest/model.pt")
model = model.cuda()
model.eval()
```

## Multi-GPU Training

### Torch Distributed Data Parallel

```bash
# With 2 GPUs
docker run --rm --runtime=nvidia --gpus all \
    -e WORLD_SIZE=2 \
    -e LOCAL_RANK=0 \
    -e NVIDIA_VISIBLE_DEVICES=0,1 \
    -v $(pwd):/data/.openclaw/workspace \
    e2ead-training:latest \
    torchrun --nproc_per_node=2 training/sft/train_waypoint.py --config configs/sft_waypoint_bc.yaml
```

### Slurm Integration

```bash
# Example sbatch script
#!/bin/bash
#SBATCH --job-name=e2ead-training
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

srun --container-image e2ead-training:latest \
    --container-mounts $(pwd):/data/.openclaw/workspace \
    torchrun --nproc_per_node=2 training/sft/train_waypoint.py --config configs/sft_waypoint_bc.yaml
```

## Troubleshooting

### GPU Not Available

```bash
# Check NVIDIA drivers
nvidia-smi

# Verify Docker GPU access
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.1-devel-ubuntu22.04 nvidia-smi

# Reinstall NVIDIA Container Toolkit
sudo apt-get install --reinstall nvidia-container-toolkit
sudo systemctl restart docker
```

### Out of Memory (OOM)

```bash
# Reduce batch size in training config
# Or use gradient accumulation
python training/sft/train_waypoint.py --config configs/sft_waypoint_bc.yaml --gradient_accumulation_steps=4
```

### CUDA Out of Memory

```bash
# Clear GPU cache in Python
import torch
torch.cuda.empty_cache()

# Or set environment variable
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### CARLA Connection Failed

```bash
# Check CARLA container status
docker compose ps

# Check CARLA logs
docker logs e2ead-carla

# Verify port is accessible
netstat -tlnp | grep 2000
```

## Building for Different Platforms

### Apple Silicon (M1/M2)

```bash
# Build for ARM64
docker buildx build --platform linux/arm64 -f Dockerfile.training -t e2ead-training:arm64 .

# Note: CUDA is not available on Apple Silicon
# Use MPS instead: pip install torch with MPS support
```

### NVIDIA Jetson

```bash
# Build for Jetson (JetPack 5.x)
docker build -f Dockerfile.training -t e2ead-training:jetson --build-arg TARGET_PLATFORM=jetson .

# Run on Jetson
docker run --rm --runtime=nvidia --gpus all e2ead-training:jetson python -c "import torch; print(torch.__version__)"
```

## Development Workflow

### 1. Edit Code

Edit files in your local directory. Changes are immediately reflected in the container.

### 2. Run Tests

```bash
docker exec e2ead-training pytest tests/ -v
```

### 3. Lint Code

```bash
docker exec e2ead-training black --check training/
docker exec e2ead-training isort --check training/
docker exec e2ead-training flake8 training/
docker exec e2ead-training mypy training/
```

### 4. Commit and Push

```bash
# Make changes to code
git add .
git commit -m "Description of changes"
git push origin feature/daily-2026-02-18-docker
```

## Image Size Optimization

The Dockerfile uses a multi-stage build to minimize the final image size:

- **Builder stage**: Includes all build dependencies
- **Runtime stage**: Only runtime dependencies (no compilers, build tools)

Typical runtime image size: ~8-10GB

### Tips for Reducing Size

```dockerfile
# Clean up in builder stage
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Use --no-cache-dir for pip
RUN pip install --no-cache-dir package

# Remove unnecessary files
RUN rm -rf /root/.cache /home/trainer/.cache
```

## Security Considerations

- Container runs as non-root user (`trainer`)
- No privileged access by default
- Secrets should be passed via environment variables or Docker secrets
- Use Docker Content Trust for image verification

## Updating Dependencies

To update dependencies:

```bash
# Edit Dockerfile.training with new versions
# Rebuild the image
docker build -f Dockerfile.training -t e2ead-training:latest .

# Test the updated image
docker compose up -d training
docker exec e2ead-training pytest tests/ -v
```

## Support

For issues and questions:

1. Check the main README.md
2. Review the docs/ directory
3. Open an issue on GitHub
