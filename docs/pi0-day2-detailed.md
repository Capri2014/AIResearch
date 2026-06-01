# π0 Series Day 2 — Detailed Baby Steps

> Goal: Run inference demo and understand the code pipeline  
> Time: ~4-5 hours  
> Prerequisites: GPU (8GB+ for inference)

---

## Step 1: Clone OpenPI Repository

### 1.1 Create working directory

```bash
# Create a clean workspace
mkdir -p ~/openpi_workspace
cd ~/openpi_workspace

# Clone the official repo
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
```

**Expected output:**
```
Cloning into 'openpi'...
remote: Enumerating objects: 1234, done.
...
```

### 1.2 Check Python environment

```bash
# Check Python version (need 3.9+)
python3 --version

# Check if CUDA is available
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

**Expected output:**
```
Python 3.10.x
CUDA: True
GPU: NVIDIA RTX 4090  (or your GPU)
```

### 1.3 Install dependencies

```bash
# Option A: Full install (recommended)
cd openpi
pip install -e .

# Option B: Minimal install
# pip install torch torchvision
# pip install transformers accelerate
# pip install pillow numpy
```

**Expected output:**
```
Successfully installed openpi-x.x.x
```

---

## Step 2: Explore Repository Structure

### 2.1 Top-level structure

```bash
cd openpi
ls -la

# Key directories
ls -la pi0/        # Main model code
ls -la scripts/    # Running scripts
ls -la configs/    # Configuration files
```

**Expected output:**
```
openpi/
├── pi0/                    # Core package
│   ├── __init__.py
│   ├── modeling/           # Model definitions
│   ├── inference/          # Inference code
│   └── data/              # Data handling
├── scripts/               # Demo scripts
├── configs/               # YAML configs
├── README.md
└── requirements.txt
```

### 2.2 Model structure

```bash
ls -la pi0/modeling/
```

**Expected output:**
```
pi0/modeling/
├── __init__.py
├── pi0_model.py       # ← START HERE: Main π0 model
├── pi0_fast_model.py  # π0-FAST variant
├── pi0_5_model.py     # π0.5 variant
└── modules/           # Building blocks
```

### 2.3 Inference scripts

```bash
ls -la scripts/
```

**Expected output:**
```
scripts/
├── run_inference.py      # ← MAIN INFERENCE SCRIPT
├── run_finetune.py       # Fine-tuning script
└── evaluation/           # Eval scripts
```

---

## Step 3: Read Key Files (Mental Model)

### 3.1 Start with README

```bash
cat README.md | head -100
```

**What to find:**
- Supported models
- Hardware requirements
- Quick start commands

### 3.2 Read main inference script

```bash
# Read the main inference script (head)
head -100 scripts/run_inference.py
```

**What to understand:**
- What arguments does it take?
- How does it load the model?
- What's the input/output format?

### 3.3 Read model definition

```bash
# Read the main π0 model (first 150 lines)
head -150 pi0/modeling/pi0_model.py
```

**What to find:**
- Model input format
- Forward pass logic
- Action generation

---

## Step 4: Download Checkpoint (If Available)

### 4.1 Check HuggingFace

```bash
# Open in browser: https://huggingface.co/collections/physicalintelligence/pi0

# Or try to list available checkpoints
# (may require authentication)
pip install huggingface_hub
huggingface-cli list-models --search pi0
```

### 4.2 Alternative: Use mock/pretrained model

```bash
# Check if there's a demo checkpoint
ls -la checkpoints/ 2>/dev/null || echo "No checkpoints directory"

# Check config for model path
cat configs/*.yaml | head -50
```

---

## Step 5: Run Minimal Inference (The Core Goal)

### 5.1 Check available scripts

```bash
# Try running help
python scripts/run_inference.py --help
```

**Expected output:**
```
usage: run_inference.py [-h] [--robot ROBOT] [--image IMAGE] [--language LANGUAGE]
                       [--checkpoint CHECKPOINT] [--output OUTPUT]
```

### 5.2 Try minimal run (with mock data if no checkpoint)

```bash
# Try to run with minimal args
python scripts/run_inference.py \
    --robot ur5e \
    --language "pick up the cup" \
    --checkpoint dummy 2>&1 | head -50
```

### 5.3 If no checkpoint, create minimal test

```python
# Create test_inference.py in scripts/
cat > scripts/test_inference.py << 'EOF'
"""
Minimal inference test - understanding the pipeline
"""
import torch
import sys
sys.path.insert(0, '.')

# Try importing the model
try:
    from pi0.modeling.pi0_model import Pi0Model
    print("✅ Successfully imported Pi0Model")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Try importing config
try:
    from pi0.modeling import Pi0Config
    print("✅ Successfully imported Pi0Config")
except ImportError as e:
    print(f"❌ Import failed: {e}")

# Try importing inference
try:
    from pi0.inference import InferencePipeline
    print("✅ Successfully imported InferencePipeline")
except ImportError as e:
    print(f"❌ Import failed: {e}")

print("\n" + "="*50)
print("IMPORT TEST COMPLETE")
print("="*50)
EOF

python scripts/test_inference.py
```

### 5.4 Trace the actual forward pass

```python
# Create trace_forward.py
cat > scripts/trace_forward.py << 'EOF'
"""
Trace the π0 forward pass - understand data flow
"""
import torch
import sys
sys.path.insert(0, '.')

print("="*60)
print("π0 INFERENCE TRACE")
print("="*60)

# === STEP 1: Input shapes ===
print("\n[STEP 1] INPUT SHAPES")
print("-"*40)

# Simulate input observation
batch_size = 1

# Image:假设前视摄像头
image_input = torch.randn(batch_size, 3, 224, 224)
print(f"Image input:    {image_input.shape}  [B, C, H, W]")

# Language: 简单指令
language_input = "pick up the cup"
print(f"Language input: '{language_input}' (string)")

# Robot state: joint positions + velocities
# UR5e has 6 joints + gripper
robot_state = torch.randn(batch_size, 14)  # [q(7) + dq(7)]
print(f"Robot state:    {robot_state.shape}  [B, 14]")

# === STEP 2: After preprocessing ===
print("\n[STEP 2] AFTER PREPROCESSING")
print("-"*40)

# Image tokens (after vision encoder)
image_tokens = torch.randn(batch_size, 256, 1024)
print(f"Image tokens:  {image_tokens.shape}  [B, seq, D]")

# Text tokens (after tokenizer)
text_tokens = torch.randint(0, 32000, (batch_size, 32))
print(f"Text tokens:   {text_tokens.shape}  [B, L]")

# State tokens (after state encoder)
state_tokens = torch.randn(batch_size, 1024)
print(f"State tokens:  {state_tokens.shape}  [B, D]")

# === STEP 3: After VLM backbone ===
print("\n[STEP 3] AFTER VLM BACKBONE")
print("-"*40)

# Hidden state after fusion
hidden_state = torch.randn(batch_size, 1024)
print(f"Hidden state:  {hidden_state.shape}  [B, D]")

# === STEP 4: Action Expert ===
print("\n[STEP 4] ACTION EXPERT OUTPUT")
print("-"*40)

# Action logits (before flow matching)
action_dim = 7  # 6 joints + gripper
chunk_length = 7  # 7 timesteps
action_logits = torch.randn(batch_size, chunk_length, action_dim)
print(f"Action logits: {action_logits.shape}  [B, T, action_dim]")

# === STEP 5: Flow Matching ===
print("\n[STEP 5] FLOW MATCHING DECODE")
print("-"*40)

# After flow matching ODE solver
action_chunk = torch.randn(batch_size, chunk_length, action_dim)
print(f"Action chunk:  {action_chunk.shape}  [B, T, action_dim]")

# === STEP 6: Robot command ===
print("\n[STEP 6] ROBOT COMMAND")
print("-"*40)

# Flatten for robot controller
robot_command = action_chunk.flatten(1)
print(f"Robot command: {robot_command.shape}  [B, T*action_dim]")
print(f"Values (first 7): {robot_command[0][:7].tolist()}")

print("\n" + "="*60)
print("INFERENCE TRACE COMPLETE")
print("="*60)

# === SUMMARY TABLE ===
print("\n" + "="*60)
print("SUMMARY: Data Flow Table")
print("="*60)
print(f"""
| Stage       | Input Shape              | Output Shape             |
|-------------|--------------------------|--------------------------|
| Image       | [1, 3, 224, 224]        | [1, 256, 1024] (tokens) |
| Language    | "string"                 | [1, 32] (tokens)        |
| State       | [1, 14]                  | [1, 1024] (tokens)       |
| VLM Output  | -                        | [1, 1024] (hidden)      |
| Action Logits| -                       | [1, 7, 7] (logits)      |
| Action Chunk| -                        | [1, 7, 7] (Δq)          |
| Robot Cmd   | -                        | [1, 49] (flat)          |
""")
EOF

python scripts/trace_forward.py
```

---

## Step 6: Understand Data Format

### 6.1 Check dataset code

```bash
ls -la pi0/data/
```

### 6.2 Read dataset definition

```bash
head -100 pi0/data/dataset.py
```

### 6.3 Understand episode structure

```python
# Create understand_data.py
cat > scripts/understand_data.py << 'EOF'
"""
Understand OpenPI / π0 data format
"""
print("="*60)
print("OPENPI DATA FORMAT")
print("="*60)

# === EXPECTED EPISODE STRUCTURE ===
print("""
An episode in OpenPI format:

Episode
├── observation_0
│   ├── images: Dict[str, Tensor]   # {"front": [3,H,W], "wrist": [3,H,W]}
│   ├── state: Tensor                # [14] = q(7) + dq(7)
│   └── language: str               # "pick up the red cup"
├── observation_1
│   └── ...
├── ...
├── observation_T
│   └── ...
├── actions: Tensor                  # [T, action_dim]
│   # For UR5e: [T, 7] = [q0,q1,q2,q3,q4,q5,gripper]
└── metadata: Dict
    ├── robot_type: str              # "ur5e"
    ├── episode_id: str
    └── ... (camera params, etc.)

In summary:
- observation: dict with images, state, language
- action: delta_q for each timestep
- Typical episode: 100-500 timesteps (3-15 seconds at 30Hz)
""")

# === DATA LOADING EXAMPLE ===
print("\n" + "="*60)
print("DATA LOADING (Pseudocode)")
print("="*60)
print("""
from pi0.data import OpenPIDataset

dataset = OpenPIDataset(
    data_path="/path/to/data",
    robot_type="ur5e",
    image_size=224,
)

# Get single episode
episode = dataset[0]

# Access
images = episode["images"]       # Dict[str, Tensor]
state = episode["state"]          # Tensor [T, 14]
actions = episode["actions"]       # Tensor [T, 7]
language = episode["language"]     # str
metadata = episode["metadata"]    # Dict

# For training: collate multiple episodes
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=8)
for batch in loader:
    # batch["image"]: [B, T, C, H, W]
    # batch["state"]: [B, T, 14]
    # batch["actions"]: [B, T, 7]
    pass
""")

print("\n" + "="*60)
print("COMMON DATASET FORMATS")
print("="*60)
print("""
| Format      | Description           | Used By      |
|-------------|----------------------|--------------|
| OpenPI     | images + state + act | π0, π0.5    |
| Bridge     | image + action       | Bridge data  |
| RLDS       | TFRecord format       | Google       |
| DAPG       | npz files            | Stanford     |
| ALOHA      | hdf5 files           | ACT          |
| LIBERO     | image + pose         | LIBERO benchmark |
""")
EOF

python scripts/understand_data.py
```

---

## Step 7: Design Mini Fine-tune Experiment

### 7.1 Identify what's needed

```bash
# Check fine-tuning script
head -50 scripts/run_finetune.py
```

### 7.2 Create experiment design

```python
# Create experiment_design.py
cat > scripts/experiment_design.py << 'EOF'
"""
Mini Fine-tune Experiment Design
"""
print("="*60)
print("MINI FINE-TUNE EXPERIMENT")
print("="*60)

print("""
GOAL: Fine-tune π0 on a small custom dataset

OPTIONS:

Option A: LIBERO (Recommended for quick start)
─────────────────────────────────────────────
- Download LIBERO dataset (simulation robot tasks)
- 10 tasks, 100 episodes each
- Simple table-top manipulation
- No real robot needed

Commands:
# Download
git clone https://github.com/LIBRERO-benchmark/LIBERO.git
cd LIBERO

# Convert to OpenPI format
python scripts/convert_libero.py --output data/libero_openpi/

# Fine-tune
python scripts/run_finetune.py \\
    --model pi0-ur5e-base \\
    --dataset data/libero_openpi \\
    --epochs 5 \\
    --lora_rank 16


Option B: ALOHA
─────────────────────────────────────────────
- Real robot teleoperation data
- Complex bimanual tasks
- Larger dataset (1000+ episodes)

Commands:
# Convert ALOHA to OpenPI
python scripts/convert_aloha.py --input data/aloha --output data/aloha_openpi


Option C: Custom Data
─────────────────────────────────────────────
- Record your own episodes
- Use keyboard/joystick teleop
- Convert to OpenPI format

Required fields:
- images/ (camera frames)
- states.json (joint positions)
- actions.json (joint velocities)
- language.txt (instruction)


HARDWARE REQUIREMENTS:
─────────────────────────────────────────────
| Task           | GPU Memory | Time    |
|----------------|------------|---------|
| Inference     | 8GB       | 5 min  |
| LoRA ft       | 22GB      | 2-4 hr |
| Full ft       | 80GB+     | 1+ day |

RECOMMENDED: Start with LIBERO + LoRA
""")
EOF

python scripts/experiment_design.py
```

---

## Day 2 Checklist

| Task | Status | Notes |
|------|--------|-------|
| Clone OpenPI | ⬜ | |
| Explore structure | ⬜ | |
| Read key files | ⬜ | README, inference script, model |
| Run inference | ⬜ | Even with mock data |
| Trace forward pass | ⬜ | Print shapes |
| Understand data format | ⬜ | Episode structure |
| Design experiment | ⬜ | Which dataset to use |

---

## Expected Outputs After Day 2

1. **Environment ready**: OpenPI installed, can import modules
2. **Mental model**: Know data flow (image → tokens → VLM → action)
3. **Data format understood**: Know what an episode looks like
4. **Next experiment chosen**: LIBERO or custom data

---

## Troubleshooting

### Common Issues

| Error | Solution |
|-------|----------|
| CUDA out of memory | Use smaller batch size |
| Import errors | Check PYTHONPATH |
| No checkpoint | Use mock data for now |
| Old PyTorch | pip install --upgrade torch |

### Ask for Help

If stuck:
1. Check OpenPI GitHub issues
2. Check HuggingFace discussions
3. Create minimal reproduction script

---

*Day 2 Complete when: You can trace image→action and know your next experiment*
