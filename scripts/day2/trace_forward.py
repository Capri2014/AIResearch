#!/usr/bin/env python3
"""
Step 5.4: Trace the π0 forward pass - understand data flow
Run: python scripts/day2/trace_forward.py
"""
import torch
import sys
print("="*60)
print("π0 INFERENCE TRACE")
print("="*60)

# === STEP 1: Input shapes ===
print("\n[STEP 1] INPUT SHAPES")
print("-"*40)

batch_size = 1
image_input = torch.randn(batch_size, 3, 224, 224)
print(f"Image input:    {image_input.shape}  [B, C, H, W]")

language_input = "pick up the cup"
print(f"Language input: '{language_input}' (string)")

robot_state = torch.randn(batch_size, 14)
print(f"Robot state:    {robot_state.shape}  [B, 14]")

# === STEP 2: After preprocessing ===
print("\n[STEP 2] AFTER PREPROCESSING")
print("-"*40)

image_tokens = torch.randn(batch_size, 256, 1024)
print(f"Image tokens:  {image_tokens.shape}  [B, seq, D]")

text_tokens = torch.randint(0, 32000, (batch_size, 32))
print(f"Text tokens:   {text_tokens.shape}  [B, L]")

state_tokens = torch.randn(batch_size, 1024)
print(f"State tokens:  {state_tokens.shape}  [B, D]")

# === STEP 3: After VLM backbone ===
print("\n[STEP 3] AFTER VLM BACKBONE")
print("-"*40)

hidden_state = torch.randn(batch_size, 1024)
print(f"Hidden state:  {hidden_state.shape}  [B, D]")

# === STEP 4: Action Expert ===
print("\n[STEP 4] ACTION EXPERT OUTPUT")
print("-"*40)

action_dim = 7
chunk_length = 7
action_logits = torch.randn(batch_size, chunk_length, action_dim)
print(f"Action logits: {action_logits.shape}  [B, T, action_dim]")

# === STEP 5: Flow Matching ===
print("\n[STEP 5] FLOW MATCHING DECODE")
print("-"*40)

action_chunk = torch.randn(batch_size, chunk_length, action_dim)
print(f"Action chunk:  {action_chunk.shape}  [B, T, action_dim]")

# === STEP 6: Robot command ===
print("\n[STEP 6] ROBOT COMMAND")
print("-"*40)

robot_command = action_chunk.flatten(1)
print(f"Robot command: {robot_command.shape}  [B, T*action_dim]")
print(f"Values (first 7): {robot_command[0][:7].tolist()}")

print("\n" + "="*60)
print("SUMMARY: Data Flow Table")
print("="*60)
print("""
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