#!/usr/bin/env python3
"""Verify that LoRA adapter weights actually changed during training.

LoRA: output = base_weight @ x + (B @ A) @ x * scaling
  - A is initialized with Kaiming uniform (non-zero)
  - B is initialized with zeros
So before training, the adapter contributes nothing. After even 1 step,
B should be non-zero if gradients flowed.
"""
import sys
import os
import torch

checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "output/tiny_sft_test/sft_lora"

print(f"Checking LoRA checkpoint: {checkpoint_dir}\n")

# Load the adapter model file
adapter_file = os.path.join(checkpoint_dir, "adapter_model.safetensors")
if not os.path.exists(adapter_file):
    # Try .bin
    adapter_file = os.path.join(checkpoint_dir, "adapter_model.bin")
    if not os.path.exists(adapter_file):
        print(f"ERROR: No adapter file found in {checkpoint_dir}")
        sys.exit(1)

from safetensors.torch import load_file
if adapter_file.endswith(".safetensors"):
    state_dict = load_file(adapter_file)
else:
    state_dict = torch.load(adapter_file, map_location="cpu")

print(f"Total LoRA parameters: {len(state_dict)}\n")

# Categorize
lora_A = {k: v for k, v in state_dict.items() if "lora_A" in k}
lora_B = {k: v for k, v in state_dict.items() if "lora_B" in k}

print(f"lora_A tensors: {len(lora_A)}")
print(f"lora_B tensors: {len(lora_B)}")

# Check lora_A (should be non-zero - initialized with Kaiming)
a_zeros = sum(1 for v in lora_A.values() if v.abs().max().item() == 0)
a_nonzero = len(lora_A) - a_zeros
print(f"\nlora_A: {a_nonzero}/{len(lora_A)} non-zero (expected: ALL non-zero)")

# Check lora_B (initialized to zero, should be non-zero AFTER training)
b_all_zero = 0
b_nonzero = 0
max_b_vals = []
sample_b = []
for k, v in lora_B.items():
    max_val = v.abs().max().item()
    if max_val == 0:
        b_all_zero += 1
    else:
        b_nonzero += 1
    max_b_vals.append(max_val)
    if len(sample_b) < 5:
        sample_b.append((k, max_val, v.abs().mean().item()))

print(f"lora_B: {b_nonzero}/{len(lora_B)} non-zero (expected: MOST non-zero)")
print(f"lora_B still-zero tensors: {b_all_zero}")

if b_nonzero > 0:
    print(f"\nlora_B max weight magnitude: {max(max_b_vals):.6f}")
    print(f"lora_B mean max magnitude:   {sum(max_b_vals)/len(max_b_vals):.6f}")
    print("\nSample lora_B tensors (name, max_abs, mean_abs):")
    for name, mx, mn in sample_b:
        print(f"  {name}: max={mx:.6f}, mean={mn:.8f}")

# Check lora_A is also non-zero (sanity)
a_max = max(v.abs().max().item() for v in lora_A.values())
print(f"\nlora_A max weight magnitude: {a_max:.6f}")

# Verdict
print("\n" + "=" * 50)
if b_nonzero > 0 and a_nonzero == len(lora_A):
    print("VERDICT: LoRA weights CHANGED — training is working!")
    print(f"  - lora_A: all {len(lora_A)} tensors non-zero (init: Kaiming)")
    print(f"  - lora_B: {b_nonzero}/{len(lora_B)} tensors non-zero (init: zeros)")
    print(f"  - Max B magnitude: {max(max_b_vals):.6f}")
    print("  The zeros→non-zeros transition in lora_B proves")
    print("  gradients flowed and weights updated.")
else:
    print("VERDICT: WARNING — lora_B is all zeros!")
    print("  Training may not have updated the adapters.")
    print("  Check: learning rate, loss curve, gradient flow.")
print("=" * 50)