# SFT Training Configurations — 24 GB vs 32 GB VRAM

Qwen3-VL-2B-Instruct QLoRA SFT for cataract surgery video understanding.  
Target: **100 frames** across 12-minute full surgical videos (~1 frame every 7 seconds).

---

## Quick Reference

| Parameter | 24 GB Config | 32 GB Config |
|---|---|---|
| `NFRAMES` | 100 | 100 |
| `VIDEO_MIN_PIXELS` | 98,304 (96×32²) | 131,072 (128×32²) |
| `VIDEO_MAX_PIXELS` | 196,608 (192×32²) | 327,680 (320×32²) |
| `BITS` | 4 (QLoRA) | 4 (QLoRA) |
| `LORA_RANK` | 64 | 64 |
| `LORA_ALPHA` | 128 | 128 |
| `BATCH_PER_DEVICE` | 1 | 2 |
| `GRAD_ACCUM` | 16 | 8 |
| `Effective Batch Size` | 16 | 16 |
| `DISABLE_FLASH_ATTN2` | 0 (enabled) | 0 (enabled) |
| Est. Visual Tokens | ~9,600 | ~16,000 |
| Est. Peak VRAM | ~18-21 GB | ~26-30 GB |

---

## 24 GB VRAM Configuration

**Philosophy:** Maximize temporal coverage (100 frames), moderate spatial resolution, batch size 1.

### VRAM Budget

| Component | Est. VRAM |
|---|---|
| Base model (4-bit NF4) | ~2.5 GB |
| LoRA adapters (rank 64, bf16) | ~0.2 GB |
| Optimizer states (AdamW, fp32 master + momentum + variance) | ~1.3 GB |
| Visual encoder forward (100 frames × 196K px) | ~2-3 GB |
| LLM activations (grad ckpt + flash-attn, ~10K seq) | ~8-10 GB |
| KV cache peak during backward | ~2-3 GB |
| **Total** | **~16-20 GB** |
| **Headroom** | **~4-8 GB** |

### Run Command

```bash
NFRAMES=100 \
VIDEO_MIN_PIXELS=$((96 * 32 * 32)) \
VIDEO_MAX_PIXELS=$((192 * 32 * 32)) \
LORA_RANK=64 \
LORA_ALPHA=128 \
BITS=4 \
DISABLE_FLASH_ATTN2=0 \
BATCH_PER_DEVICE=1 \
GRAD_ACCUM=16 \
NUM_EPOCHS=2 \
LR=1e-4 \
VISION_LR=2e-6 \
MERGER_LR=1e-5 \
SAVE_STEPS=300 \
bash train_sft.sh
```

### OOM Fallback Ladder

If you hit OOM, apply these changes **one at a time** in order:

| Step | Change | VRAM Saved |
|---|---|---|
| 1st | `VIDEO_MAX_PIXELS=$((128 * 32 * 32))` → 131K | ~2 GB |
| 2nd | `LORA_RANK=32 LORA_ALPHA=64` | ~0.7 GB |
| 3rd | `NFRAMES=80` (last resort) | ~2 GB |

---

## 32 GB VRAM Configuration

**Philosophy:** Same 100 frames, but push spatial resolution higher (320×32² = ~1.7× more detail per frame) and use batch size 2 for faster convergence and smoother gradients.

### VRAM Budget

| Component | Est. VRAM |
|---|---|
| Base model (4-bit NF4) | ~2.5 GB |
| LoRA adapters (rank 64, bf16) | ~0.2 GB |
| Optimizer states (AdamW) | ~1.3 GB |
| Visual encoder forward (100 frames × 328K px, ×2 batch) | ~5-6 GB |
| LLM activations (grad ckpt + flash-attn, ~16K seq, ×2 batch) | ~12-14 GB |
| KV cache peak during backward | ~3-4 GB |
| **Total** | **~24-28 GB** |
| **Headroom** | **~4-8 GB** |

### Run Command

```bash
NFRAMES=100 \
VIDEO_MIN_PIXELS=$((128 * 32 * 32)) \
VIDEO_MAX_PIXELS=$((320 * 32 * 32)) \
LORA_RANK=64 \
LORA_ALPHA=128 \
BITS=4 \
DISABLE_FLASH_ATTN2=0 \
BATCH_PER_DEVICE=2 \
GRAD_ACCUM=8 \
NUM_EPOCHS=2 \
LR=1e-4 \
VISION_LR=2e-6 \
MERGER_LR=1e-5 \
SAVE_STEPS=300 \
bash train_sft.sh
```

### OOM Fallback Ladder

| Step | Change | VRAM Saved |
|---|---|---|
| 1st | `BATCH_PER_DEVICE=1 GRAD_ACCUM=16` | ~6-8 GB |
| 2nd | `VIDEO_MAX_PIXELS=$((256 * 32 * 32))` → 262K | ~2 GB |
| 3rd | `VIDEO_MAX_PIXELS=$((192 * 32 * 32))` → 196K | ~2 GB |

---

## Parameters Shared by Both Configs

These are already defaults in `train_sft.sh` and do not need to be passed explicitly:

| Parameter | Value | Rationale |
|---|---|---|
| `gradient_checkpointing` | True | Essential — trades compute for ~60% activation memory savings |
| `use_liger_kernel` | True | Fused kernels reduce memory overhead on cross-entropy + RMSNorm |
| `bf16` | True | Standard mixed precision |
| `tf32` | True | Faster matmuls on Ampere+ GPUs |
| `freeze_vision_tower` | True | Only train LoRA adapters, not full vision encoder |
| `freeze_llm` | True | Only train LoRA adapters, not full LLM |
| `freeze_merger` | True | Only train LoRA adapters, not the connector |
| `max_seq_length` | 32768 | Accommodates up to ~16K visual tokens + text |
| `weight_decay` | 0.1 | Regularization |
| `lr_scheduler` | cosine | Standard for SFT |
| `warmup_steps` | 10 | Short warmup |
| `lora_dropout` | 0.05 | Light regularization |
| `dataloader_num_workers` | 4 | Parallel data loading |

---

## Why These Tradeoffs

### Temporal > Spatial for Surgical Videos

Cataract surgery recognition depends on **phase transitions across time** (capsulorhexis → hydrodissection → phacoemulsification → ...). One frame every 7 seconds (100 frames / 12 min) captures these transitions. The surgical microscope provides a zoomed, well-lit, centered view — 196K-328K pixels is sufficient to distinguish instruments and tissue structures.

### QLoRA 4-bit Over Full 16-bit

16-bit base weights would consume ~5.1 GB (vs 2.5 GB for 4-bit), eating into the VRAM budget that is better spent on frames and resolution. Research shows QLoRA matches full fine-tuning quality for domain adaptation tasks.

### Flash Attention is Non-Negotiable

At 10K-16K sequence lengths, standard SDPA materializes the full N×N attention matrix (~400M-1B float entries). Flash attention computes attention **tile-by-tile** in SRAM, using ~10-50× less GPU memory. This alone frees 2-4 GB.

### Batch Size 2 (32 GB) vs 1 (24 GB)

Batch size 2 provides smoother gradients and better GPU utilization. The effective batch size stays 16 for both configs (via gradient accumulation), so convergence behavior is identical — but batch 2 runs fewer accumulation steps per optimizer update, slightly improving throughput.

---

## Monitoring During Training

```bash
# In a separate terminal — real-time GPU monitoring
watch -n 1 nvidia-smi

# Or install nvitop for a better view
pip install nvitop && nvitop
```

Check peak VRAM during the **first 5-10 training steps**. If peak stays below 22 GB (24 GB setup) or 30 GB (32 GB setup), you're safe for the entire run.
