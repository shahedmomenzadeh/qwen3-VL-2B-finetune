# SFT/GRPO Training Configurations — 24 GB vs 32 GB VRAM (prod) + 8 GB Lite

Qwen3-VL-2B-Instruct QLoRA for cataract surgery video understanding.

**Current defaults (48 GB single GPU, `train_sft.sh`/`train.sh`): `SFT NFRAMES 60` / `GRPO NFRAMES 32 G=5 vmax 131072→262144`** (~7.6k / ~4.1k visual tok/sample, SFT `8192` limit).  
This doc’s `100`-frame configs are **high-coverage pushes** for 24/32 GB cards (~1 frame/7s on 12-min full videos). Lite `8-16` frames tested on RTX 4060 8GB (`lite_e2e_benchmark.sh`).

> **Note:** `train_sft.sh` default `60` fits `8192` at `131072` (`~7.6k`); `100/262144` needs `max_seq_length 16384` or `131072`. `GRPO_ISSUES.md P1-1` `lora_dropout 0.0` GRPO / `0.05` SFT; `use_liger_kernel True` SFT / `False` GRPO (custom loss).

---

## Quick Reference

| Parameter | 24 GB Config | 32 GB Config |
|---|---|---|
| `NFRAMES` | 100 | 100 |
| `VIDEO_MIN_PIXELS` | 98,304 (96×32²) | 131,072 (128×32²) |
| `VIDEO_MAX_PIXELS` | 196,608 (192×32²) | 327,680 (320×32²) |
| `BITS` | 4 (QLoRA) | 4 (QLoRA) |
| `LORA_RANK` | **16** (prod baseline) | **16** |
| `LORA_ALPHA` | **32** | **32** |
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
| LoRA adapters (rank 16, bf16) | ~0.05 GB |
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
LORA_RANK=16 \
LORA_ALPHA=32 \
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
| 2nd | `BITS=16` (if not QLoRA) | ~2 GB |
| 3rd | `NFRAMES=80` (last resort) | ~2 GB |

---

## 32 GB VRAM Configuration

**Philosophy:** Same 100 frames, but push spatial resolution higher (320×32² = ~1.7× more detail per frame) and use batch size 2 for faster convergence and smoother gradients.

### VRAM Budget

| Component | Est. VRAM |
|---|---|
| Base model (4-bit NF4) | ~2.5 GB |
| LoRA adapters (rank 16, bf16) | ~0.05 GB |
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
LORA_RANK=16 \
LORA_ALPHA=32 \
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
| `freeze_merger` | **False** | Merger full-trainable (prod baseline) |
| `max_seq_length` | 8192 (SFT) | Truncates; `60/262144` ~15.7k >8192 → use `131072` or raise to `16384` |
| `weight_decay` | 0.1 SFT / 0.0 GRPO | |
| `lr_scheduler` | cosine SFT / constant GRPO | |
| `warmup_steps` | 10 SFT / 0 GRPO | |
| `lora_dropout` | 0.05 SFT / 0.0 GRPO | GRPO deterministic old/current |
| `beta` | — | 0.04 GRPO KL |
| `dataloader_num_workers` | 4 | |

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
watch -n 1 nvidia-smi
pip install nvitop && nvitop
# Lite benchmark: polled @0.5s → bench_*/gpu.csv + time -v → output/lite_benchmark/report.md (GPU-hours)
```

Check first 5-10 steps. Lite `8GB` sweep `output/lite_benchmark/vram_sweep.csv`: `SFT 16/131072 3056MiB`, `GRPO G4 16/131072 7914MiB`, `16/262144 GRPO OOM`.

## Lite / GRPO Notes

- `GRPO` one-update (`ratio≈1`, `grpo_loss≈0` expected) logs `reward_mean/min/max`, `zero_std_group_fraction`, `advantage_mean/std`, `ratio_mean/std/min/max`, `clip_fraction`, `approx_kl`, `kl_loss`, `total_loss`, `comp_len_mean`, `entropy_proxy`, `grad_norm`.
- Video max `GRPO 210s` (`eAIZjIKBK_c/clip_09.mp4`), `SFT 364s`, `avg 25.9s/38s`.
- Tokens: `131072` ~`121 tok/frame`, `262144` ~`256` (`60/131072` `~7.6k`, `32/131072` `~4.1k` prompt, full `SFT 58M`/ep, `GRPO G5 98M` fwd `5.4M` loss).
