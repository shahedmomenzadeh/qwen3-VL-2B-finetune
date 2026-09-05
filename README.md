# Qwen3-VL-2B Cataract Surgery Fine-tuning (SFT + GRPO)

Fine-tune **Qwen/Qwen3-VL-2B-Instruct** with QLoRA (4-bit, `r=16 alpha=32`, merger full-trainable, `pos_embed` frozen) on cataract surgery video data (clip-level descriptions, full-video narration, and multi-choice QA for RL).

Pipeline: **Base → SFT (60 frames) → Merge → GRPO (32 frames, G=5, one-update) → Merge**

1. **SFT** on `dataset_sft` (clip + full-video visual description / CoT QA, combined stage)
2. **GRPO** on `dataset_grpo` (clip-level YouTube MCQs + 4 phase temporal tasks, `G=5` per prompt, `beta=0.04` KL to SFT reference)

GRPO is **one-update on-policy** (`generate G → old logprobs (no_grad, autocast) → per-token PPO clip ε=0.2 → KL k3 → step → discard rollout`; `ratio≈1` → `grpo_loss≈0` expected, signal is `reward/advantage/KL/total_loss`). Zero-variance groups yield `advantage=0` (not `rewards-0.5`). `lora_dropout=0.0` for GRPO (deterministic old/current), `0.05` for SFT. Custom manual token-mean loss (not `LigerFusedLinearGRPOLoss`; `use_liger_kernel False` for GRPO).

Derived from [`Qwen-VL-Series-Finetune`](https://github.com/) and adapted for Qwen3-VL (`patch 16`, merger, `mm_token_type_ids`, `video_grid_thw`).

---

## Quick Start (end-to-end)

```bash
git clone https://github.com/shahedmomenzadeh/qwen3-VL-2B-finetune.git
cd qwen3-VL-2B-finetune

# 2. Ensure dataset_sft/ and dataset_grpo/ contain Train/Validation splits

# 3. Full pipeline (48 GB default: SFT 60 frames → GRPO 32 frames G=5)
bash train_sft.sh                              # SFT only
bash train.sh                                  # SFT + GRPO (dataset_grpo)

# Smoke / lite (8 GB):
SUBSET_RATIO=0.3 bash train_sft.sh              # 30% SFT
BITS=16 NFRAMES=48 bash train_sft.sh            # 16-bit LoRA
bash lite_sft_test.sh                          # 5 samples SFT → auto-merge output/lite_sft_test/merged
bash lite_grpo_test.sh                         # 30 train /7 val GRPO G=4 nframes=8 max_completion=1024 (≈9min on RTX 4060 8GB)
GRPO_TRAIN_SAMPLES=14 GRPO_MAX_STEPS=4 bash lite_grpo_test.sh  # minimal
bash lite_e2e_benchmark.sh                     # VRAM sweep (8/16/32/48 ×131072/262144) + SFT 6 steps → Merge → GRPO 4 steps G=5 → report.md
SKIP_SWEEP=1 NFRAMES_SFT=32 NFRAMES_GRPO=16 bash lite_e2e_benchmark.sh
```

`train_sft.sh`/`train.sh` are idempotent (skip existing `.venv`/`dataset`/`output`). Lite scripts are isolated under `output/lite_*` and `data/lite_e2e/`.

---

## What `train_sft.sh` / `train.sh` do

| Step | Action | Output |
|------|--------|--------|
| 1 | `uv` → `.venv` (PyTorch cu130 + transformers main + peft + trl≥1.8 + liger-kernel + bnb + qwen-vl-utils + gdown + flash-attn) | `.venv/` |
| 2 | Verify `dataset_sft/Train,Validation` and `dataset_grpo/Train,Validation` | — |
| 3 | `data/prepare_sft.py` + `data/prepare_grpo.py` → LLaVA/GRPO JSONs (split-prefixed `Train/...`, unique IDs) | `data/*_dataset_*.json` |
| 4 | If `SUBSET_RATIO<1.0`: seeded shuffle subsample train only | — |
| 5 | **SFT** `60` frames (`131072→262144 px`) `2` epochs `batch4×grad4` → `output/sft_lora/` | `output/sft_lora/` |
| 6 | `src/merge_lora.py --model-path output/sft_lora --model-base Qwen/Qwen3-VL-2B-Instruct` | `output/sft_merged/` (final SFT) |
| 7 | **GRPO** `32` frames `G=5` `max_completion 128` `beta 0.04` → `output/grpo_lora/` (SFT-merged as base) | `output/grpo_lora/` |
| 8 | Merge GRPO LoRA onto SFT-merged | `output/grpo_merged/` (final GRPO) |

`train_sft.sh` stops at 6, `train.sh` runs 1-8. Lite `lite_e2e_benchmark.sh` mirrors 5-8 with sweep + `bench_sft/bench_grpo` logs. No full-video GRPO task.

Every training run is instrumented via `scripts/run_instrumented.sh` into `output/logs/<sft|grpo>/` (console still streams live): `train.log` (full output), `gpu.csv` (`nvidia-smi` util/mem/temp/power `@5s`), `losses.csv` + `eval_losses.csv` (parsed Trainer loss lines), `config.txt` + `cmd.txt` (exact effective config), `merge.log`, `summary.txt` (wall time, loss first/last/best, GPU peak/avg), `start_time`/`end_time`/`exit_code`.

---

## Project Structure

```
qwen3-VL-2B-finetune/
├── train_sft.sh              # SFT pipeline (env + data + train 60 frames) → output/sft_merged
├── train.sh                  # SFT + GRPO (32 frames G=5) → output/grpo_merged
├── lite_sft_test.sh          # SFT smoke (5 samples, 32 frames) → output/lite_sft_test/merged
├── lite_grpo_test.sh         # GRPO probe (30/7, G=4, nframes=8, max_completion=1024, dropout 0.0)
├── lite_e2e_benchmark.sh     # Instrumented lite E2E: VRAM sweep + SFT 6 steps → Merge → GRPO 4 steps G=5 → report.md
├── setup.sh                  # Manual env setup
├── GRPO_ISSUES.md            # GRPO audit (P0-P3) + fix status
├── ISSUES_REPORT.md          # Historical SFT issues
│
├── src/
│   ├── params.py             # TrainingArguments / GRPOArguments (GRPO lora_dropout 0.0, use_liger_loss legacy no-op)
│   ├── constants.py          # IGNORE_INDEX, vision tokens
│   ├── merge_lora.py         # Fuse LoRA adapter into base
│   ├── model/load_model.py   # Qwen3-VL load (AutoModelForImageTextToText)
│   ├── dataset/{sft,grpo}_dataset.py # SFT SupervisedDataset / GRPO GRPODataset (left-padded prompts, mm_token_type_ids)
│   ├── dataset/data_utils.py # Video probe (caps nframes), qwen_vl_utils process_vision_info
│   ├── trainer/{sft,grpo}_trainer.py # SFT + QwenGRPOTrainer (one-update, per-token clip, k3 KL, EOS-aware mask, zero-std→0)
│   └── train/{train_sft,train_grpo,reward_funcs}.py # Entrypoints + deterministic rewards (strict JSON, allow_fallback=False)
│
├── scripts/
│   ├── build_lite_benchmark_data.py  # Build balanced lite subsets (all subgroups, --grpo-train-samples 30)
│   ├── verify_qwen_logit_alignment.py # P0-2 logit alignment check (text + video greedy)
│   ├── finetune_{sft,grpo}_lora.sh / merge_lora.sh / zero*.json
│
├── configs/                  # Reference YAMLs (not auto-read, scripts are source of truth)
├── data/{prepare_sft,prepare_grpo,dataset_stats}.py
├── eval/compute_metrics.py
├── check_lora_weights.py     # Verifies lora_B 300/300 non-zero
├── dataset_sft/  dataset_grpo/ # Separated datasets (Train/Validation/Test, hardlinked videos)
├── output/{sft_lora,sft_merged,grpo_lora,grpo_merged,lite_benchmark,lite_sft_test,lite_grpo_test}/
└── README.md / overview.md / config_setup.md
```

---

## Configuration (env vars for `train_sft.sh` / `train.sh` / `lite_*`)

Defaults for 48 GB single GPU; override via env vars. Scripts are source of truth (YAMLs not read).

### Model
| Var | Default | Description |
|-----|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen3-VL-2B-Instruct` | Base model (HF ID or local) |
| `BITS` | 4 | Quant: 4/8 (QLoRA, `bnb_4bit_compute_dtype=bf16`) /16 (LoRA) |

### LoRA
| Var | Default | Description |
|-----|---------|-------------|
| `LORA_RANK` | 16 | Rank (alpha 2× rank) |
| `LORA_ALPHA` | 32 | Alpha |
| `LORA_DROPOUT` | `0.05` SFT / `0.0` GRPO | GRPO `0.0` for deterministic old/current logprobs (`GRPO_ISSUES.md P1-1`); `GRPOArguments.lora_dropout=0.0` |

### Training
| Var | Default | Description |
|-----|---------|-------------|
| `BATCH_PER_DEVICE` | 4 | Micro batch |
| `GRAD_ACCUM` | 4 | Grad accum → global 16 |
| `NUM_DEVICES` | 1 | GPUs |
| `NUM_EPOCHS` | `2` SFT / `1` GRPO | |
| `LR` | 1e-4 | LLM LoRA LR |
| `VISION_LR` | 2e-6 | Vision LoRA LR |
| `MERGER_LR` | 1e-5 | Merger LR |
| `WEIGHT_DECAY` | 0.1 | SFT (`0.0` GRPO) |
| `WARMUP_STEPS` | 10 | |
| `LR_SCHEDULER` | `cosine` SFT / `constant` GRPO | |
| `BETA` | 0.04 | GRPO KL coeff (`beta=0` disables ref) |
| `NUM_GENERATIONS` | `5` prod / `4` lite | `G` per prompt |
| `MAX_COMPLETION_LENGTH` | `128` bench / `1024` `lite_grpo_test.sh` | Decode cost linear in `G*len` |
| `TEMPERATURE` | 0.9 | GRPO sampling |

### Video
| Var | Default | Description |
|-----|---------|-------------|
| `NFRAMES` | `60` SFT / `32` GRPO prod (`8` lite) | Max frames (auto-capped to `probe_total_frames` in `data_utils.py:195`) |
| `FPS` | — | Alt to `NFRAMES` (mutually exclusive) |
| `VIDEO_MIN_PIXELS` | `128×32×32`=131072 | Min res |
| `VIDEO_MAX_PIXELS` | `256×32×32`=262144 | Max res (sweep `131072→262144`) |
| `MAX_SEQ_LENGTH` | 8192 SFT / `prompt+completion` GRPO | SFT truncates at 8192 (60/262144 ~15.7k > limit) |

### Eval / save / precision
| Var | Default | Description |
|-----|---------|-------------|
| `EVAL_STRATEGY` | `steps` | SFT `steps 300`, GRPO `no` |
| `SAVE_STRATEGY` | `steps` | `300` SFT, `steps` GRPO lite `save_steps=STEPS` |
| `BF16` | `True` | `fp16 False tf32 True` |
| `USE_LIGER_KERNEL` | `True` SFT / `False` GRPO | GRPO uses custom manual token-mean loss (not Liger, `GRPO_ISSUES.md P2-1`) |
| `GRADIENT_CHECKPOINTING` | `True` | `use_reentrant False` with `vision_lora` |

### Dataset roots
| Var | Default | Description |
|-----|---------|-------------|
| `SFT_DATASET_ROOT` | `dataset_sft` | Separated SFT dataset root |
| `GRPO_DATASET_ROOT` | `dataset_grpo` | Separated GRPO dataset root |

### Misc
| Var | Default | Description |
|-----|---------|-------------|
| `SUBSET_RATIO` | 1.0 | Use only this fraction of training data (0.0–1.0). Eval always uses full set. |
| `DISABLE_FLASH_ATTN2` | 0 | Set 1 to use SDPA instead of flash-attn (if flash-attn install fails) |
| `INSTALL_FLASH_ATTN` | 1 | Set 0 to skip flash-attn install |
| `ENABLE_GEN_EVAL` | 1 | Use generation-based eval metrics (sets `SFT_COMPUTE_METRICS=eval/compute_metrics.py`) |
| `FORCE_REPREPARE` | 0 | Set 1 to regenerate prepared JSONs even if they exist |
| `HF_TOKEN` | — | Required if downloading from private/gated models |

---

## Dataset (separated, hardlinked videos)

```
dataset_sft/  dataset_grpo/  (each Train/Validation/Test, 286 folders: 108 YT +105 PH Train, 13/22 Val)
├── <YT_ID>/  clip_*.mp4 + clip_*_{sft,grpo}.jsonl (4 SFT / 3 GRPO per clip), full_video.mp4 + full_video_sft.jsonl (SFT only)
└── PH_*/     clip_*.mp4 (1 SFT description), grpo_*.mp4 (1 GRPO record: boundary/temporal/timestamp/contextual)
```

Counts: `SFT 7663 train /954 val` (2,174 videos: 2,066 clip +108 full), `GRPO 4252 train /573 val` (1,969 unique GRPO videos, 100% deterministic). See `overview.md:3.3`.

Prep: `data/prepare_sft.py` + `data/prepare_grpo.py` → LLaVA `{id,video,conversations}` / GRPO `{id,video,conversations,correct_answer,question_type,reference_reasoning,reward_type}`. Paths prefixed `Train/...`, IDs `video_id_file_stem_line`. Warns if YT `≠4 SFT/3 GRPO` or PH `≠1`.

### Dataset format
- **SFT** LLaVA: `{id, video, conversations: [{from:"human"|"gpt", value:"<video>…"}]}` (input_ids includes prompt+response, labels mask prompt)
- **GRPO** strict JSON: `{explanation: "1-3 sent", answer: "A|B|C|D" | {timestamp} | {start,end} | "P0X"}`. Rewards `R= R_task +0.05*R_fmt` (see `overview.md:3.6`)
- Video `nframes` capped to actual frames (`probe_total_frames`); durations `GRPO 1-210s avg 25.9s`, `SFT 3-364s avg 38s`
- Lite balanced subsets: `scripts/build_lite_benchmark_data.py` (`data/lite_e2e/` 10/5 SFT +14/7 GRPO, `output/lite_grpo_test/` 30/7 for `lite_grpo_test.sh`)

---

## Stage-by-stage Manual Run

### Environment
```bash
bash setup.sh
source .venv/bin/activate
export PYTHONPATH=src:${PYTHONPATH:-} HF_HOME=$PWD/hf_cache TOKENIZERS_PARALLELISM=false
```

### Data prep
```bash
python data/prepare_sft.py --input-dir dataset_sft/Train --output data/sft_train_dataset_sft.json --data-type all
python data/prepare_sft.py --input-dir dataset_sft/Validation --output data/sft_val_dataset_sft.json --data-type all
python data/prepare_grpo.py --input-dir dataset_grpo/Train --output data/grpo_train_dataset_grpo.json --data-type all
python data/prepare_grpo.py --input-dir dataset_grpo/Validation --output data/grpo_val_dataset_grpo.json --data-type all
# Lite balanced:
python scripts/build_lite_benchmark_data.py  # → data/lite_e2e/ (10/5 SFT, 14/7 GRPO)
python scripts/build_lite_benchmark_data.py --grpo-train-samples 30  # GRPO 30/7
```

### SFT stage + merge
```bash
bash scripts/finetune_sft_lora.sh  # → output/sft_lora/
.venv/bin/python src/merge_lora.py --model-path output/sft_lora --model-base Qwen/Qwen3-VL-2B-Instruct --save-model-path output/sft_merged --safe-serialization
```

### GRPO stage (SFT-merged base, strict JSON, one-update)
```bash
# manual finetune (GRPO G=5, 32 frames):
DATA_PATH=data/grpo_train_dataset_grpo.json EVAL_PATH=data/grpo_val_dataset_grpo.json IMAGE_FOLDER=dataset_grpo \
  bash scripts/finetune_grpo_lora.sh  # → output/grpo_lora/
.venv/bin/python src/merge_lora.py --model-path output/grpo_lora --model-base output/sft_merged --save-model-path output/grpo_merged --safe-serialization

# verify logit alignment (P0-2):
HF_HOME=hf_cache .venv/bin/python scripts/verify_qwen_logit_alignment.py --bits 4
```

Or `bash lite_e2e_benchmark.sh` / `bash lite_grpo_test.sh` for instrumented lite runs.

---

## LoRA Architecture (`vision_lora True`, `freeze_vision_tower True`, `freeze_llm True`, `freeze_merger False`, `bits 4`)

Adapters on **301** modules (excluded `embed_tokens`, `lm_head` via `lora_namespan_exclude`):

| Component | # | Modules |
|---|---:|---|
| LLM 28 layers | 196 | `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj` ×28 |
| Vision 24 blocks | 96 | `attn.{qkv,proj}`, `mlp.{linear_fc1,linear_fc2}` ×24 |
| Merger | 2 | `merger.{linear_fc1,linear_fc2}` |
| Deepstack merger | 6 | `0/1/2.{linear_fc1,linear_fc2}` |
| Pos embed | 1 | `visual.pos_embed` |

Base frozen, `LoRA` on LLM `q/k/v/o`+`gate/up/down` + vision transformer linears (`292` modules), `merger` full-trainable (`1e-5`), `pos_embed`/`embed_tokens`/`lm_head` frozen. QLoRA `r=16 alpha=32` (~`10-15M` params at `r8` → `~20-30M` at `r16`). Verify: `check_lora_weights.py` (`lora_B ~292/292 non-zero`).

Norm kept `float32`, `lm_head`/`embed_tokens` `float32` to match (fixes `BFloat16 vs Float` at `lm_head` when `norm float32`). `prepare_model_for_kbit_training` + `autocast(bf16)` wraps GRPO `generate`/logprob forwards.

---

## Known Issues → Fixes (full: `ISSUES_REPORT.md` (SFT) + `GRPO_ISSUES.md` (GRPO audit P0-P3))

Production baseline: `r16 α32` LLM `q/k/v/o`+MLP + vision transformer linears, `merger` full-trainable, `pos_embed` frozen, `lora_dropout 0.05→0.0` GRPO. SFT fixes: **C1** path prefix, **C3** unique IDs, **C4** `lora_bias` dict, **H2** `bits` branching, **H4** `SFT_COMPUTE_METRICS`, **M4** `use_dora`, frame probe, TRL 1.8, DeepSpeed optional.

GRPO fixes (audit `GRPO_ISSUES.md`):
- **P0-1** zero-std `advantage=0` (was `rewards-0.5` absolute PG) + `zero_std_group_fraction` log
- **P0-2** logit alignment verified (`logits[:,prompt_len-1:-1]` → `completion_ids[:,prompt_len:]`) + `scripts/verify_qwen_logit_alignment.py`
- **P1-1** `lora_dropout 0.0` GRPO (SFT `0.05`) for deterministic old/current
- **P1-2** one-update GRPO documented (`ratio≈1`, `grpo_loss≈0` expected; focus `reward/KL/total_loss`)
- **P1-3** diagnostics `reward_min/max`, `fraction_reward_zero/one`, `zero_std_group_fraction`
- **P2-1/2** Liger flags `use_liger_kernel False` for GRPO (custom manual token-mean loss, not Liger)
- **P2-3** token-mean normalization (global `sum(loss*mask)/sum(mask)`, not "DAPO" alias)
- **P2-4** strict JSON-only rewards (`allow_fallback=False`, regex gated)
- **P2-5** EOS-aware `completion_mask` (`(cumsum==0)|(cumsum==1 & is_eos)`)
- **P3-1** strict `question_type` dispatcher
- **Dtype** `lm_head`/`embed_tokens` `float32` + `autocast(bf16)` wraps GRPO `generate`/logprobs (fixes `BFloat16 vs Float` at `lm_head`)

Rewards: MCQ exact letter, `boundary` `exp(-|Δt|/1.5)`, `temporal` `tIoU`, `phase` exact `P0X`, composite `R_task+0.05*R_fmt`.

---

## VRAM Usage & Tokens (measured RTX 4060 8GB, QLoRA `r16` `bf16`; `r32` +9 modules ≈ +0.5GB)

| Config | VRAM peak | Tokens/sample worst | Notes |
|--------|-----------|---------------------|-------|
| `BITS=4, NFRAMES=8, 131072, batch1 rank32` | `SFT 3056MiB` / `GRPO G=4 7942MiB` | SFT `~1.1-1.4k` / GRPO `~4.5k`/gen (`~22k`/step G=4) | `lite_grpo_test.sh` stable |
| `BITS=4, NFRAMES=16, 131072, batch1` | `SFT 3056MiB` / `GRPO 7914MiB` | SFT `~2.2k` / GRPO `~3.1k`/gen |  |
| `BITS=4, NFRAMES=16, 262144, batch1` | `SFT 3962MiB` / `GRPO OOM 7924MiB` | `~4.6k` / OOM | wall `G=4` |
| `BITS=4, NFRAMES=32, 131072, batch1 rank32` | `SFT ok` (`bench`) | `~4.3k` | bench SFT 6 steps |
| `BITS=4, NFRAMES=60, 262144, batch4 rank32` | `~10-15GB` est (`config_setup.md` 24/32GB) | `15.7k` > `8192` truncated → keep `131072` (~7.6k) | 48GB default |
| `BITS=4, NFRAMES=60, 131072, batch4` | `~10-12GB` | `~7.6k` fits 8192 | full SFT `58M` tok/epoch, `117M` ×2ep |

Per frame `~121 tok (131072) / 256 tok (262144)` (`ceil(H/32)*ceil(W/32)`). Text `~300` prompt + `100` response / `128-1024` completion. GRPO step `G*(prompt+completion)` fwd. Sweep `output/lite_benchmark/vram_sweep.csv`, logs `bench_*/gpu.csv` `@0.5s` + `time -v`, report `output/lite_benchmark/report.md`.

Token est for `60` SFT / `32` GRPO: SFT `7663*7.6k≈58M`/ep, GRPO `4252*5*4.6k≈98M` fwd `5.4M` loss (`256` avg) per epoch; `1024` max → `114M` fwd / `21M` loss.

---

## Testing (lite / smoke on 8 GB)

```bash
bash lite_sft_test.sh   # 5 samples SFT 32 frames → output/lite_sft_test/merged (auto-merge), ~2GB
GRPO_TRAIN_SAMPLES=30 bash lite_grpo_test.sh  # 30/7 GRPO G=4 nframes=8 max_completion=1024 (~9min, 7.9GB peak)
bash lite_e2e_benchmark.sh  # full lite E2E + VRAM sweep → output/lite_benchmark/report.md (GPU-hours est)
bash lite_sft_test.sh       # 1-video SFT smoke when running stages individually
HF_HOME=hf_cache .venv/bin/python scripts/verify_qwen_logit_alignment.py --bits 4  # P0-2 check
```

Verify: `.venv/bin/python check_lora_weights.py output/lite_sft_test/output` / `output/lite_benchmark/sft_lora` (`lora_B 300/300 non-zero`). Logs `output/lite_*/output/train.log` + `gpu.csv` + `time.log`.

---

## File Locations Summary

| Output | Path |
|--------|------|
| SFT adapter (clips + full videos) | `output/sft_lora/` |
| **Final SFT model** | `output/sft_merged/` |
| GRPO adapter (on SFT-merged) | `output/grpo_lora/` |
| **Final GRPO model** | `output/grpo_merged/` |
| Lite SFT smoke | `output/lite_sft_test/output` + `merged/` |
| Lite GRPO probe (30/7) | `output/lite_grpo_test/output` (`grpo_train.json` isolated) |
| Lite E2E bench | `output/lite_benchmark/{sft_lora,sft_merged,grpo_lora,grpo_merged,bench_*,vram_sweep.csv,report.md}` |
| Prepared JSONs | `data/sft_train_dataset_sft.json` etc (gitignored) / `data/lite_e2e/` balanced |
| GRPO video max | `210s` clip `eAIZjIKBK_c/clip_09.mp4`, SFT full `364s`, avg `GRPO 25.9s` / `SFT 38s` |
| Max len GRPO videos | `210s` (see `ffprobe` scan) |

---

## License

[Add your license here]
