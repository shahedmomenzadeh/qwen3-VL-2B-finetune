# Issues Report — Qwen3VL-Lora-SFT-GRPO

**Generated:** 2026-07-21  
**Project:** `qwen3vl_2b_finetune` (Qwen3-VL-2B LoRA SFT + GRPO for cataract surgery video understanding)  
**Reference repo:** `Qwen-VL-Series-Finetune`

---

## Summary

| Severity | Count |
|----------|-------|
| Critical | 5 |
| High | 5 |
| Medium | 14 |
| Low | 11 |

---

## Critical Issues

### C1 — `IMAGE_FOLDER` path resolution broken for split-dir structure

- **Files:** `src/dataset/sft_dataset.py:121-131`, `src/dataset/grpo_dataset.py:112-122`, `scripts/finetune_sft_lora.sh:21`, `scripts/finetune_grpo_lora.sh:13`
- **Evidence:** JSONs store paths like `YT_ID/clip_01.mp4` (relative to split dir, e.g., `dataset/Train/`). Dataset class resolves via `os.path.join(video_folder, video_file)` with `video_folder = self.data_args.image_folder`. The scripts default `IMAGE_FOLDER=dataset`, giving `dataset/YT_ID/clip_01.mp4` — **does not exist**. The correct path is `dataset/Train/YT_ID/clip_01.mp4`.
- **Impact:** **Training cannot start.** Every video path lookup returns `FileNotFoundError`.
- **Root cause:** `prepare_sft.py` / `prepare_grpo.py` output paths relative to the `--input-dir` split (e.g., `dataset/Train`), but the dataset class resolves against `image_folder` which must be the same split dir. Since train and eval data are in different splits (`Train` vs `Validation`) but share a single `image_folder` parameter, no single value works for both.
- **Workaround for `train.sh`:** Create a flat directory of symlinks (see `train.sh`).
- **Proper fix:** Either (a) use the `eval_image_folder` field in `DataArguments` (defined at `params.py:110` but unused), or (b) include the split prefix in prepared JSON paths (e.g., `Train/YT_ID/clip_01.mp4` with `image_folder=dataset`).

### C2 — `deterministic_reward` regex is case-broken; degrades to substring match

- **File:** `src/train/reward_funcs.py:14-46`
- **Evidence:** Lines 23–27 use patterns `[A-D]` (uppercase only) but search `completion_lower` (lowercased text) at line 31 with no `re.IGNORECASE`. All primary patterns are **dead code**. The function falls through to line 39: `if answer.lower() in completion_lower` — a bare substring match.
- **Impact:** **GRPO rewards are wrong.** For single-letter answers (`A`/`B`/`C`/`D`), ANY occurrence of the answer letter anywhere in the completion gives score 1.0 (e.g., completion `"This film shows BAD technique"` with correct answer `"A"` would score 1.0 because `"a" in "this film shows bad technique"`). For sequence-ordering answers (up to 10 letters A–J), correct extraction cannot work at all through the pattern path.
- **Fix:** Add `flags=re.IGNORECASE` to `re.search()`. For sequence ordering, extend pattern to `[A-J]`.

### C3 — Duplicate sample IDs in prepared JSONs (clip-type files only)

- **Files:** `data/prepare_sft.py:99`, `data/prepare_grpo.py:79`
- **Evidence:** `sample_id = f"{video_id_dir}_{line_idx}"` uses `line_idx` which restarts from 0 for each clip file in the same video directory. Result: `sft_train_clip.json` has 4564 samples but only 434 unique IDs; `grpo_train_clip.json` has 3424 samples / 325 unique IDs.
- **Impact:** IDs are non-unique, making data files not keyable and preventing per-sample tracking. If any downstream code assumed unique IDs, behavior would be undefined.
- **Fix:** Include clip filename in ID: `sample_id = f"{video_id_dir}_{os.path.splitext(basename)[0]}_{line_idx}"`

### C4 — `lora_bias="lora_only"` dict-iteration bug (iterates keys, not items)

- **File:** `src/train/train_utils.py:41-43`
- **Evidence:** `for k, t in maybe_lora_bias:` — iterating a dict without `.items()` yields **keys** (strings), not `(key, value)` tuples. `k` gets a param name string, `t` gets its first character. Additionally, the membership check at line 42 uses stale loop variable `bias_name` from line 37 instead of current `k`.
- **Impact:** When `lora_bias="lora_only"`, bias parameters for LoRA target modules are **silently excluded** from the saved state dict. This is a data-loss bug. (Current default is `lora_bias="none"`, so this path is not triggered by default — but if enabled, it would silently lose weights.)
- **Fix:** `for k, t in maybe_lora_bias.items():` and fix the membership check variable.

### C5 — GRPO dtype-cast loop runs before PEFT is applied (dead code)

- **File:** `src/train/train_grpo.py:186-198`
- **Evidence:** The loop checks `isinstance(module, LoraLayer)` but runs BEFORE `QwenGRPOTrainer.__init__` applies PEFT. At this point, no `LoraLayer` modules exist in the model — the entire loop is a no-op. (This code was copied from `train_sft.py` where it runs AFTER `get_peft_model` and works correctly.)
- **Impact:** **LoraLayer casting to `bf16` never happens in GRPO.** When using BnB quantization, LoRA parameters may remain in `float32` while the rest of the model runs in `bf16`, potentially causing dtype mismatches.
- **Fix:** Move the dtype-casting logic into `grpo_trainer.py` after PEFT is applied, or rely on TRL's built-in dtype handling.

---

## High Issues

### H1 — `non_lora_state_dict.bin` never restored on training resume

- **Files:** `src/train/train_sft.py:234-237`, `src/train/train_grpo.py:218-221`, `src/trainer/sft_trainer.py:~164`, `src/trainer/grpo_trainer.py:~955`
- **Evidence:** Checkpoints save `non_lora_state_dict.bin` (vision tower, merger, embedding weights). On resume, `trainer.train(resume_from_checkpoint=True)` loads only `pytorch_model.bin` / `adapter_model.bin` — it never loads `non_lora_state_dict.bin`.
- **Impact:** After resume, non-LoRA trainable parameters **revert to initialization values**, silently undoing training progress on those parameters. Affects both SFT and GRPO training if `freeze_vision_tower=False` or `freeze_merger=False`.

### H2 — Vision tower `.to(dtype, device)` called on BnB-quantized modules

- **File:** `src/train/train_sft.py:53`, `src/train/train_grpo.py:53`
- **Evidence:** `configure_vision_tower()` calls `vision_tower.to(dtype=compute_dtype, device=device)` at line 53. When model is loaded with BnB 4-bit quantization, vision tower `Linear` layers are wrapped in `bnb.nn.Params4bit`. Calling `.to()` on quantized modules can corrupt quantization state. Additionally, `model.to(device=device)` is redundant with `device_map`.
- **Impact:** Risk of silent vision quality degradation or crashes when running with BITS=4 (the script default).

### H3 — `llm_judge_reward` is keyword-overlap heuristic, not an LLM judge

- **File:** `src/train/reward_funcs.py:49-85`
- **Evidence:** Despite the name, this function: (1) splits reference reasoning by `'.'` into phrases, (2) checks each phrase's first 30 characters appear anywhere in the completion (substring match), (3) adds 0.3 bonus for reasoning keyword presence. **No teacher LLM is called.** A completion could copy fragments and score 1.0 without actual reasoning.
- **Impact:** 33% of GRPO training samples (those with `reward_type="llm_judge"`) are rewarded by an inadequate heuristic. The dataset README explicitly describes a teacher-LLM judge pipeline.

### H4 — SFT generation-based eval never activated (SFT_COMPUTE_METRICS not set)

- **File:** `scripts/finetune_sft_lora.sh:108-113`, `src/train/train_sft.py:214-224`
- **Evidence:** The script passes `--prediction_loss_only False` and `--eval_strategy steps`, but never sets `SFT_COMPUTE_METRICS` env var. The `compute_metrics` loading in `train_sft.py` only triggers when this env var is set. Without it, the `QwenSFTTrainer.evaluation_loop` returns loss only — **all generation-based metrics (exact_match, contains_match, reasoning_rate, avg_gen_length) are silently skipped.**
- **Impact:** SFT eval only produces loss, no quality metrics. Generation code path is untested.

### H5 — GRPO eval dataset unreachable (hardcoded to None)

- **File:** `src/dataset/grpo_dataset.py:201`
- **Evidence:** `make_grpo_data_module` calls `GRPODataset(...)` for training but sets `eval_dataset=None`. The GRPO val JSONs (`grpo_val_clip.json`, `grpo_val_video.json`) exist and contain 411+13 samples but are **never loaded**.
- **Impact:** No validation during GRPO training. Reward drift / overfitting cannot be monitored.

---

## Medium Issues

### M1 — `transformers>=4.45.0` floor too low; Qwen3-VL needs >=4.57.0

- **File:** `pyproject.toml:15`
- **Evidence:** Current `uv.lock` resolves to `transformers==4.57.6` (works), but the constraint `>=4.45.0` would allow older versions that lack Qwen3-VL support. If the lock file is regenerated on an older snapshot of PyPI, `transformers-4.45.x` could be installed and the training would fail with `Unknown model type: qwen3_vl`.
- **Fix:** Bump to `"transformers>=4.57.0"`.

### M2 — BnB 4-bit: no `bnb_4bit_quant_storage` or `modules_to_not_convert` for vision tower

- **File:** `src/train/train_sft.py:124-133`, `src/train/train_grpo.py:124-133`
- **Evidence:** `BitsAndBytesConfig` sets `llm_int8_skip_modules=["visual","lm_head"]` — this only takes effect for 8-bit. For 4-bit, there is no `bnb_4bit_quant_storage` set and no vision tower exclusion. The vision tower's `Linear` layers get quantized to 4-bit, potentially degrading visual feature quality. Scripts default `BITS=4`.
- **Fix:** Set `bnb_4bit_quant_storage` and/or use `modules_to_not_convert` for 4-bit quantization.

### M3 — `max_prompt_length` does not count video tokens (text-only truncation)

- **File:** `src/trainer/grpo_trainer.py:163-164`
- **Evidence:** The processor call sets `"max_length": self.max_prompt_length` and `"truncation": True`, but this only affects text tokens. Video tokens are added AFTER truncation by the processor and are not counted against the limit. A 60-frame clip at ≥128 tokens/frame = ≥7,680 prompt tokens, far exceeding the default `max_prompt_length=1024`.
- **Impact:** Videos silently produce prompts many times longer than the model's context. Truncation may cut off the question text or assistant header. Combined with `right` truncation, critical tokens at the end of prompts can be lost.

### M4 — `use_dora` parsed but never passed to `LoraConfig`

- **File:** `src/train/train_sft.py:168-175`, `src/train/train_grpo.py:169`
- **Evidence:** `TrainingArguments.use_dora` field exists (params.py:46, scripts pass `--use_dora False`) but the `LoraConfig()` constructor call does not include `use_dora=training_args.use_dora`. DoRA cannot be enabled even if desired.
- **Fix:** Add `use_dora=training_args.use_dora` to `LoraConfig()`.

### M5 — `nn.Embedding` included in LoRA target module discovery

- **File:** `src/train/train_sft.py:29,35`, `src/train/train_grpo.py:29,35`
- **Evidence:** `find_target_linear_names` collects both `torch.nn.Linear` AND `torch.nn.Embedding` module names. If `lora_namespan_exclude` doesn't exclude `embed_tokens`, the embedding layer gets LoRA adapters — unusual and wasteful for learned embeddings.

### M6 — `format_reward` case inconsistency

- **File:** `src/train/reward_funcs.py:88-105`
- **Evidence:** `has_reasoning` searches `completion_lower` (lowercase) at lines 95-98. `has_answer` searches `completion` (original case) at line 101 with pattern `[A-D]`. If model outputs lowercase letters in the answer section, `has_answer` becomes `False` and the function returns 0.0 instead of 0.5.

### M7 — `reward_type` and `question_type` columns unused by reward functions

- **File:** `src/train/reward_funcs.py` (all functions accept `**kwargs`)
- **Evidence:** The dataset prepares `reward_type` and `question_type` fields (passed through `GRPODataset` and TRL kwarg injection), but all three reward functions ignore them. The `reward_type` field (deterministic/llm_judge) does not gate which reward applies — instead ALL reward functions run on EVERY sample and are summed.

### M8 — Sequence ordering answers (letters E–J) not extractable

- **File:** `src/train/reward_funcs.py:23-27`
- **Evidence:** Patterns use `[A-D]` which only matches letters A–D. The `grpo_train_video.json` contains sequence-ordering questions with answers like `"F, H, I, B, D, C, J, A, E, G"` using letters A–J. Even if the case bug (C2) is fixed, letters E–J cannot be extracted.

### M9 — `eval/compute_metrics.py` regex inconsistent with `reward_funcs.py`

- **File:** `eval/compute_metrics.py:32-38`
- **Evidence:** `extract_answer()` uses `re.search(r'(answer\s+is|Answer\s*:)\s*([A-D])', text)` — the pattern `Answer\s*:` matches capital `Answer` only, not lowercase `answer:`. Does not handle sequence ordering. No `IGNORECASE`. Inconsistent with `extract_answer_from_generation` (line 71) which uses `re.IGNORECASE`.

### M10 — `merge_lora.py` may not load GRPO checkpoints correctly

- **File:** `src/merge_lora.py:5-18`
- **Evidence:** `is_lora_model()` checks for `adapter_config.json` + `adapter_model.safetensors`. GRPO checkpoints save the full model via `non_lora_state_dict.bin` with `require_grad_only=False`. If a GRPO checkpoint lacks `adapter_config.json`, it falls to the `else` branch which loads it as a standard model — but does NOT load `non_lora_state_dict.bin`. The merged model would be missing non-LoRA trainable weights.

### M11 — `configs/*.yaml` not read by any code; defaults drifted from scripts

- **Files:** `configs/sft_config.yaml`, `configs/grpo_config.yaml`
- **Evidence:** No Python code or shell script reads the YAML configs. The shell scripts are the real configuration source. Meanwhile the YAMLs say `bits: 8` while scripts default `BITS=4`; YAMLs say `fps: 1` while scripts default `NFRAMES=60`.
- **Impact:** Confusing for users who may edit the YAMLs expecting changes to take effect.

### M12 — `second_per_grid_ts` accepted but dropped in qwen3_vl forward patch

- **File:** `src/train/monkey_patch_forward.py:351, 420-430`
- **Evidence:** `qwen3_vl_mixed_modality_forward` accepts `second_per_grid_ts` in its signature (line 351) but never passes it to `compute_3d_position_ids()` (line 420-430). Compare with `qwen2_5_mixed_modality_forward` which does pass it. Same bug in `qwen3_vl_moe_mixed_modality_forward`.
- **Impact:** Temporal-aware 3D RoPE for videos is computed without per-video timing information. Inherited from the reference repo.
- **Note:** `second_per_grid_ts` is never populated in the SFT dataset anyway (M13 below), so this may be latent.

### M13 — `all_second_gird` dead code in SFT dataset

- **File:** `src/dataset/sft_dataset.py:159, 282-284`
- **Evidence:** The variable `all_second_gird` (typo: "gird") is initialized at line 159 but never appended to. The reference populates it in the `qwen2_5_vl` branch which was removed from the user's code. The `data_dict` never contains `"second_per_grid_ts"`.

### M14 — Chat template hand-built, not using `processor.apply_chat_template`

- **File:** `src/dataset/sft_dataset.py:191-195`
- **Evidence:** The chat template is constructed by string formatting with raw `<|im_start|>/<|im_end|>` tokens. For Qwen3-VL, this currently matches the expected format, but any future tokenizer/chat-template change would silently break it with no safeguard.

---

## Low Issues

### L1 — `TrainingArguments.head_lr` defined but unused in SFT

- **File:** `src/params.py:54`
- **Evidence:** `head_lr` only makes sense for classification models (used by `CLSArguments` in the reference), not for SFT. Present but silently ignored.

### L2 — `GRPOArguments` redefines TRL-internal fields without help strings

- **File:** `src/params.py:94-103`
- **Evidence:** Fields like `beta`, `temperature`, `top_p` override TRL's `GRPOConfig` defaults. Harmless now, but if TRL updates these fields, the subclass may silently shadow them.

### L3 — Vision tower `freeze_vision_tower` guard removed

- **File:** `src/train/train_sft.py:103-113`, `src/train/train_grpo.py:103-113`
- **Evidence:** Reference has `if training_args.vision_lora and not training_args.freeze_vision_tower: raise ValueError(...)`. User version removed this guard, now allowing both `vision_lora=True` AND `freeze_vision_tower=False`. The script does set both to those values, so this is intentional, but removes a safety check.

### L4 — `lora_namespan_exclude` parsing via `ast.literal_eval` has no error handling

- **File:** `src/train/train_sft.py:108`, `src/train/train_grpo.py:108`
- **Evidence:** If the string is malformed, `ast.literal_eval` raises ValueError/SyntaxError with no friendly message. Currently works because the script's value `"['lm_head','embed_tokens']"` is well-formed.

### L5 — Auto-resume triggers on any `checkpoint-*` directory

- **File:** `src/train/train_sft.py:234`
- **Evidence:** Simple `os.path.exists` check on any `checkpoint-*` in `output_dir`. No validation of checkpoint integrity. Could auto-resume from a broken/incompatible checkpoint.

### L6 — `--split` argument parsed but unused in prepare scripts

- **File:** `data/prepare_sft.py:119`, `data/prepare_grpo.py:102`
- **Evidence:** Both scripts accept `--split` but never reference `args.split`. The actual split is determined by `--input-dir`.

### L7 — `dataset_stats.py` minor issues

- **File:** `data/dataset_stats.py:26, 89`
- **Evidence:** `clip_durations` initialized but never used (dead code). Default `--dataset-dir` is relative to CWD.

### L8 — `param_group_name` typo: `"visaul"` instead of `"visual"`

- **File:** `src/trainer/grpo_trainer.py:885, 891`
- **Evidence:** Optimizer param group names have typo `"visaul_decay"`, `"visaul_non_decay"`. Cosmetic only, doesn't affect training.

### L9 — GRPO checkpoint `require_grad_only=False` saves entire base model

- **File:** `src/trainer/grpo_trainer.py:955`
- **Evidence:** At every checkpoint, ALL non-LoRA parameters (including frozen base model weights, ~2B params) are saved to `non_lora_state_dict.bin`. This can be gigabytes per checkpoint. The SFT path uses `require_grad_only=True` which is more space-efficient.

### L10 — Dead/unused pieces scattered across codebase

- **Evidence:**
  - `lazy_preprocess` flag accepted but always eager (datasets process in `__getitem__`)
  - `max_seq_length` (params.py:38, default 32768) never enforced / no truncation in dataset
  - `parse_video_path()` in `prepare_sft.py:30-31` is a no-op identity function
  - `padding` parameter in dataset classes never used
  - `monkey_patch_vision.py` only used for `qwen2_5_vl` (dead path for qwen3_vl training)

### L11 — `eval_image_folder` defined but never used

- **File:** `src/params.py:110`
- **Evidence:** `DataArguments.eval_image_folder` is defined but never referenced in `sft_dataset.py` or `grpo_dataset.py`. If implemented, it would solve the path resolution issue (C1) for eval.

---

## Verified OK

The following suspected issues were checked and found to be correct:

- **No train/validation split leakage** — Train and Validation share zero parent video IDs.
- **Video files all exist** under correct split directories (4564 train clips under `dataset/Train/`, 548 val clips under `dataset/Validation/`).
- **`_PATCHERS` correctly maps `qwen3_vl`** to only `replace_qwen3_with_mixed_modality_forward` (no vision patch — Qwen3-VL has different vision architecture).
- **`_GENERATION_MODEL_TYPES`** includes all needed Qwen3-VL variants.
- **Video metadata handling for qwen3_vl** (`return_video_metadata=True` + tuple unpacking) is correct in both datasets.
- **Patch size 16 for qwen3_vl** is correct in `data_utils.py`.
- **Label masking boundaries** correctly mask `--f<|im_start|>assistant\n` in the prompt section.
- **Post-PEFT re-unfreeze** logic in `train_sft.py:184-192` correctly re-enables training for visual/merger.
- **Optimizer param groups** (`vision_lr`, `merger_lr`) correctly constructed in both trainers.
- **Liger GRPO loss integration** correctly wired in `grpo_trainer.py`.
- **MULTIMODAL_KEYWORDS** in `constants.py` includes all Qwen3-VL-specific fields (`mm_token_type_ids`, `second_per_grid_ts`).
- **`uv.lock` resolves** to `transformers==4.57.6` (supports Qwen3-VL) despite loose constraint.

---

## Runtime Test Results (CPU)

Three categories of tests were run:

| Test | Result | Notes |
|------|--------|-------|
| `compileall` (syntax check) | **PASS** | All `.py` files in `src/` and `data/` compile without errors |
| `deterministic_reward` unit tests | **CONFIRMED BUGS** | See C2 — false positives: `"The answer is B"` with correct answer `"A"` scores 1.0 because `"a" in "the answer is b"` (the word "answer" contains "a"). `"This film shows BAD technique"` with answer `"A"` scores 1.0. Comma-separated answers partially work through fallback. |
| `format_reward` unit tests | **CONFIRMED BUG** | See M6 — lowercase answer letters cause `format_reward` to return 0.0 |
| Full import tests | **SKIPPED** | Torch fails to import on this WSL host (CUDA 13.2 driver vs. PyTorch CUDA 12.4 wheels + WSL `/mnt/d` cross-drive). Import will work on the GPU server. |
| Dataset dry-run | **SKIPPED** | Blocked by torch import failure |

## System Info

| Component | Value |
|-----------|-------|
| Python (system) | 3.14.4 |
| Python (venv) | 3.12.13 |
| uv | 0.11.24 |
| NVIDIA driver | 596.21 |
| CUDA | 13.2 |
| torch (installed) | 2.6.0+cu124 |
| transformers (installed) | 4.57.6 |
| trl (installed) | 0.11.4 |
| peft (installed) | 0.19.1 |
| liger-kernel (installed) | 0.8.0 |
| accelerate (installed) | 1.14.0 |
| deepspeed (installed) | 0.19.2 |
| bitsandbytes (installed) | 0.49.2 |

---

## Fix Recommendations (Prioritized)

1. **C1** — Fix path resolution: either use `eval_image_folder` in dataset code (add to `sft_dataset.py`, `grpo_dataset.py`), or prepend split prefix to paths in prepared JSONs. This is the **blocker** for any training.
2. **C2** — Fix `deterministic_reward` regex: add `re.IGNORECASE` flag and extend `[A-D]` to `[A-J]` for sequence ordering.
3. **C4** — Fix dict iteration in `train_utils.py:41`: change to `.items()` and fix variable name.
4. **C5** — Move GRPO dtype casting to `grpo_trainer.py` after PEFT is applied.
5. **H1** — Implement `non_lora_state_dict.bin` restoration on resume in both `train_sft.py` and `train_grpo.py`.
6. **H3** — Replace `llm_judge_reward` with an actual teacher-LLM judge, or rename and document the heuristic.
7. **H4** — Add `export SFT_COMPUTE_METRICS=eval/compute_metrics.py` to the SFT training script.
8. **H5** — Wire `eval_path` into `make_grpo_data_module` for GRPO eval.
9. **M1** — Bump `pyproject.toml` constraint to `transformers>=4.57.0`.
10. **M2** — Add `bnb_4bit_quant_storage` and vision tower exclusion for 4-bit BnB config.
