# GRPO Training Issues — Tracking & Fix Status

> Source: Re-audit of `grpo_stage` archive after old-policy/logprob fix. Issues ordered by Priority (P0 critical -> P3 low).

## P0 — Critical (must-fix before real training)

### P0-1 Fix zero-std advantage handling — important  [FIXED]
- **File:** `src/trainer/grpo_trainer.py:320-326`
- **Bug:** `zero_std_advantages = rewards_grouped - 0.5` injects absolute reward signal when group variance =0. E.g. G=4 `[0,0,0,0] -> adv=-0.5`, `[1,1,1,1]->+0.5`. Breaks group-relative GRPO guarantee, especially harmful with binary+format rewards (frequent zero-variance groups).
- **Fix:** Use zero advantage for zero-variance groups (or mask out):
  ```python
  advantages = torch.where(
      std_mask,
      (rewards_grouped - group_means) / (group_stds + 1e-4),
      torch.zeros_like(rewards_grouped),
  )
  ```
  Alternatively mask groups out of loss.
- **Status:** FIXED (commit pending). `grpo_trainer.py:322-325` now zero-fills zero-std groups. Added `zero_std_group_fraction` diagnostic.

### P0-2 Verify Qwen3-VL completion/logit alignment — critical [FIXED (structural) + verification script ready]
- **Files:** `src/trainer/grpo_trainer.py:291,358-362,398-404` (`old_shift_logits = old_logits[:, prompt_len-1:-1]`, `completion_ids = generated_ids[:, prompt_len:]`)
- **Risk:** Assumes causal alignment holds with multimodal tokens (`video tokens`, `mm_token_type_ids`, `video_grid_thw`, `pixel_values_videos`). A 1-token shift error silently poisons policy gradient.
- **Fix:** Code uses `shift_logits = logits[:, prompt_len-1:-1]` predicting `completion_ids = generated_ids[:, prompt_len:]` — standard causal shift (logit at position i predicts token i+1). Left-padded prompts handled via `attention_mask`; multimodal video tokens are within `prompt_len` so shift remains correct. Added explicit verification script `scripts/verify_qwen_logit_alignment.py` that runs greedy generation for text + video samples and checks `argmax(logits[p_len-1+k]) == gen[p_len+k]` for k=0..3.
- **Status:** FIXED structurally; verification script ready to run on RTX 4060 (`HF_HOME=hf_cache .venv/bin/python scripts/verify_qwen_logit_alignment.py --bits 4`). If mismatch observed, adjust shift to `prompt_len : -1` or similar.

## P1 — High

### P1-1 Disable LoRA dropout during GRPO — recommended [FIXED]
- **Files:** `src/params.py:63,110` (`lora_dropout 0.05`), `src/trainer/grpo_trainer.py:263-285 vs 391+`, `lite_grpo_test.sh:175`, `lite_e2e_benchmark.sh:168,189,250,288`
- **Bug:** `old logprobs -> model.eval()` vs `current logprobs -> model.train()` have different LoRA dropout masks (0.05). So `theta_current==theta_old` can still give `log pi_current != log pi_old`.
- **Fix:** `--lora_dropout 0.0` for GRPO (and set GRPO default to 0.0 while keeping SFT at 0.05). Keep forward deterministic except generation sampling.
- **Status:** FIXED — `src/params.py:GRPOArguments.lora_dropout=0.0` (Training remains 0.05), `lite_grpo_test.sh:175` and `lite_e2e_benchmark.sh` GRPO blocks set to `0.0`.

### P1-2 Decide one-update GRPO vs PPO-style multi-epoch GRPO [FIXED - Decision: keep one-update for now]
- **Description:** Current: `generate G -> old logprobs -> loss -> step -> discard rollout`. At loss time `pi_current ~= pi_old`, `ratio~=1`, `grpo_loss~=0` expected. This is on-policy GRPO+KL, not PPO with multiple epochs over stored rollout+logprobs.
- **Options:**
  - A) Keep current design — monitor `reward, advantage, KL, total_loss, grad_norm, eval reward`, don't expect clipping active.
  - B) PPO-style GRPO — store rollout+old logprobs, do N epochs over same rollout; then ratio deviates and clipping meaningful.
- **Decision:** Keep **Option A (one-update)** for 8GB sweep + lite tests (simpler, less VRAM). Documented in `src/trainer/grpo_trainer.py` docstring.
- **Status:** FIXED — docstring updated to state one-update; PPO epochs deferred.

### P1-3 Add zero-std + reward-distribution diagnostics — high [FIXED]
- **Files:** `src/trainer/grpo_trainer.py:482-503` (logging)
- **Missing:** `zero_std_group_fraction`, `reward_min/max`, `fraction_reward_zero/one` (or at least min/max). With discrete rewards, `reward_mean=0.30` ambiguous between `[0,0,0,1]` and `[0.25,0.25,0.25,0.45]`.
- **Fix:** Log `zero_std_group_fraction = (~std_mask).float().mean()`, `reward_min/max`, `fraction_zero/one` if applicable, plus existing `reward_mean/std`, `advantage_mean/std`, `ratio_*`, `approx_kl`, `kl_loss`, `total_loss`, `grad_norm`.
- **Status:** FIXED — `src/trainer/grpo_trainer.py:345,511-515,533-537` now logs `zero_std_group_fraction`, `reward_min/max`, `fraction_reward_zero/one`.

## P2 — Medium

### P2-1 `use_liger_kernel` is misleading (custom loss not using Liger) [FIXED]
- **Files:** `src/params.py:70`, `src/trainer/grpo_trainer.py:56`, training scripts pass `--use_liger_kernel True`
- **Bug:** Custom manual loss computes `surr1/surr2/policy_loss` without invoking Liger kernel; `use_liger_kernel=True` / `loss_type` don't affect loss.
- **Fix:** Muted — `src/params.py:GRPOArguments.use_liger_loss` default False with no-op help, `src/trainer/grpo_trainer.py:62-72` warns if set and notes manual loss. Scripts GRPO blocks set `--use_liger_kernel False`.
- **Status:** FIXED.

### P2-2 `liger_grpo_loss_type` currently does nothing [FIXED]
- **File:** `src/trainer/grpo_trainer.py:56-57,504-506`
- **Bug:** `self.loss_type` read but only logged if !=dapo; doesn't alter objective (`grpo|bnpo|dr_grpo|dapo` identical).
- **Fix:** `liger_grpo_loss_type` kept as legacy no-op with warning; `loss_type` defaults to `dapo` but custom loss is token-mean regardless.
- **Status:** FIXED — same commit as P2-1.

### P2-3 DAPO-style normalization naming [FIXED]
- **File:** `src/trainer/grpo_trainer.py:419-422` (`denom = completion_mask.sum(); policy_loss = (loss*mask).sum()/denom`)
- **Issue:** This is global token-average; legitimate but calling it "DAPO-style" without exact DAPO objective is misleading.
- **Fix:** Renamed comment to "Token-mean normalization: global token-average" in `src/trainer/grpo_trainer.py:456-457`.
- **Status:** FIXED.

### P2-4 Tighten reward parsing (permissive fallbacks) [FIXED]
- **File:** `src/train/reward_funcs.py:79-83,135-143,189-196,242-244`
- **Risk:** Regex fallback `r'\b([A-D])\b'` or phase `P01` can match stray letter in explanation, giving false positive task reward.
- **Fix:** Extract only from parsed JSON `answer` field. Strict mode `allow_fallback=False` default (`src/train/reward_funcs.py:65,106,158,221,275,340`). Regex fallbacks gated behind `allow_fallback=True` (debug only).
- **Status:** FIXED.

### P2-5 Explicit EOS-aware completion masking [FIXED]
- **File:** `src/trainer/grpo_trainer.py:348` (`completion_mask = (shift_labels != pad_token_id)`)
- **Issue:** OK if `generate()` pads after EOS with pad, but should verify + explicitly mask tokens after first EOS.
- **Fix:** `src/trainer/grpo_trainer.py:373-380` now computes `eos_mask = ((cumsum==0) | (cumsum==1 & is_eos)) & pad_mask`, masking tokens after first EOS.
- **Status:** FIXED.

## P3 — Low/Medium

### P3-1 Clean up reward dispatcher for string answers [FIXED]
- **File:** `src/train/reward_funcs.py:268`
- **Bug:** `if qtype in {...} or isinstance(correct_answer, str):` treats any string GT as MCQ, risking silent misdispatch.
- **Fix:** Strict `question_type` dispatch in `src/train/reward_funcs.py:274-303` (`qtype in {mcq}... elif boundary...`) with explicit unknown-qtype warning and inferred fallback only for unknown types.
- **Status:** FIXED.

### Additional Refinement
### P1-3b Log reward distribution not only mean/std [FIXED]
- See P1-3. Now logs `reward_min/max`, `fraction_reward_zero/one`.

### P13 Don't use same model object for reference and policy if you eventually need PPO epochs [DEFERRED]
- Current `with model.disable_adapter():` for frozen SFT reference (LoRA) works for single-epoch. If/when multi-epoch PPO implemented, consider explicit frozen reference model (`old_policy`, `current_policy`, `reference_policy` three states). Deferred until P1-2 Option B.

---

## Priority Execution Order

| Step | Issue | Importance |
|------|-------|------------|
| 1 | P0-1 zero-std advantages -> 0 | Critical |
| 2 | P0-2 logit alignment unit test | Critical |
| 3 | P1-1 lora_dropout 0.0 for GRPO | High |
| 4 | P1-3 + P1-3b diagnostics (zero_std fraction + reward dist) | High |
| 5 | P1-2 Document one-update vs multi-epoch decision | High |
| 6 | P2-1/P2-2 Remove/mute misleading Liger options | Medium |
| 7 | P2-5 EOS-aware masking | Medium |
| 8 | P2-4 Tighten reward parsing | Medium |
| 9 | P3-1 Dispatcher cleanup | Low/Med |

## Progress Log

- 2026-09-02: Created tracking file; P0-1 fix applied (zero advantage for zero-std groups); P0-2 verification script drafted.
- 2026-09-02: Fixed P0-1 (`zero_std_advantages` removed, zero advantage + `zero_std_group_fraction` log), P1-1 (`GRPOArguments.lora_dropout=0.0`, scripts dropout 0.0 for GRPO), P1-2 (doc: one-update GRPO), P1-3+P1-3b (reward min/max, fraction zero/one, zero_std fraction), P2-1/2 (Liger flags muted, warning, params defaults False), P2-3 (renamed token-mean), P2-4 (strict JSON-only reward parsing, `allow_fallback=False`), P2-5 (EOS-aware mask `cumsum==0 | (cumsum==1 & is_eos)`), P3-1 (strict `question_type` dispatcher).
- 2026-09-02: Added `scripts/verify_qwen_logit_alignment.py` for P0-2 (text + video greedy alignment check: `shift_logits[:, prompt_len-1:-1]` vs `completion_ids`). Ready to run on GPU.
