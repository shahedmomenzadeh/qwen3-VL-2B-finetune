# Archived scripts (not maintained)

These were superseded and are kept for reference only. Do not use for new runs.

| Archived script | Why archived | Use instead |
|---|---|---|
| `test_sft_run.sh` | 1-video SFT smoke, older conventions: hardcoded `MODEL_ID`, full-epoch (no `max_steps` cap), `NFRAMES=32`/`max_seq 16384`, always merges to a full model | `lite_sft_test.sh` |
| `sft_lite_train.sh` | Lite SFT → `output/lite/sft_merged`, orphaned: points to nonexistent `grpo_lit_train.sh`, and no pipeline stage consumes its output (GRPO lite reads `output/lite_sft_test/merged`) | `lite_sft_test.sh` (+ `lite_grpo_test.sh`) |
| `sft_train.sh` (deleted, see git history) | Stale `r64/α128` 48 GB config, predates r16 prod baseline | `train_sft.sh` |
