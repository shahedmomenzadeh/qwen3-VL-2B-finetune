# Cataract Surgery VLM Dataset — GRPO Split (Mirrored, Deterministic + JSON)

This folder is a **mirrored GRPO-only** view of `../dataset/` (active dataset). Created via `tools/build_separated_datasets.py` (copy-only, original `dataset/` untouched; videos hardlinked to both). All `clip_*_grpo.jsonl` have been transformed:

* `reward_type` `llm_judge` → `deterministic` (now 100% deterministic, 5,521 records)
* Prompt suffix `Provide your reasoning first, then state your answer.` → strict JSON instruction (parsable)

## Structure (GRPO-only, videos mirrored)

```
dataset_grpo/
├── Train/ Validation/ Test/
│   ├── {YT_ID}/              # YouTube corpus (108 Train / 13 Val / 15 Test folders)
│   │   ├── clip_01.mp4
│   │   ├── clip_01.jsonl           # metadata mirrored
│   │   └── clip_01_grpo.jsonl      # GRPO: 3 lines (JSON MCQ prompt)
│   ├── PH_*/                 # Phase corpus (105 Train / 22 Val / 23 Test folders)
│   │   ├── clip_02.mp4
│   │   ├── clip_02.jsonl           # metadata mirrored
│   │   ├── grpo_0000_temporal_localization.mp4
│   │   └── grpo_0000_temporal_localization_grpo.jsonl # GRPO: 1 line (JSON phase task)
│   └── ...
├── procedure_split_map.json  # PH_* split mapping
├── grpo_procedure_split_map.json
└── README.md
```

**Not included**: `*_sft.jsonl`, `full_video_sft.jsonl` (see `../dataset_sft/`), `archived/`/`grpo_archived/` (still in `../dataset/`).

## Counts (active, no archived)

| Split | Folders | Clip MP4 (YT + Phase) | Full Video | GRPO files | GRPO records | Reward |
|---|---:|---:|---:|---:|---:|---|
| Train | 213 | 2,894 (2,066 + 828) | 108 | 1,969 | 4,252 (3,424 + 828) | 100% deterministic |
| Val | 35 | 427 (265 + 162) | 13 | 299 | 573 (411 + 162) | 100% deterministic |
| Test | 38 | 503 (293 + 210) | 15 | 372 | 696 (486 + 210) | 100% deterministic |
| **Total** | **286** | **3,824** | **136** | **2,640** | **5,521** | **100% deterministic** |

- YouTube Track (MCQ): 1,440 files / 4,321 records (step_id: 1,441, visual_obs: 1,440, instrument_id: 1,440)
- Phase Track (JSON Phase Tasks): 1,200 files / 1,200 records (boundary_detection: 300, temporal_localization: 300, timestamp_to_phase: 300, contextual_phase_recognition: 300)
- Reward Type: 100% `deterministic` across all 5,521 samples.

## Reward Functions

All tasks use deterministic, rule-based rewards.

### 1. YouTube Track (MCQ)
* **Tasks:** `step_identification`, `visual_observation`, `instrument_identification`
* **Expected Output:** JSON `{"explanation": "...", "answer": "A|B|C|D"}`
* **Task Reward:**
  ```
  R_task = 1.0 if pred["answer"] == gold["correct_answer"] else 0.0
  ```

### 2. Phase Track
Each sample evaluates `R_task in [0,1]` + `R_fmt in {0,1}` (decoupled).

#### A. Boundary Detection (`boundary_detection`)
* **Target:** `{"timestamp": float}` (t_gt)
* **Formula:** Exponential decay on absolute error |t_pred - t_gt| with tau=1.5s:
  ```
  R_task = exp(-abs(t_pred - t_gt) / 1.5)
  ```
  *Fallback regex extraction if strict JSON fails.*
  *Exact match -> 1.0; 0.5s error -> 0.717; 1.5s error -> 0.368*

#### B. Temporal Localization (`temporal_localization`)
* **Target:** `{"start": float, "end": float}`
* **Formula:** Temporal IoU:
  ```
  inter = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
  union = (pred_end - pred_start) + (gt_end - gt_start) - inter
  R_task = inter / union
  ```

#### C. Phase Recognition (`contextual_phase_recognition`, `timestamp_to_phase`)
* **Target:** `{"phase_id": "P0X", ...}` / `{"phase": "P0X"}`
* **Formula:** Normalized exact match:
  ```
  R_task = 1.0 if normalize(pred_phase) == gt_phase else 0.0
  ```

### 3. Format Reward & Total (Phase Track)
* `R_fmt = 1.0` if strict valid task JSON else `0.0` (`is_strict_json_format`)
* `R_total = R_task + 0.05 * R_fmt`

| Outcome | R_task | R_fmt | R_total |
|---|:---:|:---:|:---:|
| Correct + Valid JSON | 1.0 | 1.0 | **1.05** |
| Correct + Malformed (fallback) | 1.0 | 0.0 | **1.00** |
| Wrong + Valid JSON | 0.0 | 1.0 | **0.05** |
| Wrong + Malformed | 0.0 | 0.0 | **0.00** |

Reference implementation: `phase_grpo_dataset/rewards.py` (`reward_phase`, `reward_localization`, `reward_boundary`, `reward_format`, `compute_reward`).

---

## New GRPO Prompt (parsable)

Old:
```
Which step is being performed?
A) Hydrodissection
B) ...
Provide your reasoning first, then state your answer.
```

New (`prompts.py:GRPO_JSON_INSTRUCTION`):
```
Which step is being performed?
A) Hydrodissection
B) ...
Respond ONLY with a JSON object with two keys: "explanation" and "answer". "explanation" is 1-3 sentences of reasoning describing what is visible (natural language, no field names). "answer" is the single letter A, B, C or D (nothing more). Example: {"explanation": "In the video the surgeon uses a keratome to create a corneal incision.", "answer": "B"}
```

## GRPO Record (new)

```json
{
  "prompt": [{"role": "user", "content": [
    {"type": "video", "video": "-Q4uQ6rEExs/clip_01.mp4"},
    {"type": "text", "text": "What primary step...?\nA) ...\n...Respond ONLY with a JSON..."}
  ]}],
  "correct_answer": "B",
  "question_type": "visual_observation",
  "reference_reasoning": "In the video the surgeon ... (used as explanation)",
  "reward_type": "deterministic"
}
```

**Reward parsing (deterministic, easy)**:
```python
import json
pred = json.loads(model_output)  # must have exactly {explanation, answer}
reward = 1 if pred["answer"] == gold["correct_answer"] and set(pred)=={"explanation","answer"} else 0
# reference_reasoning kept for logging, not needed for reward
```

## Pipeline

Future `dataset_formatter` (`stages/dataset_formatter.py:133`) now emits this JSON prompt + `reward_type="deterministic"` for all types. `tools/merge_dataset.py` with `--out-sft/--out-grpo` mirrors videos (hardlink) and routes `*_sft.jsonl` → SFT dataset, `*_grpo.jsonl` → GRPO dataset. `tools/refine_instrument_options.py` and `tools/generate_missing_grpo_sft.py` handle separated layout via `--grpo-dataset-dir` / `--target`.

## Verify

```bash
grep -r llm_judge dataset_grpo --include="*.jsonl" | wc -l  # → 0
grep -c "Respond ONLY with a JSON" dataset_grpo/Train/*/*_grpo.jsonl  # → 4321
```

Original `dataset/` untouched (still has old prompt + `llm_judge` for reference).
