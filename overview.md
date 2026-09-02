# Qwen3-VL-2B Cataract Surgery Fine-tuning

## Overview

The model is fine-tuned using **Qwen3-VL-2B-Instruct** with **LoRA/QLoRA** on cataract surgery video data.

The training pipeline consists of two learning stages:

1. **Supervised Fine-Tuning (SFT)**

   * Learns visual descriptions, surgical context, and reasoning-based question answering from the SFT dataset.
2. **Group Relative Policy Optimization (GRPO)**

   * Further optimizes the SFT-trained model using the GRPO dataset, multiple-choice question answering, temporal tasks, and deterministic reward signals.

Each dataset contains two temporal scopes:

* **Clip-level data** — short segments of surgical procedures.
* **Full-video data** — complete surgical procedures and their temporal context.

These are data scopes within the two stages, not additional pipeline stages.

The overall progression is:

**Base VLM → SFT on dataset_sft → GRPO on dataset_grpo → Final GRPO Model**

---

# 1. Overall Training Flow

The complete training process consists of two sequential stages.

### Stage 1 — Dataset SFT

The base Qwen3-VL-2B-Instruct model is fine-tuned using all applicable samples from `dataset_sft/`, including both clip-level and full-video data.

The model learns:

* Visual descriptions of surgical scenes
* Surgical context
* Instrument and procedural understanding
* Timestamped narration and temporal context
* Surgical step ordering
* Reasoning-based questions
* Multiple-choice questions related to the visual content

The SFT LoRA adapter is then merged with the base model.

**Output:** Final SFT model.

---

### Stage 2 — Dataset GRPO

The final SFT model is optimized using all applicable samples from `dataset_grpo/`, including YouTube clip-level MCQs and phase-based temporal tasks. The GRPO dataset may reference both clip-level and full-video procedural context.

The model generates answers and receives deterministic rewards based on:

* Surgical step identification
* Instrument identification
* Visual understanding
* Boundary detection
* Temporal localization
* Phase recognition
* Strict JSON format compliance

The GRPO LoRA adapter is then merged with the SFT model.

**Output:** Final GRPO model.

---

# 2. Complete Pipeline

The complete sequence is:

```text
Qwen3-VL-2B-Instruct
        │
        ▼
 SFT on dataset_sft
 (clip-level + full-video data)
        │
        ▼
 Merge SFT LoRA
        │
        ▼
 GRPO on dataset_grpo
 (clip-level + temporal data)
        │
        ▼
 Merge GRPO LoRA
        │
        ▼
   Final GRPO Model
```

---

# 3. Dataset

The dataset is organized into two dedicated, mirrored dataset views:

* **`dataset_sft/`** — Supervised Fine-Tuning split containing conversational descriptions, visual reasoning, multiple-choice questions with chain-of-thought rationale, and timestamped full-video narrations.
* **`dataset_grpo/`** — Reinforcement learning split containing 100% deterministic rule-based evaluation tasks and strict JSON prompt instructions.

Both directories are mirrored views of the active dataset (generated via `tools/build_separated_datasets.py` while keeping the master `dataset/` intact). Video files (`clip_*.mp4`, `full_video.mp4`, etc.) are **hardlinked** across both datasets to eliminate redundant disk storage.

---

## 3.1 Organization and Corpora

The dataset is partitioned into three standard splits across **286 procedure folders**:

* **Training** (213 folders)
* **Validation** (35 folders)
* **Test** (38 folders)

The dataset brings together two distinct surgical corpora:

1. **YouTube Corpus (`{YT_ID}/`)** — 136 surgical procedures (108 Train / 13 Val / 15 Test) sourced from clinical and educational cataract surgery videos. It includes both short segmented surgical clips (`clip_*.mp4`) and full-length procedure videos (`full_video.mp4`).
2. **Phase Corpus (`PH_*/`)** — 150 standardized phase procedures (105 Train / 22 Val / 23 Test) covering canonical cataract surgery phases, annotated with teacher/critic descriptions and granular temporal phase tasks.

---

## 3.2 Directory Structure

```text
VLM_qwen3_train/
├── dataset_sft/
│   ├── Train/ Validation/ Test/
│   │   ├── {YT_ID}/                        # YouTube corpus (108 Train / 13 Val / 15 Test)
│   │   │   ├── clip_01.mp4                 # Video clip
│   │   │   ├── clip_01.jsonl               # Metadata (14 keys) — mirrored
│   │   │   ├── clip_01_sft.jsonl           # SFT: 4 lines (1 description + 3 MCQs with CoT)
│   │   │   └── full_video.mp4 + full_video_sft.jsonl  # Full procedure narration & reasoning
│   │   ├── PH_*/                           # Phase corpus (105 Train / 22 Val / 23 Test)
│   │   │   ├── clip_02.mp4
│   │   │   ├── clip_02.jsonl               # Metadata (32 keys including phase_id)
│   │   │   └── clip_02_sft.jsonl           # 1 line teacher/critic description
│   │   └── ...
│   ├── grpo_procedure_split_map.json
│   └── README.md
│
├── dataset_grpo/
│   ├── Train/ Validation/ Test/
│   │   ├── {YT_ID}/                        # YouTube corpus (108 Train / 13 Val / 15 Test)
│   │   │   ├── clip_01.mp4
│   │   │   ├── clip_01.jsonl               # Metadata mirrored
│   │   │   └── clip_01_grpo.jsonl          # GRPO: 3 lines (strict JSON MCQ prompt)
│   │   ├── PH_*/                           # Phase corpus (105 Train / 22 Val / 23 Test)
│   │   │   ├── clip_02.mp4
│   │   │   ├── clip_02.jsonl               # Metadata mirrored
│   │   │   ├── grpo_0000_temporal_localization.mp4
│   │   │   └── grpo_0000_temporal_localization_grpo.jsonl # GRPO: 1 line (JSON phase task)
│   │   └── ...
│   ├── procedure_split_map.json            # PH_* split mapping
│   ├── grpo_procedure_split_map.json
│   └── README.md
```

---

## 3.3 Dataset Counts & Statistics

### SFT Dataset (`dataset_sft/`)

| Split | Folders | Clip MP4 | Full Video | Clip SFT Files | Full SFT Files | Videos Total |
|---|---:|---:|---:|---:|---:|---:|
| **Train** | 213 | 2,066 | 108 | 2,066 | 108 | 2,174 |
| **Validation** | 35 | 265 | 13 | 265 | 13 | 278 |
| **Test** | 38 | 293 | 15 | 293 | 15 | 308 |
| **Total** | **286** | **2,624** | **136** | **2,624** | **136** | **2,760** |

### GRPO Dataset (`dataset_grpo/`)

| Split | Folders | Clip MP4 (YT + Phase) | Full Video | GRPO Files | GRPO Records | Reward Type |
|---|---:|---:|---:|---:|---:|---|
| **Train** | 213 | 2,894 (2,066 + 828) | 108 | 1,969 | 4,252 (3,424 + 828) | 100% deterministic |
| **Validation** | 35 | 427 (265 + 162) | 13 | 299 | 573 (411 + 162) | 100% deterministic |
| **Test** | 38 | 503 (293 + 210) | 15 | 372 | 696 (486 + 210) | 100% deterministic |
| **Total** | **286** | **3,824** | **136** | **2,640** | **5,521** | **100% deterministic** |

### GRPO Task Breakdown (5,521 records total)

* **YouTube Track (MCQ)**: **1,440 files / 4,321 records**
  * `step_identification`: 1,441 records
  * `visual_observation`: 1,440 records
  * `instrument_identification`: 1,440 records
* **Phase Track (JSON Phase Tasks)**: **1,200 files / 1,200 records**
  * `boundary_detection`: 300 records
  * `temporal_localization`: 300 records
  * `timestamp_to_phase`: 300 records
  * `contextual_phase_recognition`: 300 records

---

## 3.4 Supervised Fine-Tuning (SFT) Data

SFT data is structured for conversational multi-modal training across two temporal resolutions:

### 1. Clip-Level SFT
* **YouTube Track (`clip_*_sft.jsonl`)**: Each clip contains **4 training items**:
  * **1× Scene Description**: General prompt (`"Describe what is happening in this surgical video clip..."`) eliciting detailed visual description of surgical actions, instruments, and anatomical structures.
  * **3× Reasoning MCQ**: Multiple-choice questions for step identification, visual observation, and instrument identification using the prompt format: `Provide your reasoning first, then state your answer.` The target response supplies full chain-of-thought reasoning (`reference_reasoning + " Therefore the answer is X) ..."`).
* **Phase Track (`clip_*_sft.jsonl`)**: **1× Expert Teacher/Critic Description** per clip providing high-density procedural and anatomical commentary.

### 2. Full-Video SFT (`full_video_sft.jsonl`)
* **Timestamped Narration**: Chronologically grounded narrations spanning the complete cataract procedure (`full_video.mp4`).
* **Procedural Reasoning**: Long-range step ordering and surgical workflow progression questions across the full operative timeline.

---

## 3.5 GRPO Reinforcement Learning Data (Deterministic + JSON)

The GRPO dataset has been completely transformed into a **100% deterministic, rule-based reward framework** with strict JSON output constraints.

### 1. YouTube Track (MCQ)
* **Tasks**: `step_identification`, `visual_observation`, `instrument_identification`
* **Prompt Instruction (`GRPO_JSON_INSTRUCTION`)**:
  ```text
  Which step is being performed?
  A) Hydrodissection
  B) ...
  Respond ONLY with a JSON object with two keys: "explanation" and "answer". "explanation" is 1-3 sentences of reasoning describing what is visible (natural language, no field names). "answer" is the single letter A, B, C or D (nothing more). Example: {"explanation": "In the video the surgeon uses a keratome to create a corneal incision.", "answer": "B"}
  ```
* **Expected Output**: JSON object `{"explanation": "...", "answer": "A|B|C|D"}`.

### 2. Phase Track (Multi-Task JSON)
* **Boundary Detection (`boundary_detection`)**: Predicts the exact transition timestamp between surgical phases.
  * *Target*: `{"timestamp": float}` ($t_{gt}$)
* **Temporal Localization (`temporal_localization`)**: Predicts the start and end timestamps of a specific surgical phase.
  * *Target*: `{"start": float, "end": float}`
* **Phase Recognition (`contextual_phase_recognition`, `timestamp_to_phase`)**: Predicts the phase identifier from video clips or given timestamps.
  * *Target*: `{"phase_id": "P0X"}` / `{"phase": "P0X"}`

---

## 3.6 Reward Formulation in GRPO

All GRPO rewards are computed deterministically via rule-based scoring without heuristic LLM judges.

### 1. MCQ Task Reward
$$R_{\text{task}} = \begin{cases} 1.0 & \text{if } \text{pred}[\text{"answer"}] = \text{gold}[\text{"correct\_answer"}] \land \text{keys}(\text{pred}) = \{\text{"explanation"}, \text{"answer"}\} \\ 0.0 & \text{otherwise} \end{cases}$$

### 2. Phase Track Task Rewards

* **Boundary Detection**: Exponential decay on absolute temporal error $|t_{\text{pred}} - t_{\text{gt}}|$ with scale parameter $\tau = 1.5\,\text{s}$:
  $$R_{\text{task}} = \exp\left(-\frac{|t_{\text{pred}} - t_{\text{gt}}|}{1.5}\right)$$
  *(Exact match $\rightarrow 1.0$; $0.5\,\text{s}$ error $\rightarrow 0.717$; $1.5\,\text{s}$ error $\rightarrow 0.368$)*

* **Temporal Localization**: Temporal Intersection-over-Union ($\text{tIoU}$):
  $$\text{Intersection} = \max\left(0, \min(t_{\text{end}}^{\text{pred}}, t_{\text{end}}^{\text{gt}}) - \max(t_{\text{start}}^{\text{pred}}, t_{\text{start}}^{\text{gt}})\right)$$
  $$\text{Union} = (t_{\text{end}}^{\text{pred}} - t_{\text{start}}^{\text{pred}}) + (t_{\text{end}}^{\text{gt}} - t_{\text{start}}^{\text{gt}}) - \text{Intersection}$$
  $$R_{\text{task}} = \frac{\text{Intersection}}{\text{Union}}$$

* **Phase Recognition**: Normalized exact match:
  $$R_{\text{task}} = \begin{cases} 1.0 & \text{if } \text{normalize}(pred\_phase) = gt\_phase \\ 0.0 & \text{otherwise} \end{cases}$$

### 3. Decoupled Format Reward & Composite Score (Phase Track)

The evaluation separates semantic task correctness ($R_{\text{task}} \in [0, 1]$) from structural JSON compliance ($R_{\text{fmt}} \in \{0, 1\}$):

* $R_{\text{fmt}} = 1.0$ if the output strictly conforms to the expected JSON schema, else $0.0$.
* Total Reward:
  $$R_{\text{total}} = R_{\text{task}} + 0.05 \times R_{\text{fmt}}$$

| Outcome | $R_{\text{task}}$ | $R_{\text{fmt}}$ | $R_{\text{total}}$ |
|---|:---:|:---:|:---:|
| **Correct + Valid JSON** | 1.0 | 1.0 | **1.05** |
| **Correct + Malformed (fallback parser)** | 1.0 | 0.0 | **1.00** |
| **Wrong + Valid JSON** | 0.0 | 1.0 | **0.05** |
| **Wrong + Malformed** | 0.0 | 0.0 | **0.00** |

---

## 3.7 Data Record Schemas

### SFT Record Example (`clip_*_sft.jsonl`)

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<video>\nWhich step is being performed?\nA) Hydrodissection\nB) Phacoemulsification\nC) Capsulorhexis\nD) Lens Insertion\nProvide your reasoning first, then state your answer."
    },
    {
      "role": "assistant",
      "content": "The phacoemulsification handpiece is sculpting the cataractous lens nucleus into quadrants while utilizing ultrasonic energy and irrigation-aspiration. Therefore the answer is B) Phacoemulsification."
    }
  ]
}
```

### GRPO Record Example (`clip_*_grpo.jsonl`)

```json
{
  "prompt": [
    {
      "role": "user",
      "content": [
        {"type": "video", "video": "-Q4uQ6rEExs/clip_01.mp4"},
        {"type": "text", "text": "What primary step is being performed?\nA) Hydrodissection\nB) Corneal Incision\nC) Phacoemulsification\nD) IOL Implantation\nRespond ONLY with a JSON object with two keys: \"explanation\" and \"answer\". \"explanation\" is 1-3 sentences of reasoning describing what is visible (natural language, no field names). \"answer\" is the single letter A, B, C or D (nothing more). Example: {\"explanation\": \"In the video the surgeon uses a keratome to create a corneal incision.\", \"answer\": \"B\"}"}
      ]
    }
  ],
  "correct_answer": "B",
  "question_type": "visual_observation",
  "reference_reasoning": "In the video the surgeon uses a keratome to create a clear corneal incision at the limbus.",
  "reward_type": "deterministic"
}
```

---

# 4. Data Preparation

Before training, the raw dataset is converted into training samples.

The preparation process is:

1. Load the training and validation surgical procedures.
2. Separate data into **clip-level** and **full-video-level** samples.
3. Convert SFT annotations into the required conversational training format.
4. Convert GRPO annotations into the required question/reward format.
5. Assign unique sample identifiers.
6. Associate each sample with its corresponding video.
7. Optionally subsample the training data.
8. Keep the complete validation set regardless of the training subsampling ratio.

The test set remains available for final evaluation.

---

# 5. Model Configuration

### Base Model

**Qwen/Qwen3-VL-2B-Instruct**

The model can be loaded from either a Hugging Face model identifier or a local model path.

---

# 6. Quantization

The default configuration uses **4-bit quantization (QLoRA)**.

| Parameter    | Default |
| ------------ | ------: |
| Quantization |   4-bit |
| Alternative  |   8-bit |
| Full LoRA    |  16-bit |

4-bit and 8-bit configurations use quantized base-model weights, while 16-bit uses standard LoRA without quantization.

The base model weights remain frozen during LoRA training.

---

# 7. LoRA Configuration

| Parameter | SFT | GRPO |
|---|---:|---:|
| LoRA Rank | 32 | 32 |
| LoRA Alpha | 64 | 64 |
| LoRA Dropout | **0.05** | **0.0** (deterministic old/current, `GRPO_ISSUES.md P1-1`) |
| Target modules | 301 (LLM 196 + vision 96 + merger 2 + deepstack 6 + pos_embed 1) |
| Excluded | `embed_tokens`, `lm_head` |
| `use_dora` | False |
| Quant | QLoRA 4-bit (`bnb_4bit_compute_dtype=bf16`, `double_quant True`) |

Alpha `≈2× rank`. Base frozen; only `LoRA A/B` + `merger` trainable. Norm `float32`, `lm_head`/`embed_tokens` `float32` + `autocast(bf16)` for GRPO generate (fixes `BFloat16 vs Float`).

LoRA adapters are applied across:

* Language-model attention projections
* Language-model MLP projections
* Vision-tower attention projections
* Vision-tower MLP projections
* Vision-language merger layers
* Deepstack merger layers
* Visual positional embeddings

The base weights remain frozen, while the LoRA parameters are trainable.

---

# 8. Training Parameters (48 GB default; lite `lite_e2e_benchmark.sh` uses 6/4 steps)

| Parameter | SFT | GRPO |
|---|---:|---:|
| Batch per device | 4 (lite 1) | 1 |
| Grad accum | 4 → eff 16 | 1 |
| Epochs | 2 | 1 |
| LR LLM / vision / merger | `1e-4` / `2e-6` / `1e-5` | same |
| Weight decay | 0.1 | 0.0 |
| Warmup | 10 | 0 |
| Scheduler | cosine | constant |
| `beta` (KL) | — | 0.04 (`0` disables ref) |
| `G` generations | — | **5** prod / **4** lite 8GB |
| `max_completion` | — | `128` bench / `1024` `lite_grpo_test.sh` |
| `temperature`/`top_p` | — | `0.9` / `1.0` |
| `bf16`/`tf32`/`grad_ckpt` | True/True/True (`use_reentrant False` with `vision_lora`) |
| `use_liger_kernel` | True | **False** (custom manual token-mean loss, `GRPO_ISSUES.md P2-1`) |
| `epsilon` clip | — | 0.2 per-token PPO |
| One-update | — | `generate→old logprobs→loss→step→discard` (`ratio≈1`, `grpo_loss≈0` expected) |
| Advantage | — | group-norm `(R-mean)/std`, zero-var → `0` + `zero_std_group_fraction` log |

Loss: token-mean `sum(loss*mask)/sum(mask)` (global), `k3 KL = exp(ref-current)-(ref-current)-1`, `total = policy + beta*KL`. Diagnostics `reward_min/max`, `fraction_zero/one`.

---

# 9. Video Configuration

| Parameter | SFT (prod) | GRPO (prod) | Lite |
|---|---:|---:|---:|
| `NFRAMES` max | **60** | **32** | `32` SFT / `8-16` GRPO (8GB) |
| `VIDEO_MIN_PIXELS` | `131072` (`128*32*32`) | same | same |
| `VIDEO_MAX_PIXELS` | `262144` (`256*32*32`) | same | sweep `131072/262144` |
| `MAX_SEQ_LENGTH` | `8192` | `prompt+completion` | — |
| Capping | `probe_total_frames` → `min(nframes, total)` (≥2) |

`FPS` alternative to `NFRAMES`. Visual tok/frame `≈121` (`131072` ~`362²`) / `≈256` (`262144` ~`512²`) (`ceil(H/32)*ceil(W/32)`). SFT `60/131072` `~7.6k`/sample (fits 8192), `60/262144` `~15.7k` >8192 truncated. GRPO `32/131072` prompt `~4.1k` → per-gen `4.5k (128)` / `5.4k (1024)`. Full dataset `GRPO max 210s` clip (`eAIZjIKBK_c/clip_09.mp4`), `avg 25.9s` GRPO / `38s` SFT, SFT full `364s`.

---

# 10. Evaluation and Checkpointing

The default evaluation and checkpoint configuration is:

| Parameter                 |         Default |
| ------------------------- | --------------: |
| Evaluation strategy       |           Steps |
| Evaluation frequency      | Every 300 steps |
| Save strategy             |           Steps |
| Save frequency            | Every 300 steps |
| Maximum saved checkpoints |               3 |
| Evaluation batch size     |               1 |

Generation-based evaluation is enabled by default for SFT.

---

# 11. Training Data Subsampling

Training can optionally be performed on a fraction of the available training data.

The parameter ranges from:

**0.0 → 1.0**

where:

* **1.0** = use the complete training set
* **0.5** = use 50% of the training set
* **0.3** = use 30% of the training set

The training samples are shuffled using a fixed seed before subsampling.

**Validation data is never subsampled.**

This provides a convenient way to perform smoke tests or smaller experiments before launching a complete training run.

---

# 12. SFT Training Process

The model starts from the original Qwen3-VL-2B-Instruct weights and is trained on `dataset_sft/` as one SFT stage. The dataset includes both clip-level and full-video samples.

The model learns:

* Surgical visual descriptions
* Visual reasoning
* Question answering
* Chain-of-thought style tasks
* Temporal relationships
* Surgical progression and step ordering
* Timestamped events
* Long-range procedural context

After SFT training, the LoRA adapter is merged into the base model. The resulting model is the **final SFT model**.

---

# 13. GRPO Training Process

GRPO begins from the final SFT model and uses `dataset_grpo/` as one GRPO stage. The dataset includes clip-level multiple-choice questions and phase-based temporal tasks.

The reward data can contain:

* Correct answer
* Question type
* Reference reasoning
* Temporal boundaries or intervals
* Phase identifiers
* Reward type

The model generates answers and receives deterministic rewards based on task correctness and, for phase tasks, JSON format compliance. The tasks cover:

* Surgical step identification
* Instrument identification
* Visual understanding
* Boundary detection
* Temporal localization
* Phase recognition
* Sequence-level reasoning over surgical procedures

The GRPO adapter is then merged into the SFT model. The resulting model is the **final GRPO model**.

---

# 14. Reward Design

GRPO employs a **100% deterministic, rule-based reward framework** (completely eliminating brittle substring matching and heuristic keyword-overlap judges).

### 1. Deterministic JSON MCQ Reward (YouTube Track)

Evaluates multiple-choice predictions directly via strict JSON parsing:
* Verifies JSON format and key constraints: exactly `{"explanation": "...", "answer": "..."}`.
* Evaluates exact answer match:
  $$R_{\text{task}} = 1.0 \quad \text{if} \quad \text{pred}[\text{"answer"}] = \text{gold}[\text{"correct\_answer"}] \quad \text{else} \quad 0.0$$

---

### 2. Fine-Grained Temporal & Phase Rewards (Phase Track)

* **Boundary Detection**: Continuous reward using exponential error decay with scale $\tau = 1.5\,\text{s}$:
  $$R_{\text{task}} = \exp\left(-\frac{|t_{\text{pred}} - t_{\text{gt}}|}{1.5}\right)$$
* **Temporal Localization**: Temporal Intersection-over-Union ($\text{tIoU}$) evaluating segment overlap:
  $$R_{\text{task}} = \frac{\text{Intersection}(\text{pred}, \text{gt})}{\text{Union}(\text{pred}, \text{gt})}$$
* **Phase Classification**: Normalized string exact match against ground truth phase ID:
  $$R_{\text{task}} = 1.0 \quad \text{if} \quad \text{normalize}(pred\_phase) = gt\_phase \quad \text{else} \quad 0.0$$

---

### 3. Decoupled Format Reward

To encourage strict adherence to structured outputs while isolating semantic reasoning:
* $R_{\text{fmt}} = 1.0$ if the output strictly parses into the expected task schema, else $0.0$.
* Composite reward: $R_{\text{total}} = R_{\text{task}} + 0.05 \times R_{\text{fmt}}$ (yielding up to $1.05$ for perfectly structured, correct answers).

---

# 15. Training Outputs

The two main training outputs are:

| Stage                | Output             |
| -------------------- | ------------------ |
| Dataset SFT          | **Final SFT model** |
| Dataset GRPO         | **Final GRPO model** |

The important final models are therefore:

**Final SFT model → Final GRPO model**

---

# 16. Recommended Training Progression

For a complete experiment, the recommended progression is:

### Step 1 — Prepare Dataset

Prepare the Train, Validation, and Test splits for both `dataset_sft/` and `dataset_grpo/`, including clip-level and full-video annotations where available.

### Step 2 — Perform Dataset SFT

Train Qwen3-VL-2B-Instruct on `dataset_sft/`, using the available clip-level and full-video samples in a single SFT stage.

### Step 3 — Merge SFT LoRA

Merge the SFT LoRA adapter into the base model. This produces the **final SFT model**.

### Step 4 — Perform Dataset GRPO

Use the final SFT model as initialization and optimize it on `dataset_grpo/` using the deterministic reward functions.

### Step 5 — Merge GRPO LoRA

Merge the GRPO LoRA adapter into the SFT model. This produces the **final GRPO model**.

---

# 17. VRAM & Tokens (RTX 4060 8GB sweep `output/lite_benchmark/vram_sweep.csv`, `nvidia-smi @0.5s`)

| Config `BITS=4 bf16 r32` | VRAM peak | Tok/sample worst |
|---|---:|---|
| `SFT 8/131072 b1` / `GRPO G4 8/131072` | `3056MiB` / `7942MiB` | `~1.4k` / `~4.5k`/gen |
| `SFT 16/131072` / `GRPO 16/131072` | `3056MiB` / `7914MiB` | `~2.5k` / `~6k`/gen |
| `SFT 16/262144` / `GRPO 16/262144` | `3962MiB` / **OOM `7924MiB`** | `~4.6k` |
| `SFT 32/131072` (bench 6 steps) | ok | `~4.3k` |
| `SFT 60/131072 b4` (48GB prod) | `~10-12GB` est | `~7.6k` fits 8192 |
| `SFT 60/262144 b4` | `~12-15GB` | `~15.7k` >8192 truncated |

Full: SFT `7663*7.6k≈58M`/ep (`×2≈117M`), GRPO `32/131072 G5 256` fwd `4252*5*4.6k≈98M`/ep loss `5.4M` (`1024→114M/21M`). Logs `bench_*/gpu.csv+time.log`, report `output/lite_benchmark/report.md` (GPU-hours `sec/step × full_steps`).

Lite: `lite_sft_test.sh` `~2GB`, `lite_grpo_test.sh G4 8/131072 1024` `~7.9GB` `~9min` 30 samples; `lite_e2e_benchmark.sh` sweep `8` trials before lite run.

---

# 18. Small-Scale Validation

Before full training, a small-scale experiment can be performed using a single surgical procedure.

The smoke test:

1. Selects a small surgical procedure.
2. Uses its clip-level samples.
3. Includes its full-video samples.
4. Performs dataset SFT using the selected clip-level and full-video samples.
5. Merges the SFT adapter.
6. Performs dataset GRPO using the corresponding GRPO samples.
7. Merges the GRPO adapter.
8. Verifies that the LoRA parameters have changed from their initialization.

This provides a quick way to verify that the complete two-stage SFT-to-GRPO pipeline is functioning before committing significant GPU time.

---

# 19. Important Considerations and Notes

* **Deterministic Reward Upgrade**: The previous limitations regarding substring false positives and keyword-overlap heuristics have been fully resolved by migrating all 5,521 GRPO records to **100% deterministic, rule-based scoring** and strict JSON output schemas.
* **JSON Syntax Compliance**: The model must maintain valid JSON syntax under exploration. The small format reward ($R_{\text{fmt}} = 0.05$) helps guide early policy exploration toward valid formatting without overshadowing task correctness.
* **Evaluation Data Configuration**: Validation during GRPO focuses on rule-based accuracy and format validity across both the YouTube and Phase evaluation splits.

These considerations do **not affect the SFT training process**.

---

# 20. Summary

The proposed training strategy progressively increases the temporal and reasoning complexity of the model.

The model first learns visual and procedural understanding from the complete `dataset_sft/`, which contains both short surgical clips and full-video context.

After supervised learning, GRPO further optimizes the model using the complete `dataset_grpo/`, deterministic task rewards, and JSON-format rewards.

The complete learning progression is:

**Visual, temporal, and procedural understanding**

→ Dataset SFT

**Reward-based visual and procedural reasoning**

→ Dataset GRPO

The final objective is a Qwen3-VL model capable of understanding cataract surgery not only at the level of individual instruments or frames, but also in terms of **surgical actions, procedural context, temporal relationships, and the overall sequence of the operation**.
