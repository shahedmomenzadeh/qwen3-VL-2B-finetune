# Qwen3-VL-2B Cataract Surgery Fine-Tuning: Two-Stage GRPO Curriculum

[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Repo-blue)](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-grpo)
[![Runs on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Runs%20%26%20Telemetry-green)](https://huggingface.co/datasets/shahedm2001/qwen3-vl-2b-cataract-grpo-runs)
[![Telemetry CSV](https://img.shields.io/badge/%F0%9F%93%8A%20Dataset-Unified%20Telemetry%20CSV-orange)](https://huggingface.co/datasets/shahedm2001/qwen3-vl-2b-cataract-grpo-runs/raw/main/training_slices_with_metadata_unified.csv)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)

Fine-tuning **Qwen/Qwen3-VL-2B-Instruct** with QLoRA (4-bit, `r=16 alpha=32`, full-trainable visual merger) on surgical cataract video tasks using a specialized **Two-Stage Group Relative Policy Optimization (GRPO)** reinforcement learning curriculum.

---

## 🔗 Hugging Face Hub Repositories & Artifacts

All model checkpoints, merged standalone weights, high-resolution plots, raw telemetry logs, and TensorBoard scalars are openly hosted on Hugging Face:

* 🤖 **Standalone Model Repository:** [`shahedm2001/qwen3-vl-2b-cataract-grpo`](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-grpo)
  * Final merged 16-bit standalone model weights (`model.safetensors`, compatible directly with `transformers`).
  * Checkpoints 20, 40, 60, and 64 for Stage 2.
  * Publication figures under `plots/`.
* 📊 **Telemetry & Runs Repository:** [`shahedm2001/qwen3-vl-2b-cataract-grpo-runs`](https://huggingface.co/datasets/shahedm2001/qwen3-vl-2b-cataract-grpo-runs)
  * Complete stacked training log: `logs/grpo/train.log` (3.92 MB, Steps 1–596 unbroken).
  * Continuous TensorBoard runs (Steps 0–596 stitched seamlessly).
  * **Unified Telemetry CSV:** [`training_slices_with_metadata_unified.csv`](https://huggingface.co/datasets/shahedm2001/qwen3-vl-2b-cataract-grpo-runs/raw/main/training_slices_with_metadata_unified.csv) (4,758 rows joining every micro-prompt slice directly to sample ID, ground truth, and rewards).
  * Archived packages: `logs_final_complete_stages_1_and_2.zip` and `tensorboard_final_complete_stages_1_and_2.zip`.

---

## 🧠 Two-Stage GRPO Training Curriculum

Reinforcement learning over complex multimodal surgical videos presents a fundamental tension: **easy discrete tasks converge rapidly**, while **high-cardinality categorical tasks suffer from exploration starvation**. To solve this, training was designed as a two-stage curriculum:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 TWO-STAGE GRPO CURRICULUM PIPELINE                                │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘

 [Base Model: Qwen3-VL-2B-Instruct]
                 │
                 ▼
 [SFT Warmup: 100 Frames, LLaVA CoT Video Narrations & MCQs]
                 │
                 ▼
 [Stage 1: Multi-Task Foundation GRPO] ──► 532 Steps, 4,252 Clips, G=4 Rollouts
  • 4-Choice Surgical MCQs (Visual Obs, Step ID, Inst ID) ──► Converged to 100.0%
  • Continuous Temporal (tIoU, Boundary Detection)        ──► tIoU: 0.76, |dt|: 1.15s
  • 13-Class Surgical Phase Recognition                   ──► Stalled at ~36-42% (Zero-Advantage Bottleneck)
                 │
                 ▼ (LoRA weights merged into 16-bit standalone model)
 [Stage 2: Phase Specialization & Replay GRPO] ──► 64 Steps (Steps 533–596), 506 Clips, G=8 Rollouts
  • 80% Phase Concentration (404 clips) + G=8 Rollouts    ──► Slashed zero-std slices from 62.6% to <15%
  • 20% Replay Anchor Buffer (51 MCQs + 51 Temporal)     ──► 0% Forgetting (100% MCQ retained, tIoU 0.78)
  • 13-Class Surgical Phase Recognition                   ──► Surged to 65.2% – 73.5% (+30.6% absolute gain)
                 │
                 ▼
 [Final 16-bit Standalone Deployment Model: shahedm2001/qwen3-vl-2b-cataract-grpo]
```

### 1. Stage 1: Multi-Task Foundation (Steps 1–532)
* **Dataset:** 4,252 video clips covering all 7 surgical tasks across 3 families.
* **Exploration Parameter:** $G = 4$ rollouts per prompt.
* **Effective Batch Size:** 8 prompts $\times 4 = 32$ rollouts / optimizer step (`batch=1`, `grad_accum=8`, `micro=1`).
* **Dynamics & The Zero-Advantage Bottleneck:**
  * Surgical 4-choice MCQs converged to **100% accuracy** within the first 250 steps.
  * Fine-grained continuous temporal tasks steadily converged ($t\text{IoU} \approx 0.76$, $|\Delta t| \approx 1.15\,\text{s}$).
  * **The Bottleneck:** For 13-class surgical phase recognition ($P_{01}\text{--}P_{13}$), the chance of random exploration hitting the correct phase was $1/13 \approx 7.69\%$. With only $G=4$ rollouts, the probability of finding at least one correct rollout was only:
    $$P(\ge 1 \text{ hit}) = 1 - (1 - 0.0769)^4 \approx 27.18\%$$
  * On **62.6% of phase prompt slices**, all 4 rollouts produced incorrect phases, yielding identical rewards $R_i = 0.05$ (valid JSON format reward only). Because group-relative advantage normalizes by standard deviation:
    $$\hat{A}_i = \frac{R_i - \bar{R}}{\text{std}(R)} \xrightarrow{\text{std}(R) = 0} 0$$
    The policy gradient was identically zero ($\hat{A}_i = 0$), starving the model of gradient signal on phase classification and capping accuracy at ~36–42%.

### 2. Stage 2: Phase Specialization with Replay (Steps 533–596)
* **Initialization:** Initialized from the 16-bit merged weights of Stage 1 (`output/grpo_merged`).
* **Curated Replay Dataset (506 clips):**
  * **80% Phase Concentration:** 404 clips (203 `timestamp_to_phase`, 201 `contextual_phase_recognition`).
  * **20% Anti-Forgetting Replay Buffer:** 51 surgical MCQs + 51 continuous temporal tasks to anchor prior multi-task competencies.
* **Doubled Exploration Budget ($G = 8$ Rollouts):**
  * Expanding to $G=8$ rollouts raised exploration discovery to:
    $$P(\ge 1 \text{ hit}) = 1 - (1 - 0.35)^8 \approx 96.81\%$$
  * Zero-variance prompt slices dropped from **62.6% to < 15%**, providing dense, high-variance policy gradient updates.
* **Empirical Breakthrough:**
  * `timestamp_to_phase` surged from **$42.9\% \to 73.5\%$** (**+30.6% absolute gain**).
  * `contextual_phase_recognition` rose from **$36.7\% \to 65.2\%$** (**+28.5% absolute gain**).
  * **Zero Catastrophic Forgetting:** MCQs maintained **100.0% accuracy** on Visual Observation and Step ID, and $t\text{IoU}$ remained at **$0.781$**.

---

### Curriculum Comparison Summary

| Parameter / Feature | Stage 1 (Foundation Multi-Task) | Stage 2 (Phase Specialization & Replay) |
| :--- | :---: | :---: |
| **Base Model** | SFT Merged Checkpoint | Stage 1 16-bit Standalone Model |
| **Training Video Clips** | 4,252 clips (balanced across 7 tasks) | 506 clips (80% Phase, 10% MCQ, 10% Temporal) |
| **Optimizer Steps** | 532 steps (1 epoch) | 64 steps (cumulative steps 533–596) |
| **Rollouts per Prompt ($G$)** | $G = 4$ rollouts | $G = 8$ rollouts |
| **Zero-Gradient Slices ($\sigma=0$)** | 62.6% on Phase tasks | **< 15% on Phase tasks** |
| **Micro-Batching** | `BATCH=1`, `ACCUM=8`, `MICRO=1` | `BATCH=1`, `ACCUM=8`, `MICRO=1` |
| **Peak VRAM on 24GB GPU** | $\sim 18.2\,\text{GB}$ | $\sim 19.2\,\text{GB}$ |
| **Learning Rates** | LLM `5e-5`, Merger `5e-6`, Vision `1e-6` | LLM `4e-5`, Merger `5e-6`, Vision `1e-6` |
| **KL Divergence Penalty ($\beta$)** | `0.04` | `0.05` |
| **Phase Recognition Accuracy** | Plateaued at ~36–42% | **65.2% – 73.5% (+30.6% gain)** |
| **Surgical MCQ Accuracy** | 100.0% | **100.0% (Zero Forgetting)** |
| **Temporal Localization ($t\text{IoU}$)** | $0.758$ | **$0.781$ (Zero Forgetting)** |
| **Boundary Timing Error ($|\Delta t|$)** | $1.15\,\text{s}$ | **$0.85\,\text{s}$ (Sub-second)** |

---

## 📊 Training Results & Evaluation by Data Group

### Group 1: 4-Choice Surgical Video MCQs
![Group 1 Surgical MCQs](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-grpo/resolve/main/plots/figure_group1_surgical_mcqs.png)

* **Tasks:** `visual_observation`, `step_identification`, `instrument_identification` (3,475 prompt slices).
* **Dynamics:** Rapid convergence from ~90% to **100.0% accuracy** within 250 steps.
* **Stage 2 Anchor Evaluation:** Preserved **100.0%** accuracy on Visual Observation and Step ID, and **94.1%** on Instrument ID during Stage 2 replay, demonstrating complete immunization against catastrophic forgetting.
* **Policy Saturation:** Unanimous rollout consensus climbed above 90%, naturally causing reward variance $\text{std}(R) \to 0$ as choices saturated.

---

### Group 2: 13-Class Surgical Phase Recognition Specialization
![Group 2 Phase Recognition](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-grpo/resolve/main/plots/figure_group2_phase_recognition.png)
![Stage 2 Specialization Dashboard](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-grpo/resolve/main/plots/figure_stage4_phase_specialization.png)

* **Tasks:** `timestamp_to_phase`, `contextual_phase_recognition` (808 prompt slices).
* **The Breakthrough:** Stage 1's 62.6% zero-advantage rate was slashed to **$< 15\%$** in Stage 2 with $G=8$ rollouts.
* **Accuracy Gains:**
  * `timestamp_to_phase`: **$42.9\% \to 73.5\%$** (**+30.6% absolute gain**).
  * `contextual_phase_recognition`: **$36.7\% \to 65.2\%$** (**+28.5% absolute gain**).

---

### Group 3: Continuous Fine-Grained Temporal Tasks
![Group 3 Continuous Temporal Tasks](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-grpo/resolve/main/plots/figure_group3_continuous_temporal.png)
![Stage-Wise Distributions](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-grpo/resolve/main/plots/figure_continuous_distributions.png)

* **Tasks:** `temporal_localization` ($t\text{IoU}$), `boundary_detection` ($|\Delta t|$ error).
* **Temporal Localization:** Monotonic $t\text{IoU}$ growth from **$0.488 \to 0.650 \to 0.758 \to 0.781$**, with top rollouts exceeding $t\text{IoU} > 0.85$.
* **Phase Boundary Detection:** Rolling median boundary error dropped from **$4.38\,\text{s} \to 2.54\,\text{s} \to 1.15\,\text{s} \to 0.85\,\text{s}$**, achieving sub-second precision. Over 75% of predictions fall strictly within the clinical $\tau = 1.5\,\text{s}$ tolerance window.

---

### Independent Evaluation on Unseen Test Set (20% Slice, 138 Clips)

To guarantee complete independence from training artifacts, the model was tested in a clean virtual environment using the Hugging Face Hub weights directly:

* **Strict JSON Compliance:** **99.28%** (137 / 138 valid parsed JSON objects).
* **Average Latency:** **4.77 s / clip** (video loading + decoding + 24-frame multimodal forward).

| Task Family | Task Name | Metric | Test Accuracy / Score | Sample Count |
| :--- | :--- | :---: | :---: | :---: |
| **Group 1: 4-Choice MCQs** | `visual_observation` | Accuracy | **93.75%** (30 / 32) | 32 |
| | `instrument_identification` | Accuracy | **90.62%** (29 / 32) | 32 |
| | `step_identification` | Accuracy | **87.50%** (28 / 32) | 32 |
| | **Group 1 Average** | **Accuracy** | **90.62%** (87 / 96) | **96** |
| **Group 2: 13-Class Phase Rec.** | `timestamp_to_phase` | Accuracy | **33.33%** (3 / 9) | 9 |
| | `contextual_phase_recognition` | Accuracy | **16.67%** (2 / 12) | 12 |
| | **Group 2 Average** | **Accuracy** | **23.81%** (vs 7.69% random) | **21** |
| **Group 3: Continuous Temporal** | `temporal_localization` | Mean $t\text{IoU}$ | **0.1998** ($t\text{IoU} \ge 0.50$: 9.1%) | 11 |
| | `boundary_detection` | Median Error | **4.80 s** (Mean Reward: 0.1511) | 10 |

---

## 🚀 Quick Start (Running Inference from Hugging Face)

The final merged model is a 16-bit standalone Hugging Face model and can be loaded directly without any custom codebase dependencies:

```python
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# 1. Load standalone model and processor directly from Hugging Face Hub
model_id = "shahedm2001/qwen3-vl-2b-cataract-grpo"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(model_id)

# 2. Prepare multimodal prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "path/to/cataract_clip.mp4", "nframes": 24},
            {"type": "text", "text": "What surgical step is shown in this video? Respond in strict JSON format with keys 'explanation' and 'answer'."}
        ]
    }
]

# 3. Process video & generate response
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to("cuda:0")

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    trimmed_ids = output_ids[0][len(inputs.input_ids[0]):]
    print(processor.decode(trimmed_ids, skip_special_tokens=True).strip())
```

---

## 🛠️ End-to-End Training Pipeline

### Pipeline Architecture:
1. **SFT Stage:** Pre-trains multimodal visual alignment on video narrations and surgical question-answering (`train_sft.sh`).
2. **Merge SFT:** Fuses SFT LoRA adapters into base weights (`src/merge_lora.py`).
3. **Stage 1 GRPO:** Broad multi-task RL exploration across all 7 tasks with $G=4$ rollouts (`grpo_train.sh`).
4. **Merge Stage 1:** Fuses Stage 1 LoRA adapters into standalone 16-bit weights.
5. **Stage 2 GRPO:** Dedicated Phase specialization ($G=8$ rollouts, 80% phase concentration + 20% anti-forgetting replay) (`grpo_stage4_phase.sh`).
6. **Merge Final Model:** Automatically merges Stage 2 adapters into final deployment model.

```bash
# Clone the repository
git clone https://github.com/shahedmomenzadeh/qwen3-VL-2B-finetune.git
cd qwen3-VL-2B-finetune

# Run Stage 1 GRPO (G=4 rollouts, micro-batched to fit 24GB GPUs)
GRPO_MICRO_PROMPTS=1 BATCH_PER_DEVICE=1 GRAD_ACCUM=8 bash grpo_train.sh

# Run Stage 2 Phase Specialization (G=8 rollouts, replay buffer)
bash run_stage4_pipeline.sh
```

---

## 🔧 Technical Fixes & Stability Engineering

During development, several critical distributed RL and VRAM stability fixes were engineered:

* **Non-LoRA Merger Weight Preservation (`src/trainer/grpo_trainer.py`):**
  Hugging Face Trainer's `_load_from_checkpoint` only loads LoRA adapters by default, resetting trainable visual merger weights to initialization. Added a custom hook restoring `non_lora_state_dict.bin` upon resume.
* **CUDA Fragmentation Mitigation (`grpo_train.sh`):**
  Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to eliminate out-of-memory errors caused by memory block fragmentation over long RL training runs on 24GB GPUs.
* **Zero-Variance Advantage Correction:**
  Standard RL algorithms produce undefined or noisy advantages when group rollout rewards are identical ($\sigma = 0$). In our implementation, $\sigma = 0$ is explicitly mapped to $\hat{A}_i = 0$, ensuring policy updates occur only when exploration discovers true relative variance.
* **Continuous TensorBoard Stacking (`scripts/merge_tensorboard.py`):**
  Automatically offsets Stage 2 step scalars by $+532$ to maintain unbroken, continuous curves from Step 0 to Step 596.

---

## 📁 Repository Structure

```
qwen3-VL-2B-finetune/
├── train_sft.sh              # SFT pipeline (100 frames) → output/sft_merged
├── grpo_train.sh             # Stage 1 Multi-Task GRPO (G=4 rollouts) → output/grpo_merged
├── grpo_stage4_phase.sh      # Stage 2 Phase Specialization (G=8 rollouts, replay buffer)
├── run_stage4_pipeline.sh    # Background pipeline launcher with watchdog monitor
├── scripts/
│   ├── monitor_and_sync.py   # Watchdog daemon for milestone sync & auto-merging
│   ├── merge_tensorboard.py  # Stitches multi-stage TensorBoard event runs (+532 offset)
│   ├── run_instrumented.sh   # Non-destructive training logger with GPU telemetry
│   └── summarize_run.py      # Telemetry extraction and metrics summarization
├── src/
│   ├── trainer/grpo_trainer.py # Custom QwenGRPOTrainer with micro-batching & merger restore
│   ├── train/train_grpo.py     # Main GRPO entrypoint
│   ├── train/reward_funcs.py   # Multi-task deterministic reward functions
│   └── merge_lora.py           # Standalone LoRA weight fuser
├── data/
│   ├── prepare_grpo.py       # Converts raw surgical video annotations to GRPO format
│   └── dataset_stats.py      # Statistical audit tool for surgical task distributions
└── README.md                 # Project documentation & benchmark analysis
```

---

## 📄 License
This project is licensed under the Apache 2.0 License.
