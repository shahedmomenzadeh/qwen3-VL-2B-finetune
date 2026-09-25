# Qwen3-VL-2B Cataract Surgery Fine-Tuning: End-to-End 4-Stage SFT & GRPO Pipeline

[![Final GRPO Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Final%20GRPO%20Model-blue)](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-grpo)
[![SFT Stage 2 Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SFT%20Stage%202%20Model-purple)](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-sft-stage2)
[![SFT Stage 1 Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SFT%20Stage%201%20Model-cyan)](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-sft)
[![Runs & Telemetry](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Telemetry%20%26%20Runs-green)](https://huggingface.co/datasets/shahedm2001/qwen3-vl-2b-cataract-grpo-runs)
[![Telemetry CSV](https://img.shields.io/badge/%F0%9F%93%8A%20Dataset-Unified%20Telemetry%20CSV-orange)](https://huggingface.co/datasets/shahedm2001/qwen3-vl-2b-cataract-grpo-runs/raw/main/training_slices_with_metadata_unified.csv)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)

Fine-tuning **Qwen/Qwen3-VL-2B-Instruct** with QLoRA (4-bit, `r=16 alpha=32`, full-trainable visual merger) on cataract surgery video tasks across an integrated **Four-Stage Curriculum**: two stages of Supervised Fine-Tuning (SFT) followed by two stages of Group Relative Policy Optimization (GRPO) Reinforcement Learning.

---

## 🧭 End-to-End Training Architecture

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE 4-STAGE END-TO-END SURGICAL CURRICULUM PIPELINE                       │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘

 [Base Model: Qwen/Qwen3-VL-2B-Instruct]
                 │
                 ▼
 ═══════════════ PART I: SUPERVISED FINE-TUNING (SFT) CURRICULUM ═══════════════
                 │
 ┌───────────────┴──────────────────────────────────────────────────────────────────┐
 │ [Stage 1 SFT: Full Dataset Perception & VQA]                                    │
 │ • 5,597 Video Clips + Full Recordings across dataset_sft (2.0 Epochs, 700 Steps) │
 │ • Cross-Entropy Loss: 2.2810 ──► 1.0420 (Avg) / 0.6804 (Min)                     │
 │ • Establishes ocular anatomy, instrument recognition, and surgical step taxonomy │
 └───────────────┬──────────────────────────────────────────────────────────────────┘
                 │ (Merged into shahedm2001/qwen3-vl-2b-cataract-sft)
                 ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │ [Stage 2 SFT: Long-Horizon Full Video Coherence & Narrative Flow]               │
 │ • 100% of All 108 Full Surgery Videos + 10% Stratified Clips (42 Steps)          │
 │ • Halved Learning Rate (5e-5), Rank 8 LoRA Adapters                              │
 │ • Cross-Entropy Loss drops from 0.9812 ──► 0.4132 (2.3× Gradient Stabilization)  │
 └───────────────┬──────────────────────────────────────────────────────────────────┘
                 │ (Merged into shahedm2001/qwen3-vl-2b-cataract-sft-stage2)
                 ▼
 ═══════════════ PART II: GROUP RELATIVE POLICY OPTIMIZATION (GRPO) ════════════
                 │
 ┌───────────────┴──────────────────────────────────────────────────────────────────┐
 │ [Stage 3 (GRPO Stage 1): Multi-Task Foundation Reinforcement Learning]           │
 │ • 4,252 Videos, 532 Optimizer Steps, G=4 Rollouts per Prompt                     │
 │ • 4-Choice Surgical MCQs (Visual Obs, Step ID, Inst ID) ──► Converged to 100.0%  │
 │ • Continuous Temporal Tasks (tIoU, Boundary Detection)  ──► tIoU: 0.76, |dt|: 1.15s│
 │ • 13-Class Phase Recognition                            ──► Stalled at ~36–42%   │
 │   [Algorithmic Discovery: 62.6% Zero-Advantage Bottleneck due to G=4 Starvation] │
 └───────────────┬──────────────────────────────────────────────────────────────────┘
                 │ (Merged into intermediate 16-bit standalone model)
                 ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │ [Stage 4 (GRPO Stage 2): Phase Specialization & Anti-Forgetting Replay]          │
 │ • 506 Clips (80% Phase Specialization + 20% Anti-Forgetting Replay Anchor)       │
 │ • Doubled Exploration Budget: G=8 Rollouts per Prompt (Hit Prob: 27.2% ──► 96.8%)│
 │ • Slashed Zero-Variance Slices from 62.6% to < 15%                               │
 │ • 13-Class Surgical Phase Recognition Surges to 65.2% – 73.5% (+30.6% GAIN)     │
 │ • Complete Retention: 100.0% MCQ Accuracy & 0.781 tIoU (Zero Forgetting)        │
 └───────────────┬──────────────────────────────────────────────────────────────────┘
                 │
                 ▼
 [Final Standalone 16-bit Deployment Model: shahedm2001/qwen3-vl-2b-cataract-grpo]
```

---

## 🔗 Hugging Face Hub Repositories & Artifacts

All models, raw telemetry archives, and publication plots are published on Hugging Face:

* 🤖 **Final Deployment Model (Stage 4 GRPO):** [`shahedm2001/qwen3-vl-2b-cataract-grpo`](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-grpo)
* 🩺 **Stage 2 SFT Merged Model:** [`shahedm2001/qwen3-vl-2b-cataract-sft-stage2`](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-sft-stage2)
* 👁️ **Stage 1 SFT Merged Model:** [`shahedm2001/qwen3-vl-2b-cataract-sft`](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-sft)
* 📊 **Runs & Telemetry Dataset:** [`shahedm2001/qwen3-vl-2b-cataract-grpo-runs`](https://huggingface.co/datasets/shahedm2001/qwen3-vl-2b-cataract-grpo-runs)
  * Full GRPO Log: `logs/grpo/train.log` (3.92 MB, Steps 1–596 unbroken).
  * **Unified GRPO Telemetry CSV:** [`training_slices_with_metadata_unified.csv`](https://huggingface.co/datasets/shahedm2001/qwen3-vl-2b-cataract-grpo-runs/raw/main/training_slices_with_metadata_unified.csv) (4,758 joined rows).
  * **Complete SFT Logs Archive:** `training_logs/sft_training_logs_stages_1_and_2.zip`.

---

## 🔬 PART I: Supervised Fine-Tuning (SFT) Dynamics

Supervised fine-tuning prepares the base multimodal model for ophthalmic surgery through two progressive curricula:

1. **Stage 1 (Full SFT):** Covers all 5,597 video clips and full recordings in `dataset_sft` over 2 full epochs (700 steps) using Rank 16 LoRA adapters and a peak learning rate of $1.0 \times 10^{-4}$. Loss declines from **2.2810** down to a plateau around **0.8230 – 1.0620** (mean: 1.0420).
2. **Stage 2 (Continued SFT):** Addresses temporal narrative flow across entire surgeries by focusing exclusively on **all 108 unclipped full videos** combined with 10% stratified clips. Using halved learning rates ($5.0 \times 10^{-5}$) and Rank 8 adapters, loss plummets from **0.9812** to **0.4132**, stabilizing gradient norms by **2.3×** (median dropping from 2.56 to 1.09).

### SFT Two-Stage Concatenated Progression (Steps 1–742)
![SFT Two-Stage Dynamics](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-sft-stage2/resolve/main/plots/figure_sft_two_stage_training_dynamics.png)

### SFT Stage 2 Standalone Dynamics (Steps 1–42)
![SFT Stage 2 Standalone](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-sft-stage2/resolve/main/plots/figure_sft_stage2_dynamics.png)

---

## 🧠 PART II: Group Relative Policy Optimization (GRPO) Curriculum

Reinforcement learning over multimodal surgical video requires overcoming the **Zero-Advantage Bottleneck** on high-cardinality tasks:

### Stage 3: Multi-Task Foundation GRPO (Steps 1–532)
* **Dataset:** 4,252 video clips covering all 7 surgical tasks across 3 families.
* **Exploration Budget:** $G = 4$ rollouts per prompt.
* **Surgical MCQs:** Converged rapidly to **100.0% accuracy** (Visual Observation, Step ID, Instrument ID).
* **Continuous Temporal Tasks:** $t\text{IoU}$ rose to **0.758**, and boundary error $|\Delta t|$ contracted to **1.15 s**.
* **The Zero-Advantage Bottleneck:** On 13-class surgical phase recognition ($P_{01}\text{--}P_{13}$), random chance hit probability was only $1/13 \approx 7.69\%$. With $G=4$, the probability of finding $\ge 1$ correct completion was only:
  $$P(\ge 1 \text{ hit}) = 1 - (1 - 0.0769)^4 \approx 27.18\%$$
  On **62.6% of phase prompt slices**, all rollouts were incorrect ($R_i = 0.05$ format reward only), causing $\text{std}(R) = 0 \implies \hat{A}_i = 0$, completely zeroing out the policy gradient and stalling phase accuracy at ~36–42%.

### Stage 4: Phase Specialization with Anti-Forgetting Replay (Steps 533–596)
* **Curated Replay Dataset (506 clips):**
  * **80% Phase Concentration:** 404 clips (203 `timestamp_to_phase`, 201 `contextual_phase_recognition`).
  * **20% Replay Anchor Buffer:** 51 MCQs + 51 temporal tasks to immunize against catastrophic forgetting.
* **Doubled Exploration Budget ($G = 8$ Rollouts):**
  * Raised discovery hit probability to:
    $$P(\ge 1 \text{ hit}) = 1 - (1 - 0.35)^8 \approx 96.81\%$$
  * Slashed zero-variance slices from **62.6% to < 15%**, unlocking dense relative policy gradients.
* **Empirical Breakthrough:**
  * `timestamp_to_phase`: **$42.9\% \to 73.5\%$** (**+30.6% absolute gain**).
  * `contextual_phase_recognition`: **$36.7\% \to 65.2\%$** (**+28.5% absolute gain**).
  * **Zero Catastrophic Forgetting:** MCQs retained **100.0%** accuracy and $t\text{IoU}$ remained at **0.781**.

---

## 📊 Comprehensive Evaluation Results & Figures

### Group 1: 4-Choice Surgical Video MCQs
![Group 1 Surgical MCQs](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-grpo/resolve/main/plots/figure_group1_surgical_mcqs.png)
* Tasks: `visual_observation`, `step_identification`, `instrument_identification` (3,475 prompt slices).
* 100.0% convergence in Stage 1, with 100% accuracy retention during Stage 2 replay.

### Group 2: 13-Class Surgical Phase Recognition Specialization
![Group 2 Phase Recognition](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-grpo/resolve/main/plots/figure_group2_phase_recognition.png)
![Stage 4 Specialization Dashboard](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-grpo/resolve/main/plots/figure_stage4_phase_specialization.png)
* Tasks: `timestamp_to_phase`, `contextual_phase_recognition` (808 prompt slices).
* Zero-variance slices dropped from 62.6% to < 15%, propelling accuracy from 42.9% to 73.5% (+30.6%).

### Group 3: Continuous Fine-Grained Temporal Tasks
![Group 3 Continuous Temporal Tasks](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-grpo/resolve/main/plots/figure_group3_continuous_temporal.png)
![Continuous Distributions](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-grpo/resolve/main/plots/figure_continuous_distributions.png)
* Tasks: `temporal_localization` ($t\text{IoU}$), `boundary_detection` ($|\Delta t|$ error).
* $t\text{IoU}$ climbs monotonically from $0.488 \to 0.781$ (top rollouts $> 0.85$).
* Median phase boundary error shrinks from $4.38\,\text{s} \to 1.15\,\text{s} \to 0.85\,\text{s}$ (sub-second clinical precision).

### Policy Diagnostics & Zero-Variance Dynamics
![Binary Diagnostics](https://huggingface.co/shahedm2001/qwen3-vl-2b-cataract-grpo/resolve/main/plots/figure_binary_diagnostics.png)
* Documents consensus saturation, $\text{std}(R)$ trajectories, and bounded trust-region KL divergence ($\le 0.035$).

---

## 🧪 Independent Evaluation on Unseen Cataract Test Set (20% Slice, 138 Clips)

Evaluated in an isolated environment directly with public `transformers`:
* **Strict JSON Compliance:** **99.28%** (137 / 138 valid parsed JSON objects).
* **Average Latency:** **4.77 s / clip** (video loading + decoding + 24-frame multimodal forward).

| Task Family | Clinical Task Name | Metric | Test Accuracy / Score | Sample Count |
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

The final model is a standalone, merged 16-bit model requiring no custom repository classes:

```python
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Load directly from Hugging Face Hub
model_id = "shahedm2001/qwen3-vl-2b-cataract-grpo"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(model_id)

# Multimodal prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "path/to/cataract_clip.mp4", "nframes": 24},
            {"type": "text", "text": "What surgical step is shown in this video? Respond in strict JSON format with keys 'explanation' and 'answer'."}
        ]
    }
]

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

## 🛠️ End-to-End Training Execution Guide

```bash
# 1. Clone repository
git clone https://github.com/shahedmomenzadeh/qwen3-VL-2B-finetune.git
cd qwen3-VL-2B-finetune

# 2. Stage 1 SFT (Full Dataset, 100 frames, 2 epochs)
bash train_sft.sh

# 3. Stage 2 SFT (All Full Videos + 0.1 Clips, 64 frames, halved LR)
STAGE2_EPOCHS=1 CLIP_FRACTION=0.1 FULL_FRACTION=1.0 \
  MODEL_ID=shahedm2001/qwen3-vl-2b-cataract-sft \
  BATCH_PER_DEVICE=4 GRAD_ACCUM=4 NFRAMES=64 bash train_sft.sh

# 4. Stage 3 GRPO (Foundation RL, G=4 rollouts, micro-batched)
GRPO_MICRO_PROMPTS=1 BATCH_PER_DEVICE=1 GRAD_ACCUM=8 bash grpo_train.sh

# 5. Stage 4 GRPO (Phase Specialization & Replay, G=8 rollouts)
bash run_stage4_pipeline.sh
```

---

## 🔧 Technical Fixes & Stability Engineering

* **Trainable Visual Merger Preservation (`src/trainer/grpo_trainer.py`):** Restores `non_lora_state_dict.bin` upon resume, preventing the merger from resetting to initial weights.
* **CUDA Fragmentation Fix (`grpo_train.sh`):** Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid out-of-memory errors over long RL runs on 24GB GPUs.
* **Zero-Variance Gradient Formulation:** Maps identical rollout rewards ($\sigma=0$) to $\hat{A}_i = 0$, guaranteeing stability during early phase exploration.
* **Continuous Multi-Stage TensorBoard Stacking (`scripts/merge_tensorboard.py`):** Offsets Stage 2 steps by $+532$ to maintain seamless, unbroken scalar curves from Step 0 to Step 596.

---

## 📁 Repository Structure

```
qwen3-VL-2B-finetune/
├── train_sft.sh              # Stage 1 & Stage 2 SFT runner → output/sft_merged
├── grpo_train.sh             # Stage 3 Foundation GRPO (G=4 rollouts) → output/grpo_merged
├── grpo_stage4_phase.sh      # Stage 4 Phase Specialization (G=8 rollouts, replay buffer)
├── run_stage4_pipeline.sh    # Background pipeline launcher with watchdog monitor
├── scripts/
│   ├── monitor_and_sync.py   # Watchdog daemon for milestone sync & auto-merging
│   ├── merge_tensorboard.py  # Stitches multi-stage TensorBoard event runs (+532 offset)
│   ├── run_instrumented.sh   # Non-destructive training logger with GPU telemetry
│   └── build_epoch_subset.py # Stratified stage-2 epoch sampler (clips + full videos)
├── src/
│   ├── trainer/grpo_trainer.py # Custom QwenGRPOTrainer with micro-batching & merger restore
│   ├── train/train_grpo.py     # Main GRPO entrypoint
│   ├── train/reward_funcs.py   # Multi-task deterministic reward functions
│   └── merge_lora.py           # Standalone LoRA weight fuser
├── data/
│   ├── prepare_sft.py        # Converts raw surgical video annotations to LLaVA format
│   └── prepare_grpo.py       # Converts raw surgical video annotations to GRPO format
└── README.md                 # Complete 4-stage documentation & benchmark report
```

---

## 📄 License
This project is licensed under the Apache 2.0 License.
