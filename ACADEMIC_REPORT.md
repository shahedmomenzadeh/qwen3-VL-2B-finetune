# Fine-tuning Qwen3-VL-2B for Cataract Surgery Video Understanding via Supervised Fine-tuning and Group Relative Policy Optimization

**Shahed Momenzadeh — Qwen3-VL-2B Finetune (SFT + GRPO) — September 2026**

> Base model: `Qwen/Qwen3-VL-2B-Instruct` (`qwen3_vl`, patch 16, 2B params). Adaptation: QLoRA 4-bit (`r=32, α=64`, 301 adapter modules). Pipeline: **Base → SFT (dataset_sft, 60 frames) → GRPO (dataset_grpo, 32 frames, G=5, one-update, β=0.04) → Final GRPO Model**. Codebase: `grpo-stage-refactor` branch, `GRPO_ISSUES.md` P0–P3 audit.

---

## Abstract

We present a two-stage adaptation of Qwen3-VL-2B-Instruct to the cataract surgery video domain. Stage 1 performs supervised fine-tuning (SFT) on clip-level and full-video narration and chain-of-thought multiple-choice questions. Stage 2 performs Group Relative Policy Optimization (GRPO) on a 100% deterministic, rule-based reward suite covering three YouTube multiple-choice tasks and four phase-temporal tasks, with strict JSON structured outputs. We detail the algorithmic corrections required for a correct VLM-GRPO implementation: per-token PPO clipping with separate old-policy forwards, unbiased $k_3$ KL regularization to a frozen SFT reference via LoRA-disabled forwards, group-relative advantage with zero-variance handling, EOS-aware completion masking, and deterministic dropout-free policy log-probabilities. We quantify the visual token budget ($\approx$121 tok/frame at 131K pixels, 256 tok/frame at 262K), the end-to-end token load (SFT $58\,$M/epoch, GRPO $98\,$M forward / $5.4\,$M loss per epoch at the production 60/32-frame setting), and measured VRAM scaling on an RTX 4060 Laptop 8 GB (SFT 16 frames 3.0 GB, GRPO $G=4$ 8 frames 7.9 GB, 16 frames 262K OOM). A lite benchmark harness (`lite_e2e_benchmark.sh`) provides instrumented sweeps and GPU-hour projections for full-scale training.

---

## 1. Introduction

Vision-language models (VLMs) pre-trained on web-scale data lack the procedural granularity required for cataract surgery: instrument identification, step ordering, phase boundaries, and temporal localization of surgical phases. We address this by domain-adapting Qwen3-VL-2B-Instruct, a 2-billion-parameter mixed-modality transformer, through two stages of increasing reasoning demand.

**Contributions:**

1. A separated two-dataset design (`dataset_sft` for SFT, `dataset_grpo` for RL) with mirrored hardlinked videos, LLaVA-formatted SFT targets and GRPO targets that retain `correct_answer`, `question_type`, and `reference_reasoning` for deterministic scoring.
2. A corrected Qwen3-VL GRPO trainer (`src/trainer/grpo_trainer.py`) that implements per-token PPO ratios, reference KL, and multimodal logit alignment, fixing the $\text{ratio}=1$ detachment bug, the zero-variance absolute-reward bias, the LoRA-dropout nondeterminism, and the post-EOS masking error (audit `GRPO_ISSUES.md`).
3. A deterministic multi-task reward formulation yielding $R_{\text{total}} \in \{0, 0.05, 1.0, 1.05\}$ for MCQ and continuous $[0,1]$/$[0,1.05]$ for temporal tasks, with strict JSON-only parsing.
4. Token and compute accounting that maps $n_{\text{frames}}$ and $v_{\max}$ to trainable tokens and GPU-hours, validated by an instrumented lite benchmark.

---

## 2. Related Work

**VLM adaptation for surgical video.** Prior work fine-tunes CLIP-style encoders or VLMs on Cholec80/Cataract-101 for phase recognition and instrument detection, typically as classification. We extend to generative VLM QA with temporal grounding and reward-based optimization.

**Parameter-efficient fine-tuning.** LoRA (Hu et al., 2021) and QLoRA (Dettmers et al., 2023) enable adaptation of 2B+ models on single GPUs via 4-bit NormalFloat quantization and low-rank adapters $W = W_0 + B A$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$, $A \in \mathbb{R}^{r \times d_{\text{in}}}$.

**RL for VLM reasoning.** GRPO (DeepSeekMath, Shao et al., 2024) replaces the PPO value baseline with group-relative normalization over $G$ completions per prompt. DAPO, Dr. GRPO, and BNPO vary the token-level normalization. We adopt a one-update on-policy GRPO variant with token-mean normalization and $k_3$ KL, documenting the trade-off against PPO-style multi-epoch reuse (Section 4.6).

---

## 3. Method

### 3.1 Model Architecture: Qwen3-VL-2B-Instruct

Qwen3-VL (`model_type \in \{qwen3\_vl, qwen3\_vl\_moe, qwen3\_5, qwen3\_5\_moe\}$) comprises:

- **Vision tower** $\mathcal{V}$: 24 transformer blocks, patch size $p=16$, with 3D position embeddings `visual.pos_embed` and a merger $\mathcal{M}$ that compresses $2 \times 2$ patch neighborhoods.
- **Language model** $\mathcal{L}$: 28 decoder layers, hidden size $d$, causal LM head.
- **Connective:** `mm_token_type_ids` distinguishes text ($0$), image ($1$), and video ($2$) tokens; `video_grid_thw = (t, h, w)` encodes the spatio-temporal grid after merging, where

$$h = \left\lceil \frac{H}{p \cdot 2} \right\rceil,\quad w = \left\lceil \frac{W}{p \cdot 2} \right\rceil = \left\lceil \frac{H}{32} \right\rceil \cdot \left\lceil \frac{W}{32} \right\rceil$$

and $t = n_{\text{frames}}$ (capped to the video's actual frame count via `probe_total_frames`, `src/dataset/data_utils.py:182$).

With square resizing to $\sqrt{v_{\max}}$ side length:

- $v_{\max}=131\,072\approx 362^2 \Rightarrow 11 \times 11 \approx 121$ tokens/frame,
- $v_{\max}=262\,144\approx 512^2 \Rightarrow 16 \times 16 = 256$ tokens/frame.

Total visual tokens per sample $N_{\text{vis}} = t \cdot h \cdot w$. The processor interleaves `<|vision_start|><video><|vision_end|>` placeholder tokens whose count equals $N_{\text{vis}}$; `prompt_input_ids` length therefore already includes visual tokens.

### 3.2 Parameter-Efficient Adaptation: QLoRA

We quantize the base weights $W_0$ to 4-bit NF4 (`BitsAndBytesConfig`, `bnb_4bit_compute_dtype = \text{bf16}$, `double_quant = \text{True}$) and freeze them. Trainable parameters are:

- LoRA adapters on 301 `nn.Linear`/`nn.Embedding` modules (discovered via `find_target_linear_names`, excluding `embed_tokens` and `lm_head`): LLM 196 + vision 96 + merger 2 + deepstack merger 6 + `visual.pos_embed` 1.
- The merger itself when `freeze_merger = \text{False}$ (production), trained at $1\!\times\!10^{-5}$ vs. $1\!\times\!10^{-4}$ (LLM) and $2\!\times\!10^{-6}$ (vision).

Rank $r=32$, $\alpha=64$ ($\alpha/r = 2$), $\text{dropout}=0.05$ for SFT and $0.0$ for GRPO. The GRPO zero-dropout choice is algorithmic, not merely regularizational (Section 4.5). LayerNorm modules are kept in `float32` for stability; `lm_head` and `embed_tokens` are kept in `float32` to match (`src/train/train_{sft,grpo}.py:217$), with GRPO forwards wrapped in `torch.autocast(bf16)` to ensure `hidden$ (float32 from norm) and `lm_head.weight` (float32) execute in bf16 without the `BFloat16 vs Float` mismatch.

Effective trainable scalars $\approx 40$–$60\,$M ($r=32$), vs. $\approx 10$–$15\,$M at $r=8$.

### 3.3 Stage 1: Supervised Fine-tuning

SFT is standard causal language modeling with label masking.

Given dataset $\mathcal{D}_{\text{SFT}} = \{(x_i, y_i)\}_{i=1}^{N}$, where $x_i$ is the multimodal prompt (video + instruction) and $y_i$ the assistant response, we minimize

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{|\mathcal{D}|} \sum_{i=1}^{|\mathcal{D}|} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \pi_{\theta}(y_{i,t} \mid x_i, y_{i,<t}) \tag{1}$$

with `labels = input_ids` masked to `IGNORE_INDEX = -100` on prompt tokens (`src/dataset/sft_dataset.py:264$). The trainer `QwenSFTTrainer` supports per-group learning rates and generation-based evaluation (`compute_metrics.py`: `exact_match`, `contains_match`, `reasoning_rate`).

**Data:** `dataset_sft` (SFT) contains 2,760 videos (2,624 clip + 136 full). Prepared JSONs: `sft_train_dataset_sft.json` (7,663 samples), `sft_val_dataset_sft.json` (954). YouTube clips contribute 4 SFT items each (1 description + 3 CoT MCQs); phase clips contribute 1 teacher/critic description; full videos contribute timestamped narration. Lite balanced subset `data/lite_e2e/sft_train.json` (10: 2 youtube_full / 4 youtube_clip / 4 phase_clip) mirrors all subgroups.

**Hyperparameters (production, $48\,$GB single GPU):** $n_{\text{frames}}=60$, $v_{\min}=131\,072$, $v_{\max}=262\,144$, $\text{bf16}=\text{True}$, $\text{grad ckpt}=\text{True}$ (`use_reentrant=False` when `vision_lora`), effective batch $4 \times 4 = 16$, $2$ epochs, $\text{lr}=1\mathrm{e}{-4}$ / $2\mathrm{e}{-6}$ / $1\mathrm{e}{-5}$, $\text{weight decay}=0.1$, $\text{cosine}$ with $10$ warmup steps, `max_seq_length=8192`.

Note: at $60/262144$, worst-case $60 \cdot 256 = 15\,360$ visual + $\sim 400$ text $\approx 15.7\,$k tokens exceeds $8192$ and is truncated; at $60/131072$ ($\approx 7.6\,$k) it fits.

### 3.4 Stage 2: Group Relative Policy Optimization

#### 3.4.1 Problem Formulation

We have a prompt distribution $x \sim \mathcal{D}_{\text{GRPO}}$ and a reference policy $\pi_{\text{ref}} = \pi_{\theta_{\text{SFT}}}$ (frozen SFT-merged model, realized as the base model with LoRA adapters disabled). For each prompt $x$, we sample $G$ completions

$$y^{(g)} \sim \pi_{\theta_{\text{old}}}(\cdot \mid x),\quad g=1,\dots,G \tag{2}$$

where $\theta_{\text{old}}$ is the policy at rollout time. A deterministic reward $R(x, y^{(g)}) \in [0, 1.05]$ is computed (Section 3.5).

#### 3.4.2 Group-Relative Advantage

Rewards are grouped per prompt, $R_{i,g} = R(x_i, y_i^{(g)})$, reshaped to $(B, G)$ and normalized:

$$\mu_i = \frac{1}{G}\sum_{g=1}^{G} R_{i,g},\quad \sigma_i = \text{std}_g(R_{i,g}) \tag{3}$$

$$A_{i,g} = \begin{cases}
\frac{R_{i,g} - \mu_i}{\sigma_i + \epsilon}, & \sigma_i > \delta \\
0, & \sigma_i \le \delta
\end{cases},\quad \epsilon = 10^{-4},\ \delta = 10^{-6} \tag{4}$$

**Correction P0-1:** The prior implementation used $A_{i,g} = R_{i,g} - 0.5$ when $\sigma_i \le \delta$, injecting an absolute-reward gradient (e.g., $[0,0,0,0] \to -0.5$, $[1,1,1,1] \to +0.5$). The corrected form yields $A_{i,g}=0$ for zero-variance groups, preserving the group-relative guarantee. This matters because the reward is discrete (binary MCQ $+$ $0.05$ format bonus), so zero-variance groups occur frequently.

We log the diagnostic

$$\rho_{\text{zero-std}} = \frac{1}{B}\sum_{i=1}^{B} \mathbb{I}[\sigma_i \le \delta] \tag{5}$$

which measures the fraction of prompts providing no learning signal. If $\rho_{\text{zero-std}} \approx 0.7$, $70\%$ of prompts are uninformative in that step, indicating weak reward diversity or insufficient $G$.

Broadcasting: $A_{i,g}$ is expanded to per-token shape $A_{i,g} \cdot \mathbf{1}_{L_g}$ where $L_g$ is the completion length.

#### 3.4.3 Per-Token Log-Probabilities and Alignment

For each prompt-completion pair we have `generated_ids = [prompt_ids ; completion_ids]` of length $L_{\text{tot}} = L_{\text{prompt}} + L_{\text{comp}}$ and `full_mm_token_type_ids` (zeros appended for completion positions). Log-probabilities are obtained via a causal forward:

$$\text{logits} = \text{LM}_\theta(\text{generated\_ids}, \text{mm\_type}, \text{pixel\_videos}, \text{grid\_thw}) \in \mathbb{R}^{(B\cdot G) \times L_{\text{tot}} \times V} \tag{6}$$

$$\text{shift\_logits} = \text{logits}[:, L_{\text{prompt}}-1 : -1, :],\quad
\text{shift\_labels} = \text{completion\_ids} \tag{7}$$

so that $\text{shift\_logits}[b, k, :]$ predicts $\text{completion\_ids}[b, k]$. Verification script `scripts/verify_qwen_logit_alignment.py` checks greedily that $\arg\max \text{shift\_logits}[k] = \text{completion\_ids}[k]$ for $k=0,\dots,3$ on text-only and video prompts, ensuring no off-by-one error despite `mm_token_type_ids` left-padding and vision token expansion (P0-2).

Per-token log-probabilities with completion masking $M \in \{0,1\}^{(B\cdot G) \times L_{\text{comp}}}$ (Section 3.4.6):

$$\ell_{\theta}(b, k) = \log \pi_\theta(\text{completion\_ids}[b,k] \mid \text{prefix}) \cdot M_{b,k} \tag{8}$$

We compute three variants:

- $\ell_{\text{old}}$: `no_grad`, LoRA enabled (rollout policy),
- $\ell_{\text{ref}}$: `no_grad`, `model.disable_adapter()` (SFT reference, when $\beta>10^{-9}$),
- $\ell_{\theta}$: with `grad`, `model.train()` (current policy),

all under `torch.autocast(bf16)` to match `LayerNorm(float32) \to lm_head(float32)` dtypes.

#### 3.4.4 Per-Token PPO Surrogate and Clipping

Per-token ratio:

$$r_{b,k} = \exp\left(\ell_{\theta}(b,k) - \ell_{\text{old}}(b,k)\right) \tag{9}$$

**Correction (old-policy bug):** Prior code set $\ell_{\text{old}} = \ell_{\theta}.\text{detach}()$, hence $r\equiv 1$ and the surrogate was $-\,\mathbb{E}[A]$ ($=0$ because $A$ is zero-mean per group), yielding $1\mathrm{e}{-8}$ loss and dead clipping. The corrected code runs a separate `no_grad` forward before the `grad` forward.

Surrogate with $\epsilon=0.2$:

$$s^{(1)}_{b,k} = r_{b,k} A_{b},\quad s^{(2)}_{b,k} = \text{clip}(r_{b,k}, 1-\epsilon, 1+\epsilon)\, A_{b} \tag{10}$$

$$s_{b,k} = -\min(s^{(1)}_{b,k}, s^{(2)}_{b,k}) \tag{11}$$

Clipping is meaningful only when $r \neq 1$, which does not occur at step start under one-update GRPO (Section 3.4.7).

#### 3.4.5 Token-Mean Normalization and KL Regularization

Denominator over valid completion tokens:

$$D = \sum_{b,k} M_{b,k} \ge 1 \tag{12}$$

Policy loss (global token-average, previously mislabeled “DAPO-style”):

$$\mathcal{L}_{\text{policy}} = \frac{1}{D}\sum_{b,k} s_{b,k}\, M_{b,k} \tag{13}$$

Per-token $k_3$ KL estimator (unbiased, non-negative) to the reference:

$$\text{kl}_{b,k} = \exp(\Delta_{b,k}) - \Delta_{b,k} - 1,\quad \Delta_{b,k} = \ell_{\text{ref}}(b,k) - \ell_{\theta}(b,k) \tag{14}$$

$$\hat{\text{KL}} = \frac{1}{D}\sum_{b,k} \text{kl}_{b,k} M_{b,k} \tag{15}$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + \beta\, \hat{\text{KL}},\quad \beta=0.04 \text{ (default), } 0 \text{ disables reference.} \tag{16}$$

The reference is the SFT model ($W_0 + \text{no LoRA}$) rather than a separate snapshot; for PPO-style multi-epoch reuse an explicit frozen `old` and `ref` would be cleaner (Section 4.6).

#### 3.4.6 EOS-Aware Completion Masking

`generate` pads after EOS with `pad_token_id` when `pad ≠ eos`, but to avoid including tokens after the first EOS (e.g., a second EOS or stray token) we compute

$$E_{b,k} = \mathbb{I}[\text{completion\_ids}[b,k]=\text{eos}],\quad C_{b,k}=\sum_{j\le k} E_{b,j} \tag{17}$$

$$M^{\text{eos}}_{b,k} = \mathbb{I}[C_{b,k}=0] \lor \big(\mathbb{I}[C_{b,k}=1] \land E_{b,k}\big),\quad M_{b,k} = \mathbb{I}[\text{id}_{b,k}\neq \text{pad}] \cdot M^{\text{eos}}_{b,k} \tag{18}$$

so only tokens up to and including the first EOS are counted.

#### 3.4.7 One-Update vs. Multi-Epoch GRPO

Current implementation (Option A):

$$\text{rollout }G \to \ell_{\text{old}} \to \ell_{\theta} \to \mathcal{L} \to \text{step} \to \text{discard} \tag{19}$$

At loss time $\theta \approx \theta_{\text{old}}$, hence

$$\mathbb{E}[r] \approx 1,\quad \mathcal{L}_{\text{policy}} \approx 0,\quad \mathbb{E}[\text{clip\_fraction}] \approx 0 \tag{20}$$

$\mathcal{L}_{\text{policy}}\approx 0$ is expected: the gradient $\nabla_\theta \mathcal{L}_{\text{policy}} = -\,\mathbb{E}[A \cdot \nabla_\theta \log \pi_\theta]$ remains non-zero because $A$ is zero-mean but not zero. Learning is driven by $A$, $\hat{\text{KL}}$, and $\|\nabla\|$, evaluated via `reward_mean`, `advantage_std`, `approx_kl`, `grad_norm`, and validation reward.

Option B (conventional PPO-style GRPO) would buffer the rollout and $\ell_{\text{old}}$ and optimize for $E=4$ epochs over the same batch, making $r \neq 1$ and clipping active. We retain A on 8 GB hardware for memory simplicity; B is a drop-in extension via a `num_ppo_epochs` buffer.

#### 3.4.8 Diagnostics

Per logging step we record:

$$\begin{aligned}
&\text{grpo\_loss}=\mathcal{L}_{\text{policy}},\ \text{kl\_loss}=\beta\hat{\text{KL}},\ \text{approx\_kl}=\hat{\text{KL}},\ \text{total\_loss}=\mathcal{L}_{\text{total}},\\
&\text{reward\_mean},\ \text{reward\_std},\ \text{reward\_min/max},\ \text{fraction}_{R<0.01},\ \text{fraction}_{R>0.99},\ \rho_{\text{zero-std}},\\
&\text{adv\_mean/std},\ \text{ratio\_mean/std/min/max},\ \text{clip\_fraction},\ \text{comp\_len\_mean},\ \text{entropy} \approx -\mathbb{E}[\ell_{\theta}],\ \|\nabla\|.
\end{aligned}\tag{21}$$

### 3.5 Reward Formulation

All rewards are deterministic, JSON-gated, and sum to $R_{\text{total}} = R_{\text{task}} + w_{\text{fmt}} R_{\text{fmt}}$, $w_{\text{fmt}}=0.05$. Parsing is strict by default (`allow_fallback=False`, `src/train/reward_funcs.py`): only the `answer` field of the parsed JSON object counts; regex fallbacks (`\b[A-D]\b`, `P0?([1-9]|1[0-3])`) are gated for debugging. Dispatch is strictly by `question_type`, not by `isinstance(correct_answer, str)` (P3-1).

#### MCQ: `step_identification`, `visual_observation`, `instrument_identification`

Prompt requires `{"explanation": str, "answer": "A"|"B"|"C"|"D"}`. Let $g$ be the gold letter and $p$ the prediction:

$$R_{\text{fmt}}^{\text{mcq}} = \mathbb{I}[\text{keys}(J)=\{\text{explanation},\text{answer}\} \land \text{is\_str}(J.\text{explanation})],\quad
R_{\text{task}}^{\text{mcq}} = \mathbb{I}[R_{\text{fmt}}=1 \land \text{upper}(J.\text{answer}[0]) = g] \tag{22}$$

where $J = \text{extract\_json}(\text{completion})$ (fast-path `json.loads`, then ```json blocks, then outermost braces).

#### Boundary Detection: `boundary_detection`

Gold $t_{\text{gt}}$, prediction $t_{\text{pred}}$ from `answer.timestamp`:

$$R_{\text{task}}^{\text{bd}} = \exp\!\left(-\frac{|t_{\text{pred}}-t_{\text{gt}}|}{\tau}\right),\quad \tau=1.5\text{s} \tag{23}$$

Exact match $\to 1.0$, $0.5\,$s error $\to 0.717$, $1.5\,$s $\to 0.368$, $3\,$s $\to 0.135$. $R_{\text{fmt}}=1$ iff the timestamp was parsed from JSON.

#### Temporal Localization: `temporal_localization`

Gold $[s_{\text{gt}}, e_{\text{gt}}]$, prediction $[s_{\text{p}}, e_{\text{p}}]$ (swapped if $s_{\text{p}}>e_{\text{p}}$):

$$\text{Inter}=\max(0,\min(e_{\text{p}},e_{\text{gt}})-\max(s_{\text{p}},s_{\text{gt}})),\quad
\text{Union}=(e_{\text{p}}-s_{\text{p}})+(e_{\text{gt}}-s_{\text{gt}})-\text{Inter} \tag{24}$$

$$R_{\text{task}}^{\text{tl}} = \begin{cases} \text{Inter}/\text{Union}, & \text{Union}>0\\ 0, & \text{otherwise} \end{cases} \tag{25}$$

#### Phase Recognition: `timestamp_to_phase`, `contextual_phase_recognition`

Gold `P_{\text{gt}} \in \{P01,\dots,P13\}$ (canonical `P{\small\text{02d}}$`), prediction normalized via $\text{normalize\_phase}(s)=\texttt{P}\{\text{int}(m_1):02\text{d}\}$ on the regex `P0?([1-9]|1[0-3])\b$:

$$R_{\text{task}}^{\text{phase}} = \mathbb{I}[\text{normalize}(P_{\text{p}})=\text{normalize}(P_{\text{gt}})] \lor \mathbb{I}[\text{lower}(name_{\text{p}}) \approx \text{lower}(name_{\text{gt}})] \tag{26}$$

Strict mode returns $0$ if no JSON `answer` field.

#### Composite and Distribution

$$R_{\text{total}} = R_{\text{task}} + 0.05\,R_{\text{fmt}} \in \{0,0.05,1.0,1.05\} \text{ (MCQ/phase) or } [0,1.05] \text{ (continuous)}. \tag{27}$$

We log $R$’s distribution beyond mean/std to disambiguate, e.g., $\mathbb{E}[R]=0.3$ from $[0,0,0,1]$ vs. $[0.25,0.25,0.25,0.45]$:

$$\min R,\ \max R,\ \Pr(R<0.01),\ \Pr(R>0.99). \tag{28}$$

---

## 4. Dataset and Statistics

### 4.1 Separated Design

The master `dataset/` is split by `tools/build_separated_datasets.py` into `dataset_sft/` and `dataset_grpo/`, with video files hardlinked to avoid duplication. Paths in prepared JSONs are split-prefixed (`Train/YtId/clip_01.mp4`) for root-agnostic loading.

| Corpus | Folders (Train/Val/Test) | Description |
|---|---|---:|
| YouTube `YT_ID` | 108 / 13 / 15 (136) | Clinical cataract videos: `clip_*.mp4` + `full_video.mp4` |
| Phase `PH_*` | 105 / 22 / 23 (150) | Standardized phases: `clip_*.mp4` + `grpo_*.mp4` temporal tasks |
| **Total** | **213 / 35 / 38 (286)** | Mirrored across `dataset_sft` / `dataset_grpo` |

### 4.2 Prepared Sample Counts

| Dataset | Split | Videos (`\approx$clip+full) | Samples (LLaVA/GRPO JSON) |
|---|---|---:|---:|
| `dataset_sft` | Train | 2,174 (2,066 clip + 108 full) | **7,663** (`sft_train_dataset_sft.json`) |
|  | Val | 278 | **954** |
|  | Test | 308 | 1,050 (est.) |
| `dataset_grpo` | Train | 1,969 unique GRPO videos | **4,252** (`grpo_train_dataset_grpo.json`) = 3,424 YT clip MCQ + 828 phase temporal |
|  | Val | 299 | **573** = 411 + 162 |
|  | Test | 372 | 696 |

GRPO Train breakdown (deterministic): `step_id 1,142`, `visual_obs 1,141`, `instrument 1,141`, `temporal 214`, `boundary 210`, `timestamp_to_phase 203`, `contextual 201`. Lite balanced subsets `data/lite_e2e/` (10/5 SFT, 14/7 GRPO) and `output/lite_grpo_test/grpo_train.json` (30/7, $\approx$4–5 per task) cover all subgroups.

### 4.3 Video Duration Distribution

Measured via `ffprobe` on the prepared GRPO Train set (1,969 unique videos): $\min 1.0\,$s, $\mathbf{max}\ 210.0\,$s (`Train/eAIZjIKBK_c/clip_09.mp4`), $\text{mean}\ 25.9\,$s, $p_{95}=59\,$s, $p_{99}=107\,$s. Phase GRPO clips $11$–$39\,$s ($\text{mean}\ 25.2\,$s). SFT Train $3$–$364\,$s ($\text{mean}\ 38.7\,$s, $p_{95}=186\,$s) including $108$ full videos $\sim 12\,$min. Short clips ($1$–$3\,$s) cap $n_{\text{frames}}$ to their actual frame count via `probe_total_frames` (decord → cv2 fallback).

### 4.4 Token Statistics

**Visual tokens per frame** (Section 3.1):

| $v_{\max}$ | Side $\approx \sqrt{v_{\max}}$ | $h \times w = \lceil H/32\rceil \cdot \lceil W/32\rceil$ | Tok/frame |
|---|---|---|---:|
| $131\,072$ | $362$ | $11 \times 11$ | **121** |
| $262\,144$ | $512$ | $16 \times 16$ | **256** |

Text tokens (word $\approx$ token, measured via `AutoProcessor` tokenizer on 20-sample Monte Carlo, `add_special_tokens=False` + $\sim 20$ template overhead for `<|im_start|>` etc.):

- SFT prompt $\approx 150$–$300$, response $\approx 50$–$150$.
- GRPO question $\approx 200$–$350$ ($\approx 60$–$90$ word instruction + MCQ choices), completion budget $\texttt{max\_completion\_length}=128$ / $256$ / $1024$.

Worst-case $N_{\text{tok}} = N_{\text{vis}} + N_{\text{text}}$ per sample:

| Setting | $N_{\text{vis}}$ | $N_{\text{text}}$ | Total |
|---|---:|---|---:|
| SFT $60/131072$ | $60\cdot121=7\,260$ | $\sim 400$ | $\mathbf{\sim 7.6\,$k}$ (fits `max_seq_length=8192`) |
| SFT $60/262144$ | $15\,360$ | $\sim 400$ | $\sim 15.7\,$k ($>8192$ truncated) |
| SFT $32/131072$ | $3\,872$ | $\sim 400$ | $\sim 4.3\,$k |
| GRPO prompt $32/131072$ | $3\,872$ | $\sim 250$ | $\sim 4.1\,$k |
| GRPO per-gen $(prompt+comp)$ $32/131072$, $128$ | $3\,872$ | $\sim 250+128=378$ | $\sim 4.5\,$k |
|  | $1024$ |  | $\sim 5.4\,$k |
| GRPO per-prompt step $G=5$, $128$ | $5 \cdot 4.5\,$k |  | $\mathbf{\sim 22\,$k}$ forward, $640$ loss |
|  | $1024$ |  | $\sim 27\,$k fwd, $5\,120$ loss |

Averages are lower because short videos cap $t = \min(n_{\text{frames}}, n_{\text{total}})$; e.g., a $2\,$s clip at 30 fps has $60$ frames total, so $60 \to 60$, but a $1\,$s clip caps to $30$.

**End-to-end volume (production):**

- SFT: $7\,663 \cdot 7.6\,$k $\approx \mathbf{58\,M}$ tokens/epoch, train-label $\approx 0.8\,$M; $\times 2$ epochs $\approx \mathbf{117\,M}$ / $1.6\,$M.
- GRPO ($32/131072$, $\text{comp}_{\text{avg}}\approx 256$): forward $4\,252 \cdot 5 \cdot (4.1\,$k$+256) \approx \mathbf{98\,M}$/epoch, loss $4\,252 \cdot 5 \cdot 256 \approx \mathbf{5.4\,M}$; at $1024$ max, $114\,$M fwd / $21.7\,$M loss.

These determine step time and GPU-hours (Section 6.3).

---

## 5. Training Protocol

### 5.1 Optimization

HuggingFace `Trainer` with `adamw_torch`, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, per-group learning rates (LLM $10^{-4}$, vision $2\!\times\!10^{-6}$, merger $10^{-5}$), `bf16=True`, `tf32=True`, `gradient_checkpointing=True` (`use_reentrant=False` when `vision_lora`), `dataloader_num_workers=0` (lite) / $4$ (prod), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

SFT: `cosine` with $10$ warmup steps, `weight_decay=0.1`, effective batch $4\times4=16$, `2$ epochs, `max_steps=6$ in lite bench (vs. $7\,663$ full). GRPO: `constant` with $0$ warmup, `weight_decay=0.0`, batch $1$, `grad_accum=1$, $1$ epoch, `beta=0.04$, $G=5$ (prod) / $4$ (8 GB lite, since $G=5$ multi-step OOMs `CUDA driver error: device not ready` at $n_{\text{frames}}\ge 8$), `max_completion_length=128$ (bench) / $1024$ (`lite_grpo_test.sh`).

### 5.2 Quantization and Memory

QLoRA 4-bit (`bits=4`, `nf4`, `double_quant`) keeps the base $2\,$B params at $\sim 2.5\,$GB vs. $5.1\,$GB at 16-bit, freeing budget for frames. The 8 GB RTX 4060 Laptop profile runs $n_{\text{frames}}=32$ for SFT and $8$ for GRPO $G=4$ within budget; see Section 6.

### 5.3 Data Pipeline

`data/prepare_sft.py` / `data/prepare_grpo.py` generate LLaVA/GRPO JSONs with validation that each YouTube clip contributes expected $4$ SFT / $3$ GRPO records and each phase clip $1$. `scripts/build_lite_benchmark_data.py` samples balanced lite subsets by shuffling within each `question_type` or `youtube_full/youtube_clip/phase_clip` stratum and distributing remainder randomly (seed $42$), with `--grpo-train-samples 30$ yielding $\approx 4$–$5$ per 7 tasks.

Multimodal collation left-pads `prompt_input_ids` and `prompt_mm_token_type_ids` ($0$ pad), concatenates `pixel_values_videos` and `video_grid_thw`, and retains `second_per_grid_ts` for Qwen3-VL temporal RoPE.

---

## 6. Experimental Setup & Resource Analysis

### 6.1 Hardware and Environment

Single-GPU measurements on **RTX 4060 Laptop 8 GB** ($8188\,$MiB), CUDA $12.x$, `transformers` main ($5.15.0.dev0$), `trl≥1.8$, `peft 0.19$, `bitsandbytes 0.49$, `liger-kernel`, `qwen-vl-utils` (decord). Lite runs under `HF_HUB_OFFLINE=1` from `hf_cache` ($\approx$625 shards). SFT merging via `src/merge_lora.py` (`--safe-serialization`) chains `output/lite_sft_test/merged → output/lite_grpo_test/output$ for GRPO.

### 6.2 Lite Benchmark Harness

`lite_e2e_benchmark.sh` instruments an end-to-end run:

1. **Balanced subset check** (`data/lite_e2e/`).
2. **VRAM sweep** ($8$ trials: $n_{\text{frames}}\in\{8,16,32,48\}\times v_{\max}\in\{131072,262144\}$, single-step SFT/GRPO $G=5$ probes, `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw` $@0.5\,$s + `/usr/bin/time -v$).
3. **Stage 1 SFT Lite** ($6$ steps, $32$ frames) → `bench_sft/train.log+gpu.csv+time.log$.
4. **Merge SFT**.
5. **Stage 2 GRPO Lite** ($4$ steps, $G=5$, $16$ frames, $128$ comp) → `bench_grpo/*$.
6. **Final merge** → `report.md` with wall-clock, VRAM $p_{95}$/avg, and GPU-hour projection:

$$\text{steps}_{\text{full}} = \left\lceil\frac{N_{\text{samples}}}{\text{eff\_batch}}\right\rceil \cdot \text{epochs},\quad
\text{GPU-hours} = \text{steps}_{\text{full}} \cdot t_{\text{step}} / 3600 \tag{29}$$

with $t_{\text{step}}$ measured on the lite bench at fixed $n_{\text{frames}}$, $v_{\max}$, $G$.

`lite_grpo_test.sh` is the isolated GRPO probe ($30/7$, $G=4$, $8$ frames, $1024$ comp, $8$ steps, $\approx 9\,$min) that mirrors the sweep-stable configuration.

### 6.3 VRAM Scaling (Measured)

| $n_{\text{frames}}$ | $v_{\max}$ | SFT status | SFT peak MiB | GRPO $G=5$ status | GRPO peak MiB |
|---:|---:|---|---:|---|---:|
| 8 | 131072 | ok | 3056 | ok | 7942 |
| 8 | 262144 | ok | 3122 | ok | 7934 |
| 16 | 131072 | ok | 3056 | ok | 7914 |
| 16 | 262144 | ok | 3962 | **OOM** | 7924 |
| 32 | 131072 | ok (6 steps) | — | — | — |
| 32 | 262144 | — | — | — | — |

Multi-step GRPO at $G=5$, $n_{\text{frames}}\ge 8$, $\text{comp}=128$ fails with `CUDA driver error: device not ready` (OOM) even after single-step sweep success; $G=4$ with $n_{\text{frames}}=8$, $\text{comp}=1024$ is stable ($\approx 7.9\,$GB peak, $11.7\,$s/step). $v_{\max}$ doubling from $131\,$k to $262\,$k adds $\approx 2\times$ visual tokens and $\sim 900\,$MiB for SFT at $16$ frames.

### 6.4 GPU-Hour Projection (Example)

From a representative lite bench ($t_{\text{SFT}}\approx 5\,$s/step at $32/131072$, $t_{\text{GRPO}}\approx 12\,$s/step at $8/131072\ G=4\ 1024$ — substitute measured $t$ from `bench_*/end_time-start_time$):

- SFT full $7\,663 \cdot 2 / 1 \approx 15\,326$ steps $\Rightarrow 15\,326 \cdot 5/3600 \approx \mathbf{21.3\,$h}$ single-GPU ($5.3\,$h on $4\times$).
- GRPO full $4\,252 \cdot 1 \approx 4\,252$ steps $\Rightarrow 4\,252 \cdot 12/3600 \approx \mathbf{14.2\,$h}$ ($3.5\,$h on $4\times$).
- **Total $\approx 35.5\,$h** single-GPU, $8.9\,$h on $4\times$ A100/NVLink, $+5$–$10\%$ for eval/save/merge. Re-running the sweep at $n_{\text{frames}}=48/64$ on $80\,$GB cards allows $B=2$–$4$ and $\sim 2$–$3\times$ frame budget within the same wall-clock.

---

## 7. Evaluation

### 7.1 SFT Evaluation

Generation-based: for each validation prompt we generate with greedy decoding and compute `exact_match` (normalized answer equality), `contains_match` (substring containment, case-insensitive), `reasoning_rate` (fraction containing `Therefore`/`answer is`), and `avg_gen_length`. Training and validation cross-entropy (`eval_loss`) are also tracked (lite SFT: `train 1.85 \to 1.65$, `eval 2.04$).

### 7.2 GRPO Evaluation

`QwenGRPOTrainer.prediction_step` generates greedily on validation prompts and scores via `compute_grpo_rewards` (same deterministic functions as training), reporting `eval_reward` mean. Training logs additionally expose the diagnostics of Section 3.4.8; key health signals are $\rho_{\text{zero-std}}$ (if $>0.7$, most prompts are uninformative), $\hat{\text{KL}}$ (should rise slowly from $0$, not explode), and $\text{grad\_norm}$ ($2$–$8$ observed in corrected runs vs. $0$ when zero-std was mishandled and loss $\approx 10^{-8}$).

---

## 8. Discussion

**Algorithmic correctness vs. scale.** The one-update GRPO design makes `ratio≈1$ at step start a *feature*, not a bug: the policy gradient $-\mathbb{E}[A \nabla \log \pi]$ is non-zero even when $\mathcal{L}_{\text{policy}}\approx 0$. Treating $\mathcal{L}_{\text{policy}}$ magnitude as a convergence metric is therefore misleading; reward mean, advantage spread, and KL are the faithful progress indicators. Multi-epoch reuse would make clipping and ratio statistics meaningful at the cost of a rollout buffer and $E\times$ memory.

**Reward sparsity.** Strict JSON-only parsing removes false positives from reasoning text (e.g., a stray `A` or `P01` mention) but requires the SFT stage to have taught the `{"explanation","answer"}` schema reliably; otherwise early GRPO sees $\rho_{\text{zero-std}}\to1$ and $\nabla\to0$. The $0.05$ format bonus provides a minimal gradient toward well-formed outputs even when the task is wrong ($\approx 5\%$ of a correct answer), guiding exploration without dominating the signal.

**VLM-specific pitfalls.** Three of the P0/P1 fixes are unique to vision-language GRPO: (i) logit alignment must account for `mm_token_type_ids` left-padding and the vision placeholder length $L_{\text{prompt}}$; (ii) LoRA dropout must be disabled to keep `ℓ_{\text{old}}$ and $\ell_{\theta}$ comparable when $\theta_{\text{old}}=\theta$; (iii) `pixel_values_videos` and `video_grid_thw` must be replicated $G\times$ and extended with zero-typed completion placeholders (`full_mm`) for the logprob forwards.

**Resource envelope.** The $8\,$GB profile is severely $G$-bound: $5\times$ generation ($5\cdot256$ tokens decode) plus $3\times$ forward ($G_{\text{old}}+G_{\text{ref}}+G_{\text{current}}$) exceeds the $262\,$k $16$-frame budget. Reducing $G$ to $4$ and $v_{\max}$ to $131\,$k recovers headroom at $\approx 2\times$ loss in reward variance (and thus noisier advantages).

---

## 9. Conclusion

We have adapted Qwen3-VL-2B to the cataract surgery domain through a disciplined SFT+GRPO pipeline whose GRPO stage is now algorithmically faithful: per-token PPO with separate old-policy estimation, $k_3$ KL to a LoRA-disabled reference, zero-variance-aware advantages, EOS-correct masking, and dropout-free policy ratios. The accompanying lite harness makes the full-scale cost predictable from a sub-$10$-minute $30$-sample probe, enabling practitioners to trade frames, resolution, and $G$ against a measured $8\,$GB ceiling before committing $30$+ GPU-hours.

Future work includes PPO-style multi-epoch GRPO with an explicit `old`/`ref`/`current` decomposition, and Liger's fused `LigerFusedLinearGRPOLoss` variants (`grpo`/`bnpo`/`dr_grpo`/`dapo`) once their normalization is pinned to the token-mean baseline established here.

---

## References

- Hu, E. J. et al. LoRA: Low-Rank Adaptation of Large Language Models. *ICLR* 2022.
- Dettmers, T. et al. QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS* 2023.
- Shao, Z. et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. *arXiv:2402.03300*, 2024.
- Schulman, J. et al. Proximal Policy Optimization Algorithms. *arXiv:1707.06347*, 2017.
- Bai, J. et al. Qwen2.5-VL Technical Report. 2025; Qwen3-VL Technical Report. 2025.
- Hsu, H. et al. Liger Kernel: Efficient Triton Kernels for LLM Training. 2024.

---

## Appendix A: Hyperparameter Summary

| Group | Param | Value |
|---|---|---|
| Model | `model_id` | `Qwen/Qwen3-VL-2B-Instruct` |
|  | `attn` | `sdpa` (`disable_flash_attn2 True` in lite) |
| LoRA | `r` / `α` / `dropout` | `32` / `64` / `0.05` SFT, `0.0` GRPO |
|  | `target_modules` | 301, `lora_namespan_exclude=[lm_head, embed_tokens]` |
| Quant | `bits` | `4` (`nf4`, `double_quant`, `bnb_4bit_compute_dtype=bf16`) |
| SFT | `nframes` / `v_min` / `v_max` / `max_seq` | `60` / `131072` / `262144` / `8192` |
|  | `batch` / `grad_accum` / `epochs` | `4` / `4` (16 eff) / `2` |
|  | `lr` / `vision_lr` / `merger_lr` | `1e-4` / `2e-6` / `1e-5` |
|  | `optimizer` / `scheduler` / `warmup` | `adamw_torch` / `cosine` / `10` |
| GRPO | `nframes` / `v_max` | `32` / `131072` (prod), `8` / `131072` (lite 8GB) |
|  | `G` / `max_completion` / `beta` / `temp` | `5` / `128` / `0.04` / `0.9` (lite `4/1024`) |
|  | `epsilon` / `top_p` / `weight_decay` / `warmup` | `0.2` / `1.0` / `0.0` / `0` |
| Common | `bf16` / `tf32` / `grad_ckpt` | `True` |

## Appendix B: File Map

- `src/params.py`: `TrainingArguments` / `GRPOArguments` (dropout $0.0$, `use_liger_loss=False` legacy no-op).
- `src/trainer/grpo_trainer.py`: per-token ratio, EOS mask, KL $k_3$, diagnostics.
- `src/train/reward_funcs.py`: `extract_json_object`, `score_mcq/boundary/temporal/phase`, `compute_grpo_rewards` (`allow_fallback=False`).
- `lite_e2e_benchmark.sh`: sweep + `bench_sft`/`bench_grpo` + `report.md`.
- `lite_grpo_test.sh`: `30/7` probe (`GRPO_TRAIN_SAMPLES=30`, `GRPO_MAX_STEPS=8`).
- `scripts/verify_qwen_logit_alignment.py`: greedy `argmax(shift_logits)==completion` check.

## Appendix C: Dataset Schema

```json
// SFT
{"id": "...", "video": "Train/YTId/clip_01.mp4", "conversations": [{"from":"human","value":"<video>\n..."},{"from":"gpt","value":"..."}]}
// GRPO
{"id": "...", "video": "Train/YTId/clip_01.mp4", "conversations": [...], "correct_answer": "B" | {"timestamp":12.3} | {"start":2.1,"end":5.4} | {"phase_id":"P06"}, "question_type": "visual_observation|...", "reference_reasoning": "...", "reward_type": "deterministic"}
```

---

*Codebase commit history: `c4b6bac` (refactor) → `9adae32` (old-policy fix, 30-sample lite) → `344e7f7` (P0–P3 audit) → `c49fa79` (dtype fix) → `97fc0dc` (docs). Report generated from `output/lite_benchmark/vram_sweep.csv` and `ffprobe` scans of `dataset_{sft,grpo}`.*
