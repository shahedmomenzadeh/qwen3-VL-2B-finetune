import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Dict, Any
from collections import defaultdict
from contextlib import nullcontext

from transformers import Trainer, GenerationConfig
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    PREFIX_CHECKPOINT_DIR,
    logger,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from torch.utils.data import DataLoader

from train.train_utils import (
    extract_tensor_cpu,
    get_peft_state_dict,
    get_peft_state_non_lora,
)
from train.reward_funcs import compute_grpo_rewards


class QwenGRPOTrainer(Trainer):
    """
    Custom GRPO Trainer for Qwen3-VL supporting:
    - Multi-modal vision-language inputs (video/image/text).
    - Group Relative Policy Optimization with G sampled completions per prompt.
    - Deterministic rule-based scoring (MCQ accuracy, temporal IoU, exponential boundary decay, phase exact match).
    - Per-parameter-group learning rates (LLM, Vision Tower, Merger).
    - QLoRA (4-bit/8-bit) and 16-bit LoRA optimization.
    - One-update GRPO: each rollout is used for a single optimizer step (on-policy + KL),
      not PPO-style multi-epoch reuse. With this design ratio ~=1 at step start is expected
      and clipping is rarely active; learning signal comes from advantage * grad(logp) and KL.
      If multi-epoch PPO-GRPO is desired, a rollout buffer with stored old logprobs must be added.
    Fixed implementation (2026-09):
    - Old policy logprobs computed via separate no_grad forward (not detach of current).
    - Per-token PPO ratio with clipping (not sequence-mean ratio).
    - Reference KL via base SFT (LoRA disabled) when beta>0.
    - Zero-variance groups yield zero advantage (not rewards-0.5).
    - Proper diagnostics: ratio stats, clip_fraction, approx_kl, zero_std_group_fraction, reward min/max.
    - Token-mean normalization is global token-average (not aliased as DAPO without DAPO objective).
    """

    def __init__(
        self,
        *args,
        processing_class=None,
        **kwargs,
    ):
        super(QwenGRPOTrainer, self).__init__(*args, **kwargs)
        self.processor = processing_class or getattr(self, "processing_class", None)
        self.num_generations = getattr(self.args, "num_generations", 4)
        self.max_completion_length = getattr(self.args, "max_completion_length", 256)
        self.beta = getattr(self.args, "beta", 0.04)
        self.temperature = getattr(self.args, "temperature", 0.9)
        self.top_p = getattr(self.args, "top_p", 1.0)
        # Legacy Liger loss type flag — kept for backward compat but not used;
        # our GRPO loss is custom manual (not LigerFusedLinearGRPOLoss). If you need Liger,
        # integrate LigerFusedLinearGRPOLoss explicitly. Warn once if user sets non-default.
        self.loss_type = getattr(self.args, "liger_grpo_loss_type", None) or "dapo"
        _liger_requested = getattr(self.args, "liger_grpo_loss_type", None) is not None
        _use_liger_kernel = getattr(self.args, "use_liger_kernel", False) or getattr(self.args, "use_liger_loss", False)
        if _liger_requested or _use_liger_kernel:
            logger.warning(
                "QwenGRPOTrainer uses custom manual GRPO loss (token-mean PPO clip + k3 KL), "
                "not LigerFusedLinearGRPOLoss. liget_* flags are no-ops here (see GRPO_ISSUES.md P2-1/2)."
            )
        # PPO clip epsilon
        self.epsilon = 0.2

    def create_optimizer(self):
        """Setup optimizer with parameter-group specific learning rates."""
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "visual" in name and "merger" not in name
                ]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "merger" in name
                ]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters

                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in special_lr_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in special_lr_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                ]

                if visual_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n in decay_parameters
                                        and n in visual_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n not in decay_parameters
                                        and n in visual_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )

                if merger_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n in decay_parameters
                                        and n in merger_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n not in decay_parameters
                                        and n in merger_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if not getattr(self.args, "lora_enable", False):
            return

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        non_lora = get_peft_state_non_lora(
            self.model.named_parameters(),
            require_grad_only=True,
        )

        if self.args.should_save:
            if non_lora:
                torch.save(non_lora, os.path.join(output_dir, "non_lora_state_dict.bin"))
            if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "config"):
                self.model.base_model.config.to_json_file(os.path.join(output_dir, "config.json"))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Computes GRPO Policy Loss (corrected):
        1. Sample G completions per prompt with old policy (no_grad).
        2. Score deterministic task+format rewards.
        3. Group-normalize rewards -> advantages (zero-mean per group).
        4. Compute OLD per-token logprobs via no_grad forward (old policy).
           Optionally compute REFERENCE per-token logprobs via LoRA-disabled forward.
        5. Compute CURRENT per-token logprobs via grad forward.
        6. Per-token PPO ratio = exp(current - old), clipped surrogate, KL penalty.
        Note: with single on-policy epoch per rollout, ratio==1 at step start -> policy_loss ~0
              but gradient = -mean(A * d logp) remains non-zero. KL becomes non-zero after divergence.
        """
        device = self.args.device
        tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor

        prompt_input_ids = inputs["prompt_input_ids"].to(device)
        prompt_attention_mask = inputs["prompt_attention_mask"].to(device)
        prompt_mm_token_type_ids = inputs.get("prompt_mm_token_type_ids")
        if prompt_mm_token_type_ids is not None:
            prompt_mm_token_type_ids = prompt_mm_token_type_ids.to(device)

        batch_size = prompt_input_ids.shape[0]
        G = self.num_generations

        # Forward multimodal kwargs
        forward_kwargs = {}
        if "pixel_values_videos" in inputs:
            forward_kwargs["pixel_values_videos"] = inputs["pixel_values_videos"].to(device)
            forward_kwargs["video_grid_thw"] = inputs["video_grid_thw"].to(device)
        elif "pixel_values" in inputs:
            forward_kwargs["pixel_values"] = inputs["pixel_values"].to(device)
            forward_kwargs["image_grid_thw"] = inputs["image_grid_thw"].to(device)
        if "second_per_grid_ts" in inputs:
            forward_kwargs["second_per_grid_ts"] = inputs["second_per_grid_ts"]

        # ── Step 1: Generate G completions per prompt ───────────
        model.eval()
        gen_kwargs = {
            "input_ids": prompt_input_ids.repeat_interleave(G, dim=0),
            "attention_mask": prompt_attention_mask.repeat_interleave(G, dim=0),
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if prompt_mm_token_type_ids is not None:
            gen_kwargs["mm_token_type_ids"] = prompt_mm_token_type_ids.repeat_interleave(G, dim=0)

        for k, v in forward_kwargs.items():
            if isinstance(v, torch.Tensor):
                gen_kwargs[k] = v.repeat_interleave(G, dim=0)
            elif isinstance(v, list):
                expanded_list = []
                for item in v:
                    expanded_list.extend([item] * G)
                gen_kwargs[k] = expanded_list

        with torch.no_grad():
            unwrapped_model = self.accelerator.unwrap_model(model)
            generated_ids = unwrapped_model.generate(**gen_kwargs)

        prompt_len = prompt_input_ids.shape[1]
        completion_ids = generated_ids[:, prompt_len:]

        # Decode completions
        completion_texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # ── Step 2: Score completions with deterministic rewards ───────────
        correct_answers = inputs["correct_answers"]
        question_types = inputs["question_types"]

        expanded_gold_answers = []
        expanded_qtypes = []
        for ans in correct_answers:
            expanded_gold_answers.extend([ans] * G)
        for qt in question_types:
            expanded_qtypes.extend([qt] * G)

        rewards = compute_grpo_rewards(
            completions=completion_texts,
            correct_answers=expanded_gold_answers,
            question_types=expanded_qtypes,
            fmt_weight=0.05,
        )
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)

        # ── Step 3: Compute Advantages per prompt group G ───────────
        # Shape: (batch_size, G)
        rewards_grouped = rewards_tensor.view(batch_size, G)
        group_means = rewards_grouped.mean(dim=-1, keepdim=True)
        group_stds = rewards_grouped.std(dim=-1, keepdim=True)

        std_mask = group_stds > 1e-6
        # Group-relative normalization; zero-variance groups get zero advantage (no relative preference).
        # Previously used (rewards-0.5) which injected absolute-reward PG signal; removed (see GRPO_ISSUES.md P0-1).
        advantages = torch.where(
            std_mask,
            (rewards_grouped - group_means) / (group_stds + 1e-4),
            torch.zeros_like(rewards_grouped),
        )
        # Diagnostic: fraction of groups with zero variance (no learning signal)
        zero_std_group_fraction = (~std_mask.squeeze(-1)).float().mean() if std_mask.numel() > 0 else torch.tensor(0.0, device=device)
        advantages = advantages.view(-1)  # (batch_size * G)

        # ── Step 4: Prepare full forward kwargs & masks ───────────
        full_attention_mask = (generated_ids != tokenizer.pad_token_id).to(torch.long)
        full_forward_kwargs = {}
        if prompt_mm_token_type_ids is not None:
            comp_len = completion_ids.shape[1]
            comp_mm = torch.zeros((generated_ids.shape[0], comp_len), dtype=torch.long, device=device)
            full_mm = torch.cat([gen_kwargs["mm_token_type_ids"], comp_mm], dim=1)
            full_forward_kwargs["mm_token_type_ids"] = full_mm

        for k, v in forward_kwargs.items():
            if isinstance(v, torch.Tensor):
                full_forward_kwargs[k] = v.repeat_interleave(G, dim=0)
            elif isinstance(v, list):
                expanded_list = []
                for item in v:
                    expanded_list.extend([item] * G)
                full_forward_kwargs[k] = expanded_list

        # Helper to compute per-token logprobs for a given model state
        # Mask: exclude PAD. If EOS exists, tokens after first EOS are already PAD when using
        # generate(pad_token_id != eos_token_id), but we also explicitly mask after first EOS for safety.
        shift_labels = completion_ids.contiguous()  # (B*G, L)
        # EOS-aware mask: valid until and including first EOS, then 0. Falls back to PAD-only if no EOS.
        # Note: generate(pad_token_id != eos_token_id) already pads after EOS with PAD, but we mask explicitly
        # to avoid any stray tokens after EOS contributing to loss.
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and completion_ids.numel() > 0:
            is_eos = (completion_ids == eos_id)  # (B*G, L) bool
            eos_cumsum = is_eos.cumsum(dim=-1)  # 0 before first EOS, 1 at/after first EOS, 2 after second etc.
            # Valid if before first EOS (cumsum==0) OR at first EOS (cumsum==1 & is_eos). Tokens after first EOS (cumsum>=1 but not first EOS) masked.
            eos_mask = ((eos_cumsum == 0) | ((eos_cumsum == 1) & is_eos)).float()  # (B*G, L)
            pad_mask = (shift_labels != tokenizer.pad_token_id).float()
            completion_mask = (pad_mask * eos_mask).to(torch.float32)  # (B*G, L)
        else:
            completion_mask = (shift_labels != tokenizer.pad_token_id).to(torch.float32)  # (B*G, L)

        # ── 4a: OLD policy per-token logprobs (no_grad, LoRA enabled) ───────────
        with torch.no_grad():
            old_outputs = model(
                input_ids=generated_ids,
                attention_mask=full_attention_mask,
                **full_forward_kwargs,
            )
            old_logits = old_outputs.logits  # (B*G, seq_len, vocab)
            old_shift_logits = old_logits[:, prompt_len - 1 : -1, :].contiguous()
            old_log_probs = F.log_softmax(old_shift_logits, dim=-1)
            old_per_token_logps = torch.gather(
                old_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)  # (B*G, L)
            old_per_token_logps = old_per_token_logps * completion_mask
            # free
            del old_outputs, old_logits, old_shift_logits, old_log_probs

        # ── 4b: REFERENCE policy per-token logprobs for KL (no_grad, LoRA disabled) ───────────
        ref_per_token_logps = None
        if self.beta is not None and self.beta > 1e-9:
            with torch.no_grad():
                # Disable LoRA adapters to get SFT reference
                if hasattr(model, "disable_adapter"):
                    ctx = model.disable_adapter()
                else:
                    ctx = nullcontext()
                with ctx:
                    ref_outputs = model(
                        input_ids=generated_ids,
                        attention_mask=full_attention_mask,
                        **full_forward_kwargs,
                    )
                    ref_logits = ref_outputs.logits
                    ref_shift_logits = ref_logits[:, prompt_len - 1 : -1, :].contiguous()
                    ref_log_probs = F.log_softmax(ref_shift_logits, dim=-1)
                    ref_per_token_logps = torch.gather(
                        ref_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
                    ).squeeze(-1)
                    ref_per_token_logps = ref_per_token_logps * completion_mask
                    del ref_outputs, ref_logits, ref_shift_logits, ref_log_probs

        # ── 4c: CURRENT policy per-token logprobs (with grad) ───────────
        model.train()
        outputs = model(
            input_ids=generated_ids,
            attention_mask=full_attention_mask,
            **full_forward_kwargs,
        )
        logits = outputs.logits  # (batch_size * G, seq_len, vocab_size)
        shift_logits = logits[:, prompt_len - 1 : -1, :].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        per_token_logps = per_token_logps * completion_mask

        # ── Step 5: Per-token PPO/GRPO clipped loss ───────────
        eps = self.epsilon
        # Per-token ratio: exp(current - old)
        # For masked positions old==0 current==0 -> diff 0 -> ratio 1 (masked out later)
        per_token_ratio = torch.exp(per_token_logps - old_per_token_logps)
        # Clamp ratio for stability before clipping
        # Expand advantages to (B*G, L)
        advantages_expanded = advantages.unsqueeze(-1)  # (B*G, 1) -> broadcast to L

        surr1 = per_token_ratio * advantages_expanded
        surr2 = torch.clamp(per_token_ratio, 1.0 - eps, 1.0 + eps) * advantages_expanded
        per_token_policy_loss = -torch.min(surr1, surr2)  # (B*G, L)

        # Token-mean normalization: global token-average (sum(loss*mask)/sum(mask)).
        # This is the correct stable choice for variable-length completions.
        # Previously labeled "DAPO-style"; renamed to avoid implying full DAPO objective.
        denom = completion_mask.sum().clamp_min(1.0)
        policy_loss = (per_token_policy_loss * completion_mask).sum() / denom

        # ── Step 6: KL penalty to reference ───────────
        total_loss = policy_loss
        approx_kl = torch.tensor(0.0, device=device)
        kl_loss_val = torch.tensor(0.0, device=device)
        if ref_per_token_logps is not None:
            # Unbiased k3 estimator KL(current || ref): k3 = exp(ref - current) - (ref - current) - 1  >=0
            # This is the standard approximation used in PPO/GRPO for KL.
            # Alternative k1 = current - ref can be negative; k3 is always non-negative.
            log_ratio_ref = ref_per_token_logps - per_token_logps  # log pi_ref - log pi_current
            # Clamp for numerical stability
            log_ratio_ref = torch.clamp(log_ratio_ref, min=-20, max=20)
            per_token_kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
            per_token_kl = per_token_kl * completion_mask
            approx_kl = per_token_kl.sum() / denom
            # Also compute simple mean diff for logging (k1)
            # KL penalty added to loss
            kl_loss_val = approx_kl
            total_loss = policy_loss + self.beta * kl_loss_val
        else:
            # No reference -> KL 0
            pass

        # ── Diagnostics (no_grad) ───────────
        with torch.no_grad():
            # Ratio stats over valid tokens
            valid_ratio = per_token_ratio[completion_mask.bool()] if completion_mask.sum() > 0 else per_token_ratio.view(-1)
            if valid_ratio.numel() == 0:
                ratio_mean = torch.tensor(1.0, device=device)
                ratio_std = torch.tensor(0.0, device=device)
                ratio_min = torch.tensor(1.0, device=device)
                ratio_max = torch.tensor(1.0, device=device)
                clip_fraction = torch.tensor(0.0, device=device)
            else:
                ratio_mean = valid_ratio.mean()
                ratio_std = valid_ratio.std()
                ratio_min = valid_ratio.min()
                ratio_max = valid_ratio.max()
                # Clip fraction: fraction of tokens where ratio outside [1-eps, 1+eps]
                clipped = (valid_ratio < 1.0 - eps) | (valid_ratio > 1.0 + eps)
                clip_fraction = clipped.float().mean()

            # Completion length stats
            comp_lens = completion_mask.sum(dim=-1).float()
            comp_len_mean = comp_lens.mean()

            # Advantage stats
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            # Reward stats already available
            mean_reward = rewards_tensor.mean()
            # Use group_stds.mean for reward_std logging compatibility
            reward_std_mean = group_stds.mean() if group_stds.numel() > 0 else torch.tensor(0.0, device=device)
            reward_min = rewards_tensor.min() if rewards_tensor.numel() > 0 else torch.tensor(0.0, device=device)
            reward_max = rewards_tensor.max() if rewards_tensor.numel() > 0 else torch.tensor(0.0, device=device)
            # Discrete reward fractions (binary-like rewards common)
            fraction_reward_zero = (rewards_tensor < 0.01).float().mean() if rewards_tensor.numel() > 0 else torch.tensor(0.0, device=device)
            fraction_reward_one = (rewards_tensor > 0.99).float().mean() if rewards_tensor.numel() > 0 else torch.tensor(0.0, device=device)

            # Entropy approximation: -mean(logp) over completion tokens (lower = more confident)
            # Use current per_token_logps
            valid_logps = per_token_logps[completion_mask.bool()] if completion_mask.sum() > 0 else per_token_logps.view(-1)
            entropy_proxy = -valid_logps.mean() if valid_logps.numel() > 0 else torch.tensor(0.0, device=device)

        # Log metrics to trainer state
        if self.state.global_step % self.args.logging_steps == 0:
            # All values detached and converted to float for logging
            self.log(
                {
                    "grpo_loss": policy_loss.detach().item(),
                    "kl_loss": kl_loss_val.detach().item() if isinstance(kl_loss_val, torch.Tensor) else float(kl_loss_val),
                    "approx_kl": approx_kl.detach().item() if isinstance(approx_kl, torch.Tensor) else float(approx_kl),
                    "total_loss": total_loss.detach().item(),
                    "reward_mean": mean_reward.detach().item() if isinstance(mean_reward, torch.Tensor) else float(mean_reward),
                    "reward_std": reward_std_mean.detach().item() if isinstance(reward_std_mean, torch.Tensor) else float(reward_std_mean),
                    "reward_min": reward_min.detach().item(),
                    "reward_max": reward_max.detach().item(),
                    "fraction_reward_zero": fraction_reward_zero.detach().item(),
                    "fraction_reward_one": fraction_reward_one.detach().item(),
                    "zero_std_group_fraction": zero_std_group_fraction.detach().item() if isinstance(zero_std_group_fraction, torch.Tensor) else float(zero_std_group_fraction),
                    "advantage_mean": adv_mean.detach().item(),
                    "advantage_std": adv_std.detach().item(),
                    "ratio_mean": ratio_mean.detach().item(),
                    "ratio_std": ratio_std.detach().item(),
                    "ratio_min": ratio_min.detach().item(),
                    "ratio_max": ratio_max.detach().item(),
                    "clip_fraction": clip_fraction.detach().item(),
                    "comp_len_mean": comp_len_mean.detach().item(),
                    "entropy_proxy": entropy_proxy.detach().item(),
                }
            )
            # Also log loss_type for debugging if non-dapo
            if self.loss_type != "dapo":
                logger.info(f"GRPO loss_type={self.loss_type} policy_loss={policy_loss.item():.4f} kl={approx_kl.item():.4f}")

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Evaluation step computing deterministic reward accuracy across validation set."""
        device = self.args.device
        tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor

        prompt_input_ids = inputs["prompt_input_ids"].to(device)
        prompt_attention_mask = inputs["prompt_attention_mask"].to(device)
        prompt_mm_token_type_ids = inputs.get("prompt_mm_token_type_ids")
        if prompt_mm_token_type_ids is not None:
            prompt_mm_token_type_ids = prompt_mm_token_type_ids.to(device)

        forward_kwargs = {}
        if "pixel_values_videos" in inputs:
            forward_kwargs["pixel_values_videos"] = inputs["pixel_values_videos"].to(device)
            forward_kwargs["video_grid_thw"] = inputs["video_grid_thw"].to(device)
        elif "pixel_values" in inputs:
            forward_kwargs["pixel_values"] = inputs["pixel_values"].to(device)
            forward_kwargs["image_grid_thw"] = inputs["image_grid_thw"].to(device)
        if "second_per_grid_ts" in inputs:
            forward_kwargs["second_per_grid_ts"] = inputs["second_per_grid_ts"]

        model.eval()
        gen_kwargs = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "max_new_tokens": self.max_completion_length,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if prompt_mm_token_type_ids is not None:
            gen_kwargs["mm_token_type_ids"] = prompt_mm_token_type_ids
        for k, v in forward_kwargs.items():
            gen_kwargs[k] = v

        with torch.no_grad():
            unwrapped_model = self.accelerator.unwrap_model(model)
            generated_ids = unwrapped_model.generate(**gen_kwargs)

        prompt_len = prompt_input_ids.shape[1]
        completion_ids = generated_ids[:, prompt_len:]
        completion_texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        correct_answers = inputs["correct_answers"]
        question_types = inputs["question_types"]

        rewards = compute_grpo_rewards(
            completions=completion_texts,
            correct_answers=correct_answers,
            question_types=question_types,
            fmt_weight=0.05,
        )
        avg_reward = sum(rewards) / max(len(rewards), 1)
        loss = torch.tensor(1.0 - avg_reward, device=device)

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, None, None)
