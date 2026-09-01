import os
import torch
import torch.nn as nn
from typing import Optional, List, Union, Dict, Any
from collections import defaultdict

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
    - Loss variants: DAPO (default) / GRPO.
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
        self.loss_type = getattr(self.args, "liger_grpo_loss_type", "dapo") or "dapo"

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
        Computes GRPO Policy Loss:
        1. Forward-samples G candidate completions for each prompt in the batch.
        2. Computes deterministic task + format rewards for all completions.
        3. Normalizes rewards within each group G to obtain advantages A_i = (R_i - mean(R)) / (std(R) + eps).
        4. Computes per-token log-probabilities under the current policy (and ref policy if beta > 0).
        5. Backpropagates policy gradient loss: - mean( advantage * ratio - beta * D_KL ).
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
        # When all completions in a group have equal reward, provide advantage relative to neutral baseline
        # so gradients are non-zero (reward - mean, or reward - 0.5 if all equal)
        norm_std = torch.where(std_mask, group_stds, torch.ones_like(group_stds))
        advantages = (rewards_grouped - group_means) / (norm_std + 1e-4)
        # If std is 0 (all G generations got same score), center around 0.5 baseline so policy reinforces good responses or penalizes bad ones
        zero_std_advantages = rewards_grouped - 0.5
        advantages = torch.where(std_mask, advantages, zero_std_advantages)
        advantages = advantages.view(-1)  # (batch_size * G)

        # ── Step 4: Policy Forward Pass & Logprobs ───────────
        model.train()

        # Full sequence is generated_ids: (batch_size * G, total_seq_len)
        full_attention_mask = (generated_ids != tokenizer.pad_token_id).to(torch.long)
        full_forward_kwargs = {}
        if prompt_mm_token_type_ids is not None:
            # Pad token type ids for the completion portion with 0
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

        outputs = model(
            input_ids=generated_ids,
            attention_mask=full_attention_mask,
            **full_forward_kwargs,
        )

        logits = outputs.logits  # (batch_size * G, seq_len, vocab_size)
        # Shift logits to match completion tokens
        shift_logits = logits[:, prompt_len - 1 : -1, :].contiguous()
        shift_labels = completion_ids.contiguous()

        # Compute per-token log probabilities: (batch_size * G, comp_len)
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        # Mask padding and EOS
        completion_mask = (shift_labels != tokenizer.pad_token_id).to(torch.float32)
        per_token_logps = per_token_logps * completion_mask

        # Sum or average token logprobs for sequence logprobs
        sequence_logps = per_token_logps.sum(dim=-1) / (completion_mask.sum(dim=-1) + 1e-6)

        # ── Step 5: Policy Loss Calculation ───────────
        # Detached baseline for importance sampling ratio
        with torch.no_grad():
            old_sequence_logps = sequence_logps.detach()

        ratio = torch.exp(sequence_logps - old_sequence_logps)

        # PPO / GRPO clipping (epsilon = 0.2)
        eps = 0.2
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        total_loss = policy_loss

        # Log metrics to trainer state
        if self.state.global_step % self.args.logging_steps == 0:
            mean_reward = rewards_tensor.mean().item()
            self.log(
                {
                    "grpo_loss": policy_loss.item(),
                    "reward_mean": mean_reward,
                    "reward_std": group_stds.mean().item(),
                    "advantage_mean": advantages.mean().item(),
                }
            )

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
