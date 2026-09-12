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
    High-Efficiency Custom GRPO Trainer for Qwen3-VL:
    - Multi-modal vision-language inputs (video/image/text).
    - Group Relative Policy Optimization with G sampled completions per prompt.
    - Deterministic rule-based scoring (MCQ accuracy, temporal IoU, exponential boundary decay, phase exact match).
    - Per-parameter-group learning rates (LLM, Vision Tower, Merger).
    - QLoRA (4-bit/8-bit) and 16-bit LoRA optimization.
    - Memory-Optimized execution:
      * Prompt-by-prompt generation and backward pass to bound peak VRAM to G completions for 1 video.
      * Passing logits_to_keep=comp_len+1 to avoid computing logits on thousands of visual/prompt tokens.
      * Correct slicing of flattened multimodal video/image patches (no patch-level interleaving corruption).
    - One-update GRPO:
      * Old policy logprobs computed via separate no_grad forward.
      * Reference KL via base SFT (LoRA disabled) when beta>0.
      * Zero-variance groups yield zero advantage.
      * EOS-aware completion masking.
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

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        output_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(
            output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        )
        self.save_model(output_dir, _internal_call=True)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving model checkpoint to {output_dir}")

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)

        if self.processor is not None:
            self.processor.save_pretrained(output_dir)

        non_lora = get_peft_state_non_lora(
            self.model.named_parameters(),
            require_grad_only=True,
        )

        if self.args.should_save:
            if non_lora:
                torch.save(non_lora, os.path.join(output_dir, "non_lora_state_dict.bin"))
            if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "config"):
                self.model.base_model.config.to_json_file(os.path.join(output_dir, "config.json"))

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Executes GRPO training step with peak memory bounded to 1 prompt * G generations.
        Iterates over each prompt in the micro-batch, computes prompt rewards, advantages,
        logprobs (with logits_to_keep=comp_len+1), calls backward immediately to free
        intermediate graphs, and accumulates gradients across the micro-batch.
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

        # Compute video slice offsets
        video_thws = inputs.get("video_grid_thw")
        pixel_videos = inputs.get("pixel_values_videos")
        video_offsets = [0]
        if video_thws is not None:
            video_thws = video_thws.to(device)
            pixel_videos = pixel_videos.to(device)
            for thw in video_thws:
                video_offsets.append(video_offsets[-1] + int(thw[0] * thw[1] * thw[2]))

        # Compute image slice offsets
        image_thws = inputs.get("image_grid_thw")
        pixel_images = inputs.get("pixel_values")
        image_offsets = [0]
        if image_thws is not None:
            image_thws = image_thws.to(device)
            pixel_images = pixel_images.to(device)
            for thw in image_thws:
                image_offsets.append(image_offsets[-1] + int(thw[0] * thw[1]))

        compute_dtype = (
            torch.bfloat16
            if getattr(self.args, "bf16", False)
            else (torch.float16 if getattr(self.args, "fp16", False) else torch.float32)
        )
        use_autocast = getattr(self.args, "bf16", False) or getattr(self.args, "fp16", False)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=compute_dtype)
            if use_autocast and torch.cuda.is_available()
            else nullcontext()
        )

        unwrapped_model = self.accelerator.unwrap_model(model)
        eps = self.epsilon
        gas = self.current_gradient_accumulation_steps

        # Accumulated metrics for step logging
        step_metrics = defaultdict(list)
        total_unscaled_loss = 0.0

        for i in range(batch_size):
            p_id = prompt_input_ids[i:i + 1]
            valid_mask = (p_id != tokenizer.pad_token_id)
            p_id_unpadded = p_id[valid_mask].unsqueeze(0)
            p_len = p_id_unpadded.shape[1]

            gen_p_id = p_id_unpadded.repeat_interleave(G, dim=0)
            gen_attn = torch.ones_like(gen_p_id)

            f_kwargs = {}
            if video_thws is not None:
                thw_i = video_thws[i:i + 1]
                f_kwargs["video_grid_thw"] = thw_i.repeat(G, 1)
                start, end = video_offsets[i], video_offsets[i + 1]
                v_patches = pixel_videos[start:end]
                f_kwargs["pixel_values_videos"] = torch.cat([v_patches] * G, dim=0)
            elif image_thws is not None:
                thw_i = image_thws[i:i + 1]
                f_kwargs["image_grid_thw"] = thw_i.repeat(G, 1)
                start, end = image_offsets[i], image_offsets[i + 1]
                img_patches = pixel_images[start:end]
                f_kwargs["pixel_values"] = torch.cat([img_patches] * G, dim=0)

            if prompt_mm_token_type_ids is not None:
                p_mm_unpadded = prompt_mm_token_type_ids[i:i + 1][valid_mask].unsqueeze(0)
                f_kwargs["mm_token_type_ids"] = p_mm_unpadded.repeat_interleave(G, dim=0)

            # ── 1. Generate G completions ──
            model.eval()
            with torch.no_grad():
                with autocast_ctx:
                    gen_out = unwrapped_model.generate(
                        input_ids=gen_p_id,
                        attention_mask=gen_attn,
                        max_new_tokens=self.max_completion_length,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        **f_kwargs,
                    )

            comp_ids = gen_out[:, p_len:]
            comp_len = comp_ids.shape[1]
            completion_texts = tokenizer.batch_decode(comp_ids, skip_special_tokens=True)

            # ── 2. Rewards & Advantages ──
            gold_answers = [inputs["correct_answers"][i]] * G
            qtypes = [inputs["question_types"][i]] * G
            rewards = compute_grpo_rewards(
                completions=completion_texts,
                correct_answers=gold_answers,
                question_types=qtypes,
                fmt_weight=0.05,
            )
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            group_mean = rewards_tensor.mean()
            group_std = rewards_tensor.std()
            std_mask = group_std > 1e-6
            adv = (
                (rewards_tensor - group_mean) / (group_std + 1e-4)
                if std_mask
                else torch.zeros_like(rewards_tensor)
            )

            # ── Masks ──
            shift_labels = comp_ids.contiguous()
            pad_mask = (shift_labels != tokenizer.pad_token_id).float()
            eos_id = tokenizer.eos_token_id
            if eos_id is not None and comp_ids.numel() > 0:
                is_eos = (comp_ids == eos_id)
                eos_cumsum = is_eos.cumsum(dim=-1)
                eos_mask = ((eos_cumsum == 0) | ((eos_cumsum == 1) & is_eos)).float()
                comp_mask = pad_mask * eos_mask
            else:
                comp_mask = pad_mask

            full_attn = (gen_out != tokenizer.pad_token_id).long()
            if "mm_token_type_ids" in f_kwargs:
                f_kwargs["mm_token_type_ids"] = torch.cat(
                    [
                        f_kwargs["mm_token_type_ids"],
                        torch.zeros((G, comp_len), dtype=torch.long, device=device),
                    ],
                    dim=1,
                )

            # ── 3. Old policy logprobs (no_grad) ──
            with torch.no_grad():
                with autocast_ctx:
                    old_out = model(
                        input_ids=gen_out,
                        attention_mask=full_attn,
                        logits_to_keep=comp_len + 1,
                        **f_kwargs,
                    )
                old_shift = old_out.logits[:, :-1, :].contiguous()
                old_lp = F.log_softmax(old_shift, dim=-1)
                old_per_token_lp = torch.gather(
                    old_lp, dim=-1, index=shift_labels.unsqueeze(-1)
                ).squeeze(-1) * comp_mask
                del old_out, old_shift, old_lp
            torch.cuda.empty_cache()

            # ── 4. Reference policy logprobs for KL (no_grad, disable_adapter) ──
            ref_per_token_lp = None
            if self.beta > 1e-9:
                with torch.no_grad():
                    if hasattr(model, "disable_adapter"):
                        ctx = model.disable_adapter()
                    else:
                        ctx = nullcontext()
                    with ctx:
                        with autocast_ctx:
                            ref_out = model(
                                input_ids=gen_out,
                                attention_mask=full_attn,
                                logits_to_keep=comp_len + 1,
                                **f_kwargs,
                            )
                        ref_shift = ref_out.logits[:, :-1, :].contiguous()
                        ref_lp = F.log_softmax(ref_shift, dim=-1)
                        ref_per_token_lp = torch.gather(
                            ref_lp, dim=-1, index=shift_labels.unsqueeze(-1)
                        ).squeeze(-1) * comp_mask
                        del ref_out, ref_shift, ref_lp
                torch.cuda.empty_cache()

            # ── 5. Current policy logprobs (with grad) ──
            model.train()
            with autocast_ctx:
                curr_out = model(
                    input_ids=gen_out,
                    attention_mask=full_attn,
                    logits_to_keep=comp_len + 1,
                    **f_kwargs,
                )
                curr_shift = curr_out.logits[:, :-1, :].contiguous()
                curr_lp = F.log_softmax(curr_shift, dim=-1)
                curr_per_token_lp = torch.gather(
                    curr_lp, dim=-1, index=shift_labels.unsqueeze(-1)
                ).squeeze(-1) * comp_mask

                # PPO clipped surrogate
                ratio = torch.exp(curr_per_token_lp - old_per_token_lp)
                adv_exp = adv.unsqueeze(-1)
                surr1 = ratio * adv_exp
                surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv_exp
                per_token_policy_loss = -torch.min(surr1, surr2)

                denom = comp_mask.sum().clamp_min(1.0)
                policy_loss = (per_token_policy_loss * comp_mask).sum() / denom

                if ref_per_token_lp is not None:
                    log_ratio_ref = torch.clamp(
                        ref_per_token_lp - curr_per_token_lp, min=-20, max=20
                    )
                    per_token_kl = (
                        torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
                    ) * comp_mask
                    approx_kl = per_token_kl.sum() / denom
                    prompt_loss = policy_loss + self.beta * approx_kl
                else:
                    approx_kl = torch.tensor(0.0, device=device)
                    prompt_loss = policy_loss

                # Scale by batch_size and gradient accumulation steps
                loss_to_backward = prompt_loss / (batch_size * gas)

            # ── 6. Backward immediately to free prompt graph ──
            self.accelerator.backward(loss_to_backward)

            total_unscaled_loss += prompt_loss.detach().item()

            # Record metrics
            with torch.no_grad():
                valid_ratio = ratio[comp_mask.bool()] if comp_mask.sum() > 0 else ratio.view(-1)
                step_metrics["policy_loss"].append(policy_loss.detach().item())
                step_metrics["approx_kl"].append(approx_kl.detach().item())
                step_metrics["reward_mean"].append(group_mean.detach().item())
                step_metrics["reward_std"].append(group_std.detach().item())
                step_metrics["reward_min"].append(rewards_tensor.min().detach().item())
                step_metrics["reward_max"].append(rewards_tensor.max().detach().item())
                step_metrics["fraction_reward_zero"].append(
                    (rewards_tensor < 0.01).float().mean().detach().item()
                )
                step_metrics["fraction_reward_one"].append(
                    (rewards_tensor > 0.99).float().mean().detach().item()
                )
                step_metrics["zero_std_fraction"].append(0.0 if std_mask else 1.0)
                step_metrics["advantage_mean"].append(adv.mean().detach().item())
                step_metrics["advantage_std"].append(adv.std().detach().item())
                if valid_ratio.numel() > 0:
                    step_metrics["ratio_mean"].append(valid_ratio.mean().detach().item())
                    clipped = (valid_ratio < 1.0 - eps) | (valid_ratio > 1.0 + eps)
                    step_metrics["clip_fraction"].append(clipped.float().mean().detach().item())
                step_metrics["comp_len_mean"].append(comp_mask.sum(dim=-1).float().mean().detach().item())

            # Free prompt memory
            del (
                gen_out,
                comp_ids,
                shift_labels,
                comp_mask,
                full_attn,
                curr_out,
                curr_shift,
                curr_lp,
                curr_per_token_lp,
                ratio,
                old_per_token_lp,
            )
            torch.cuda.empty_cache()

        avg_loss = total_unscaled_loss / batch_size

        # Log metrics to trainer state
        if self.state.global_step % self.args.logging_steps == 0:
            def avg(lst):
                return sum(lst) / max(len(lst), 1)

            self.log(
                {
                    "loss": avg_loss,
                    "grpo_loss": avg(step_metrics["policy_loss"]),
                    "approx_kl": avg(step_metrics["approx_kl"]),
                    "reward_mean": avg(step_metrics["reward_mean"]),
                    "reward_std": avg(step_metrics["reward_std"]),
                    "reward_min": min(step_metrics["reward_min"]) if step_metrics["reward_min"] else 0.0,
                    "reward_max": max(step_metrics["reward_max"]) if step_metrics["reward_max"] else 0.0,
                    "fraction_reward_zero": avg(step_metrics["fraction_reward_zero"]),
                    "fraction_reward_one": avg(step_metrics["fraction_reward_one"]),
                    "zero_std_group_fraction": avg(step_metrics["zero_std_fraction"]),
                    "advantage_mean": avg(step_metrics["advantage_mean"]),
                    "advantage_std": avg(step_metrics["advantage_std"]),
                    "ratio_mean": avg(step_metrics["ratio_mean"]) if step_metrics["ratio_mean"] else 1.0,
                    "clip_fraction": avg(step_metrics["clip_fraction"]) if step_metrics["clip_fraction"] else 0.0,
                    "comp_len_mean": avg(step_metrics["comp_len_mean"]),
                }
            )

        # Return loss normalized by gradient accumulation steps as expected by Trainer
        return torch.tensor(avg_loss / gas, device=device, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Fallback compute_loss delegating scalar evaluation."""
        return torch.tensor(0.0, device=self.args.device, requires_grad=True)

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
