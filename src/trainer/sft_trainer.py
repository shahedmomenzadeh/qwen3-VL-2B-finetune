import os
import torch
import torch.nn as nn
from typing import Optional, List, Union, Dict, Any
from dataclasses import dataclass

from transformers import Trainer, GenerationConfig
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy,
    has_length,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_utils import EvalLoopOutput
from torch.utils.data import DataLoader
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3

from constants import IGNORE_INDEX


_DS_AVAILABLE = False
try:
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    _DS_AVAILABLE = True
except Exception:
    pass


def maybe_zero_3(param, ignore_status=False, name=None):
    if _DS_AVAILABLE and hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


@dataclass
class GenerativeEvalPrediction:
    predictions: List[str]
    references: List[str]


class QwenSFTTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(QwenSFTTrainer, self).__init__(*args, **kwargs)

    def create_optimizer(self):
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
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters

                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

                if visual_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )

                if merger_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)

        if not self.args.lora_enable:
            return

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        non_lora = get_peft_state_non_lora_maybe_zero_3(
            self.model.named_parameters(),
            require_grad_only=True,
        )

        if self.args.should_save:
            torch.save(non_lora, os.path.join(output_dir, "non_lora_state_dict.bin"))
            self.model.base_model.config.to_json_file(os.path.join(output_dir, "config.json"))

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.get("labels") if "labels" in inputs else None

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else None
            logits = outputs.logits if hasattr(outputs, "logits") else None

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)

    def _extract_prompt_and_reference(self, input_ids, labels, tokenizer):
        label_mask = labels != IGNORE_INDEX

        if label_mask.any():
            answer_start_idx = label_mask.nonzero(as_tuple=True)[0][0].item()
        else:
            answer_start_idx = len(input_ids)

        prompt_ids = input_ids[:answer_start_idx]
        answer_ids = labels[label_mask]
        reference_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

        return prompt_ids, reference_text

    def _prepare_generation_inputs(self, batch_prompt_ids, original_inputs, tokenizer, device):
        batch_size = len(batch_prompt_ids)

        max_prompt_len = max(p.shape[0] for p in batch_prompt_ids)

        padded_prompts = torch.full(
            (batch_size, max_prompt_len),
            tokenizer.pad_token_id,
            dtype=batch_prompt_ids[0].dtype,
            device=device,
        )
        attention_masks = torch.zeros(
            (batch_size, max_prompt_len),
            dtype=torch.long,
            device=device,
        )

        for i, prompt in enumerate(batch_prompt_ids):
            prompt_len = len(prompt)
            padded_prompts[i, :prompt_len] = prompt
            attention_masks[i, :prompt_len] = 1

        gen_inputs = {
            "input_ids": padded_prompts,
            "attention_mask": attention_masks,
        }

        if "mm_token_type_ids" in original_inputs:
            padded_mm_token_type_ids = torch.zeros(
                (batch_size, max_prompt_len),
                dtype=original_inputs["mm_token_type_ids"].dtype,
                device=device,
            )
            for i, prompt in enumerate(batch_prompt_ids):
                prompt_len = len(prompt)
                padded_mm_token_type_ids[i, :prompt_len] = original_inputs["mm_token_type_ids"][i, :prompt_len]
            gen_inputs["mm_token_type_ids"] = padded_mm_token_type_ids

        if "pixel_values" in original_inputs:
            gen_inputs["pixel_values"] = original_inputs["pixel_values"]
        if "image_grid_thw" in original_inputs:
            gen_inputs["image_grid_thw"] = original_inputs["image_grid_thw"]
        if "pixel_values_videos" in original_inputs:
            gen_inputs["pixel_values_videos"] = original_inputs["pixel_values_videos"]
        if "video_grid_thw" in original_inputs:
            gen_inputs["video_grid_thw"] = original_inputs["video_grid_thw"]
        if "second_per_grid_ts" in original_inputs:
            gen_inputs["second_per_grid_ts"] = original_inputs["second_per_grid_ts"]

        return gen_inputs

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        args = self.args

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        if prediction_loss_only or self.compute_metrics is None:
            return super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only,
                ignore_keys,
                metric_key_prefix,
            )

        logger.info(f"\n***** Running {description} (Generation Mode) *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        logger.info(f"  Batch size = {self.args.eval_batch_size}")

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        tokenizer = self.processing_class.tokenizer

        generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=getattr(args, 'generation_max_new_tokens', 512),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        unwrapped_model = self.accelerator.unwrap_model(model)

        all_predictions = []
        all_references = []
        all_losses = []

        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)

            batch_input_ids = inputs["input_ids"]
            batch_labels = inputs["labels"]
            batch_size = batch_input_ids.shape[0]

            with torch.no_grad():
                outputs = model(**inputs)
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss.detach()
                    loss = self.accelerator.gather(loss.repeat(batch_size))
                    all_losses.append(loss.cpu())

            batch_prompt_ids = []
            batch_references = []

            for i in range(batch_size):
                prompt_ids, reference_text = self._extract_prompt_and_reference(
                    batch_input_ids[i],
                    batch_labels[i],
                    tokenizer,
                )
                batch_prompt_ids.append(prompt_ids)
                batch_references.append(reference_text)

            gen_inputs = self._prepare_generation_inputs(
                batch_prompt_ids,
                inputs,
                tokenizer,
                batch_input_ids.device,
            )

            with torch.no_grad():
                generated_ids = unwrapped_model.generate(
                    **gen_inputs,
                    generation_config=generation_config,
                )

            for i in range(batch_size):
                prompt_len = len(batch_prompt_ids[i])
                new_tokens = generated_ids[i][prompt_len:]
                pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                all_predictions.append(pred_text)

            all_references.extend(batch_references)

            if step % 10 == 0:
                logger.info(f"  Eval step {step}/{len(dataloader)}")

        if self.args.world_size > 1:
            all_predictions = self._gather_predictions(all_predictions)
            all_references = self._gather_predictions(all_references)

        eval_prediction = GenerativeEvalPrediction(
            predictions=all_predictions,
            references=all_references,
        )

        metrics = self.compute_metrics(eval_prediction)

        if all_losses:
            avg_loss = torch.cat(all_losses).mean().item()
            metrics[f"{metric_key_prefix}_loss"] = avg_loss

        metrics = {
            f"{metric_key_prefix}_{k}" if not k.startswith(metric_key_prefix) else k: v
            for k, v in metrics.items()
        }

        self.log(metrics)

        return EvalLoopOutput(
            predictions=all_predictions,
            label_ids=all_references,
            metrics=metrics,
            num_samples=len(all_predictions),
        )

    def _gather_predictions(self, predictions: List[str]) -> List[str]:
        import torch.distributed as dist

        if not dist.is_initialized():
            return predictions

        world_size = dist.get_world_size()

        gathered = [None] * world_size
        dist.all_gather_object(gathered, predictions)

        all_predictions = []
        for preds in gathered:
            all_predictions.extend(preds)

        return all_predictions
