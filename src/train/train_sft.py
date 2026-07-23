import os
import torch
from peft import LoraConfig, get_peft_model
import ast
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from model.load_model import get_qwen_vl_generation_backbone, load_qwen_vl_generation_model
from trainer import QwenSFTTrainer
from dataset import make_supervised_data_module
from params import DataArguments, ModelArguments, TrainingArguments
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
import importlib
import sys

local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    backbone = get_qwen_vl_generation_backbone(model)
    vision_tower = backbone.visual

    # When NOT quantized: cast dtype/device and toggle requires_grad on base weights.
    # When quantized (bits in [4, 8]): BnB already placed weights on device with the
    # right compute dtype; base vision/merger weights are packed as Params4bit (uint8)
    # and CANNOT have requires_grad=True. Forcing them frozen here is safe — LoRA
    # adapters (added later by get_peft_model) are the trainable components for vision.
    if training_args.bits not in [4, 8]:
        vision_tower.to(dtype=compute_dtype, device=device)
        set_requires_grad(backbone.visual.parameters(), not training_args.freeze_vision_tower)
        set_requires_grad(backbone.visual.merger.parameters(), not training_args.freeze_merger)
        if hasattr(backbone.visual, "deepstack_merger_list"):
            set_requires_grad(
                backbone.visual.deepstack_merger_list.parameters(),
                not training_args.freeze_merger,
            )
    else:
        # Quantized: force base vision/merger frozen (Params4bit can't be trained directly).
        set_requires_grad(backbone.visual.parameters(), False)
        set_requires_grad(backbone.visual.merger.parameters(), False)
        if hasattr(backbone.visual, "deepstack_merger_list"):
            set_requires_grad(backbone.visual.deepstack_merger_list.parameters(), False)


def configure_llm(model, training_args):
    backbone = get_qwen_vl_generation_backbone(model)
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = backbone.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def unfreeze_topk_layers(model, k_llm: int = 0, k_vis: int = 0):
    backbone = get_qwen_vl_generation_backbone(model)

    if k_llm and hasattr(backbone, "language_model") and hasattr(backbone.language_model, "layers"):
        for layer in backbone.language_model.layers[-k_llm:]:
            for p in layer.parameters():
                p.requires_grad = True

    if k_vis and hasattr(backbone, "visual") and hasattr(backbone.visual, "blocks"):
        for blk in backbone.visual.blocks[-k_vis:]:
            for p in blk.parameters():
                p.requires_grad = True


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError("You cannot set both `nframes` and `fps` at the same time. Please set only one of them.")

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."

    # Match the reference repo's safety guard: when LoRA is applied to the vision
    # tower, the base vision weights must be frozen. Otherwise `set_requires_grad`
    # would attempt to enable gradients on quantized Params4bit tensors (when bits
    # in [4, 8]), which raises "only Tensors of floating point and complex dtype
    # can require gradients". LoRA adapters are the trainable vision components.
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError(
            "If `vision_lora` is True, `freeze_vision_tower` must also be True. "
            "LoRA adapters train on top of the (frozen) vision tower; unfreezing the "
            "base vision weights conflicts with LoRA and is unsupported under quantization."
        )

    if training_args.lora_namespan_exclude is not None:
        training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
    else:
        training_args.lora_namespan_exclude = []

    if not training_args.vision_lora:
        training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["visual", "lm_head"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            ),
        ))

    model = load_qwen_vl_generation_model(
        model_args.model_id,
        dtype=compute_dtype,
        attn_implementation="sdpa" if training_args.disable_flash_attn2 else "flash_attention_2",
        **bnb_model_from_pretrained_args,
    )

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    unfreeze_topk_layers(
        model_to_configure,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )

    if training_args.gradient_checkpointing:
        if training_args.vision_lora:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        else:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

        model.enable_input_require_grads()

    if training_args.bits in [4, 8]:
        model.config.dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing,
                                                gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude,
                                                     num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            use_dora=training_args.use_dora,
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

        # Re-unfreeze non-LoRA vision/merger weights only when NOT quantized.
        # Under 4/8-bit quantization these are Params4bit and cannot have
        # requires_grad=True; LoRA adapters already provide the gradient path.
        if not training_args.freeze_vision_tower and training_args.bits not in [4, 8]:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True

        if not training_args.freeze_merger and training_args.bits not in [4, 8]:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)

            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(model_id=model_args.model_id,
                                               processor=processor,
                                               data_args=data_args)

    # Load custom compute_metrics if provided via environment variable
    compute_metrics_fn = None
    compute_metrics_path = os.environ.get("SFT_COMPUTE_METRICS", "")
    if compute_metrics_path:
        spec = importlib.util.spec_from_file_location("custom_metrics", compute_metrics_path)
        if spec:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "compute_metrics"):
                compute_metrics_fn = mod.compute_metrics
                rank0_print(f"Loaded custom compute_metrics from {compute_metrics_path}")

    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        compute_metrics=compute_metrics_fn,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    # ── Save training metadata for plotting/tracking ──────────────────
    if local_rank == 0 or local_rank is None or local_rank == -1:
        import json as _json
        from dataclasses import asdict as _asdict

        # Per-step metrics (log_history is already in trainer_state.json,
        # but save a clean standalone copy for easy loading in plotting scripts)
        log_history = getattr(trainer.state, "log_history", [])
        with open(os.path.join(training_args.output_dir, "log_history.json"), "w") as f:
            _json.dump(log_history, f, indent=2, default=float)

        # Training / model / data arguments
        with open(os.path.join(training_args.output_dir, "training_args.json"), "w") as f:
            _json.dump(_asdict(training_args), f, indent=2, default=str)
        with open(os.path.join(training_args.output_dir, "model_args.json"), "w") as f:
            _json.dump(_asdict(model_args), f, indent=2, default=str)
        with open(os.path.join(training_args.output_dir, "data_args.json"), "w") as f:
            _json.dump(_asdict(data_args), f, indent=2, default=str)

        rank0_print(f"Training metadata saved to {training_args.output_dir}")

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
