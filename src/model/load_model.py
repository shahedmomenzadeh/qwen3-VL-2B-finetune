from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoModelForImageTextToText, PretrainedConfig

_GENERATION_MODEL_TYPES = {
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_vl",
    "qwen3_vl_moe",
}


def get_qwen_vl_generation_backbone(model):
    if not hasattr(model, "model"):
        raise TypeError(f"Unsupported generation model wrapper: {type(model)!r}")
    return model.model


def load_qwen_vl_generation_model(
    model_name_or_path: str, *, config: PretrainedConfig | None = None, **kwargs: Any
):
    if config is None:
        config = AutoConfig.from_pretrained(model_name_or_path)
    if config.model_type not in _GENERATION_MODEL_TYPES:
        supported = ", ".join(sorted(_GENERATION_MODEL_TYPES))
        raise ValueError(
            f"Unsupported Qwen-VL generation model_type: {config.model_type}. Supported: {supported}"
        )

    return AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        config=config,
        **kwargs,
    )
