import os
from importlib import import_module

__all__ = ["QwenSFTTrainer", "QwenGRPOTrainer", "GenerativeEvalPrediction"]


def __getattr__(name):
    if name in {"QwenSFTTrainer", "GenerativeEvalPrediction"}:
        module = import_module(".sft_trainer", __name__)
        return getattr(module, name)
    if name == "QwenGRPOTrainer":
        module = import_module(".grpo_trainer", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
