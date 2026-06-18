from dataclasses import dataclass, field
from typing import Optional

try:
    from accelerate.utils import ParallelismConfig as _PC
except Exception:
    class _PC:
        pass

import transformers.training_args as _ta
if not hasattr(_ta, "ParallelismConfig"):
    _ta.ParallelismConfig = _PC

from transformers import TrainingArguments as HFTrainingArguments
from trl import GRPOConfig as GRPOConfigTRL


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen3-VL-2B-Instruct")


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    unfreeze_topk_llm: int = 0
    unfreeze_topk_vision: int = 0

    max_seq_length: int = field(default=32768)

    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)

    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    head_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    use_liger_kernel: bool = True

    generation_max_new_tokens: int = field(default=512)


@dataclass
class GRPOArguments(GRPOConfigTRL):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    unfreeze_topk_llm: int = 0
    unfreeze_topk_vision: int = 0

    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)

    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1

    beta: float = field(default=0.04)
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    max_completion_length: int = 256
    max_prompt_length: int = 512
    use_liger_loss: bool = True
    liger_grpo_loss_type: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    eval_path: str = field(default=None)
    eval_image_folder: Optional[str] = field(default=None)
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    video_resized_width: int = field(default=None)
    video_resized_height: int = field(default=None)
    fps: Optional[int] = field(default=None)
    nframes: Optional[int] = field(default=None)
    enable_reasoning: bool = field(default=False)
