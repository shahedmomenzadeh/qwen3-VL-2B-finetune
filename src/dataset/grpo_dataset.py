import copy
import os
from typing import Dict, List, Any
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from params import DataArguments
from constants import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
)

from .data_utils import (
    get_image_info,
    get_mm_token_type_ids,
    get_qwen_multimodal_settings,
    get_video_info,
    llava_to_openai,
    pad_sequence,
    use_default_system_message,
)


class GRPODataset(Dataset):
    """
    Dataset for Qwen3-VL GRPO reinforcement learning fine-tuning.
    Formats multi-modal prompts (images, videos, text) and retains gold metadata
    (correct_answer, question_type, reference_reasoning, reward_type) for reward computation.
    """

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id: str,
        padding: bool = True,
    ):
        super(GRPODataset, self).__init__()
        if isinstance(data_path, str):
            with open(data_path, "r", encoding="utf-8") as f:
                list_data_dict = json.load(f)
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.nframes = data_args.nframes

        self.model_type, self.image_patch_size, self.return_video_metadata = get_qwen_multimodal_settings(
            self.model_id
        )

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, Any]:
        sources = self.list_data_dict[i]
        processor = self.processor
        is_video = False

        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            for image_file in image_files:
                if not os.path.exists(image_file) and not image_file.startswith("http"):
                    image_file = os.path.join(image_folder, image_file)
                image_input = get_image_info(
                    image_file,
                    self.image_min_pixel,
                    self.image_max_pixel,
                    self.image_resized_w,
                    self.image_resized_h,
                    self.image_patch_size,
                )
                images.append(image_input)

        elif "video" in sources:
            is_video = True
            images = None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file) and not video_file.startswith("http"):
                    video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(
                    video_file,
                    self.video_min_pixel,
                    self.video_max_pixel,
                    self.video_resized_w,
                    self.video_resized_h,
                    self.data_args.fps,
                    self.data_args.nframes,
                    self.image_patch_size,
                    return_video_metadata=self.return_video_metadata,
                )
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images = None
            videos = None

        conversations = copy.deepcopy(llava_to_openai(sources["conversations"], is_video=is_video))

        all_input_ids = []
        all_mm_token_type_ids = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_grid = []

        image_curr_count = 0
        video_curr_count = 0

        if len(SYSTEM_MESSAGE) > 0 and use_default_system_message(self.model_type):
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_input_ids = processor.tokenizer(
                system_message, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]
            all_input_ids.append(system_input_ids.squeeze(0))
            all_mm_token_type_ids.append(
                torch.zeros_like(system_input_ids, dtype=torch.long).squeeze(0)
            )

        # For GRPO, prompt consists of user prompt formatted with assistant header ready for completion
        for _, j in enumerate(range(0, len(conversations), 2)):
            user_turn = conversations[j]
            user_prompt = (
                f"{DEFAULT_IM_START_TOKEN}{user_turn['role']}\n{user_turn['content']}{DEFAULT_IM_END_TOKEN}\n"
                f"{DEFAULT_IM_START_TOKEN}assistant\n"
            )

            if DEFAULT_IMAGE_TOKEN in user_prompt:
                num_images = user_prompt.count(DEFAULT_IMAGE_TOKEN)
                images_turn = images[image_curr_count : image_curr_count + num_images]
                inputs = processor(
                    text=[user_prompt],
                    images=images_turn,
                    videos=videos,
                    padding=False,
                    do_resize=False,
                    return_tensors="pt",
                )
                prompt_input_ids = inputs["input_ids"]
                prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, prompt_input_ids)
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                image_curr_count += num_images

            elif DEFAULT_VIDEO_TOKEN in user_prompt:
                num_videos = user_prompt.count(DEFAULT_VIDEO_TOKEN)
                videos_turn = videos[video_curr_count : video_curr_count + num_videos]

                if self.model_type == "qwen2_5_vl":
                    inputs = processor(
                        text=[user_prompt],
                        images=images,
                        videos=videos_turn,
                        padding=False,
                        do_resize=False,
                        return_tensors="pt",
                        **video_kwargs,
                    )
                    prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, inputs["input_ids"])
                    all_second_grid.extend(inputs.get("second_per_grid_ts", []))
                elif self.model_type in {"qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"}:
                    video_datas, video_metadatas = zip(*videos_turn)
                    inputs = processor(
                        text=[user_prompt],
                        images=images,
                        videos=list(video_datas),
                        padding=False,
                        do_resize=False,
                        return_tensors="pt",
                        **video_kwargs,
                        video_metadata=list(video_metadatas),
                    )
                    prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, inputs["input_ids"])
                else:
                    inputs = processor(
                        text=[user_prompt],
                        images=images,
                        videos=videos_turn,
                        padding=False,
                        do_resize=False,
                        return_tensors="pt",
                    )
                    prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, inputs["input_ids"])

                prompt_input_ids = inputs["input_ids"]
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                video_curr_count += num_videos

            else:
                prompt_input_ids = processor.tokenizer(
                    user_prompt, add_special_tokens=False, padding=False, return_tensors="pt"
                )["input_ids"]
                prompt_mm_token_type_ids = torch.zeros_like(prompt_input_ids, dtype=torch.long)

            all_input_ids.append(prompt_input_ids.squeeze(0))
            all_mm_token_type_ids.append(prompt_mm_token_type_ids.squeeze(0))

        prompt_input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        prompt_mm_token_type_ids = torch.cat(all_mm_token_type_ids, dim=0).to(torch.long)
        prompt_attention_mask = (prompt_input_ids > -1000000).to(torch.long)

        data_dict = dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_mm_token_type_ids=prompt_mm_token_type_ids,
            correct_answer=sources.get("correct_answer", ""),
            question_type=sources.get("question_type", ""),
            reference_reasoning=sources.get("reference_reasoning", ""),
            reward_type=sources.get("reward_type", "deterministic"),
            sample_id=sources.get("id", f"grpo_{i}"),
        )

        if pixel_key and grid_key and len(all_pixel_values) > 0:
            data_dict[pixel_key] = torch.cat(all_pixel_values, dim=0)
            data_dict[grid_key] = torch.cat(all_image_grid_thw, dim=0)

        if len(all_second_grid) > 0:
            data_dict["second_per_grid_ts"] = all_second_grid

        return data_dict


class DataCollatorForGRPODataset(object):
    """Collate examples for GRPO policy optimization."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_prompt_input_ids = []
        batch_prompt_mm_token_type_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []

        correct_answers = []
        question_types = []
        reference_reasonings = []
        reward_types = []
        sample_ids = []

        for example in examples:
            batch_prompt_input_ids.append(example["prompt_input_ids"])
            batch_prompt_mm_token_type_ids.append(example["prompt_mm_token_type_ids"])

            if "pixel_values_videos" in example:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in example:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            if "second_per_grid_ts" in example:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

            correct_answers.append(example.get("correct_answer", ""))
            question_types.append(example.get("question_type", ""))
            reference_reasonings.append(example.get("reference_reasoning", ""))
            reward_types.append(example.get("reward_type", ""))
            sample_ids.append(example.get("sample_id", ""))

        # Pad prompt sequence to left for causal generation
        prompt_input_ids = pad_sequence(
            batch_prompt_input_ids, padding_side="left", padding_value=self.pad_token_id
        )
        prompt_attention_mask = (prompt_input_ids != self.pad_token_id).to(torch.long)
        prompt_mm_token_type_ids = pad_sequence(
            batch_prompt_mm_token_type_ids, padding_side="left", padding_value=0
        )

        batch = {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "prompt_mm_token_type_ids": prompt_mm_token_type_ids,
            "correct_answers": correct_answers,
            "question_types": question_types,
            "reference_reasonings": reference_reasonings,
            "reward_types": reward_types,
            "sample_ids": sample_ids,
        }

        if len(batch_pixel_video_values) > 0:
            batch["pixel_values_videos"] = torch.cat(batch_pixel_video_values, dim=0)
            batch["video_grid_thw"] = torch.cat(batch_video_thw, dim=0)

        if len(batch_pixel_values) > 0:
            batch["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
            batch["image_grid_thw"] = torch.cat(batch_image_thw, dim=0)

        if len(batch_second_per_grid_ts) > 0:
            batch["second_per_grid_ts"] = batch_second_per_grid_ts

        return batch


def make_grpo_data_module(model_id: str, processor: transformers.ProcessorMixin, data_args: DataArguments) -> Dict[str, Any]:
    """Create GRPO training and evaluation datasets and data collator."""
    train_dataset = GRPODataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        model_id=model_id,
    )

    eval_dataset = None
    if data_args.eval_path is not None:
        eval_dataset = GRPODataset(
            data_path=data_args.eval_path,
            processor=processor,
            data_args=data_args,
            model_id=model_id,
        )

    data_collator = DataCollatorForGRPODataset(pad_token_id=processor.tokenizer.pad_token_id)

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
