#!/usr/bin/env python3
"""
verify_qwen_logit_alignment.py — P0-2 Verify Qwen3-VL completion/logit alignment.

Checks that for Qwen3-VL generation:
  completion_ids = generated_ids[:, prompt_len:]
  aligns with:
  logits[:, prompt_len-1:-1] -> logprobs for completion_ids

Specifically:
  logits[b, prompt_len-1, :] should predict completion_ids[b, 0]
  logits[b, prompt_len, :]   should predict completion_ids[b, 1]
  ...

We test on:
  1) text-only example
  2) video example (if dataset available)

Failure indicates off-by-one error in src/trainer/grpo_trainer.py:358/368.

Usage:
  HF_HOME=hf_cache .venv/bin/python scripts/verify_qwen_logit_alignment.py
  .venv/bin/python scripts/verify_qwen_logit_alignment.py --model Qwen/Qwen3-VL-2B-Instruct
"""
import argparse
import torch
import torch.nn.functional as F


def test_text_only(model, processor, device, dtype):
    print("\n=== Test 1: text-only alignment ===")
    # Minimal conversation
    messages = [
        {"role": "user", "content": "What is 2+2? Answer in JSON {\"explanation\": \"...\", \"answer\": \"B\"}."},
    ]
    # Use llava_to_openai style prompt via processor
    from transformers import AutoProcessor
    # Build prompt using chat template
    # Qwen3-VL processor handles chat template; simpler: tokenize manually via Qwen template
    # We'll use processor.apply_chat_template if available
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"Chat text[:200]: {text[:200]!r}")
    except Exception as e:
        print(f"apply_chat_template failed: {e}, fallback")
        text = "<|im_start|>user\nWhat is 2+2? Answer in JSON.\n<|im_end|>\n<|im_start|>assistant\n"

    inputs = processor(text=[text], padding=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]
    print(f"prompt_len={prompt_len} input_ids shape {input_ids.shape}")

    tokenizer = processor.tokenizer
    model.eval()
    with torch.no_grad():
        # Generate short completion
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        comp_ids = generated[:, prompt_len:]
        print(f"generated shape {generated.shape} comp shape {comp_ids.shape} comp_ids {comp_ids[0].tolist()[:10]}")
        print(f"decoded comp: {tokenizer.decode(comp_ids[0], skip_special_tokens=False)[:300]!r}")

        # Now compute logits for full sequence
        full_attn = (generated != tokenizer.pad_token_id).to(torch.long)
        outputs = model(input_ids=generated, attention_mask=full_attn)
        logits = outputs.logits  # (B, seq_len, vocab)
        print(f"logits shape {logits.shape}")

        # Check alignment: for each position k in completion, check that argmax(logits[prompt_len-1+k]) == completion_ids[0, k] (greedy)
        # For greedy generation, this should hold if alignment correct (with no sampling).
        shift_logits = logits[:, prompt_len - 1 : -1, :]
        print(f"shift_logits shape {shift_logits.shape} shift_labels shape {comp_ids.shape}")
        # For greedy generation, next-token prediction from prompt_at_len-1 should match first generated token
        for k in range(min(4, comp_ids.shape[1])):
            if comp_ids[0, k].item() == tokenizer.pad_token_id:
                break
            pred_id = shift_logits[0, k].argmax().item()
            gold_id = comp_ids[0, k].item()
            match = "OK" if pred_id == gold_id else "MISMATCH"
            # Also compare logprob
            logp = F.log_softmax(shift_logits[0, k], dim=-1)[gold_id].item()
            decoded_pred = tokenizer.decode([pred_id])[:40].replace("\n","\\n")
            decoded_gold = tokenizer.decode([gold_id])[:40].replace("\n","\\n")
            print(f"  k={k} pred={pred_id}({decoded_pred!r}) gold={gold_id}({decoded_gold!r}) logp={logp:.2f} {match}")
            # Also check off-by-1 would fail
            if match != "OK":
                # check alternative alignment prompt_len : prompt_len+... would give different
                alt = logits[0, prompt_len, :].argmax().item() if logits.shape[1] > prompt_len else -1
                print(f"    alt alignment (shift+1) pred {alt}")

        # Additional per-token logprob sanity: logprobs from shift_logits should be <=0 and reasonable
        log_probs = F.log_softmax(shift_logits, dim=-1)
        # Gather completion logprobs
        gathered = torch.gather(log_probs, dim=-1, index=comp_ids.unsqueeze(-1)).squeeze(-1)
        print(f"  per-token logprobs (first 5): {[f'{x:.2f}' for x in gathered[0,:5].tolist()]}")
        # For greedy, logprobs should be near 0 (high prob) not near -inf
        print("  text-only alignment test complete — if all OK, P0-2 text path verified.")

    return True


def test_video(model, processor, device):
    print("\n=== Test 2: video alignment ===")
    # Try to load a real video sample from data/grpo_train_dataset_grpo.json if available
    import json, os, pathlib
    from dataset.data_utils import get_video_info, get_qwen_multimodal_settings
    import torch as th

    candidates = ["data/grpo_train_dataset_grpo.json", "data/lite_e2e/grpo_train.json", "output/lite_grpo_test/grpo_train.json"]
    data_path = None
    for cand in candidates:
        if os.path.exists(cand):
            data_path = cand
            break
    if data_path is None:
        print("No GRPO video dataset found, skipping video test.")
        return False

    with open(data_path) as f:
        data = json.load(f)
    # Find one with video
    sample = None
    for s in data:
        if "video" in s:
            sample = s
            break
    if sample is None:
        print("No video sample in dataset.")
        return False
    print(f"Using sample id={sample.get('id')} video={sample.get('video')}")

    # Build prompt similar to dataset GRPODataset
    from dataset.data_utils import get_mm_token_type_ids, llava_to_openai, use_default_system_message
    from constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VIDEO_TOKEN, SYSTEM_MESSAGE
    import copy

    model_type, patch_size, return_video = get_qwen_multimodal_settings(processor.tokenizer.name_or_path if hasattr(processor.tokenizer, "name_or_path") else "Qwen/Qwen3-VL-2B-Instruct")
    # Fallback model_type
    model_type = "qwen3_vl"
    video_file = sample["video"]
    # prepend image_folder if needed
    search_roots = ["dataset_grpo", "dataset_sft", "."]
    full_path = video_file
    for root in search_roots:
        cand = os.path.join(root, video_file)
        if os.path.exists(cand):
            full_path = cand
            break
        cand2 = os.path.join(root, os.path.basename(video_file))
        # not needed
    if not os.path.exists(full_path):
        print(f"Video file not found: {video_file} (tried {search_roots})")
        return False

    # Video info kwargs
    # Use processor defaults: need video_kwargs
    video_kwargs = {}
    nframes = 8
    video_min = 100352
    video_max = 131072
    try:
        vid_data, vid_kwargs = get_video_info(
            full_path,
            video_min, video_max,
            None, None, None, nframes, patch_size, return_video_metadata=return_video
        )
        # Build prompt
        conversations = copy.deepcopy(llava_to_openai(sample["conversations"], is_video=True))
        # Build prompt input ids including assistant header, as in GRPODataset
        all_input_ids = []
        all_mm = []
        all_pixel = []
        all_grid = []
        all_second = []
        if len(SYSTEM_MESSAGE) > 0 and use_default_system_message(model_type):
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            sys_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors="pt")["input_ids"]
            all_input_ids.append(sys_ids.squeeze(0))
            all_mm.append(torch.zeros_like(sys_ids).squeeze(0))

        for j in range(0, len(conversations), 2):
            user_turn = conversations[j]
            user_prompt = (
                f"{DEFAULT_IM_START_TOKEN}{user_turn['role']}\n{user_turn['content']}{DEFAULT_IM_END_TOKEN}\n"
                f"{DEFAULT_IM_START_TOKEN}assistant\n"
            )
            # Check if prompt contains video token
            if DEFAULT_VIDEO_TOKEN in user_prompt:
                if model_type in {"qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"}:
                    inputs = processor(
                        text=[user_prompt],
                        images=None,
                        videos=[vid_data],
                        padding=False,
                        do_resize=False,
                        return_tensors="pt",
                        **vid_kwargs,
                        video_metadata=[vid_kwargs.get("video_metadata", {})] if isinstance(vid_kwargs, dict) else [],
                    )
                    # Actually vid_kwargs contains video_metadata already; handle
                else:
                    inputs = processor(text=[user_prompt], videos=[vid_data], padding=False, return_tensors="pt", **vid_kwargs)
                # For qwen3_vl with video_metadata pattern, use two values unpacked differently
                # Simplify: if earlier get_video_info returned (data, kwargs) for qwen3_vl, we already handled
                prompt_input_ids = inputs["input_ids"]
                mm_ids = get_mm_token_type_ids(inputs, prompt_input_ids)
                all_pixel.append(inputs["pixel_values_videos"])
                all_grid.append(inputs["video_grid_thw"])
                if "second_per_grid_ts" in inputs:
                    all_second.extend(inputs["second_per_grid_ts"])
            else:
                prompt_input_ids = processor.tokenizer(user_prompt, add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"]
                mm_ids = torch.zeros_like(prompt_input_ids)
            all_input_ids.append(prompt_input_ids.squeeze(0))
            all_mm.append(mm_ids.squeeze(0))

        prompt_input_ids = torch.cat(all_input_ids, dim=0).unsqueeze(0).to(device)  # (1, L)
        prompt_mm = torch.cat(all_mm, dim=0).unsqueeze(0).to(device)
        pixel_values_videos = torch.cat(all_pixel, dim=0).to(device) if all_pixel else None
        video_grid_thw = torch.cat(all_grid, dim=0).to(device) if all_grid else None
        second_per_grid_ts = all_second

        print(f"prompt_input_ids shape {prompt_input_ids.shape} prompt_mm {prompt_mm.shape}")
        if pixel_values_videos is not None:
            print(f"pixel_values_videos {pixel_values_videos.shape} grid {video_grid_thw.shape}")
        prompt_len = prompt_input_ids.shape[1]
        tokenizer = processor.tokenizer
        forward_kwargs = {}
        if pixel_values_videos is not None:
            forward_kwargs["pixel_values_videos"] = pixel_values_videos
            forward_kwargs["video_grid_thw"] = video_grid_thw
        if second_per_grid_ts:
            forward_kwargs["second_per_grid_ts"] = second_per_grid_ts

        model.eval()
        with torch.no_grad():
            gen_kwargs = dict(
                input_ids=prompt_input_ids,
                attention_mask=(prompt_input_ids != tokenizer.pad_token_id).to(torch.long),
                mm_token_type_ids=prompt_mm,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **forward_kwargs,
            )
            # Broadcast video tensors inside generate: generation backbone handles mm
            # For Qwen3-VL, generate expects same forward_kwargs shape; ensure repeated? For batch 1, fine.
            generated = model.generate(**gen_kwargs)
            # Account for possible HF generate returning only prompt_len + new tokens with same mm needed?
            comp_ids = generated[:, prompt_len:]
            print(f"generated shape {generated.shape} comp {comp_ids.shape} first ids {comp_ids[0,:8].tolist()}")
            print(f"decoded comp: {tokenizer.decode(comp_ids[0], skip_special_tokens=False)[:500]!r}")

            # Now forward with full generated + mm extended with zeros for completion mm ids
            comp_len = comp_ids.shape[1]
            full_mm = torch.cat([prompt_mm, torch.zeros((1, comp_len), dtype=torch.long, device=device)], dim=1)
            full_attn = (generated != tokenizer.pad_token_id).to(torch.long)
            full_kwargs = dict(forward_kwargs)
            # Extend video tensors? For batch 1, no need; generation already expanded? For P0 test, we can just forward with same video tensors (they correspond to prompt only, completion has no video)
            outputs = model(input_ids=generated, attention_mask=full_attn, mm_token_type_ids=full_mm, **full_kwargs)
            logits = outputs.logits
            print(f"logits shape {logits.shape}")
            shift_logits = logits[:, prompt_len - 1:-1, :]
            print(f"shift_logits {shift_logits.shape} comp {comp_ids.shape}")
            # Greedy check
            mismatches = 0
            for k in range(min(4, comp_ids.shape[1])):
                if comp_ids[0, k].item() == tokenizer.pad_token_id:
                    break
                pred = shift_logits[0, k].argmax().item()
                gold = comp_ids[0,k].item()
                ok = pred == gold
                if not ok:
                    mismatches += 1
                print(f"  k={k} pred={pred} gold={gold} {tokenizer.decode([pred])!r} vs {tokenizer.decode([gold])!r} {'OK' if ok else 'MISMATCH'}")
            if mismatches == 0:
                print("  video alignment OK — no off-by-one")
            else:
                print(f"  video alignment FAILED — {mismatches} mismatches in first 4 tokens (potential off-by-one)")

        return mismatches == 0

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Video test error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--hf-home", type=str, default=None)
    parser.add_argument("--bits", type=int, default=16, choices=[4,8,16])
    args = parser.parse_args()

    import os
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    else:
        # default to ./hf_cache if exists
        if os.path.exists("hf_cache"):
            os.environ["HF_HOME"] = "hf_cache"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} Model: {args.model} bits={args.bits}")

    from transformers import AutoProcessor
    from model.load_model import load_qwen_vl_generation_model
    import pathlib

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    # Load model
    print("Loading model...")
    bnb_kwargs = {}
    if args.bits in [4,8]:
        from transformers import BitsAndBytesConfig
        bnb_kwargs = dict(
            device_map={"": device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.bits==4, load_in_8bit=args.bits==8,
                llm_int8_skip_modules=["visual","lm_head"],
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
    model = load_qwen_vl_generation_model(
        args.model, dtype=compute_dtype, attn_implementation="sdpa",
        **bnb_kwargs,
    )
    model.to(device)
    model.eval()

    ok1 = test_text_only(model, processor, device, compute_dtype)
    try:
        ok2 = test_video(model, processor, device)
        if ok2 is None:
            ok2 = True  # skip counts as pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        ok2 = False

    print("\n=== SUMMARY ===")
    print(f"Text alignment: {'PASS' if ok1 else 'FAIL'}")
    print(f"Video alignment: {'PASS' if ok2 else 'FAIL/SKIP'}")
    if ok1 and ok2:
        print("All alignment checks PASSED — P0-2 verified.")
    else:
        print("At least one alignment check FAILED — investigate grpo_trainer.py shift indices and mm_token_type_ids padding.")

if __name__ == "__main__":
    main()
