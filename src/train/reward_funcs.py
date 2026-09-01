import re
import math
import json
from typing import Any, Dict, List, Optional, Tuple, Union


def extract_json_object(text: str) -> Optional[dict]:
    """
    Extracts the first valid JSON object from model completion text.
    Handles raw JSON, markdown-wrapped JSON (```json ... ```),
    and JSON embedded in conversational output.
    """
    if not isinstance(text, str):
        return None
    text = text.strip()

    # Fast path: entire text is valid JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Markdown code blocks ```json { ... } ``` or ``` { ... } ```
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            data = json.loads(code_block_match.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Find outermost matching curly braces
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        json_str = text[first_brace:last_brace + 1]
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


def normalize_phase(val: Any) -> Optional[str]:
    """
    Normalizes phase identifier to canonical 'P01' - 'P13' format or integer string.
    """
    if val is None:
        return None
    val_str = str(val).strip().upper()
    # Match P01..P13 or P1..P13
    m = re.search(r"P0?([1-9]|1[0-3])\b", val_str)
    if m:
        num = int(m.group(1))
        return f"P{num:02d}"
    return val_str


def score_mcq_reward(completion_text: str, gold_answer: str) -> Tuple[float, float]:
    """
    Evaluates MCQ tasks (step_identification, visual_observation, instrument_identification).
    Prompt instruction requires: {"explanation": "...", "answer": "A|B|C|D"}

    Returns:
        (task_reward, format_reward)
        task_reward: 1.0 if answer matches gold_answer and schema is valid, else 0.0
        format_reward: 1.0 if JSON strictly conforms to expected schema, else 0.0
    """
    gold_answer = str(gold_answer).strip().upper()
    parsed = extract_json_object(completion_text)

    if parsed is None:
        # Fallback regex for task reward if malformed JSON
        match = re.search(r'\b([A-D])\b', completion_text.upper())
        if match and match.group(1) == gold_answer:
            return 0.0, 0.0  # In MCQ, task reward requires valid JSON according to spec
        return 0.0, 0.0

    # Format check: keys should be {"explanation", "answer"}
    keys = set(parsed.keys())
    is_fmt_valid = (keys == {"explanation", "answer"} and isinstance(parsed.get("explanation"), str))
    format_reward = 1.0 if is_fmt_valid else 0.0

    raw_answer = str(parsed.get("answer", "")).strip().upper()
    # If answer contains extra text like "B) Phacoemulsification", take first letter
    m = re.match(r"^([A-D])\b", raw_answer)
    pred_letter = m.group(1) if m else raw_answer

    if is_fmt_valid and pred_letter == gold_answer:
        task_reward = 1.0
    else:
        task_reward = 0.0

    return task_reward, format_reward


def score_boundary_detection(completion_text: str, gold_answer: Dict[str, Any], tau: float = 1.5) -> Tuple[float, float]:
    """
    Evaluates boundary detection.
    Gold: {"timestamp": float}
    Pred JSON: {"explanation": "...", "answer": {"timestamp": float}} or {"timestamp": float}
    Continuous reward: exp(-|t_pred - t_gt| / 1.5)
    """
    try:
        gt_t = float(gold_answer["timestamp"])
    except (KeyError, TypeError, ValueError):
        return 0.0, 0.0

    parsed = extract_json_object(completion_text)
    pred_t = None
    format_reward = 0.0

    if parsed is not None:
        # Check standard answer schema
        ans_obj = parsed.get("answer", parsed)
        if isinstance(ans_obj, dict) and "timestamp" in ans_obj:
            try:
                pred_t = float(ans_obj["timestamp"])
                format_reward = 1.0
            except (TypeError, ValueError):
                pass
        elif "timestamp" in parsed:
            try:
                pred_t = float(parsed["timestamp"])
                format_reward = 1.0
            except (TypeError, ValueError):
                pass

    if pred_t is None:
        # Regex fallback for timestamp
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\b", completion_text)
        if m:
            try:
                pred_t = float(m.group(1))
            except ValueError:
                pass

    if pred_t is None:
        return 0.0, 0.0

    err = abs(pred_t - gt_t)
    task_reward = math.exp(-err / tau)
    return task_reward, format_reward


def score_temporal_localization(completion_text: str, gold_answer: Dict[str, Any]) -> Tuple[float, float]:
    """
    Evaluates temporal localization.
    Gold: {"start": float, "end": float}
    Pred JSON: {"explanation": "...", "answer": {"start": float, "end": float}}
    Reward: tIoU (Intersection over Union)
    """
    try:
        gt_start = float(gold_answer["start"])
        gt_end = float(gold_answer["end"])
    except (KeyError, TypeError, ValueError):
        return 0.0, 0.0

    parsed = extract_json_object(completion_text)
    pred_start, pred_end = None, None
    format_reward = 0.0

    if parsed is not None:
        ans_obj = parsed.get("answer", parsed)
        if isinstance(ans_obj, dict) and "start" in ans_obj and "end" in ans_obj:
            try:
                pred_start = float(ans_obj["start"])
                pred_end = float(ans_obj["end"])
                format_reward = 1.0
            except (TypeError, ValueError):
                pass
        elif "start" in parsed and "end" in parsed:
            try:
                pred_start = float(parsed["start"])
                pred_end = float(parsed["end"])
                format_reward = 1.0
            except (TypeError, ValueError):
                pass

    if pred_start is None or pred_end is None:
        # Regex fallback: find two numbers representing start and end
        nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", completion_text)
        if len(nums) >= 2:
            try:
                pred_start = float(nums[0])
                pred_end = float(nums[1])
            except ValueError:
                return 0.0, 0.0
        else:
            return 0.0, 0.0

    if pred_start > pred_end:
        pred_start, pred_end = pred_end, pred_start

    # Compute tIoU
    inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = (pred_end - pred_start) + (gt_end - gt_start) - inter
    if union <= 0:
        task_reward = 0.0
    else:
        task_reward = inter / union

    return task_reward, format_reward


def score_phase_recognition(completion_text: str, gold_answer: Union[Dict[str, Any], str]) -> Tuple[float, float]:
    """
    Evaluates phase recognition / timestamp_to_phase.
    Gold: {"phase_id": "P0X", "phase_name": "..."} or "P0X"
    Reward: 1.0 if normalized phase matches, else 0.0
    """
    if isinstance(gold_answer, dict):
        gt_phase = normalize_phase(gold_answer.get("phase_id", gold_answer.get("phase", "")))
        gt_name = str(gold_answer.get("phase_name", "")).strip().lower()
    else:
        gt_phase = normalize_phase(gold_answer)
        gt_name = ""

    parsed = extract_json_object(completion_text)
    pred_phase = None
    pred_name = ""
    format_reward = 0.0

    if parsed is not None:
        ans_obj = parsed.get("answer", parsed)
        if isinstance(ans_obj, dict):
            pred_phase = normalize_phase(ans_obj.get("phase_id", ans_obj.get("phase")))
            pred_name = str(ans_obj.get("phase_name", "")).strip().lower()
            if pred_phase is not None:
                format_reward = 1.0
        elif isinstance(ans_obj, str):
            pred_phase = normalize_phase(ans_obj)
            if pred_phase is not None:
                format_reward = 1.0

    if pred_phase is None:
        # Regex search for P01..P13
        pred_phase = normalize_phase(completion_text)

    is_correct = False
    if gt_phase and pred_phase and gt_phase == pred_phase:
        is_correct = True
    elif gt_name and pred_name and (gt_name in pred_name or pred_name in gt_name):
        is_correct = True

    task_reward = 1.0 if is_correct else 0.0
    return task_reward, format_reward


def compute_reward_single(
    completion_text: str,
    correct_answer: Any,
    question_type: str,
    fmt_weight: float = 0.05,
) -> Dict[str, float]:
    """
    Computes total reward, task reward, and format reward for a single completion.
    Total reward = task_reward + fmt_weight * format_reward
    """
    qtype = (question_type or "").strip().lower()

    if qtype in {"step_identification", "visual_observation", "instrument_identification"} or isinstance(correct_answer, str):
        task_r, fmt_r = score_mcq_reward(completion_text, correct_answer)
    elif qtype == "boundary_detection":
        task_r, fmt_r = score_boundary_detection(completion_text, correct_answer)
    elif qtype == "temporal_localization":
        task_r, fmt_r = score_temporal_localization(completion_text, correct_answer)
    elif qtype in {"timestamp_to_phase", "contextual_phase_recognition"}:
        task_r, fmt_r = score_phase_recognition(completion_text, correct_answer)
    else:
        # Generic fallback
        if isinstance(correct_answer, str):
            task_r, fmt_r = score_mcq_reward(completion_text, correct_answer)
        elif isinstance(correct_answer, dict):
            if "timestamp" in correct_answer:
                task_r, fmt_r = score_boundary_detection(completion_text, correct_answer)
            elif "start" in correct_answer and "end" in correct_answer:
                task_r, fmt_r = score_temporal_localization(completion_text, correct_answer)
            elif "phase_id" in correct_answer or "phase" in correct_answer:
                task_r, fmt_r = score_phase_recognition(completion_text, correct_answer)
            else:
                task_r, fmt_r = 0.0, 0.0
        else:
            task_r, fmt_r = 0.0, 0.0

    total_r = task_r + fmt_weight * fmt_r
    return {
        "reward": total_r,
        "task_reward": task_r,
        "format_reward": fmt_r,
    }


def compute_grpo_rewards(
    completions: List[str],
    correct_answers: List[Any],
    question_types: List[str],
    fmt_weight: float = 0.05,
) -> List[float]:
    """
    Computes rewards for a batch of completions.
    """
    rewards = []
    for comp, ans, qtype in zip(completions, correct_answers, question_types):
        res = compute_reward_single(comp, ans, qtype, fmt_weight=fmt_weight)
        rewards.append(res["reward"])
    return rewards
