import re
from src.trainer import GenerativeEvalPrediction


def compute_metrics(eval_pred: GenerativeEvalPrediction):
    """
    Custom evaluation metrics for cataract surgery SFT model.

    This function is loaded automatically when SFT_COMPUTE_METRICS
    environment variable points to this file.
    """
    predictions = eval_pred.predictions
    references = eval_pred.references

    n = len(predictions)
    if n == 0:
        return {}

    # Exact match (for MCQs)
    exact_matches = sum(
        1 for p, r in zip(predictions, references)
        if p.strip().lower() == r.strip().lower()
    )

    # Contains match
    contains_matches = sum(
        1 for p, r in zip(predictions, references)
        if r.strip().lower() in p.strip().lower()
    )

    # Extract answer letter from predictions and references
    def extract_answer(text):
        match = re.search(r'(?:answer\s+is|Answer\s*:)\s*([A-D])', text)
        if match:
            return match.group(1)
        match = re.search(r'\b([A-D])\b', text)
        if match:
            return match.group(1)
        return None

    answer_matches = 0
    for p, r in zip(predictions, references):
        pred_ans = extract_answer(p)
        ref_ans = extract_answer(r)
        if pred_ans and ref_ans and pred_ans == ref_ans:
            answer_matches += 1

    # Reasoning quality
    reasoning_scores = []
    for p in predictions:
        has_reasoning = bool(re.search(
            r'(?:because|since|as|therefore|hence|thus|consequently)',
            p, re.IGNORECASE,
        ))
        reasoning_scores.append(1.0 if has_reasoning else 0.0)

    # Average generation length
    avg_length = sum(len(p.split()) for p in predictions) / n

    return {
        "exact_match": exact_matches / n,
        "contains_match": contains_matches / n,
        "answer_extract_match": answer_matches / n,
        "reasoning_rate": sum(reasoning_scores) / n,
        "avg_gen_length": avg_length,
    }


def extract_answer_from_generation(text: str) -> str:
    """Extract the answer (single letter) from a generated response."""
    match = re.search(r'(?:answer\s+is|Answer\s*:)\s*([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r'\b([A-D])\b', text)
    if match:
        return match.group(1)
    return ""
