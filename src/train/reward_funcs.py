import re


def deterministic_reward(completions, correct_answer, **kwargs):
    """Reward function for deterministic answer matching (step_identification, instrument_identification, sequence_ordering).

    Matches:
    - "Therefore the answer is B"
    - "Answer: A"
    - "E, A, B, C, D" (for sequence ordering)
    - "The answer is B" -> extracts 'B'
    """
    rewards = []
    for completion, answer in zip(completions, correct_answer):
        if not answer:
            rewards.append(0.0)
            continue

        answer = answer.strip()
        completion_lower = completion.strip().lower()

        # Try to extract answer after "answer is" or "Answer:"
        answer_patterns = [
            r'(?:answer\s+is)\s*([A-D](?:\s*,\s*[A-D])*)',
            r'answer\s*:\s*([A-D](?:\s*,\s*[A-D])*)',
            r'\b([A-D](?:\s*,\s*[A-D])*)\b',
        ]

        pred = ""
        for pattern in answer_patterns:
            match = re.search(pattern, completion_lower)
            if match:
                candidate = match.group(1).replace(" ", "")
                # Validate that extracted answer matches expected format
                if len(candidate) <= len(answer.replace(" ", "")):
                    pred = candidate
                    break

        if not pred:
            # Fallback: check if answer appears anywhere in completion
            if answer.lower() in completion_lower:
                pred = answer.replace(" ", "")

        rewards.append(1.0 if pred == answer.replace(" ", "") else 0.0)

    return rewards


def llm_judge_reward(completions, reference_reasoning, **kwargs):
    """Reward function that checks if the completion contains key reasoning elements from the reference.

    This is a heuristic-based approach. For production, consider using a teacher LLM.
    """
    rewards = []
    for completion, reference in zip(completions, reference_reasoning):
        if not reference:
            rewards.append(0.0)
            continue

        completion_lower = completion.strip().lower()
        reference_lower = reference.strip().lower()

        # Check for key phrases from reference reasoning
        key_phrases = reference_lower.split('.')
        key_phrases = [p.strip() for p in key_phrases if len(p.strip()) > 20]

        if not key_phrases:
            rewards.append(0.5)  # No reference to compare
            continue

        # Score based on how many key reasoning concepts are mentioned
        matched = sum(1 for phrase in key_phrases if phrase[:30] in completion_lower)
        score = matched / len(key_phrases)

        # Bonus for having reasoning structure
        has_reasoning = bool(re.search(
            r'(?:because|since|as|therefore|hence|thus|consequently|this\s+(?:means|indicates|shows|suggests))',
            completion_lower,
        ))
        if has_reasoning:
            score = min(1.0, score + 0.3)

        rewards.append(score)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has proper reasoning format."""
    rewards = []
    for completion in completions:
        completion_lower = completion.strip().lower()

        # Check for reasoning indicators followed by answer
        has_reasoning = bool(re.search(
            r'(?:because|since|as|therefore|hence|thus|consequently|the\s+answer\s+is)',
            completion_lower,
        ))

        # Check for answer letter
        has_answer = bool(re.search(r'\b[A-D]\b', completion))

        rewards.append(1.0 if (has_reasoning and has_answer) else 0.0)

    return rewards
