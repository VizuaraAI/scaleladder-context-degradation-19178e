"""Scoring and degradation metrics.

Three things live here:

  1. answer extraction and exact match / token F1
  2. the retrieval probe, which asks whether the model reproduced the
     answer-bearing sentence before answering
  3. the two degradation scores that the whole project turns on

The second degradation score, D_hn, is the contribution of this study. Prior
work divides the accuracy drop by the short-context score, which is a number
that differs by tens of points across a scale ladder. Dividing by the available
headroom instead makes the comparison between a 0.5B model and a 32B model
mean something.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

ANSWER_RE = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)
EVIDENCE_RE = re.compile(r"EVIDENCE:\s*(.+)", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def normalise(text: str) -> str:
    """Lowercase, strip articles, punctuation and extra whitespace (SQuAD style)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_answer(output: str, task: str) -> str:
    """Pull the answer out of a generation, tolerating chatty models."""
    match = ANSWER_RE.search(output)
    if match:
        candidate = match.group(1).strip()
    else:
        lines = output.strip().splitlines()
        candidate = lines[0].strip() if lines else ""
    if task == "math_wordproblems":
        numbers = NUMBER_RE.findall(candidate) or NUMBER_RE.findall(output)
        return numbers[-1].replace(",", "") if numbers else candidate
    return candidate


def extract_evidence(output: str) -> str:
    match = EVIDENCE_RE.search(output)
    return match.group(1).strip() if match else ""


def exact_match(prediction: str, gold: str, task: str) -> float:
    if task == "math_wordproblems":
        try:
            return float(abs(float(prediction) - float(gold)) < 1e-6)
        except (TypeError, ValueError):
            return 0.0
    return float(normalise(prediction) == normalise(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalise(prediction).split()
    gold_tokens = normalise(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    overlap = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def retrieval_hit(output: str, evidence: list[str], threshold: float = 0.6) -> float:
    """Did the model reproduce an answer-bearing sentence?

    Scored with token F1 against the closest gold evidence sentence, so a
    paraphrase still counts. Returns 0.0 when the model produced no EVIDENCE
    line at all, which is itself an informative failure.
    """
    recited = extract_evidence(output)
    if not recited or not evidence:
        return 0.0
    best = max(token_f1(recited, sentence) for sentence in evidence)
    return float(best >= threshold)


def score_record(output: str, gold: str, task: str, evidence: list[str]) -> dict[str, float]:
    prediction = extract_answer(output, task)
    return {
        "em": exact_match(prediction, gold, task),
        "f1": token_f1(prediction, gold),
        "retrieval_hit": retrieval_hit(output, evidence),
    }


# --------------------------------------------------------------------- degradation


@dataclass
class Degradation:
    short_accuracy: float
    long_accuracy: float
    floor_accuracy: float
    d_rel: float | None
    d_hn: float | None
    headroom: float


def degradation(
    short_accuracy: float,
    long_accuracy: float,
    floor_accuracy: float = 0.0,
    min_headroom: float = 5.0,
) -> Degradation:
    """Both degradation scores, in percentage points of the input scale.

    Accuracies are expected on a 0-100 scale. `d_rel` is the quantity reported
    by prior work. `d_hn` divides by the headroom (short-context score minus the
    measured no-evidence floor) and is set to None when there is too little
    headroom for the ratio to be stable.
    """
    drop = short_accuracy - long_accuracy
    d_rel = 100.0 * drop / short_accuracy if short_accuracy > 0 else None
    headroom = short_accuracy - floor_accuracy
    d_hn = 100.0 * drop / headroom if headroom >= min_headroom else None
    return Degradation(
        short_accuracy=short_accuracy,
        long_accuracy=long_accuracy,
        floor_accuracy=floor_accuracy,
        d_rel=d_rel,
        d_hn=d_hn,
        headroom=headroom,
    )
