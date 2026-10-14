"""Task loading.

Two tasks, both chosen because the answer depends on a short, identifiable
evidence span. That property is what lets the harness pad the prompt to any
length while keeping the underlying question exactly the same, and what lets
metrics.py run a retrieval probe afterwards.

Datasets are downloaded from the Hugging Face Hub on first use. Nothing large
is committed to this repository.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset


@dataclass
class Example:
    """A single task instance, independent of any context-length condition."""

    uid: str
    task: str
    question: str
    answer: str
    evidence: list[str]            # sentences that must be present to answer
    distractors: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def _hotpot_sentences(example: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Split HotpotQA's distractor context into supporting and non-supporting text."""
    context = example["context"]
    titles = list(context["title"])
    sentence_lists = list(context["sentences"])

    supporting = set(
        zip(example["supporting_facts"]["title"], example["supporting_facts"]["sent_id"])
    )

    evidence: list[str] = []
    distractors: list[str] = []
    for title, sentences in zip(titles, sentence_lists):
        for idx, sentence in enumerate(sentences):
            text = sentence.strip()
            if not text:
                continue
            if (title, idx) in supporting:
                evidence.append(f"{title}: {text}")
            else:
                distractors.append(f"{title}: {text}")
    return evidence, distractors


def load_multihop_qa(n_examples: int, seed: int = 1234) -> list[Example]:
    """HotpotQA distractor split: two-hop questions with gold supporting sentences."""
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n_examples * 3, len(dataset)))

    examples: list[Example] = []
    for i in indices:
        row = dataset[i]
        evidence, distractors = _hotpot_sentences(row)
        # Keep only clean two-hop items: at least two evidence sentences and a
        # short extractive answer, so exact match is a fair metric.
        if len(evidence) < 2 or len(row["answer"]) > 60:
            continue
        examples.append(
            Example(
                uid=f"hotpot-{row['id']}",
                task="multihop_qa",
                question=row["question"].strip(),
                answer=row["answer"].strip(),
                evidence=evidence,
                distractors=distractors,
                meta={"level": row.get("level", "unknown")},
            )
        )
        if len(examples) >= n_examples:
            break
    return examples


def load_math_wordproblems(n_examples: int, seed: int = 1234) -> list[Example]:
    """GSM8K: grade-school math. The problem statement is the evidence span."""
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n_examples, len(dataset)))

    examples: list[Example] = []
    for i in indices:
        row = dataset[i]
        answer = row["answer"].split("####")[-1].strip()
        examples.append(
            Example(
                uid=f"gsm8k-{i}",
                task="math_wordproblems",
                question=row["question"].strip(),
                answer=answer,
                evidence=[row["question"].strip()],
                distractors=[],
                meta={"rationale": row["answer"]},
            )
        )
    return examples


LOADERS = {
    "multihop_qa": load_multihop_qa,
    "math_wordproblems": load_math_wordproblems,
}


def load_task(name: str, n_examples: int, seed: int = 1234) -> list[Example]:
    if name not in LOADERS:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(LOADERS)}")
    return LOADERS[name](n_examples=n_examples, seed=seed)


def filler_corpus(n_sentences: int = 4000, seed: int = 1234) -> list[str]:
    """In-domain but irrelevant prose, used for the realistic filler condition.

    WikiText-103 is public, English, and stylistically close to the Wikipedia
    paragraphs HotpotQA is built from, which is what makes it a fair "plausible
    but irrelevant" filler rather than obvious noise.
    """
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    rng = random.Random(seed)
    sentences: list[str] = []
    for i in rng.sample(range(len(dataset)), min(n_sentences * 4, len(dataset))):
        text = dataset[i]["text"].strip()
        if len(text) < 80 or text.startswith("="):
            continue
        sentences.append(" ".join(text.split()))
        if len(sentences) >= n_sentences:
            break
    return sentences
