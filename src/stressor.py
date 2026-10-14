"""The context stress generator.

This is the heart of the experiment. Given one task instance it builds a family
of prompts that differ along exactly one of four dials:

  dial 1  input length      how many tokens of filler surround the evidence
  dial 2  filler type       whitespace / in-domain prose / hard distractors
  dial 3  evidence position  start / middle / end of the padded context
  dial 4  output length      how many answer items the model must produce

The invariant that makes the study a controlled experiment: the question, the
evidence sentences and the instruction wording never change across conditions.
Only the padding around them does.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from .data import Example

FILLER_TYPES = ("whitespace", "in_domain_shuffled", "hard_distractor")
EVIDENCE_POSITIONS = ("start", "middle", "end")

INSTRUCTION = (
    "You are given a document followed by a question. The document may contain "
    "irrelevant material. Read carefully, then answer using only the document.\n"
    "Reply with the answer only, on a single line, prefixed by 'ANSWER: '."
)

RECITE_SUFFIX = (
    "First copy the one sentence from the document that contains the answer, "
    "prefixed by 'EVIDENCE: '. Then give the answer prefixed by 'ANSWER: '."
)


@dataclass
class Condition:
    """One cell of the sweep."""

    input_length: int
    filler_type: str = "whitespace"
    evidence_position: str = "middle"
    output_length: int = 1
    probe_retrieval: bool = False
    drop_evidence: bool = False      # the no-evidence control that measures a_floor

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_length": self.input_length,
            "filler_type": self.filler_type,
            "evidence_position": self.evidence_position,
            "output_length": self.output_length,
            "probe_retrieval": self.probe_retrieval,
            "drop_evidence": self.drop_evidence,
        }


@dataclass
class Prompt:
    uid: str
    text: str
    answer: str
    evidence: list[str]
    condition: dict[str, Any]
    actual_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


def one_dial_at_a_time(
    input_lengths: Sequence[int],
    filler_types: Sequence[str],
    evidence_positions: Sequence[str],
    output_lengths: Sequence[int],
) -> list[Condition]:
    """Build the sweep: a full length curve at the default setting of the other
    dials, then one length curve per alternative value of each other dial.

    A full cross product would be `len(lengths) * len(fillers) * len(positions)`
    cells, which is the quickest way to run out of GPU hours in week two. The
    one-dial design keeps every comparison interpretable and the cost linear.
    """
    base_filler, base_position, base_output = filler_types[0], evidence_positions[0], output_lengths[0]
    conditions: list[Condition] = []
    seen: set[tuple] = set()

    def add(condition: Condition) -> None:
        key = (
            condition.input_length,
            condition.filler_type,
            condition.evidence_position,
            condition.output_length,
        )
        if key not in seen:
            seen.add(key)
            conditions.append(condition)

    for length in input_lengths:
        add(Condition(length, base_filler, base_position, base_output))
    for filler in filler_types[1:]:
        for length in input_lengths:
            add(Condition(length, filler, base_position, base_output))
    for position in evidence_positions[1:]:
        for length in input_lengths:
            add(Condition(length, base_filler, position, base_output))
    for output_length in output_lengths[1:]:
        for length in input_lengths:
            add(Condition(length, base_filler, base_position, output_length))
    return conditions


class ContextStressor:
    """Renders (example, condition) pairs into prompts of a target token length."""

    def __init__(
        self,
        tokenizer,
        filler_sentences: Sequence[str] | None = None,
        seed: int = 1234,
    ) -> None:
        self.tokenizer = tokenizer
        self.filler_sentences = list(filler_sentences or [])
        self.seed = seed

    # ------------------------------------------------------------------ filler
    def _filler_pool(self, example: Example, filler_type: str) -> list[str]:
        if filler_type == "whitespace":
            # Minimally distracting padding: length without semantic content.
            return ["." + " " * 32]
        if filler_type == "in_domain_shuffled":
            if not self.filler_sentences:
                raise ValueError(
                    "filler_type 'in_domain_shuffled' needs filler_sentences; "
                    "pass data.filler_corpus() into ContextStressor."
                )
            return self.filler_sentences
        if filler_type == "hard_distractor":
            # Topically adjacent sentences from the same task instance, which is
            # the condition the RAG noise literature finds most damaging.
            pool = list(example.distractors)
            if not pool:
                pool = list(self.filler_sentences)
            if not pool:
                raise ValueError(
                    f"No distractors available for {example.uid} and no filler corpus loaded."
                )
            return pool
        raise ValueError(f"Unknown filler_type '{filler_type}'. Expected one of {FILLER_TYPES}.")

    def _n_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _pad_to(self, target_tokens: int, pool: Sequence[str], rng: random.Random) -> str:
        """Sample from `pool` until the rendered text reaches target_tokens."""
        if target_tokens <= 0:
            return ""
        chunks: list[str] = []
        total = 0
        # Estimate once, then correct, instead of re-tokenising after each line.
        sample = pool[rng.randrange(len(pool))]
        per_chunk = max(1, self._n_tokens(sample))
        while total < target_tokens:
            chunk = pool[rng.randrange(len(pool))]
            chunks.append(chunk)
            total += per_chunk
        text = "\n".join(chunks)
        # Trim overshoot so the realised length matches the requested one.
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > target_tokens:
            text = self.tokenizer.decode(ids[:target_tokens], skip_special_tokens=True)
        return text

    # ------------------------------------------------------------------ prompts
    def render(self, example: Example, condition: Condition, seed: int | None = None) -> Prompt:
        rng = random.Random((seed if seed is not None else self.seed) ^ hash(example.uid) % (2**31))

        evidence_block = "\n".join(example.evidence)
        instruction = INSTRUCTION
        if condition.probe_retrieval:
            instruction = f"{INSTRUCTION}\n{RECITE_SUFFIX}"

        question_block = f"Question: {example.question}"
        if condition.output_length > 1:
            question_block = (
                f"Answer the following question {condition.output_length} times, "
                f"numbered 1 to {condition.output_length}, giving the same answer each time. "
                f"Use one 'ANSWER: ' line per item.\n{question_block}"
            )

        if condition.drop_evidence:
            # Floor control: everything except the answer-bearing sentences, so
            # whatever the model scores here it scored without reading them.
            evidence_block = ""

        fixed_tokens = self._n_tokens(f"{instruction}\n\n{evidence_block}\n\n{question_block}")
        filler_budget = max(0, condition.input_length - fixed_tokens)

        pool = self._filler_pool(example, condition.filler_type)
        if condition.evidence_position == "start":
            before, after = 0, filler_budget
        elif condition.evidence_position == "end":
            before, after = filler_budget, 0
        elif condition.evidence_position == "middle":
            before = filler_budget // 2
            after = filler_budget - before
        else:
            raise ValueError(
                f"Unknown evidence_position '{condition.evidence_position}'. "
                f"Expected one of {EVIDENCE_POSITIONS}."
            )

        parts = [instruction, "", "Document:"]
        if before:
            parts.append(self._pad_to(before, pool, rng))
        if evidence_block:
            parts.append(evidence_block)
        if after:
            parts.append(self._pad_to(after, pool, rng))
        parts += ["", question_block]
        text = "\n".join(p for p in parts if p != "")

        return Prompt(
            uid=example.uid,
            text=text,
            answer=example.answer,
            evidence=list(example.evidence),
            condition=condition.as_dict(),
            actual_tokens=self._n_tokens(text),
            meta={"task": example.task, "fixed_tokens": fixed_tokens},
        )

    def render_all(
        self,
        examples: Sequence[Example],
        conditions: Sequence[Condition],
        seed: int | None = None,
    ) -> list[Prompt]:
        return [self.render(ex, cond, seed) for cond in conditions for ex in examples]
