"""Evaluation entry point.

    python -m src.evaluate --config configs/default.yaml

Loads each model in the ladder once, renders every (example, condition) prompt
with that model's own tokenizer, generates, scores, and appends one JSON line
per generation to runs/<run_name>/results.jsonl. The file is append-only and
keyed by (model, task, condition, seed, uid), so an interrupted sweep can be
resumed without recomputing anything.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from .config import Config, load_config
from .data import filler_corpus, load_task
from .metrics import score_record
from .models import build_runner, load_tokenizer
from .stressor import Condition, ContextStressor, one_dial_at_a_time


def record_key(row: dict) -> tuple:
    condition = row["condition"]
    return (
        row["model"],
        row["task"],
        condition["input_length"],
        condition["filler_type"],
        condition["evidence_position"],
        condition["output_length"],
        condition["probe_retrieval"],
        condition["drop_evidence"],
        row["seed"],
        row["uid"],
    )


def load_done(path: Path) -> set[tuple]:
    """Keys already present in results.jsonl, so reruns are cheap."""
    done: set[tuple] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(record_key(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def build_conditions(config: Config) -> list[Condition]:
    stress = config.stress
    conditions = one_dial_at_a_time(
        stress.input_lengths,
        stress.filler_types,
        stress.evidence_positions,
        stress.output_lengths,
    )
    # Two controls the analysis cannot do without:
    #   probe_retrieval=True -> separates retrieval failure from reasoning failure
    #   drop_evidence=True   -> measures a_floor, the denominator of D_hn
    base_filler = stress.filler_types[0]
    base_position = stress.evidence_positions[0]
    for length in stress.input_lengths:
        conditions.append(
            Condition(length, base_filler, base_position, 1, probe_retrieval=True)
        )
    conditions.append(
        Condition(
            stress.input_lengths[0], base_filler, base_position, 1, drop_evidence=True
        )
    )
    return conditions


def run(config: Config, limit: int | None = None) -> Path:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    results_path = config.results_path
    done = load_done(results_path)
    if done:
        print(f"Resuming: {len(done)} generations already recorded in {results_path}")

    conditions = build_conditions(config)
    tasks = {
        spec.name: load_task(spec.name, spec.n_examples, seed=config.seed)
        for spec in config.tasks
    }
    # hard_distractor falls back to the corpus for tasks without their own
    # distractors (GSM8K has none), so load it for either filler type.
    needs_corpus = bool(
        {"in_domain_shuffled", "hard_distractor"} & set(config.stress.filler_types)
    )
    corpus = filler_corpus(seed=config.seed) if needs_corpus else None

    with (config.run_dir / "config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(config.raw, handle, indent=2)

    n_written = 0
    for spec in config.models:
        print(f"\n=== {spec.id} ({spec.params_b}B, family={spec.family}) ===")
        tokenizer = load_tokenizer(spec.id, config.generation.trust_remote_code)
        stressor = ContextStressor(tokenizer, filler_sentences=corpus, seed=config.seed)
        runner = None  # loaded lazily so a fully cached model costs nothing

        for task_name, examples in tasks.items():
            for condition in conditions:
                for seed in config.seeds:
                    pending = []
                    for example in examples[: limit or len(examples)]:
                        prompt = stressor.render(example, condition, seed=seed)
                        key = (
                            spec.id,
                            task_name,
                            condition.input_length,
                            condition.filler_type,
                            condition.evidence_position,
                            condition.output_length,
                            condition.probe_retrieval,
                            condition.drop_evidence,
                            seed,
                            example.uid,
                        )
                        if key not in done:
                            pending.append(prompt)
                    if not pending:
                        continue

                    if runner is None:
                        runner = build_runner(spec, config.generation)

                    control = (
                        " [floor]" if condition.drop_evidence
                        else " [probe]" if condition.probe_retrieval
                        else ""
                    )
                    label = (
                        f"{task_name} L={condition.input_length} "
                        f"{condition.filler_type}/{condition.evidence_position} "
                        f"G={condition.output_length} seed={seed}{control}"
                    )
                    started = time.time()
                    outputs = runner.generate(
                        [p.text for p in pending], config.generation.max_new_tokens
                    )
                    elapsed = time.time() - started

                    with results_path.open("a", encoding="utf-8") as handle:
                        for prompt, output in zip(tqdm(pending, desc=label, leave=False), outputs):
                            scores = score_record(
                                output, prompt.answer, task_name, prompt.evidence
                            )
                            row = {
                                "model": spec.id,
                                "family": spec.family,
                                "params_b": spec.params_b,
                                "task": task_name,
                                "uid": prompt.uid,
                                "seed": seed,
                                "condition": prompt.condition,
                                "actual_tokens": prompt.actual_tokens,
                                "gold": prompt.answer,
                                "output": output[:2000],
                                **scores,
                            }
                            handle.write(json.dumps(row) + "\n")
                            done.add(record_key(row))
                            n_written += 1
                    print(f"  {label}: {len(pending)} items in {elapsed:.1f}s")

        del runner

    print(f"\nWrote {n_written} new generations to {results_path}")
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ScaleLadder context sweep.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-name", default=None, help="Override run_name from the config.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Use only the first N examples per task."
    )
    args = parser.parse_args()

    config = load_config(args.config, overrides={"run_name": args.run_name})
    run(config, limit=args.limit)


if __name__ == "__main__":
    main()
