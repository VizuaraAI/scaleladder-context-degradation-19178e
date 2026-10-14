"""Configuration loading.

One YAML file describes the whole experiment: which models form the scale
ladder, which tasks are used, and how the four stress dials are swept. Keeping
all of that in one place is what makes a run reproducible three weeks later.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelSpec:
    """One rung of a scale ladder."""

    id: str
    family: str
    params_b: float

    @property
    def slug(self) -> str:
        return self.id.replace("/", "__")


@dataclass
class TaskSpec:
    name: str
    n_examples: int = 60


@dataclass
class GenerationSpec:
    max_new_tokens: int = 96
    temperature: float = 0.0
    dtype: str = "bfloat16"
    device_map: str = "auto"
    batch_size: int = 4
    trust_remote_code: bool = False


@dataclass
class StressSpec:
    input_lengths: list[int] = field(default_factory=lambda: [1000, 4000, 16000])
    filler_types: list[str] = field(default_factory=lambda: ["whitespace"])
    evidence_positions: list[str] = field(default_factory=lambda: ["middle"])
    output_lengths: list[int] = field(default_factory=lambda: [1])


@dataclass
class AnalysisSpec:
    short_length: int = 1000
    bootstrap_samples: int = 2000
    floor_mode: str = "measured"
    min_headroom: float = 5.0


@dataclass
class Config:
    run_name: str
    output_dir: Path
    seed: int
    seeds: list[int]
    models: list[ModelSpec]
    tasks: list[TaskSpec]
    generation: GenerationSpec
    stress: StressSpec
    analysis: AnalysisSpec
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name

    @property
    def results_path(self) -> Path:
        return self.run_dir / "results.jsonl"


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Read a YAML config and return a typed Config.

    `overrides` is a shallow dict applied on top of the YAML, which is how the
    command line flags in evaluate.py stay thin.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    if overrides:
        raw = {**copy.deepcopy(raw), **{k: v for k, v in overrides.items() if v is not None}}

    models = [ModelSpec(**m) for m in raw.get("models", [])]
    if not models:
        raise ValueError(f"{path} lists no models; the scale ladder cannot be empty.")

    tasks = [TaskSpec(**t) for t in raw.get("tasks", [])]
    if not tasks:
        raise ValueError(f"{path} lists no tasks.")

    seeds = raw.get("seeds") or [raw.get("seed", 1234)]

    return Config(
        run_name=str(raw.get("run_name", "default")),
        output_dir=Path(raw.get("output_dir", "runs")),
        seed=int(raw.get("seed", 1234)),
        seeds=[int(s) for s in seeds],
        models=models,
        tasks=tasks,
        generation=GenerationSpec(**raw.get("generation", {})),
        stress=StressSpec(**raw.get("stress", {})),
        analysis=AnalysisSpec(**raw.get("analysis", {})),
        raw=raw,
    )
