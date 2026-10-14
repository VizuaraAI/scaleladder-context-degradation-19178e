"""Aggregation and the size-effect estimate.

    python -m src.analyze --config configs/default.yaml

Reads runs/<run_name>/results.jsonl and writes three files next to it:

  summary.csv      one row per (model, task, condition), accuracy with CI
  degradation.csv  one row per (model, task, length), D_rel and D_hn
  scale_fit.txt    the regression of degradation on log10(params)

The last file is the actual answer to the research question, so read it last
and read it carefully. A coefficient without its interval means nothing here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .metrics import degradation


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.evaluate` first.")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.update({f"cond_{k}": v for k, v in row.pop("condition").items()})
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"{path} has no usable rows.")
    return frame


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap over items. Returns (low, high) on the 0-100 scale."""
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n, values.size), replace=True).mean(axis=1)
    return (float(np.percentile(draws, 2.5) * 100), float(np.percentile(draws, 97.5) * 100))


CONDITION_KEYS = [
    "cond_input_length",
    "cond_filler_type",
    "cond_evidence_position",
    "cond_output_length",
    "cond_probe_retrieval",
    "cond_drop_evidence",
]


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    groups = ["model", "family", "params_b", "task", *CONDITION_KEYS]
    rows = []
    for key, group in frame.groupby(groups, dropna=False):
        values = group["em"].to_numpy(dtype=float)
        low, high = bootstrap_ci(values)
        record = dict(zip(groups, key))
        record.update(
            {
                "n": int(values.size),
                "accuracy": float(values.mean() * 100),
                "ci_low": low,
                "ci_high": high,
                "retrieval_rate": float(group["retrieval_hit"].mean() * 100),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["family", "params_b", "task", "cond_input_length"])


def degradation_table(summary: pd.DataFrame, config: Config) -> pd.DataFrame:
    """D_rel and D_hn per (model, task, length), relative to the short condition.

    The floor is taken from the probe-free condition with the evidence removed if
    the sweep contains the no-evidence control rows; otherwise it falls back to
    zero and D_hn collapses to D_rel, which the `floor_source` column records.
    """
    default_filler = config.stress.filler_types[0]
    default_position = config.stress.evidence_positions[0]
    base = summary[
        (~summary["cond_probe_retrieval"])
        & (~summary["cond_drop_evidence"])
        & (summary["cond_filler_type"] == default_filler)
        & (summary["cond_evidence_position"] == default_position)
        & (summary["cond_output_length"] == config.stress.output_lengths[0])
    ]

    floors = _load_floors(summary)
    rows = []
    for (model, task), group in base.groupby(["model", "task"]):
        short = group[group["cond_input_length"] == config.analysis.short_length]
        if short.empty:
            continue
        a_short = float(short["accuracy"].iloc[0])
        floor_key = (model, task)
        a_floor = floors.get(floor_key, 0.0)
        for _, row in group.iterrows():
            scores = degradation(
                a_short,
                float(row["accuracy"]),
                floor_accuracy=a_floor,
                min_headroom=config.analysis.min_headroom,
            )
            rows.append(
                {
                    "model": model,
                    "family": row["family"],
                    "params_b": row["params_b"],
                    "task": task,
                    "input_length": row["cond_input_length"],
                    "accuracy": row["accuracy"],
                    "a_short": a_short,
                    "a_floor": a_floor,
                    "headroom": scores.headroom,
                    "d_rel": scores.d_rel,
                    "d_hn": scores.d_hn,
                    "floor_source": "measured" if floor_key in floors else "assumed_zero",
                }
            )
    return pd.DataFrame(rows)


def _load_floors(summary: pd.DataFrame) -> dict[tuple[str, str], float]:
    """a_floor from the no-evidence control rows of the same sweep."""
    floor_rows = summary[summary["cond_drop_evidence"]]
    return {
        (row["model"], row["task"]): float(row["accuracy"])
        for _, row in floor_rows.iterrows()
    }


def fit_scale_effect(table: pd.DataFrame, score: str, config: Config) -> str:
    """Regress degradation on log10(params) with family and task as fixed effects.

    Statsmodels OLS with family dummies is the honest small-sample choice here: a
    mixed model wants more groups than four families to estimate a variance
    component well. The family dummies still do the job that matters, which is
    forcing the size effect to be estimated within recipes rather than across
    them.
    """
    import statsmodels.formula.api as smf

    usable = table.dropna(subset=[score]).copy()
    usable = usable[usable["input_length"] > config.analysis.short_length]
    if usable["model"].nunique() < 3:
        return f"{score}: not enough models with a usable {score} to fit a size effect.\n"

    usable["log_params"] = np.log10(usable["params_b"])
    usable["log_length"] = np.log10(usable["input_length"])
    formula = f"{score} ~ log_params + C(family) + C(task) + log_length"
    model = smf.ols(formula, data=usable).fit()

    beta = model.params.get("log_params", float("nan"))
    low, high = model.conf_int().loc["log_params"].tolist()
    pvalue = model.pvalues.get("log_params", float("nan"))
    verdict = (
        "H1 supported: degradation falls as parameter count rises."
        if high < 0
        else "H2 supported: no size effect distinguishable from zero."
        if low < 0 < high
        else "Degradation rises with parameter count, which neither reading predicted."
    )
    return (
        f"=== {score} ===\n"
        f"n = {len(usable)} cells across {usable['model'].nunique()} models, "
        f"{usable['family'].nunique()} families\n"
        f"beta(log10 params) = {beta:.3f}  95% CI [{low:.3f}, {high:.3f}]  p = {pvalue:.4f}\n"
        f"adjusted R^2 = {model.rsquared_adj:.3f}\n"
        f"{verdict}\n\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ScaleLadder results.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    config = load_config(args.config, overrides={"run_name": args.run_name})
    frame = load_results(config.results_path)

    summary = summarise(frame)
    summary.to_csv(config.run_dir / "summary.csv", index=False)

    table = degradation_table(summary, config)
    table.to_csv(config.run_dir / "degradation.csv", index=False)

    report = "".join(fit_scale_effect(table, score, config) for score in ("d_rel", "d_hn"))
    if (table["floor_source"] == "assumed_zero").any():
        report += (
            "WARNING: some floors were assumed to be zero, so d_hn is not "
            "trustworthy for those rows. Make sure the no-evidence control "
            "condition ran for every model and task.\n"
        )
    (config.run_dir / "scale_fit.txt").write_text(report, encoding="utf-8")

    print(report)
    print(f"Wrote summary.csv, degradation.csv and scale_fit.txt to {config.run_dir}")


if __name__ == "__main__":
    main()
