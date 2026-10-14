"""Plots built from real results.

    python -m src.plots --config configs/default.yaml

Produces the two panels that replace Figure 2 of the paper draft:

  fig2a_length_curves.png   accuracy against input length, one line per rung
  fig2b_degradation.png     D_rel next to D_hn for every model

Keep the visual language of the draft: a sequential blue ramp for the ordered
size ladder, two distinct hues for the two degradation metrics, and confidence
intervals drawn on everything. If a panel looks surprising, check the
corresponding rows of summary.csv before believing it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config

LADDER = ["#bcd2ea", "#8db5db", "#5d93c6", "#3570ab", "#1b4b80", "#0d2f52"]
WARM = "#b5541f"
ACCENT = "#1f5fa8"
MUTED = "#5b6670"
RULE = "#c7ced6"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": RULE,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def length_curves(summary: pd.DataFrame, task: str, out_path: Path) -> None:
    frame = summary[
        (summary["task"] == task)
        & (~summary["cond_probe_retrieval"])
        & (~summary["cond_drop_evidence"])
    ]
    if frame.empty:
        print(f"no rows for task {task}, skipping {out_path.name}")
        return

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for i, (model, group) in enumerate(frame.groupby("model")):
        group = group.sort_values("cond_input_length")
        colour = LADDER[min(i, len(LADDER) - 1)]
        label = f"{model.split('/')[-1]} ({group['params_b'].iloc[0]}B)"
        ax.plot(
            group["cond_input_length"], group["accuracy"],
            marker="o", markersize=3.5, linewidth=1.4, color=colour, label=label, zorder=3,
        )
        ax.fill_between(
            group["cond_input_length"], group["ci_low"], group["ci_high"],
            color=colour, alpha=0.18, linewidth=0, zorder=2,
        )

    ax.set_xscale("log")
    ax.set_xlabel("input length (tokens)")
    ax.set_ylabel(f"{task} accuracy (%)")
    ax.grid(axis="y", color=RULE, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6, frameon=False, loc="best")
    ax.set_title(f"Accuracy against input length: {task}", fontsize=9)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def degradation_bars(table: pd.DataFrame, task: str, out_path: Path) -> None:
    frame = table[(table["task"] == task) & (table["input_length"] == table["input_length"].max())]
    frame = frame.dropna(subset=["d_rel"]).sort_values("params_b")
    if frame.empty:
        print(f"no degradation rows for task {task}, skipping {out_path.name}")
        return

    labels = [f"{m.split('/')[-1]}\n{p}B" for m, p in zip(frame["model"], frame["params_b"])]
    x = np.arange(len(frame))
    width = 0.36

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.bar(x - width / 2, frame["d_rel"], width, color=WARM, label="$D_{rel}$", zorder=3)
    ax.bar(
        x + width / 2, frame["d_hn"].fillna(0.0), width, color=ACCENT,
        label="$D_{hn}$ (headroom-normalised)", zorder=3,
    )
    for xi, value in zip(x + width / 2, frame["d_hn"]):
        if pd.isna(value):
            ax.text(xi, 1.0, "no headroom", rotation=90, fontsize=5.5, color=MUTED, ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel(f"degradation at {int(frame['input_length'].max())} tokens (%)")
    ax.grid(axis="y", color=RULE, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title(f"Degradation by metric: {task}", fontsize=9)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ScaleLadder results.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    config = load_config(args.config, overrides={"run_name": args.run_name})
    summary_path = config.run_dir / "summary.csv"
    table_path = config.run_dir / "degradation.csv"
    if not summary_path.exists() or not table_path.exists():
        raise FileNotFoundError("Run `python -m src.analyze` before plotting.")

    summary = pd.read_csv(summary_path)
    table = pd.read_csv(table_path)
    figures_dir = config.run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    for spec in config.tasks:
        length_curves(summary, spec.name, figures_dir / f"fig2a_length_{spec.name}.png")
        degradation_bars(table, spec.name, figures_dir / f"fig2b_degradation_{spec.name}.png")


if __name__ == "__main__":
    main()
