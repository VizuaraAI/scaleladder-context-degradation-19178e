#!/usr/bin/env bash
# Thirty-minute sanity check: smallest model, two lengths, ten items.
# If this does not finish cleanly, do not start the real sweep.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.evaluate --config configs/smoke.yaml --limit 10
python -m src.analyze  --config configs/smoke.yaml
python -m src.plots    --config configs/smoke.yaml

echo
echo "Smoke test done. Look at:"
echo "  runs/smoke/results.jsonl     one line per generation"
echo "  runs/smoke/summary.csv       accuracy per condition with bootstrap CIs"
echo "  runs/smoke/scale_fit.txt     the size effect (meaningless at this scale, but it should run)"
echo "  runs/smoke/figures/          the plots"
