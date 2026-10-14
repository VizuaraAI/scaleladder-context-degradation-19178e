#!/usr/bin/env bash
# The real sweep. Resumable: results.jsonl is append-only and keyed by
# (model, task, condition, seed, item), so a crash costs you only the cell that
# was in flight. Run it under tmux or nohup.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/default.yaml}"

python -m src.evaluate --config "$CONFIG"
python -m src.analyze  --config "$CONFIG"
python -m src.plots    --config "$CONFIG"

echo
echo "Read runs/<run_name>/scale_fit.txt. The coefficient on log10(params) with"
echo "its confidence interval is the result the paper turns on. Then copy the"
echo "numbers into experiments/log_template.md before you forget the setup."
