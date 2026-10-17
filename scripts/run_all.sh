#!/usr/bin/env bash
# Full XAI-ChestDiag experiment. Usage:  bash scripts/run_all.sh [config] [seeds...]
#   e.g. bash scripts/run_all.sh configs/default.yaml 42 43 44
set -euo pipefail
CONFIG=${1:-configs/default.yaml}
shift || true
SEEDS=${@:-42}
MODELS=$(python -c "import yaml;print(' '.join(yaml.safe_load(open('$CONFIG'))['models']))")

echo "== 1. patient-level split"
python -m src.prepare_splits --config "$CONFIG"

for SEED in $SEEDS; do
  for M in $MODELS; do
    echo "== 2. train $M (seed $SEED)"
    python -m src.train    --config "$CONFIG" --model "$M" --seed "$SEED"
    echo "== 3. evaluate $M (seed $SEED)"
    python -m src.evaluate --config "$CONFIG" --model "$M" --seed "$SEED"
  done
done

FIRST=$(echo $SEEDS | awk '{print $1}')
echo "== 4. throughput"
python -m src.throughput --config "$CONFIG" --batch-sizes 1 64
echo "== 5. comparison + DeLong"
python -m src.compare --config "$CONFIG" --seed "$FIRST"
echo "== 6. pointing game"
python -m src.pointing_game --config "$CONFIG" --model densenet121 --seed "$FIRST"
echo "== 7. figures"
python -m src.plots --config "$CONFIG" --seed "$FIRST"
python -m src.gradcam_figure --config "$CONFIG" --model densenet121 --seed "$FIRST"
echo "Done. Results in outputs/summary, figures in outputs/figures"
