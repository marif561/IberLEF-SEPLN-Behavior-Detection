#!/usr/bin/env bash
# Train and evaluate both DA-VINCIS tasks.
# Usage: bash scripts/run_all.sh <data_dir>   (expects train.csv, dev.csv and optionally test.csv
#        with columns text, label (Task 1) and category (Task 2))
set -euo pipefail
D=${1:?"data directory required"}
TEST=""; [[ -f "$D/test.csv" ]] && TEST="--test $D/test.csv"

python -m davincis.train --task 1 --label-col label    --train "$D/train.csv" --dev "$D/dev.csv" $TEST
python -m davincis.train --task 2 --label-col category --train "$D/train.csv" --dev "$D/dev.csv" $TEST
echo "Done. Results in outputs/"
