#!/bin/bash
set -euo pipefail
# Run from the project root. Require a new directory under night11.
run_dir="${1:?usage: bash night11/reproduce_all.sh night11/new_run}"
if [[ ! "$run_dir" =~ ^night11/[A-Za-z0-9_-]+$ ]]; then exit 2; fi
if [[ -e "$run_dir" ]]; then exit 3; fi
mkdir -p "$run_dir"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python_bin="final_package/.build/venv/bin/python"
"$python_bin" -B night11/reproduce_nested.py --output "$run_dir/measurements" > "$run_dir/nested.log" 2>&1
"$python_bin" -B night11/verify_temporal.py --measurements "$run_dir/measurements" --output "$run_dir/temporal" > "$run_dir/temporal.log" 2>&1
"$python_bin" -B night11/review_events.py --measurements "$run_dir/measurements" --output "$run_dir/events" --review "$run_dir/review.csv" --candidate "$run_dir/submission_events_hedged15.csv" > "$run_dir/events.log" 2>&1
