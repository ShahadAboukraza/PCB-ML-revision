#!/usr/bin/env bash
# Reproduces every new number, table and figure in the revised manuscript.
# Total run time on one CPU core: roughly 30-45 minutes. Quick check: SMOKE=1 ./run_all.sh
set -e
cd "$(dirname "$0")/src"
python 01_feature_leakage.py
python 02_sensitivity_analyses.py
python 03_figures.py
echo "Done."
