#!/bin/zsh
# Re-run every experiment, then build tables, figures and Grad-CAM.
#   zsh code/experiments/run_all.sh
# CNNs (MPS/CUDA) and classical pipelines (CPU) run as two parallel queues.
# Runs that already have outputs in runs/ are skipped by cnn_queue.sh.
set -e
cd "${0:A:h}"
PY=../../.venv/bin/python
LOG=../../runs/logs
mkdir -p $LOG
$PY common.py
./cnn_queue.sh > $LOG/cnn_queue.log 2>&1 &
$PY classical.py longtail balanced rebalanced > $LOG/classical.log 2>&1 &
wait
$PY analyze.py
$PY gradcam.py longtail
