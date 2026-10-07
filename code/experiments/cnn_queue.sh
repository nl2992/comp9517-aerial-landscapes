#!/bin/zsh
# CNN runs, sequential (one GPU). Used by run_all.sh; safe to run on its own.
cd "${0:A:h}"
PY=../../.venv/bin/python; LOG=../../runs/logs; mkdir -p $LOG
for s in longtail balanced rebalanced; do
  for m in resnet18 effnet_b3; do
    [[ -f ../../runs/$s/$m/test_prob.npy ]] && continue
    $PY cnn.py $m $s > $LOG/${s}_${m}.log 2>&1 || echo "FAILED $s $m"
  done
done
for m in resnet18 effnet_b3; do
  [[ -f ../../runs/longtail/${m}_cw/test_prob.npy ]] && continue
  $PY cnn.py $m longtail --class-weighted > $LOG/longtail_${m}_cw.log 2>&1 || echo "FAILED cw $m"
done
echo CNN-QUEUE-DONE
