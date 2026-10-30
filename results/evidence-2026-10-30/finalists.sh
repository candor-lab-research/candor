#!/bin/bash
# After the unedited model's measurement: trial 18's range-end check, then each finalist exported and measured.
set -euo pipefail
cd /root/candor
while kill -0 -- -"$(cat /root/logs/measure-base.pid)" 2>/dev/null; do sleep 30; done
[ "$(cat /root/logs/measure-base.exit)" = 0 ] || { echo "the unedited measurement failed"; exit 1; }
echo "beyond: $(date -u +%H:%M)"
python results/evidence-2026-10-29/beyond-spark.py runs/spark > results/evidence-2026-10-29/beyond-spark.txt ||
    echo "beyond failed"
for trial in 53 70 18; do
    echo "export $trial: $(date -u +%H:%M)"
    candor export runs/spark --trial $trial
    echo "measure $trial: $(date -u +%H:%M)"
    candor measure runs/spark --trial $trial --exported
done
echo "done: $(date -u +%H:%M)"
