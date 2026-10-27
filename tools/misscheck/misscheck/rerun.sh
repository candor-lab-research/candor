#!/bin/bash
# Re-run the whole check from scratch (about 1.5 h on 12 cores). Numbers only; no model is started.
set -e
W=$(cd "$(dirname "$0")" && pwd)
cd "$W"
REPO=$(cd "$W/../../.." && pwd)
# candor's search code as at HEAD 49fc36e (the working tree was being edited by another session).
[ -d candor-head/src ] || { mkdir -p candor-head && git -C "$REPO" archive 49fc36e src | tar -x -C candor-head; }
export PYTHONPATH="$W/candor-head/src:$W"
PY="uv run --project $REPO python"
J=${JOBS:-10}
$PY scenarios.py refs --jobs=$J   # references and fronts, cached in cache/
A="base,rand1,rand2,rand3,rand4,rand5,rand6,basex2.4,rand4x1.8,rand6x1.8"
AF="fbase,frand1,frand2,frand3,frand4,frand5,frand6,fbasex2.6,frand4x1.8,frand6x1.8"
B="a-same,b-answer,c-think,d-high"
BF="fa-same,fb-answer,fc-think,fd-high"
$PY runs.py --scenarios $A --designs narrow,wide --seeds 100 --jobs $J
$PY runs.py --scenarios $AF --designs narrow,wide --seeds 60 --jobs $J
$PY runs.py --scenarios $A,$AF --designs widelog,grow --seeds 60 --jobs $J
$PY runs.py --scenarios $B,$BF --designs D1,D2,D3 --seeds 100 --jobs $J
$PY runs.py --scenarios $BF --designs D1log,D3log --seeds 100 --jobs $J
$PY runs.py --scenarios $BF --designs D3x,D4 --seeds 60 --jobs $J
$PY runs.py --scenarios $B,$BF --designs D3grow --seeds 60 --jobs $J
$PY summary.py | tee summary.txt
# Per-landscape detail, e.g.: $PY report.py --scenarios $A --designs narrow,wide,widelog,grow
