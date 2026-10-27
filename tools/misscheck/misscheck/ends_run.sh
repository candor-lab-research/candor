#!/bin/bash
# The peak-floor and reach-ceiling check (about 1.5 h on 10 cores with the references). Numbers only; no model is started.
set -e
W=$(cd "$(dirname "$0")" && pwd)
cd "$W"
REPO=$(cd "$W/../../.." && pwd)
# candor's search code at d2043bc, the commit the check ran on.
[ -d candor-d2043bc/src ] || { mkdir -p candor-d2043bc && git -C "$REPO" archive d2043bc src | tar -x -C candor-d2043bc; }
export PYTHONPATH="$W/candor-d2043bc/src:$W"
PY="uv run --project $REPO python"
J=${JOBS:-10}
A="base,rand1,rand2,rand3,rand4,rand5,rand6,basex2.4,rand4x1.8,rand6x1.8,fbase,frand1,frand2,frand3,frand4,frand5,frand6,fbasex2.6,frand4x1.8,frand6x1.8"
B="a-same,b-answer,c-think,d-high,fa-same,fb-answer,fc-think,fd-high"
$PY scenarios.py refs --jobs=$J
$PY runs.py --scenarios $A --designs grow,floor-grow,ends-grow --seeds 100 --jobs $J
$PY runs.py --scenarios $B --designs D3grow,floor-D3grow,ends-D3grow --seeds 100 --jobs $J
$PY ends_summary.py | tee ends-tables.txt
