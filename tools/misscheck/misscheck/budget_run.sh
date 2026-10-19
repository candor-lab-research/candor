#!/bin/bash
# The search.max_budget_hits check, end to end (about 6 h on 10 cores). Numbers only; no model is started.
set -e
W=$(cd "$(dirname "$0")" && pwd)
cd "$W"
REPO=$(cd "$W/../../.." && pwd)
# candor's search code at 8dfc0f5, the commit that added search.max_budget_hits.
[ -d candor-8dfc0f5/src ] || { mkdir -p candor-8dfc0f5 && git -C "$REPO" archive 8dfc0f5 src | tar -x -C candor-8dfc0f5; }
export PYTHONPATH="$W/candor-8dfc0f5/src:$W"
PY="uv run --project $REPO python"
J=${JOBS:-10}
B="a-same,b-answer,c-think,d-high,fa-same,fb-answer,fc-think,fd-high"
A="base,rand1,rand2,rand3,rand4,rand5,rand6,basex2.4,rand4x1.8,rand6x1.8,fbase,frand1,frand2,frand3,frand4,frand5,frand6,fbasex2.6,frand4x1.8,frand6x1.8"
$PY budget.py truth --scenarios $B,$A --stress $B,$A --cliffs $B,$A --jobs $J   # truths and bumps, cached in cache/
$PY budget_sim.py check --jobs $J | tee budget-check.txt
$PY budget_sim.py run --variant base --scenarios $B,$A --designs D3grow,D3,grow --arms off,on,switch --seeds ${BASE_SEEDS:-200} --jobs $J
$PY budget_sim.py run --variant stress --scenarios $B,$A --designs D3grow,D3,grow --arms off,on,switch --seeds ${STRESS_SEEDS:-120} --jobs $J
$PY budget_sim.py run --variant cliffs --scenarios $B,$A --designs D3grow,D3,grow --arms off,on,switch --seeds ${CLIFF_SEEDS:-100} --jobs $J
$PY budget_report.py > budget-tables.txt   # budget-report.txt is the written summary of these
