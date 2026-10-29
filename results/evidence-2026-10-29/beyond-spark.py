"""Is there a better edit past the search's range ends? Spark-X2.5-4B's per-layer finalist, trial 18,
presses the MLP strength's floor (0.912 against 0.8), so it is scored again as the search scores its
trials, as it is and with that strength moved past the floor, one value at a time. Each change is
compared with the trial as it is, scored in the same session and paired by prompt on the same test
prompts. Numbers only.

    uv run python results/evidence-2026-10-29/beyond-spark.py runs/spark \
        > results/evidence-2026-10-29/beyond-spark.txt
"""

import sys
from dataclasses import replace
from pathlib import Path

from candor import search
from candor.run import Run

# Each trial with the settings it presses on an end, and the values past the range to try: down to the
# widest floor a search may widen strength to.
PROBES = {18: {"mlp.strength": (0.65, 0.5)}}

run = Run.open(Path(sys.argv[1]), report=lambda _: None)
found = search.load(run)
layers = run.layers
edits, labels = [], []
for number, moves in PROBES.items():
    params = found.trials[number].params
    edits.append(search.edit_from(params, number, layers))
    labels.append((number, None, None))
    for setting, values in moves.items():
        for value in values:
            edit = search.edit_from({**params, setting: value}, number, layers)
            edits.append(replace(edit, name=f"trial-{number} {setting} {value:g}"))
            labels.append((number, setting, value))

with run.locked():
    scored = search.Search(run).score_locked(edits, base=False, beside_search=False)
print(f"Scored as the search scores its trials, on its {scored.prompts} test prompts.")
as_is = {}
for (number, setting, value), score in zip(labels, scored.edits, strict=True):
    if setting is None:
        as_is[number] = score
        searched = search.searched_score(found, number)
        print(
            f"trial {number} as it is: refused {score.refusals}, keywords {score.keyword_refusals}, "
            f"KL {score.kl:.4f}, at the budget {score.forced} (in the search: {searched.refusals}, "
            f"KL {searched.kl:.4f}, at the budget {searched.forced})"
        )
        continue
    gap = search.kl_gap(score, as_is[number])
    paired = "" if gap is None else f", KL gap {gap[0]:+.4f} (SE {gap[1]:.4f}{', a tie' if gap[2] else ''})"
    print(
        f"  {setting} {value:g}: refused {score.refusals}, keywords {score.keyword_refusals}, "
        f"KL {score.kl:.4f}, at the budget {score.forced}{paired}"
    )
