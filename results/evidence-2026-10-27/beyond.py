"""Is there a better edit past the search's range ends? rude-llama's best trials that press the peak's
floor or reach's ceiling, scored again as the search scores its trials, as they are and with that end
moved past its range, one setting at a time. Each change is compared with its trial as it is, scored
in the same session and paired by prompt on the same test prompts. The search was made with older
candor code, so these scores aren't set beside its trials. Numbers only.

    caffeinate -is uv run python results/evidence-2026-10-27/beyond.py runs/rude \
        > results/evidence-2026-10-27/beyond.txt
"""

import sys
from dataclasses import replace
from pathlib import Path

from candor import search
from candor.run import Run

# Each trial with the settings it presses on an end, and the values past the range to try.
PROBES = {
    535: {"attn.peak_layer": (0.5, 0.4), "attn.reach": (0.8, 1.0)},
    295: {"attn.peak_layer": (0.5, 0.4)},
    345: {"attn.peak_layer": (0.5, 0.4)},
    381: {"attn.reach": (0.8, 1.0)},
    463: {"mlp.reach": (0.8, 1.0)},
}

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
            f"KL {score.kl:.4f} (in the search, with older code: {searched.refusals}, KL {searched.kl:.4f})"
        )
        continue
    gap = search.kl_gap(score, as_is[number])
    paired = "" if gap is None else f", KL gap {gap[0]:+.4f} (SE {gap[1]:.4f}{', a tie' if gap[2] else ''})"
    print(
        f"  {setting} {value:g}: refused {score.refusals}, keywords {score.keyword_refusals}, "
        f"KL {score.kl:.4f}{paired}"
    )
