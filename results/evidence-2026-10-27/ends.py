"""Which range ends each search's best edits within the cap press, by candor's own edge test
(search.at_edges, on the ranges each kind was drawn from), read from the study journals. Numbers only.

    uv run python results/evidence-2026-10-27/ends.py runs/rude runs/qwen runs/k2 runs/qwen27b-medium
"""

import sys
from collections import Counter
from pathlib import Path

from candor import search, source
from candor.run import Run

TOP = 10


def end(pressed: str) -> str:
    """An end at_edges reports, without the trial's own value: "attn.reach, at its ceiling 0.6"."""
    setting, where = pressed.split(",", 1)
    return f"{setting.rsplit(' ', 1)[0]},{where}"


for folder in sys.argv[1:]:
    run = Run.open(Path(folder), report=lambda _: None)
    found = search.load(run)
    space = run.settings.search
    _, layers = source.sizes(source.read_config(run._model_without_download))
    print(f"## {folder}: {len(found.trials)} trials, {layers} layers")
    for scope in search.SCOPES:
        best = search.within_cap(found, scope, space)[:TOP]
        if not best:
            continue
        drawn_from = search.kind_space(space, found, scope)
        ends = [[end(e) for e in search.at_edges(c.params, drawn_from, layers)] for c in best]
        print(f"  {scope}, strength {drawn_from.strength[0]:g}-{drawn_from.strength[1]:g}:")
        pressed = "; ".join(ends[0]) or "presses no end"
        print(f"    best, trial {best[0].trial} at KL {best[0].kl:.4f}: {pressed}")
        counts = Counter(e for trial in ends for e in trial)
        for e, n in counts.most_common():
            print(f"    top {len(best)}: {n} press{'es' if n == 1 else ''} {e}")
        floor = [any("peak_layer, at its floor" in e for e in trial) for trial in ends]
        ceiling = [any("reach, at its ceiling" in e for e in trial) for trial in ends]
        either = sum(f or c for f, c in zip(floor, ceiling, strict=True))
        print(
            f"    trials of the top {len(best)} pressing a peak's floor {sum(floor)}, a reach's ceiling "
            f"{sum(ceiling)}, either {either}"
        )
