"""Where each search's best edits within the cap put their lowest peak layer, read from the study
journals. Numbers only.

    uv run python results/evidence-2026-10-23/floor.py runs/rude runs/qwen runs/k2 runs/qwen27b-medium"""

import sys
from pathlib import Path

from candor import search
from candor.run import Run

BANDS = ((0.6, 0.65), (0.65, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0))

for folder in sys.argv[1:]:
    run = Run.open(Path(folder), report=lambda _: None)
    found = search.load(run)
    space = run.settings.search
    low, high = space.peak_layer
    edge = low + search.EDGE * (high - low)
    print(f"## {folder}: peak_layer {low}-{high}, {len(found.trials)} trials")
    for scope in search.SCOPES:
        best = sorted(search.within_cap(found, scope, space), key=lambda c: c.kl)
        if not best:
            continue
        peaks = [min(v for k, v in c.params.items() if k.endswith("peak_layer")) for c in best]
        pressing = "presses the floor" if peaks[0] <= edge else "doesn't press the floor"
        print(
            f"  {scope}: {len(best)} within the cap; best KL {best[0].kl:.4f} at lowest peak {peaks[0]:.3f},"
        )
        print(f"    which {pressing} by candor's edge test (at or below {edge:.2f})")
        print(f"    top 10 by KL at or below {edge:.2f}: {sum(p <= edge for p in peaks[:10])}")
        for start, end in BANDS:
            kls = [c.kl for c, p in zip(best, peaks, strict=True) if start <= p < end or p == end == high]
            if kls:
                print(f"    lowest peak {start:.2f}-{end:.2f}: {len(kls)} trials, best KL {min(kls):.4f}")
