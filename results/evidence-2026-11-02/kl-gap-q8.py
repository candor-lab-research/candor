"""Does storing the edited weights at Q8_0 close the export's KL gap? kl-gap.txt found trial 53's
export reading about 2.5 times its adapter, on both prompt sets, so the gap is the export's own
rounding of the 31 edited weights, not the prompts. Here the same edit is exported with every
edited weight written unrounded and stored as Q8_0, as export does for a weight that keeps too
little of its edit, under another name so the measured export is untouched, and read on the Mac
against the unedited Q4_K_M as kl-gap.py reads. Numbers only.

    uv run python results/evidence-2026-11-02/kl-gap-q8.py runs/spark 53 \
        > results/evidence-2026-11-02/kl-gap-q8.txt
"""

import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path

from candor import export, kl, search
from candor.run import KL_SETS, Run

SLOTS = 4

run = Run.open(Path(sys.argv[1]), report=lambda line: print(line, file=sys.stderr))
number = int(sys.argv[2])
found = search.load(run)
assert found is not None
edit = replace(search.edit_from(found.trials[number].params, number, run.layers), name=f"q8-{number}")

# export keeps a weight apart when _short lists it: listing every edited weight on its first two
# calls, the check before keeping apart and the first round of it, stores every one as Q8_0.
real_short, calls = export._short, []


def every_weight_short(held: dict[str, float]) -> dict[str, float]:
    calls.append(1)
    return dict(held) if len(calls) <= 2 else real_short(held)


export._short = every_weight_short
shipped = export.export(run, edit)
export._short = real_short

with run.locked():
    _, quantized = run.gguf()
    ids = run.prompt_ids()
    sets = {name: run.kl_ids(ids, name) for name in KL_SETS}
    room = max(len(p) for prompts in sets.values() for p in prompts) + 8
    tokens = run.settings.scoring.kl_tokens
    reads = {}
    with run.server(quantized, context_per_slot=room, parallel=SLOTS) as server:
        for name, prompts in sets.items():
            reads["unedited", name] = server.top_logprobs(prompts, tokens)
    with run.server(shipped, context_per_slot=room, parallel=SLOTS) as server:
        for name, prompts in sets.items():
            reads["q8", name] = server.top_logprobs(prompts, tokens)


def summary(divergences: list[float]) -> str:
    se = statistics.stdev(divergences) / math.sqrt(len(divergences))
    return (
        f"mean {statistics.fmean(divergences):.4f} (SE {se:.4f}), median {statistics.median(divergences):.4f}"
    )


print(f"Trial {number}, read on the Mac at {SLOTS} slots, KL over the top {tokens} tokens.")
print(f"{shipped.name}: {shipped.stat().st_size:,} bytes")
for name, prompts in sets.items():
    divergences = list(map(kl.from_top, reads["unedited", name], reads["q8", name]))
    print(f"{name}, {len(prompts)} prompts: the export, edited weights at Q8_0: {summary(divergences)}")
