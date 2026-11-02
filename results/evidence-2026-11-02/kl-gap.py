"""Why does Spark-X2.5-4B's held-out KL read about 3 times the search's (a median 5 times)? Trial 53 is
read on the Mac on the search's harmless test prompts and on the held-out ones, as its adapter on the
unedited Q4_K_M (as the search scores it) and as its export (as it ships), each against the unedited
model read in the same session, after the unedited model's own thinking as both read it. If the
adapter reads low on both sets, the export adds the gap; if it reads low only on the test prompts,
the search's choice of trial does. Numbers only.

    uv run python results/evidence-2026-11-02/kl-gap.py runs/spark 53 \
        > results/evidence-2026-11-02/kl-gap.txt
"""

import math
import statistics
import sys
from pathlib import Path

from candor import kl, search
from candor.run import KL_SETS, Run

SLOTS = 4

run = Run.open(Path(sys.argv[1]), report=lambda line: print(line, file=sys.stderr))
number = int(sys.argv[2])
found = search.load(run)
assert found is not None
edit = search.edit_from(found.trials[number].params, number, run.layers)
exported = next(run.path("exports", f"trial-{number}").glob("*-Q4_K_M.gguf"))

with run.locked():
    _, quantized = run.gguf()
    ids = run.prompt_ids()
    sets = {name: run.kl_ids(ids, name) for name in KL_SETS}
    folder = run.path("scores", "kl-gap")
    folder.mkdir(parents=True, exist_ok=True)
    adapter, _ = run.edit_adapter(folder, edit)
    room = max(len(p) for prompts in sets.values() for p in prompts) + 8
    tokens = run.settings.scoring.kl_tokens
    reads = {}
    with run.server(quantized, [adapter], context_per_slot=room, parallel=SLOTS) as server:
        for name, prompts in sets.items():
            reads["unedited", name] = server.top_logprobs(prompts, tokens)
            reads["adapter", name] = server.top_logprobs(prompts, tokens, {adapter: 1.0})
            reads["unedited again", name] = server.top_logprobs(prompts, tokens)
    with run.server(exported, context_per_slot=room, parallel=SLOTS) as server:
        for name, prompts in sets.items():
            reads["export", name] = server.top_logprobs(prompts, tokens)


def summary(divergences: list[float]) -> str:
    se = statistics.stdev(divergences) / math.sqrt(len(divergences))
    return (
        f"mean {statistics.fmean(divergences):.4f} (SE {se:.4f}), median {statistics.median(divergences):.4f}"
    )


print(f"Trial {number}, read on the Mac at {SLOTS} slots, KL over the top {tokens} tokens.")
for name, prompts in sets.items():
    print(f"{name}, {len(prompts)} prompts:")
    for variant in ("adapter", "export", "unedited again"):
        divergences = list(map(kl.from_top, reads["unedited", name], reads[variant, name]))
        print(f"  {variant} against the unedited: {summary(divergences)}")
    between = list(map(kl.from_top, reads["adapter", name], reads["export", name]))
    print(f"  export against the adapter: {summary(between)}")
searched = search.searched_score(found, number)
print(f"In the search, on the A40: KL {searched.kl:.4f}")
