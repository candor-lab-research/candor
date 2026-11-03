"""Does an export onto the run's Q4_K_M ship the model the search scored? Trial 53 is exported with
deployment.export_onto = "quantized", under another name so the measured export is untouched, and read
on the Mac beside its adapter on the unedited Q4_K_M and its export as shipped, each against the
unedited Q4_K_M read in the same session, on the search's test prompts and the held-out ones. If it is
the scored model, it reads as the adapter does, and near 0 against it. Numbers only.

    uv run python results/evidence-2026-11-03/kl-onto.py runs/spark 53 \
        > results/evidence-2026-11-03/kl-onto.txt
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
edit = search.edit_from(found.trials[number].params, number, run.layers)
files = {"export as shipped": next(run.path("exports", f"trial-{number}").glob("*-Q4_K_M.gguf"))}
run.settings.deployment.export_onto = "quantized"
files["export onto the Q4_K_M"] = export.export(run, replace(edit, name=f"onto-{number}"))

with run.locked():
    _, quantized = run.gguf()
    ids = run.prompt_ids()
    sets = {name: run.kl_ids(ids, name) for name in KL_SETS}
    adapter, _ = run.edit_adapter(run.path("scores", "kl-gap"), edit)
    room = max(len(p) for prompts in sets.values() for p in prompts) + 8
    tokens = run.settings.scoring.kl_tokens
    reads = {}
    with run.server(quantized, [adapter], context_per_slot=room, parallel=SLOTS) as server:
        for name, prompts in sets.items():
            reads["unedited", name] = server.top_logprobs(prompts, tokens)
            reads["adapter", name] = server.top_logprobs(prompts, tokens, {adapter: 1.0})
    for label, model in files.items():
        with run.server(model, context_per_slot=room, parallel=SLOTS) as server:
            for name, prompts in sets.items():
                reads[label, name] = server.top_logprobs(prompts, tokens)


def summary(values: list[float]) -> str:
    se = statistics.stdev(values) / math.sqrt(len(values))
    return f"{statistics.fmean(values):.4f} (SE {se:.4f}), median {statistics.median(values):.4f}"


def divergences(reference: str, variant: str, name: str) -> list[float]:
    return list(map(kl.from_top, reads[reference, name], reads[variant, name]))


print(f"Trial {number}, read on the Mac at {SLOTS} slots, KL over the top {tokens} tokens: mean (SE), median")
for label, path in files.items():
    print(f"{label}: {path.name}, {path.stat().st_size:,} bytes")
for name, prompts in sets.items():
    print(f"{name}, {len(prompts)} prompts:")
    print("  against the unedited Q4_K_M:")
    for variant in ("adapter", *files):
        print(f"    {variant}: {summary(divergences('unedited', variant, name))}")
    for label in files:
        gaps = [
            edited - scored
            for edited, scored in zip(
                divergences("unedited", label, name), divergences("unedited", "adapter", name), strict=True
            )
        ]
        print(f"    {label} less the adapter, paired by prompt: {summary(gaps)}")
    print("  against the adapter on the unedited Q4_K_M:")
    for label in files:
        print(f"    {label}: {summary(divergences('adapter', label, name))}")
