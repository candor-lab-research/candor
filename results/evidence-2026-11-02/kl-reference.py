"""Which of trial 53's files is nearest the model it stands for? Against the unedited Q4_K_M, its
export read 0.044 held out and its adapter 0.018 (kl-gap.txt); with the edited weights at Q8_0 the
export read 0.033 (kl-gap-q8.txt). The adapter keeps the edited weights' Q4_K rounding, which the
unedited Q4_K_M shares, while an export rounds them afresh from 16 bits, so KL from the Q4_K_M may
count an export's being more exact as drift. So each is read here against the unedited model in
BF16 as well: the unedited Q4_K_M, the adapter on it and on the BF16, and both exports, on the Mac,
on the search's test prompts and the held-out ones. Numbers only.

    uv run python results/evidence-2026-11-02/kl-reference.py runs/spark 53 \
        > results/evidence-2026-11-02/kl-reference.txt
"""

import math
import statistics
import sys
from pathlib import Path

from candor import kl
from candor.run import KL_SETS, Run

run = Run.open(Path(sys.argv[1]), report=lambda line: print(line, file=sys.stderr))
number = int(sys.argv[2])
adapter = run.path("scores", "kl-gap", "edit.gguf")
exports = {
    "export": next(run.path("exports", f"trial-{number}").glob("*-Q4_K_M.gguf")),
    "export, edited at Q8_0": next(run.path("exports", f"q8-{number}").glob("*-Q4_K_M.gguf")),
}

with run.locked():
    bf16, quantized = run.gguf()
    ids = run.prompt_ids()
    sets = {name: run.kl_ids(ids, name) for name in KL_SETS}
    room = max(len(p) for prompts in sets.values() for p in prompts) + 8
    tokens = run.settings.scoring.kl_tokens
    reads = {}
    # BF16 at 2 slots leaves the 18 GB Mac room for the KV cache.
    for model, slots, label in ((bf16, 2, "BF16"), (quantized, 4, "Q4_K_M")):
        with run.server(model, [adapter], context_per_slot=room, parallel=slots) as server:
            for name, prompts in sets.items():
                reads[f"unedited {label}", name] = server.top_logprobs(prompts, tokens)
                reads[f"adapter on the {label}", name] = server.top_logprobs(prompts, tokens, {adapter: 1.0})
    for label, model in exports.items():
        with run.server(model, context_per_slot=room, parallel=4) as server:
            for name, prompts in sets.items():
                reads[label, name] = server.top_logprobs(prompts, tokens)


def summary(divergences: list[float]) -> str:
    se = statistics.stdev(divergences) / math.sqrt(len(divergences))
    return f"{statistics.fmean(divergences):.4f} (SE {se:.4f}), median {statistics.median(divergences):.4f}"


variants = [
    "unedited Q4_K_M",
    "adapter on the BF16",
    "adapter on the Q4_K_M",
    "export",
    "export, edited at Q8_0",
]
print(f"Trial {number}, read on the Mac, KL over the top {tokens} tokens: mean (SE), median.")
for name, prompts in sets.items():
    print(f"{name}, {len(prompts)} prompts:")
    for reference in ("unedited BF16", "unedited Q4_K_M"):
        print(f"  against the {reference}:")
        for variant in variants:
            if variant != reference:
                divergences = list(map(kl.from_top, reads[reference, name], reads[variant, name]))
                print(f"    {variant}: {summary(divergences)}")
    print("  against the adapter on the BF16 (the edit, exact):")
    for variant in variants[2:]:
        divergences = list(map(kl.from_top, reads["adapter on the BF16", name], reads[variant, name]))
        print(f"    {variant}: {summary(divergences)}")
