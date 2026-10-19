"""Miss rates per design and landscape from a results file."""

import json
import math
import sys
from collections import defaultdict

import numpy as np

rows = list({(r["landscape"], r["design"], r["budget"], r["seed"]): r for r in (
    json.loads(line) for row_file in sys.argv[1].split(",") for line in open(row_file))}.values())
optima = json.load(open("optima.json"))
optima["shifted"] = optima["union"]
# Reference: the best KL any run of any design found at each cap (best known), not the GP's optimum,
# which sits at box corners where the GP extrapolates.
truth = {(l, c): min(optima[l][f"{s}@{c}"]["kl"] for s in ("one", "per")) for l in optima for c in (0, 2, 5)}
known = defaultdict(lambda: math.inf)
for row in rows:
    for cap, (ratio, _) in row["result"].items():
        if ratio is not None and not math.isnan(ratio):
            known[(row["landscape"], cap)] = min(known[(row["landscape"], cap)], ratio * truth[(row["landscape"], int(cap))])
for row in rows:
    for cap, (ratio, scope) in list(row["result"].items()):
        if ratio is not None and not math.isnan(ratio):
            row["result"][cap] = [ratio * truth[(row["landscape"], int(cap))] / known[(row["landscape"], cap)], scope]
json.dump({f"{l}@{c}": v for (l, c), v in known.items()}, open("known.json", "w"))
for (l, c), v in sorted(known.items()):
    print(f"best known {l} cap {c}: KL {v:.4f} (GP optimum {truth[(l, int(c))]:.4f})")
groups = defaultdict(list)
for row in rows:
    groups[(row["landscape"], row["budget"], row["design"])].append(row)

caps = sys.argv[2].split(",") if len(sys.argv) > 2 else ["0", "2", "5"]
print(f"{'landscape':8s} {'budget':>6s} {'design':32s} n  " + "  ".join(
    f"cap{c}: med  P>5% P>10% P>25%" for c in caps) + "   s/run")
for (land, budget, design), runs in sorted(groups.items()):
    parts = []
    for cap in caps:
        ratios = np.array([r["result"][cap][0] for r in runs], dtype=float)
        ratios = np.where(np.isnan(ratios), 1.0, ratios)  # nothing feasible exists at this cap
        if all(math.isnan(r["result"][cap][0]) for r in runs):
            parts.append(f"cap{cap}:   (none feasible)      ")
            continue
        parts.append(
            f"cap{cap}: {np.median(ratios):5.2f} {np.mean(ratios > 1.05):4.0%} {np.mean(ratios > 1.10):5.0%} {np.mean(ratios > 1.25):5.0%}"
        )
    # T170 scored on the surrogate (layer 12): 1 refusal, KL 0.0782 on union and hybrid.
    if land in ("union", "hybrid"):
        kl2 = np.array([r["result"]["2"][0] * known[(land, "2")] for r in runs])
        one = np.mean([r["result"]["2"][1] == "one" for r in runs])
        parts.append(f"P(cap2 worse than T170)={np.mean(kl2 > 0.0782):4.0%} one-layer best {one:4.0%}")
    secs = np.mean([r["seconds"] for r in runs])
    print(f"{land:8s} {budget:6d} {design:32s} {len(runs):2d} " + "  ".join(parts) + f"  {secs:5.1f}")
