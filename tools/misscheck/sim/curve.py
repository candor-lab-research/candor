"""Miss rate against trials per family, from the prefixes of full 400-trial traces (restarts=1)."""
import json, math, sys
from collections import defaultdict
import numpy as np

CAP = 2
T170 = {"union": 0.0782, "hybrid": 0.0783}
runs = [json.loads(l) for f in sys.argv[1].split(",") for l in open(f)]
runs = [r for r in runs if r["restarts"] == 1 and r["budget"] == 400]
known = {}
for k, v in json.load(open("known.json")).items():
    land, cap = k.split("@"); known[(land, int(cap))] = v
for run in runs:
    b = min((k for _, _, r, k in run["trials"] if r <= CAP), default=math.inf)
    known[(run["landscape"], CAP)] = min(known.get((run["landscape"], CAP), math.inf), b)
print("best known at cap 2:", {l: round(v, 4) for (l, c), v in known.items() if c == CAP})
def prefix_best(run, k):
    """Least KL at the cap among each family's first k trials."""
    rows = [x for f in ("one", "per") for x in [y for y in run["trials"] if y[0] == f][:k]]
    return min((x[3] for x in rows if x[2] <= CAP), default=math.inf)


groups = defaultdict(list)
for run in runs:
    groups[(run.get("sampler", "tpe"), run["landscape"])].append(run)
for (sampler, land), sel in sorted(groups.items()):
    out = []
    for k in (60, 80, 100, 125, 150, 200):
        best = np.array([prefix_best(r, k) for r in sel])
        miss = best / known[(land, CAP)] - 1
        t = f" worse-than-T170 {np.mean(best > T170[land]):.0%}" if land in T170 else ""
        out.append(f"{k}/family: >10% {np.mean(miss > 0.10):.0%} >25% {np.mean(miss > 0.25):.0%}{t}")
    print(f"{sampler} {land:8s} (n={len(sel)}): " + " | ".join(out))
