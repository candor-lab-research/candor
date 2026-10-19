"""Emulate stopping rules on full traces (restarts=1): each family stops at the first trial t >= floor
where its best at the cap improved by less than `tol` over its last `window` share of trials, or at
the end of the trace. Reports the miss at the stop and the trials used."""
import json, math, sys
from collections import defaultdict
import numpy as np

CAP = 2
known = {}
runs = [json.loads(l) for f in sys.argv[1].split(",") for l in open(f)]
runs = [r for r in runs if r["restarts"] == 1]
for k, v in json.load(open("known.json")).items():
    land, cap = k.split("@"); known[(land, int(cap))] = v
for run in runs:
    b = min((k for _, _, r, k in run["trials"] if r <= CAP), default=math.inf)
    known[(run["landscape"], CAP)] = min(known.get((run["landscape"], CAP), math.inf), b)


def key(rows):
    return min(((max(r - CAP, 0), k) for _, _, r, k in rows), default=(math.inf, math.inf))


def stop_at(rows, floor, window, tol):
    for t in range(floor, len(rows) + 1):
        now, then = key(rows[:t]), key(rows[: int(t * (1 - window))])
        if now[0] == then[0] == 0 and now[1] >= then[1] * (1 - tol):
            return t
    return len(rows)


for floor, window, tol in [(60, 0.3, 0.01), (80, 0.3, 0.02), (100, 0.3, 0.01), (100, 0.4, 0.01)]:
    groups = defaultdict(list)
    for run in runs:
        if run["budget"] != 400:
            continue
        used, best = 0, []
        for fam in ("one", "per"):
            rows = [x for x in run["trials"] if x[0] == fam]
            t = stop_at(rows, floor, window, tol)
            used += t
            best += rows[:t]
        miss = min((k for _, _, r, k in best if r <= CAP), default=math.inf) / known[(run["landscape"], CAP)] - 1
        groups[(run.get("sampler", "tpe"), run["landscape"])].append((miss, used))
    print(f"rule: each family >= {floor} trials, stop when the best improved < {tol:.0%} over the last {window:.0%}; max 200 per family")
    for (sampler, land), vals in sorted(groups.items()):
        m = np.array([v[0] for v in vals]); u = np.array([v[1] for v in vals])
        print(f"  {sampler} {land:8s}: trials used median {np.median(u):.0f} (max {u.max()}), P(miss>10%) {np.mean(m > 0.10):.0%}, P(miss>25%) {np.mean(m > 0.25):.0%}, n={len(vals)}")
