"""From traces: (1) how often the family leading after k trials is the family that wins in the end,
(2) whether two restarts agreeing predicts a miss (the measured miss signal)."""

import json
import math
import sys
from collections import defaultdict

import numpy as np

CAP = 2
WINNER = {"union": "one", "hybrid": "one", "heretic": "one", "candor": "per"}


def key(rows):
    """Lexicographic best: fewest refusals over the cap, then least KL."""
    return min(((max(r - CAP, 0), k) for _, _, r, k in rows), default=(math.inf, math.inf))


def best_at(rows, cap):
    return min((k for _, _, r, k in rows if r <= cap), default=math.inf)


runs = [json.loads(line) for line in open(sys.argv[1])]
print("sampler:", {r.get("sampler", "tpe") for r in runs})
known = defaultdict(lambda: math.inf)
for run in runs:
    for cap in (0, 2, 5):
        known[(run["landscape"], cap)] = min(known[(run["landscape"], cap)], best_at(run["trials"], cap))
# The design comparison's best known values too, so both use the same reference.
for name, value in json.load(open("known.json")).items():
    land, cap = name.split("@")
    known[(land, int(cap))] = min(known[(land, int(cap))], value)

print("(1) Family leading after k trials per family == eventual true winner (restarts=1 runs)")
for land in sorted({r["landscape"] for r in runs}):
    for budget in sorted({r["budget"] for r in runs}):
        sel = [r for r in runs if r["landscape"] == land and r["budget"] == budget and r["restarts"] == 1]
        if not sel:
            continue
        per = budget // 2
        out = []
        for k in (10, 20, 30, 45, 60, per):
            if k > per:
                continue
            right = 0
            for run in sel:
                fam = {s: [row for row in run["trials"] if row[0] == s][:k] for s in ("one", "per")}
                lead = min(fam, key=lambda s: key(fam[s]))
                right += lead == WINNER[land]
            out.append(f"k={k}: {right / len(sel):.0%}")
        print(f"  {land:8s} budget {budget}: " + ", ".join(out) + f"  (n={len(sel)})")

print("\n(2) Two restarts per family: agreement of the winning family's restart bests at cap 2")
for land in sorted({r["landscape"] for r in runs}):
    for budget in sorted({r["budget"] for r in runs}):
        sel = [r for r in runs if r["landscape"] == land and r["budget"] == budget and r["restarts"] == 2]
        if not sel:
            continue
        stats = defaultdict(list)
        for run in sel:
            pooled = best_at(run["trials"], CAP)
            miss = pooled / known[(land, CAP)] - 1
            fam = min(("one", "per"), key=lambda s: best_at([x for x in run["trials"] if x[0] == s], CAP))
            a, b = (best_at([x for x in run["trials"] if x[0] == fam and x[1] == j], CAP) for j in (0, 1))
            gap = abs(a - b) / min(a, b) if math.isfinite(min(a, b)) and math.isfinite(max(a, b)) else math.inf
            for tol in (0.05, 0.10):
                stats[(tol, gap <= tol)].append(miss)
            stats["right family"].append(fam == WINNER[land])
        line = [f"right family {np.mean(stats['right family']):.0%}"]
        for tol in (0.05, 0.10):
            for agree in (True, False):
                m = np.array(stats[(tol, agree)])
                if len(m):
                    line.append(f"gap<={tol:.0%} {'agree' if agree else 'differ'}: n={len(m)} P(miss>10%)={np.mean(m > 0.10):.0%} P(miss>25%)={np.mean(m > 0.25):.0%}")
        print(f"  {land:8s} budget {budget} (n={len(sel)}):\n    " + "\n    ".join(line))

print("\n(3) Single run per family: did the best at cap 2 improve by >2% in the last 25% of trials? Miss rates either way")
for land in sorted({r["landscape"] for r in runs}):
    for budget in sorted({r["budget"] for r in runs}):
        sel = [r for r in runs if r["landscape"] == land and r["budget"] == budget and r["restarts"] == 1]
        if not sel:
            continue
        stats = defaultdict(list)
        for run in sel:
            trials = run["trials"]
            miss = best_at(trials, CAP) / known[(land, CAP)] - 1
            late = False
            for fam in ("one", "per"):
                rows = [x for x in trials if x[0] == fam]
                cut = int(len(rows) * 0.75)
                before, after = best_at(rows[:cut], CAP), best_at(rows, CAP)
                late |= after < before * 0.98
            stats[late].append(miss)
        line = []
        for late in (False, True):
            m = np.array(stats[late])
            if len(m):
                line.append(f"{'still improving' if late else 'settled'}: n={len(m)} P(miss>10%)={np.mean(m > 0.10):.0%} P(miss>25%)={np.mean(m > 0.25):.0%}")
        print(f"  {land:8s} budget {budget}: " + "; ".join(line))

print("\n(4) Chance the family's explore phase has reached the basin (a trial with <= 10 refusals) by trial k of the family")
for land in sorted({r["landscape"] for r in runs}):
    for budget in sorted({r["budget"] for r in runs}):
        sel = [r for r in runs if r["landscape"] == land and r["budget"] == budget and r["restarts"] == 1]
        if not sel:
            continue
        per = budget // 2
        out = []
        for k in (20, 40, 60, 80, 100, 150, 200):
            if k > per:
                continue
            hit = np.mean([min([x[2] for x in r["trials"] if x[0] == WINNER[land]][:k]) <= 10 for r in sel])
            out.append(f"k={k}: {hit:.0%}")
        print(f"  {land:8s} budget {budget} family '{WINNER[land]}': " + ", ".join(out))
