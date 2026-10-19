"""Split misses into 'never reached the winning family's good region' and 'reached it but refined poorly'."""
import json, math, sys
from collections import defaultdict
import numpy as np

CAP = 2
WINNER = {"union": "one", "hybrid": "one", "heretic": "one", "candor": "per"}
known = {tuple([k.split("@")[0], int(k.split("@")[1])]): v for k, v in json.load(open("known.json")).items()}
runs = [json.loads(l) for f in sys.argv[1].split(",") for l in open(f)]
for run in runs:
    for cap in (0, 2, 5):
        b = min((k for _, _, r, k in run["trials"] if r <= cap), default=math.inf)
        known[(run["landscape"], cap)] = min(known.get((run["landscape"], cap), math.inf), b)
groups = defaultdict(list)
for run in runs:
    if run["restarts"] == 1:
        groups[(run.get("sampler", "tpe"), run["landscape"], run["budget"])].append(run)
for (sampler, land, budget), sel in sorted(groups.items()):
    win = WINNER[land]
    fam_best = np.array([min((k for s, _, r, k in r_["trials"] if s == win and r <= CAP), default=math.inf) for r_ in sel])
    ratio = fam_best / known[(land, CAP)]
    reached = np.isfinite(ratio)
    print(f"{sampler} {land:8s} {budget}: winning family reached cap {CAP} in {reached.mean():.0%} of runs; "
          f"its best / best known: median {np.median(ratio[reached]):.2f}, P>1.10 {np.mean(ratio[reached] > 1.10):.0%}, P>1.25 {np.mean(ratio[reached] > 1.25):.0%} (n={len(sel)})")
