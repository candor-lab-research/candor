"""H2 check: GP-BO per family in the default box vs a wider box (strength 0.6-1.8, peak 0.4-1.0,
reach up to 0.8), on union (best inside the default box) and shifted (best below the peak floor)."""
import json, math
import numpy as np

CAP = 2
TRUE_BEST = 0.0559  # union's best known at cap 2; shifted is union translated, so the same value


def best(run, k):
    rows = [x for f in ("one", "per") for x in [y for y in run["trials"] if y[0] == f][:k]]
    return min((x[3] for x in rows if x[2] <= CAP), default=math.inf)


def load(path, land):
    return {r["seed"]: r for r in map(json.loads, open(path))
            if r["landscape"] == land and r["restarts"] == 1 and r["budget"] == 400}


wide = {land: load("tr_gpwide.jsonl", land) for land in ("union", "shifted")}
base = {"union": load("tr_gp.jsonl", "union"), "shifted": load("tr_gp2.jsonl", "shifted")}
for land in ("union", "shifted"):
    seeds = sorted(wide[land])
    for k in (100, 150, 200):
        w = np.array([best(wide[land][s], k) for s in seeds])
        b = np.array([best(base[land][s], k) for s in seeds])
        print(f"{land:7s} {k}/family: default box median {np.median(b):.4f}, >10% over best {np.mean(b > TRUE_BEST * 1.1):.0%} | "
              f"wide box median {np.median(w):.4f}, >10% {np.mean(w > TRUE_BEST * 1.1):.0%}  (n={len(seeds)})")
