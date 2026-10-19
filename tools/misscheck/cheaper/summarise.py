import json, sys, glob, numpy as np
rows = [r for f in sys.argv[1:] for r in json.load(open(f))]
arms = list(dict.fromkeys(r["arm"] for r in rows))
for arm in arms:
    rs = [r for r in rows if r["arm"] == arm]
    regret = [r["scored_kl"] / r["optimum"] - 1 if r["scored_kl"] else np.inf for r in rs]
    near = [r["clean_kl"] is not None and r["clean_refusals"] <= 8 and r["clean_kl"] <= 1.10 * r["optimum"] for r in rs]
    print(f"{arm:10s} n={len(rs)} trials {np.mean([r['trials'] for r in rs]):6.1f}  best is one-layer {sum(r['one_layer'] for r in rs):2d}/{len(rs)}  "
          f"true optimum region (<=8 refusals, KL within 10%) {sum(near):2d}/{len(rs)}  median scored regret {100*np.median(regret):5.1f}%")
