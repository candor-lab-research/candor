"""Where a strategy's trials went: per scope, how many reached few true refusals, the best found, and
how its best one-layer edit's settings compare with the landscape's best."""
import sys

import numpy as np

from analyze import load
from evaluate import found_gap
from landscape import ONE_LAYER
from variants import front, landscapes, reference

name, strategy, total = sys.argv[1], sys.argv[2], int(sys.argv[3])
landscape = landscapes()[name]
found = reference(landscape)
best = found["best"]
curve = front(landscape)
keys = [k for k in best.params if "." in k]
print("true best:", {k: round(best.params[k], 3) for k in keys}, "layer", best.params["direction_layer"])
rows = []
for i, trials in enumerate(load(name, strategy)):
    trials = trials[:total]
    one = [t for t in trials if t["params"]["direction_scope"] == ONE_LAYER]
    per = [t for t in trials if t["params"]["direction_scope"] != ONE_LAYER]
    low_one = [t for t in one if t["true refusals"] <= 2]
    low_per = [t for t in per if t["true refusals"] <= 2]
    best_one = min(low_one, key=lambda t: t["true kl"], default=None)
    best_per = min(low_per, key=lambda t: t["true kl"], default=None)
    gap = found_gap(trials, curve)
    layers = np.bincount([t["params"]["direction_layer"] for t in one[40:]], minlength=27)
    top_layers = np.argsort(-layers)[:3]
    meas_low_one = sum(t["refusals"] <= 2 and t["prompts"] == 100 for t in one)
    print(f"seed {i:2d} gap {gap:+.1%}  one: {len(low_one):3d} true<=2 ({meas_low_one:3d} measured<=2), best "
          f"{best_one['true kl'] if best_one else float('nan'):.4f}; per: {len(low_per):3d}, best "
          f"{best_per['true kl'] if best_per else float('nan'):.4f}; aimed one-layer layers {list(top_layers)} "
          f"x{list(layers[top_layers])}")
    if best_one and i < 6:
        print("    best one:", {k: round(best_one["params"][k], 3) for k in keys}, "layer",
              best_one["params"]["direction_layer"], f"true ref {best_one['true refusals']:.2f}")
