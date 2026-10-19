"""Where a landscape's best sits when KL isn't penalised for any strength (the fitted model extrapolated)."""
import sys, time
from dataclasses import replace
import torch
from optimum import best_of, describe
from scenarios import scenarios, space, _hints
from landscape import ONE_LAYER, HERETIC
torch.set_num_threads(1)
name, layer = sys.argv[1], float(sys.argv[2])
land = replace(scenarios()[name].positions[None], support=replace(HERETIC, strength=(0.01, 100.0)))
per, one = _hints(land)
hints = [h for h in one if abs(h["direction_layer"] - layer) <= 1]
t = time.time()
b = best_of(land, space((0.1, 4.0)), ONE_LAYER, layer=layer, max_refusals=2.0, hints=hints, samples=20000, steps=2000, seeds=2)
print(f"{name} free layer {layer:g}: {describe(b)} [{time.time()-t:.0f} s]", flush=True)
