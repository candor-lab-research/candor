"""Does a descent over the wide strength range stall where one over the penalty-free zone does not?"""
import sys, time
import torch
from optimum import best_of, describe
from scenarios import WIDE, NARROW, scenarios, space, _hints
from landscape import ONE_LAYER
torch.set_num_threads(1)
name, layer, low, high = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
land = scenarios()[name].positions[None]
per, one = _hints(land)
hints = [h for h in one if abs(h["direction_layer"] - layer) <= 1]
t = time.time()
b = best_of(land, space((low, high)), ONE_LAYER, layer=layer, max_refusals=2.0, hints=hints)
print(f"{name} layer {layer:g} strength {low}-{high}: {describe(b)} [{time.time()-t:.0f} s]", flush=True)
