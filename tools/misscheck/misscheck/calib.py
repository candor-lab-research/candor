"""Quick calibration: base's best one-layer edit at layer 12 (<= 2 true refusals, 0.5-3.0) per efficiency and stretch."""
import sys, time
import torch
from optimum import best_of, describe
from scenarios import WIDE, NARROW, _harness, scaled, space, _hints
from landscape import ONE_LAYER, PER_LAYER
torch.set_num_threads(1)
base = _harness()["base"]
name, e, k = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
land = scaled(base, name, stretch=k, efficiency=e)
per, one = _hints(land)
t = time.time()
b = best_of(land, space(WIDE), ONE_LAYER, layer=12.0, max_refusals=2.0, hints=[h for h in one if abs(h["direction_layer"] - 12) <= 1], samples=20000, steps=1500, seeds=2)
print(f"e {e} k {k}: {describe(b)}  [{time.time()-t:.0f} s]", flush=True)
