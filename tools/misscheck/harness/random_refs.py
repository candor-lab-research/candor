"""Find (and cache) the best edits and fronts of the random landscapes, and say what each is like."""
import sys
import time

from optimum import describe
from variants import front, landscapes, reference

for name in sys.argv[1].split(","):
    started = time.time()
    landscape = landscapes()[name]
    found = reference(landscape)
    curve = front(landscape)
    one, per = found["one layer best"], found["per layer"]
    print(f"{name}: shift {landscape.shift:+g}, one-layer KL x{landscape.one_layer_kl:.2f}, {time.time() - started:.0f} s", flush=True)
    print("  per layer:", describe(per))
    print("  one layer:", describe(one))
    print("  front:", "  ".join(f"{r:g}:{kl:.4f}" for r, kl in curve.items()), flush=True)
