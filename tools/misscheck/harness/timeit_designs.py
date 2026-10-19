"""Time one run of each design on base, to plan the evaluation."""
import sys
import time

import torch

from strategies import STRATEGIES
from variants import landscapes

torch.set_num_threads(1)
land = landscapes()["base"]
for name in sys.argv[1].split(","):
    budget = int(sys.argv[2])
    start = time.time()
    trials = STRATEGIES[name](land.scorer(0), budget, 0)
    rates, kls = land.true([t.params for t in trials])
    best = min((k for r, k in zip(rates, kls) if r <= 0.02), default=float("inf"))
    hours = sum(t.seconds for t in trials) / 3600
    print(f"{name}: {budget} trials in {time.time() - start:.1f} s, {hours:.2f} sim hours, "
          f"best true KL at <=2 true refusals {best:.4f}, prompts mean {sum(t.prompts for t in trials) / len(trials):.0f}", flush=True)
