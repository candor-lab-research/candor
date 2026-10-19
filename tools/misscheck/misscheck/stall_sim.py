"""Replays a stall stop on the phase 3 miss-rate simulator's saved runs: how often the pick is within
10% of the true best when the search stops at a fixed count, or once both kinds have stalled.

A kind stalls when its best measured KL within the cap (judged in full, refusals <= 2) is less than GAIN
below its best WINDOW of its own trials before. Numbers only.

    PYTHONPATH=misscheck/candor-head/src:misscheck python stall_sim.py
"""
import numpy as np

from report import judge, load, wilson
from summary import judging
from scenarios import fronts, load_refs

GROUPS = {
    "positions, penalised (D3grow)": (["a-same", "b-answer", "c-think", "d-high"], "D3grow"),
    "positions, free (D3grow)": (["fa-same", "fb-answer", "fc-think", "fd-high"], "D3grow"),
    "positions, penalised (D3, wide)": (["a-same", "b-answer", "c-think", "d-high"], "D3"),
    "positions, free (D3, wide)": (["fa-same", "fb-answer", "fc-think", "fd-high"], "D3"),
    "strength, all (grow)": ([*"base rand1 rand2 rand3 rand4 rand5 rand6 basex2.4 rand4x1.8 rand6x1.8".split(),
                              *"fbase frand1 frand2 frand3 frand4 frand5 frand6 fbasex2.6 frand4x1.8 frand6x1.8".split()],
                             "grow"),
}
RULES = [("fixed", 300, None, None), ("fixed", 400, None, None), ("fixed", 600, None, None),
         ("stall", 600, 50, 0.10), ("stall", 300, 50, 0.10), ("stall", 600, 40, 0.10),
         ("stall", 600, 75, 0.10), ("stall", 600, 50, 0.05)]


def stop(trials: np.ndarray, cap: int, window: int, gain: float) -> int:
    """The trial count at which both kinds have stalled, or cap."""
    best = {0.0: [], 1.0: []}
    running = {0.0: None, 1.0: None}
    for n, row in enumerate(trials[:cap], 1):
        kind = row[0]
        if row[2] == 100 and row[3] <= 2 and (running[kind] is None or row[4] < running[kind]):
            running[kind] = row[4]
        best[kind].append(running[kind])
        stalled = []
        for k, seq in best.items():
            if len(seq) <= window or seq[-1] is None or seq[-1 - window] is None:
                stalled.append(False)
            else:
                stalled.append(seq[-1] > (1 - gain) * seq[-1 - window])
        if all(stalled):
            return n
    return min(cap, len(trials))


refs, curves = load_refs(), fronts()
for label, (names, design) in GROUPS.items():
    print(label)
    for rule, cap, window, gain in RULES:
        hits = n = 0
        stops = []
        for name in names:
            curve, kl, position = judging(name, refs, curves)
            for t in load(name, design):
                at = cap if rule == "fixed" else stop(t, cap, window, gain)
                stops.append(at)
                hits += judge(t, at, kl, position, curve)["gap"] <= 0.10
                n += 1
        low, high = wilson(hits, n)
        what = f"fixed {cap}" if rule == "fixed" else f"stall {window}/{gain:.0%}, at most {cap}"
        print(f"  {what:28s} within 10%: {hits / n:4.0%} ({low:.0%}-{high:.0%}), {n} runs; "
              f"trials mean {np.mean(stops):.0f}, median {np.median(stops):.0f}")
