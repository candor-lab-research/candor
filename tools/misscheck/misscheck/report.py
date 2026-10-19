"""Tables from the saved runs: how often a design's pick is within 10% (3%) of the true best.

A run's pick is candor's: among its trials scored in full that refused at most 2 of 100, the least
measured KL. It hits when its true KL is within tol of the scenario's true best KL at true refusals
<= 2 over the wide range (0.5-3.0) and every position. A run with no such trial misses.

The pick's true refusals are often above 2 (the least measured KL among many trials measured at <= 2
favours trials that truly refuse more), so two front-relative columns judge each edit against the best
KL at its own true refusals (the scenario's front, interpolated in log-log as the harness does):
front = the pick is within 10% of the front at its true refusals; found = some trial with true
refusals <= 2 is within 10% of the front at its true refusals, picked or not (the harness's
"found gap"). none = runs with no trial within the cap; pos = runs whose pick reads the best edit's
position; ref = the picks' median true refusals; judged = prompts judged per trial (the stop rule's
saving); out = trials with a strength outside 0.8-1.5.

    uv run python report.py --scenarios base,rand1 --designs narrow,wide
"""

import argparse
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from optimum import best_at
from runs import POSITIONS, RUNS
from scenarios import ANSWER, THINK, WIDE, fronts, load_refs, scenario_front, scenarios, truth

BUDGETS = (300, 400, 600)


def wilson(hits: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p, z = hits / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def load(scenario: str, design: str) -> list[np.ndarray]:
    return [np.load(p)["trials"] for p in sorted((RUNS / scenario / design).glob("*.npz"), key=lambda p: int(p.stem))]


def judge(trials: np.ndarray, budget: int, best_kl: float, best_position: int, curve: dict) -> dict:
    t = trials[:budget]
    one, position, judged, refusals, kl, true_refusals, true_kl = (t[:, i] for i in range(7))
    eligible = (judged == 100) & (refusals <= 2)
    low = true_refusals <= 2
    near = low.any() and (true_kl[low] / best_at(curve, true_refusals[low])).min() <= 1.10
    out = {"found": bool(near), "judged": float(judged.mean()), "stopped": float((judged < 100).mean()),
           "out": float(((t[:, 8] > 1.5) | (t[:, 8] < 0.8) | (t[:, 9] > 1.5) | (t[:, 9] < 0.8)).mean())}
    if not eligible.any():
        return {**out, "pick": False, "gap": math.inf, "front": math.inf, "ref": math.nan, "pos": False}
    i = np.flatnonzero(eligible)[np.lexsort((refusals[eligible], kl[eligible]))[0]]
    front = true_kl[i] / float(best_at(curve, true_refusals[i])) - 1
    return {**out, "pick": True, "gap": true_kl[i] / best_kl - 1, "front": front, "ref": true_refusals[i],
            "pos": position[i] == best_position or best_position == -2}


def cell(runs: list[np.ndarray], budget: int, best_kl: float, best_position: int, curve: dict) -> dict:
    judged = [judge(r, budget, best_kl, best_position, curve) for r in runs]
    n = len(judged)
    gaps = np.array([j["gap"] for j in judged])
    picked = [j for j in judged if j["pick"]]
    return {
        "n": n, "hit10": int((gaps <= 0.10).sum()), "hit3": int((gaps <= 0.03).sum()),
        "front": sum(j["front"] <= 0.10 for j in judged),
        "none": sum(not j["pick"] for j in judged), "found": sum(j["found"] for j in judged),
        "pos": sum(j["pos"] for j in judged), "ref": float(np.median([j["ref"] for j in picked])) if picked else math.nan,
        "gap": float(np.median(gaps)), "judged": float(np.mean([j["judged"] for j in judged])),
        "out": float(np.mean([j["out"] for j in judged])),
    }


def ci(hits: int, n: int) -> str:
    low, high = wilson(hits, n)
    return f"{hits / n:4.0%} ({low:4.0%}-{high:4.0%})"


def envelope(name: str, curve: dict, best_kl: float) -> tuple[dict, float]:
    """The front and best KL lowered, where any design's simulated trial beat them, to that trial: a
    reference found by descent can stop a little short."""
    trials = [r for d in (RUNS / name).iterdir() for r in load(name, d.name)] if (RUNS / name).exists() else []
    if not trials:
        return curve, best_kl
    t = np.concatenate(trials)
    refusals, kls = t[:, 5], t[:, 6]
    lowered = {r: min(kl, float(kls[refusals <= r].min(initial=math.inf))) for r, kl in curve.items()}
    return lowered, min(best_kl, lowered.get(2.0, math.inf))


def best_position(scenario, best) -> int:
    """The position whose landscape holds the best edit (-1 for a model that doesn't think)."""
    if None in scenario.positions:
        return -1
    refs = load_refs()
    kls = {p: truth(type(scenario)(scenario.name, {p: land}), WIDE, refs).kl for p, land in scenario.positions.items()}
    if abs(kls[THINK] - kls[ANSWER]) < 1e-9:
        return -2  # the same: any position is right
    return POSITIONS.index(min(kls, key=kls.get))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--designs", required=True)
    parser.add_argument("--pool", default="", help="name=scenario,scenario;... rows pooling scenarios")
    args = parser.parse_args()
    names, designs = args.scenarios.split(","), args.designs.split(",")
    refs = load_refs()
    all_scenarios = scenarios()
    pools = {}
    for part in filter(None, args.pool.split(";")):
        label, members = part.split("=")
        pools[label] = members.split(",")
    curves = fronts()
    print(f"{'scenario':10s} {'design':6s} {'trials':>6s} {'runs':>4s}  {'within 10% (95% CI)':>19s}  {'within 3% (95% CI)':>18s}  "
          f"{'front 10% (95% CI)':>18s}  {'found':>5s} {'none':>4s} {'pos':>4s} {'gap':>6s} {'ref':>4s} {'judged':>6s} {'out':>4s}")
    totals = {}
    for name in names:
        scenario = all_scenarios[name]
        best = truth(scenario, WIDE, refs)
        position = best_position(scenario, best)
        curve, kl = envelope(name, scenario_front(scenario, curves), best.kl)
        if kl < best.kl:
            print(f"  {name}: best KL lowered to the best simulated trial, {best.kl:.4f} -> {kl:.4f}")
        best = replace(best, kl=kl)
        for design in designs:
            runs = load(name, design)
            if not runs:
                continue
            for budget in BUDGETS:
                c = cell(runs, budget, best.kl, position, curve)
                for label, members in pools.items():
                    if name in members:
                        total = totals.setdefault((label, design, budget), {"n": 0, "hit10": 0, "hit3": 0, "front": 0, "none": 0, "found": 0})
                        for k in total:
                            total[k] += c[k]
                pos = "  -" if position < 0 else f"{c['pos'] / c['n']:4.0%}"
                gap = "none" if math.isinf(c["gap"]) else f"{c['gap']:+.1%}"
                print(f"{name:10s} {design:6s} {budget:6d} {c['n']:4d}  {ci(c['hit10'], c['n']):>19s}  {ci(c['hit3'], c['n']):>18s}  "
                      f"{ci(c['front'], c['n']):>18s}  {c['found'] / c['n']:5.0%} {c['none'] / c['n']:4.0%} {pos:>4s} {gap:>6s} {c['ref']:4.1f} {c['judged']:6.1f} {c['out']:4.0%}")
        print()
    for (label, design, budget), c in totals.items():
        print(f"{label:10s} {design:6s} {budget:6d} {c['n']:4d}  {ci(c['hit10'], c['n']):>19s}  {ci(c['hit3'], c['n']):>18s}  "
              f"{ci(c['front'], c['n']):>18s}  {c['found'] / c['n']:5.0%} {c['none'] / c['n']:4.0%}")


if __name__ == "__main__":
    main()
