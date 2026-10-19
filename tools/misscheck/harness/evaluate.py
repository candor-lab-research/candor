"""Run search strategies on landscapes many times, and report how often each misses the best edit.

    uv run python evaluate.py --seeds 30                       # every strategy on every landscape
    uv run python evaluate.py --landscapes base --strategies candor --seeds 20 --budgets 100,200,400
    uv run python evaluate.py --report                         # tables from saved runs only
    uv run python evaluate.py --report --hours 1.6,3.2,6.4     # judged by time spent instead of trials
    uv run python evaluate.py --report --landscapes outside --anywhere   # against the best outside the ranges

(Run from the candor repo with PYTHONPATH pointing at this folder, so candor and these modules import.)

Each run (landscape, strategy, seed) is saved under sims/ with every trial's measured and true scores,
so the report can be redone with other tolerances without running again. An anytime strategy runs
once at the longest budget and is judged on its first n trials; any other runs once per budget.
A strategy that scores some edits on fewer prompts is best judged by --hours, as its trials differ in cost.

A run is judged two ways. What it found: how close its trials came to the landscape's best trade-off
near zero refusals, free of measurement noise (the least, over trials with true refusals <= 2 out of
100, of true KL over the best KL at that many refusals); a hit is within 3% (and 10%). What a user
would pick: among trials scored on all prompts with at most 1% refused, the lowest measured KL; its
true KL and true refusals show what the noise of 100 prompts adds on top.
"""

import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from landscape import HERETIC, ONE_LAYER, WIDE
from optimum import NEAR_ZERO, gaps
from strategies import STRATEGIES
from variants import front, key, landscapes, reference

HERE = Path(__file__).parent
SIMS = HERE / "sims"


def folder(name: str, strategy: str) -> Path:
    """Where a landscape's runs are kept: by its settings too, so a changed landscape starts afresh."""
    return SIMS / f"{name}-{key(landscapes()[name])}" / strategy


def run_path(name: str, strategy: str, seed: int, budget: int) -> Path:
    anytime = getattr(STRATEGIES[strategy], "anytime", False)
    return folder(name, strategy) / (f"{seed}.json" if anytime else f"{seed}-{budget}.json")


def run_one(task: tuple[str, str, int, int]) -> str:
    name, strategy_name, seed, budget = task
    path = run_path(name, strategy_name, seed, budget)
    if path.exists() and len(json.loads(path.read_text())["trials"]) >= budget:
        return f"{name}/{strategy_name}/{seed}: kept"
    torch.set_num_threads(1)
    landscape = landscapes()[name]
    started = time.time()
    trials = STRATEGIES[strategy_name](landscape.scorer(seed), budget, seed)
    rates, kls = landscape.true([t.params for t in trials])
    record = {
        "budget": budget,
        "wall seconds": time.time() - started,
        "trials": [
            {**vars(t), "true refusals": 100 * float(r), "true kl": float(k)}
            for t, r, k in zip(trials, rates, kls, strict=True)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return f"{name}/{strategy_name}/{seed}: {record['wall seconds']:.1f} s"


def pick(trials: list[dict], full: int = 100) -> dict | None:
    eligible = [
        t for t in trials if t["prompts"] >= full and not math.isnan(t["kl"]) and t["refusals"] <= 0.01 * t["prompts"]
    ]
    return min(eligible, key=lambda t: t["kl"]) if eligible else None


def found_gap(trials: list[dict], curve: dict[float, float]) -> float:
    """How close a run's trials came to the landscape's best trade-off near zero refusals: the least,
    over trials with true refusals <= NEAR_ZERO, of true KL over the best KL at that many refusals,
    minus 1. inf if no trial got that low."""
    if not trials:
        return math.inf
    return float(gaps(curve, [t["true refusals"] for t in trials], [t["true kl"] for t in trials]).min())


def wilson(hits: int, n: int) -> tuple[float, float]:
    """95% interval for a hit rate."""
    if n == 0:
        return 0.0, 1.0
    p, z = hits / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def within(trials: list[dict], budget: float, hours: bool) -> list[dict]:
    if not hours:
        return trials[: int(budget)]
    spent, kept = 0.0, []
    for t in trials:
        spent += t["seconds"]
        if spent > budget * 3600:
            break
        kept.append(t)
    return kept


def quantile(values: list[float], q: float) -> float:
    """A quantile that treats inf (nothing found) as the worst value rather than as nan."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def summarise(name: str, strategy: str, budgets: list[float], hours: bool, tols: tuple[float, float],
              anywhere: bool) -> list[dict]:
    landscape = landscapes()[name]
    found = reference(landscape)
    best = found["best anywhere" if anywhere else "best"]
    curve = front(landscape, WIDE if anywhere else HERETIC)
    best_one = best.params["direction_scope"] == ONE_LAYER
    anytime = getattr(STRATEGIES[strategy], "anytime", False)
    rows = []
    for budget in budgets:
        # A strategy run once per budget is judged on its own run; by hours, on its longest.
        pattern = "*.json" if anytime else ("*-*.json" if hours else f"*-{int(budget)}.json")
        paths = sorted(folder(name, strategy).glob(pattern))
        if not anytime and hours:
            longest = {}
            for path in paths:
                seed, size = map(int, path.stem.split("-"))
                if size > longest.get(seed, (0, None))[0]:
                    longest[seed] = (size, path)
            paths = [path for _, path in longest.values()]
        distances, picks, spent = [], [], []
        for trials in (json.loads(path.read_text())["trials"] for path in paths):
            if not hours and len(trials) < budget:  # too short to judge at this budget
                continue
            kept = within(trials, budget, hours)
            distances.append(found_gap(kept, curve))
            picks.append(pick(kept))
            spent.append(sum(t["seconds"] for t in kept) / 3600)
        n = len(distances)
        if n == 0:
            continue
        chosen = [p for p in picks if p is not None]
        hits = [sum(d <= tol for d in distances) for tol in tols]
        rows.append({
            "landscape": name, "strategy": strategy, "budget": budget, "runs": n,
            "hit": hits[0] / n, "hit ci": wilson(hits[0], n), "hit wide": hits[1] / n,
            "median gap": quantile(distances, 0.5), "p90 gap": quantile(distances, 0.9),
            "family": sum((p["params"]["direction_scope"] == ONE_LAYER) == best_one for p in chosen) / n,
            "pick gap": float(np.median([p["true kl"] / best.kl - 1 for p in chosen])) if chosen else math.nan,
            "pick refusals": float(np.median([p["true refusals"] for p in chosen])) if chosen else math.nan,
            "no pick": (n - len(chosen)) / n, "hours": float(np.median(spent)), "best kl": best.kl,
        })
    return rows


def _gap(value: float) -> str:
    return "  none" if math.isinf(value) else f"{value:+6.1%}"


def table(rows: list[dict], hours: bool, tols: tuple[float, float]) -> str:
    unit = "hours" if hours else "trials"
    lines = [
        f"found gap = how far above the landscape's best trade-off (KL at the same true refusals, <= {NEAR_ZERO:g}/100) "
        f"a run's closest trial was; hit = within {tols[0]:.0%} ({tols[1]:.0%} for the wide column).\n"
        "pick = the lowest measured KL at <= 1/100 measured, as a user picks: its true KL over the best at <= 1 "
        "refused, its true refusals; family = it is in the best edit's family (one layer or per layer)",
        f"{'landscape':10s} {'strategy':8s} {unit:>6s} {'runs':>4s}  {'hit ' + format(tols[0], '.0%'):>6s} "
        f"{'(95% CI)':>11s} {'hit ' + format(tols[1], '.0%'):>7s}  {'found gap':>9s} {'p90':>6s}  "
        f"{'family':>6s} {'pick gap':>8s} {'pick refused':>12s} {'no pick':>7s} {'hours':>5s}",
    ]
    for r in rows:
        low, high = r["hit ci"]
        lines.append(
            f"{r['landscape']:10s} {r['strategy']:8s} {r['budget']:6g} {r['runs']:4d}  {r['hit']:6.0%} "
            f"({low:4.0%}-{high:4.0%}) {r['hit wide']:7.0%}  {_gap(r['median gap']):>9s} {_gap(r['p90 gap']):>6s}  "
            f"{r['family']:6.0%} {r['pick gap']:+8.1%} {r['pick refusals']:12.1f} {r['no pick']:7.0%} {r['hours']:5.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landscapes", default="base,per-layer,outside,needle")
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--budgets", default="100,200,400", help="trials; the longest is what each run makes")
    parser.add_argument("--hours", default="", help="judge by hours spent instead, e.g. 1.6,3.2,6.4")
    parser.add_argument("--tols", default="0.03,0.10", help="how far above the best KL counts as a hit")
    parser.add_argument("--anywhere", action="store_true", help="judge against the best even outside today's ranges")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--report", action="store_true", help="only report saved runs")
    parser.add_argument("--runs-only", action="store_true", help="run, but leave the report for later")
    args = parser.parse_args()
    names, strategies = args.landscapes.split(","), args.strategies.split(",")
    budgets = [float(b) for b in args.budgets.split(",")]
    landscapes()  # computes and caches the best edits once, before the workers start
    if not args.report:
        tasks = []
        for strategy in strategies:
            anytime = getattr(STRATEGIES[strategy], "anytime", False)
            sizes = [int(max(budgets))] if anytime else sorted({int(b) for b in budgets})
            for name in names:
                for seed in range(args.first_seed, args.first_seed + args.seeds):
                    tasks += [(name, strategy, seed, size) for size in sizes]
        started = time.time()
        with Pool(args.jobs) as pool:
            for i, line in enumerate(pool.imap_unordered(run_one, tasks), 1):
                if i % 10 == 0 or i == len(tasks):
                    print(f"{i}/{len(tasks)} runs, {time.time() - started:.0f} s ({line})", flush=True)
        if args.runs_only:
            return
    hours = bool(args.hours)
    judged = [float(h) for h in args.hours.split(",")] if hours else budgets
    tols = tuple(float(t) for t in args.tols.split(","))
    rows = [
        row
        for name in names
        for strategy in strategies
        for row in summarise(name, strategy, judged, hours, tols, args.anywhere)
    ]
    print(table(rows, hours, tols))


if __name__ == "__main__":
    main()
