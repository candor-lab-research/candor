"""Compare saved runs of every strategy: hit rates by trials and by hours, the best KL found and picked
at <= 1 refusal, best-of-two repeats and how well two repeats agreeing predicts a hit.

Columns: hit = some trial within 3% (10%) of the best trade-off at its true refusals (<= 2/100);
found<=1 = lowest true KL at <= 1 true refusal over the best there; pick = lowest measured KL at
<= 1/100 measured, its true KL over the best at <= 1, and its true refusals; pick ok = the pick truly
refuses <= 2/100 and is within 10% of the best trade-off at that; accept = the pick passes the
rude-llama check's analogue as measured (<= 1/100 at KL within 3% of Trial 170's), which noise can fool.

    uv run python analyze.py [--landscapes base,needle] [--strategies candor,gp] [--section all]

Reads sims/ only (run evaluate.py --runs-only first). Every strategy here is anytime, so a run's first
n trials, or its trials within h hours, are what it would have done with that budget.
"""

import argparse
import itertools
import json
import math

import numpy as np

from evaluate import folder, found_gap, pick, wilson, within
from landscape import HERETIC, ONE_LAYER, WIDE
from variants import front, landscapes, reference

# Trial 170 read at layer 12 on the base landscape (calibrate.log): its true KL. The rude-llama
# acceptance check passes a pick at <= 1/100 measured within 3% of Trial 170's KL.
T170_KL = 0.0759


def load(name: str, strategy: str, budget: int | None = None) -> list[list[dict]]:
    """Each seed's run: for a strategy run once per budget, its run at that budget, or its longest."""
    longest = {}
    for path in folder(name, strategy).glob("*.json"):
        seed, _, size = path.stem.partition("-")
        size = int(size) if size else 0
        if budget is not None and size and size != budget:
            continue
        if size >= longest.get(int(seed), (-1, None))[0]:
            longest[int(seed)] = (size, path)
    return [json.loads(longest[s][1].read_text())["trials"] for s in sorted(longest)]


def found_at_one(trials: list[dict]) -> float:
    """The lowest true KL among trials with true refusals <= 1."""
    return min((t["true kl"] for t in trials if t["true refusals"] <= 1.0), default=math.inf)


def ci(hits: int, n: int) -> str:
    low, high = wilson(hits, n)
    return f"{hits / n:4.0%} ({low:3.0%}-{high:3.0%})"


def budgets_table(name, strategies, budgets, hours, anywhere=False):
    landscape = landscapes()[name]
    found = reference(landscape)
    best = found["best anywhere" if anywhere else "best"]
    curve = front(landscape, WIDE if anywhere else HERETIC)
    unit = "hours" if hours else "trials"
    print(f"\n== {name}: by {unit}; best KL at <= 1 true refusal {best.kl:.4f} "
          f"({'one layer' if best.params['direction_scope'] == ONE_LAYER else 'per layer'})"
          f"{' (anywhere)' if anywhere else ''}")
    print(f"{'strategy':9s} {unit:>6s} {'n':>3s}  {'hit 3% (95% CI)':>16s}  {'hit 10% (95% CI)':>17s}  "
          f"{'gap med':>7s} {'p90':>7s}  {'found<=1':>8s}  {'pick med':>8s} {'pick p90':>8s} {'pick ref':>8s}  "
          f"{'pick ok':>7s} {'accept':>6s} {'trials':>6s} {'hours':>5s}")
    for strategy in strategies:
        for budget in budgets:
            runs = load(name, strategy, None if hours else int(budget))
            if not runs:
                continue
            kept = [within(r, budget, hours) for r in runs]
            if not hours and any(len(k) < budget for k in kept):
                continue
            gaps = np.array([found_gap(k, curve) for k in kept])
            n = len(gaps)
            at_one = np.array([found_at_one(k) / best.kl - 1 for k in kept])
            picks = [pick(k) for k in kept]
            pick_gaps = np.array([p["true kl"] / best.kl - 1 if p else math.inf for p in picks])
            pick_ref = np.median([p["true refusals"] for p in picks if p]) if any(picks) else math.nan
            accept = sum(1 for p in picks if p and p["kl"] <= 1.03 * T170_KL)
            good = sum(1 for p in picks if p and found_gap([p], curve) <= 0.10)
            print(
                f"{strategy:9s} {budget:6g} {n:3d}  {ci(int((gaps <= 0.03).sum()), n):>16s}  "
                f"{ci(int((gaps <= 0.10).sum()), n):>17s}  {fmt(np.median(gaps)):>7s} {fmt(q(gaps, 0.9)):>7s}  "
                f"{fmt(np.median(at_one)):>8s}  {fmt(np.median(pick_gaps)):>8s} {fmt(q(pick_gaps, 0.9)):>8s} "
                f"{pick_ref:8.1f}  {good / n:7.0%} {accept / n:6.0%} {np.median([len(k) for k in kept]):6.0f} "
                f"{np.median([sum(t['seconds'] for t in k) / 3600 for k in kept]):5.2f}"
            )


def q(values, quantile):
    ordered = np.sort(values)
    return ordered[min(len(ordered) - 1, int(round(quantile * (len(ordered) - 1))))]


def fmt(value: float) -> str:
    return "none" if math.isinf(value) else f"{value:+.1%}"


def repeats_table(name, strategies, total, tol_agree=0.05):
    """k independent repeats of total/k trials each (k = 1, 2, 4), against one run of total. Repeats
    are formed from every combination of distinct seeds, so rates are averages over combinations;
    their effective sample size is still the number of seeds. Agreement: every repeat's pick (lowest
    measured KL at <= 1/100 measured) is within tol_agree of the best pick, in measured KL."""
    landscape = landscapes()[name]
    best = reference(landscape)["best"]
    curve = front(landscape, HERETIC)
    print(f"\n== {name}: {total} trials as k repeats; agree = every repeat's pick within {tol_agree:.0%} of the best pick")
    print(f"{'strategy':9s} {'k':>2s} {'combos':>6s}  {'hit 3%':>6s} {'hit 10%':>7s}  {'pick med':>8s}  "
          f"{'agree':>5s}  {'miss10|agree':>12s} {'miss10|disagree':>15s}  {'miss3|agree':>11s}")
    for strategy in strategies:
        runs = load(name, strategy, total)
        if not runs or any(len(r) < total for r in runs):
            continue
        for k in (1, 2, 4):
            size = total // k
            heads = [r[:size] for r in runs]
            gaps = [found_gap(h, curve) for h in heads]
            picks = [pick(h) for h in heads]
            combos = list(itertools.combinations(range(len(runs)), k))
            if len(combos) > 3000:
                rng = np.random.default_rng(0)
                combos = [tuple(sorted(rng.choice(len(runs), k, replace=False))) for _ in range(3000)]
            hit3 = hit10 = agree = miss10_agree = miss10_dis = miss3_agree = 0
            pick_gaps = []
            for combo in combos:
                gap = min(gaps[i] for i in combo)
                hit3 += gap <= 0.03
                hit10 += gap <= 0.10
                chosen = [picks[i] for i in combo]
                measured = [p["kl"] if p else math.inf for p in chosen]
                top = int(np.argmin(measured))
                pick_gaps.append(chosen[top]["true kl"] / best.kl - 1 if chosen[top] else math.inf)
                agreed = all(m <= (1 + tol_agree) * measured[top] for m in measured)
                agree += agreed
                if agreed:
                    miss10_agree += gap > 0.10
                    miss3_agree += gap > 0.03
                else:
                    miss10_dis += gap > 0.10
            n = len(combos)
            n_dis = n - agree
            print(
                f"{strategy:9s} {k:2d} {n:6d}  {hit3 / n:6.0%} {hit10 / n:7.0%}  {fmt(float(np.median(pick_gaps))):>8s}  "
                f"{agree / n:5.0%}  {rate(miss10_agree, agree):>12s} {rate(miss10_dis, n_dis):>15s}  "
                f"{rate(miss3_agree, agree):>11s}"
            )


def still_improving(trials: list[dict], share: float = 0.25, gain: float = 0.02) -> bool:
    """Whether the pick (lowest measured KL at <= 1/100 measured) improved by more than gain over the
    last share of the trials, or there was no pick before then."""
    before = pick(trials[: int(len(trials) * (1 - share))])
    now = pick(trials)
    if now is None:
        return True
    return before is None or now["kl"] < (1 - gain) * before["kl"]


def diagnostics_table(name, strategies, total):
    """How well a run's own warning flags its misses: 'still improving' over its last quarter."""
    landscape = landscapes()[name]
    curve = front(landscape, HERETIC)
    print(f"\n== {name}: one run of {total}; flag = its pick improved > 2% in the last 25% of trials")
    print(f"{'strategy':9s} {'n':>3s}  {'flagged':>7s}  {'miss10|no flag':>14s} {'miss10|flag':>11s}  {'miss3|no flag':>13s}")
    for strategy in strategies:
        runs = load(name, strategy, total)
        if not runs or any(len(r) < total for r in runs):
            continue
        heads = [r[:total] for r in runs]
        flags = [still_improving(h) for h in heads]
        gaps = [found_gap(h, curve) for h in heads]
        quiet = [g for g, f in zip(gaps, flags) if not f]
        loud = [g for g, f in zip(gaps, flags) if f]
        print(f"{strategy:9s} {len(heads):3d}  {sum(flags) / len(heads):7.0%}  "
              f"{rate(sum(g > 0.10 for g in quiet), len(quiet)):>14s} {rate(sum(g > 0.10 for g in loud), len(loud)):>11s}  "
              f"{rate(sum(g > 0.03 for g in quiet), len(quiet)):>13s}")


def confirmed_pick(trials: list[dict], landscape, seed: int, k: int, extra: int, cap: int = 2) -> dict | None:
    """The pick after re-scoring finalists on more prompts: the k lowest measured KL among trials with
    <= cap/100 measured are scored on extra fresh prompts, and the pick is the lowest KL whose refusal
    rate over all their prompts is <= 1%, else the lowest rate."""
    finalists = sorted((t for t in trials if t["prompts"] >= 100 and t["refusals"] <= cap), key=lambda t: t["kl"])[:k]
    if not finalists:
        return None
    scorer = landscape.scorer(seed + 100_000, prompts=extra)
    rated = []
    for t in finalists:
        total = t["refusals"] + scorer.score(t["params"], kl_pass=False).refusals
        rated.append((total / (t["prompts"] + extra), t))
    passing = [t for r, t in rated if r <= 0.01]
    return min(passing, key=lambda t: t["kl"]) if passing else min(rated, key=lambda rt: rt[0])[1]


def confirm_table(name, strategies, total, ks=(5, 10), extra=288):
    landscape = landscapes()[name]
    best = reference(landscape)["best"]
    curve = front(landscape, HERETIC)
    print(f"\n== {name}: pick after confirming the k best finalists on {extra} more prompts ({total} trials)")
    print(f"{'strategy':9s} {'k':>3s}  {'pick med':>8s} {'pick ref':>8s} {'pick ok':>7s}")
    for strategy in strategies:
        runs = load(name, strategy, total)
        if not runs or any(len(r) < total for r in runs):
            continue
        for k in (0, *ks):
            picks = []
            for seed, r in enumerate(runs):
                head = r[:total]
                picks.append(pick(head) if k == 0 else confirmed_pick(head, landscape, seed, k, extra))
            chosen = [p for p in picks if p]
            gaps = [p["true kl"] / best.kl - 1 for p in chosen]
            ok = sum(found_gap([p], curve) <= 0.10 for p in chosen)
            print(f"{strategy:9s} {k:3d}  {fmt(float(np.median(gaps))):>8s} "
                  f"{np.median([p['true refusals'] for p in chosen]):8.1f} {ok / len(runs):7.0%}")


def rate(a: int, n: int) -> str:
    return f"{a / n:.0%} of {n}" if n else "-"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landscapes", default="base,needle,per-layer,outside")
    parser.add_argument("--strategies", default="candor,heretic,random,tpesplit,tpewide,gp,gpwide,gpstop")
    parser.add_argument("--section", default="all", help="trials, hours, repeats, anywhere or all")
    parser.add_argument("--repeat-total", type=int, default=400)
    args = parser.parse_args()
    names, strategies = args.landscapes.split(","), args.strategies.split(",")
    for name in names:
        if args.section in ("all", "trials"):
            budgets_table(name, strategies, [100, 200, 300, 400], hours=False)
        if args.section in ("all", "hours"):
            budgets_table(name, strategies, [1.6, 3.2, 4.8, 6.4], hours=True)
        if args.section in ("all", "anywhere") and name == "outside":
            budgets_table(name, strategies, [200, 400], hours=False, anywhere=True)
            budgets_table(name, strategies, [3.2, 6.4], hours=True, anywhere=True)
        if args.section in ("all", "repeats"):
            repeats_table(name, strategies, args.repeat_total)
        if args.section in ("all", "confirm"):
            confirm_table(name, strategies, args.repeat_total)
        if args.section in ("all", "diagnostics"):
            diagnostics_table(name, strategies, args.repeat_total)


if __name__ == "__main__":
    main()
