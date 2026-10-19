"""Replay candor's 200 rude-llama trials under staged scoring and exact stop rules.

Only the per-trial totals are known (refusals out of 100, KL). Which prompts a trial refuses is
modelled: either uniformly at random, or by a nested model in which each prompt has a difficulty
b_i and a trial refuses its R hardest prompts after per-trial noise of size sigma (sigma 0: every
trial refuses a subset of what a weaker trial refuses; large sigma: random).
"""

import csv
import math
import sys
from pathlib import Path

import numpy as np

D = str(Path(__file__).resolve().parent.parent) + "/"
rows = list(csv.DictReader(open(D + "candor_search_rude.csv")))
R = np.array([int(x["refusals_judge"]) for x in rows])
KL = np.array([float(x["kl"]) for x in rows])
N, BASE, TARGET, STARTUP = 100, 99, 0.01, 60
JUDGE_BATCH, LOOKAHEAD = 16, 12


def costs(model):
    """Seconds per trial part: fixed (KL + overhead), judge per answer, generation per refused and
    per answered prompt. Fitted so a fully scored trial costs 45.5 + 0.286 * (100 - refusals)."""
    if model == "3B":
        return dict(fixed=5.5, judge=0.10, refused=0.30, answered=0.586)
    if model == "3B, KL 10 s":
        return dict(fixed=10.5, judge=0.10, refused=0.25, answered=0.536)
    if model.startswith("9B thinking"):
        length = int(model.split("x")[0].split()[-1])
        return dict(fixed=14.0, judge=0.18, refused=0.30 * 2.7 * length, answered=0.586 * 2.7 * length)
    raise ValueError(model)


def outcomes(order_model, sigma, rng):
    """refused[t, i]: whether trial t refuses prompt i, and the prompt order each trial is scored in."""
    refused = np.zeros((len(R), N), bool)
    b = rng.standard_normal(N)
    for t, r in enumerate(R):
        if order_model == "random":
            refused[t, rng.choice(N, r, replace=False)] = True
        else:
            score = b + sigma * rng.standard_normal(N)
            refused[t, np.argsort(-score)[:r]] = True
    return refused


def orders(refused, ordering, rng):
    """The order each trial's prompts are scored in. 'fixed': one random order for the whole search.
    'hardest first': the startup trials are scored in that fixed order, then every later trial in
    descending order of how often the startup trials refused each prompt (frozen once)."""
    fixed = rng.permutation(N)
    if ordering == "fixed":
        return [fixed] * len(R)
    freq = refused[:STARTUP].sum(0)
    hardest = np.lexsort((np.arange(N), -freq))
    return [fixed if t < STARTUP else hardest for t in range(len(R))]


def objectives(r, kl):
    share = r / BASE
    return (kl if kl >= TARGET else share * TARGET, share)


def surely_dominated_count(done, k, kl):
    """How many fully scored trials surely dominate a trial with exact KL kl and at least k refusals,
    in the sampler's own objectives."""
    lo1, lo2 = objectives(k, kl)
    count = 0
    for o1, o2 in done:
        if o1 <= lo1 and o2 <= lo2 and (o1 < lo1 or o2 < lo2):
            count += 1
    return count


def beaten_on_front(done_rk, k, kl):
    """Whether some fully scored trial beats one with at least k refusals and KL kl, as candor shows it."""
    shown = round(kl, 4)
    return any(r <= k and round(y, 4) <= shown and (r < k or round(y, 4) < shown) for r, y in done_rk)


def stop_after(rule, t, seq, done, done_rk, feasible_kl, cap, g):
    """Prompts judged before the rule stops trial t (N if it never does)."""
    kl = KL[t]
    k = 0
    checkpoints = [0] + list(range(JUDGE_BATCH, N, JUDGE_BATCH))
    cum = np.concatenate([[0], np.cumsum(seq)])
    for n in checkpoints:
        k = cum[n]
        if rule == "none":
            return N
        if rule == "MO exact, sampler unchanged":
            if surely_dominated_count(done, k, kl) >= g:
                return n
        elif rule == "MO exact for the front":
            if beaten_on_front(done_rk, k, kl):
                return n
        elif rule == "cap, sampler unchanged":
            if k > cap:
                beaters = sum(1 for r, _ in done_rk if r <= cap or r < k)
            else:
                beaters = sum(1 for y in feasible_kl if y < kl)
            if beaters >= g:
                return n
        elif rule == "cap only":
            if k > cap:
                return n
        elif rule.startswith("cap, top "):
            m = int(rule.split()[2])
            if k > cap or sum(1 for y in feasible_kl if y < kl) >= m:
                return n
        else:
            raise ValueError(rule)
    return N


def replay(rule, refused, order, cost, cap=5, g=20):
    total, judged_total, pruned, skipped_all = 0.0, 0, 0, 0
    done, done_rk, feasible_kl = [], [], []
    for t in range(len(R)):
        seq = refused[t, order[t]]
        n = stop_after(rule, t, seq, done, done_rk, feasible_kl, cap, g)
        generated = N if n == N else min(N, n + (LOOKAHEAD if n else 0))
        gen = seq[:generated]
        total += cost["fixed"] + cost["judge"] * n + cost["refused"] * gen.sum() + cost["answered"] * (len(gen) - gen.sum())
        judged_total += n
        if n < N:
            pruned += 1
            skipped_all += n == 0
        else:
            done.append(objectives(R[t], KL[t]))
            done_rk.append((R[t], KL[t]))
            if R[t] <= cap:
                feasible_kl.append(KL[t])
    return total / len(R), judged_total / len(R), pruned, skipped_all


def main():
    rng_seed = 0
    reps = 20
    scenarios = [("random", 0.0, "fixed"), ("nested", 0.5, "hardest first"), ("nested", 1.0, "hardest first"), ("nested", 2.0, "hardest first"), ("nested", 1.0, "fixed")]
    rules = [
        ("none", {}),
        ("MO exact, sampler unchanged", {"g": 20}),
        ("MO exact, sampler unchanged", {"g": 25}),
        ("MO exact for the front", {}),
        ("cap, sampler unchanged", {"cap": 5, "g": 20}),
        ("cap, sampler unchanged", {"cap": 10, "g": 20}),
        ("cap, sampler unchanged", {"cap": 5, "g": 25}),
        ("cap only", {"cap": 5}),
        ("cap only", {"cap": 10}),
        ("cap, top 5", {"cap": 5}),
        ("cap, top 1", {"cap": 5}),
        ("cap, top 5", {"cap": 0}),
    ]
    models = sys.argv[1:] or ["3B"]
    for model in models:
        cost = costs(model)
        print(f"\n=== cost model {model}: {cost}")
        for om, sigma, ordering in scenarios:
            print(f"--- refusals {om}{'' if om == 'random' else f' sigma {sigma}'}, order {ordering}")
            base_time = None
            for rule, kw in rules:
                times, judged, pruned, skipped = [], [], [], []
                for rep in range(reps):
                    rng = np.random.default_rng(rng_seed + rep)
                    refused = outcomes(om, sigma, rng)
                    order = orders(refused, ordering, rng)
                    a, b, c, d = replay(rule, refused, order, cost, **kw)
                    times.append(a); judged.append(b); pruned.append(c); skipped.append(d)
                mt = float(np.mean(times))
                base_time = base_time or mt
                label = rule + "".join(f" {k}={v}" for k, v in kw.items())
                print(f"{label:45s} s/trial {mt:7.1f}  ({100 * (1 - mt / base_time):4.1f}% saved, x{base_time / mt:.2f} trials)"
                      f"  prompts judged {np.mean(judged):5.1f}  pruned {np.mean(pruned):5.1f}/200  KL-only {np.mean(skipped):4.1f}")


if __name__ == "__main__":
    main()
