"""Does telling TPE an estimated violation for curtailed trials mislead it? Same made-up landscape
as budget.py; constrained TPE (min KL, refusals <= CAP). Curtailed arms measure KL only for trials
that finish and check every 8 judged answers, with 8 answers wasted at a stop.

  full       200 fully scored trials; its cost is every other arm's wall-clock budget
  exact      stop once refusals > CAP and 25 fully scored trials have fewer (TPE proposes as 'full')
  cap-lb     stop once refusals > CAP and 1 fully scored trial has fewer; told (k - CAP) / BASE
  cap-est    as cap-lb; told (k * N / n - CAP) / BASE, the refusal rate so far over all prompts
  full-same  full scoring for as many trials as cap-est ran on that seed: full information
"""

import json
import math
import sys
import time

import numpy as np
import optuna

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from budget import BASE, CAP, COST, N, noiseless, optimum, suggest, truth  # noqa: E402

CHUNK, LOOKAHEAD = 8, 8


def cost(seq, judged):
    if judged == N:
        return COST["fixed"] + COST["judge"] * N + COST["refused"] * seq.sum() + COST["answered"] * (N - seq.sum())
    gen = seq[: min(N, judged + LOOKAHEAD)]
    return 0.5 + COST["judge"] * judged + COST["refused"] * gen.sum() + COST["answered"] * (len(gen) - gen.sum())


def run(arm, seed, budget=None, trials=None):
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=60, n_ei_candidates=128, multivariate=True, seed=seed,
        constraints_func=lambda t: (t.user_attrs["violation"],),
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    done_r, spent, count, best = [], 0.0, 0, None
    g = {"exact": 25, "cap-lb": 1, "cap-est": 1}.get(arm)
    while (trials is not None and count < trials) or (budget is not None and spent < budget):
        trial = study.ask()
        suggest(trial)
        seq, kl = truth(trial.params, (seed, trial.number))
        stop = None
        if g is not None:
            cum = np.concatenate([[0], np.cumsum(seq)])
            for n in range(CHUNK, N, CHUNK):
                k = int(cum[n])
                if k > CAP and sum(r < k for r in done_r) >= g:
                    stop = (n, k)
                    break
        if stop is None:
            r = int(seq.sum())
            spent += cost(seq, N)
            done_r.append(r)
            trial.set_user_attr("violation", (r - CAP) / BASE)
            study.tell(trial, kl)
            if r <= CAP and (best is None or kl < best[0]):
                best = (kl, dict(trial.params))
        else:
            n, k = stop
            spent += cost(seq, n)
            violation = {"cap-lb": (k - CAP) / BASE, "cap-est": (k * N / n - CAP) / BASE}.get(arm, math.inf)
            trial.set_user_attr("violation", violation)
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        count += 1
    return best, count


def main():
    seeds = range(int(sys.argv[1]), int(sys.argv[2]))
    started = time.perf_counter()
    opt = optimum(50_000)
    rows = []
    for seed in seeds:
        best, _ = run("full", seed, trials=200)
        results = {"full": (best, 200)}
        sampler_budget = full_cost(seed)  # what those 200 trials cost
        for arm in ("exact", "cap-lb", "cap-est"):
            results[arm] = run(arm, seed, budget=sampler_budget)
        results["full-same"] = run("full", seed, trials=results["cap-est"][1])
        for arm, (b, count) in results.items():
            r, kl = noiseless(b[1]) if b else (N, None)
            rows.append(dict(arm=arm, seed=seed, trials=count, one_layer=bool(b and b[1]["direction_scope"] == "one layer"),
                             scored_kl=b[0] if b else None, clean_refusals=r, clean_kl=kl, optimum=opt))
    json.dump(rows, open(f"budget2_{seeds.start}_{seeds.stop}.json", "w"))
    print(f"took {time.perf_counter() - started:.0f} s")


def full_cost(seed):
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=60, n_ei_candidates=128, multivariate=True, seed=seed,
        constraints_func=lambda t: (t.user_attrs["violation"],),
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    spent = 0.0
    for _ in range(200):
        trial = study.ask()
        suggest(trial)
        seq, kl = truth(trial.params, (seed, trial.number))
        spent += cost(seq, N)
        trial.set_user_attr("violation", (int(seq.sum()) - CAP) / BASE)
        study.tell(trial, kl)
    return spent


if __name__ == "__main__":
    main()
