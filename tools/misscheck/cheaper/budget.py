"""Equal wall-clock comparison on a made-up landscape shaped like the rude-llama story: a narrow
one-layer region (direction layer ~12, MLP peak ~0.62, at the edge of its range) that beats a broad
per-layer region. Objective: min KL subject to refusals <= CAP of 100 (constrained TPE).

Arms, all given the wall-clock that 200 fully scored trials took on the same seed:
  full      every trial scored on all 100 prompts
  exact     stop a trial once >= 25 fully scored trials surely rank ahead of it in TPE's split
            (the sampler proposes exactly what it would under full scoring)
  top5-inf  stop once refusals > CAP, or once 5 feasible trials have lower KL; told as infeasible
            with unknown (infinite) violation
  top5-lb   as top5-inf, but a trial stopped for refusals is told its violation so far
"""

import math
import sys
import time
import warnings

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.simplefilter("ignore")

N, BASE, CAP, CHUNK, LOOKAHEAD = 100, 99, 5, 16, 12
COST = dict(fixed=5.5, judge=0.10, refused=0.30, answered=0.586)
DIFFICULTY = np.random.default_rng(123).standard_normal(N)
ORDER = np.argsort(-DIFFICULTY)


def suggest(trial):
    trial.suggest_categorical("direction_scope", ["one layer", "per layer"])
    trial.suggest_int("direction_layer", 11, 24)
    for c in ("attn", "mlp"):
        trial.suggest_float(f"{c}.strength", 0.8, 1.5)
        trial.suggest_float(f"{c}.peak_layer", 0.6, 1.0)
        trial.suggest_float(f"{c}.edge_strength", 0.0, 1.0)
        trial.suggest_float(f"{c}.reach", 1 / 27, 0.6)


def landscape(p):
    mass = sum(p[f"{c}.strength"] * (0.3 + p[f"{c}.reach"]) * (0.5 + 0.5 * p[f"{c}.edge_strength"]) for c in ("attn", "mlp"))
    if p["direction_scope"] == "one layer":
        fit = (0.3 * math.exp(-((p["direction_layer"] - 14) / 5.0) ** 2)
               + 0.85 * math.exp(-((p["direction_layer"] - 12) / 2.0) ** 2)
               * math.exp(-((p["mlp.peak_layer"] - 0.62) / 0.07) ** 2)
               * (0.5 + 0.5 * p["mlp.reach"] / 0.6))
    else:
        fit = 0.8 * math.exp(-((p["mlp.peak_layer"] - 0.95) / 0.3) ** 2)
    theta = -1.5 + 2.8 * fit * mass
    kl = 0.045 * mass + 0.02 * (1 - min(fit, 1.0)) * mass
    return theta, kl


def truth(p, key):
    theta, kl = landscape(p)
    rng = np.random.default_rng(key)
    refused = (DIFFICULTY + 0.4 * rng.standard_normal(N)) > theta
    return refused[ORDER], kl * (1 + 0.03 * rng.standard_normal())


def noiseless(p):
    theta, kl = landscape(p)
    return int((DIFFICULTY > theta).sum()), kl


def trial_cost(seq, judged):
    generated = N if judged == N else min(N, judged + (LOOKAHEAD if judged else 0))
    gen = seq[:generated]
    return COST["fixed"] + COST["judge"] * judged + COST["refused"] * gen.sum() + COST["answered"] * (len(gen) - gen.sum())


def stop_at(arm, seq, kl, done):
    if arm == "full":
        return None
    for n in range(0, N, CHUNK):
        k = int(seq[:n].sum())
        if arm == "exact":
            if k > CAP:
                beaters = sum(r < k for r, _ in done)
            else:
                beaters = sum(r <= CAP and y < kl for r, y in done)
            if beaters >= 25:
                return n, k
        elif arm.startswith("cap-"):
            if k > CAP:
                return n, k
        else:
            if k > CAP or sum(r <= CAP and y < kl for r, y in done) >= 5:
                return n, k
    return None


def run(arm, seed, budget=None, trials=200):
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=60, n_ei_candidates=128, multivariate=True, seed=seed,
        constraints_func=lambda t: (t.user_attrs["violation"],),
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    done, spent, count = [], 0.0, 0
    best = None
    while (budget is None and count < trials) or (budget is not None and spent < budget):
        trial = study.ask()
        suggest(trial)
        seq, kl = truth(trial.params, (seed, trial.number))
        stop = stop_at(arm, seq, kl, done)
        if stop is None:
            r = int(seq.sum())
            spent += trial_cost(seq, N)
            done.append((r, kl))
            trial.set_user_attr("violation", (r - CAP) / BASE)
            study.tell(trial, kl)
            if r <= CAP and (best is None or kl < best[0]):
                best = (kl, dict(trial.params))
        else:
            n, k = stop
            spent += trial_cost(seq, n)
            violation = (k - CAP) / BASE if arm.endswith("-lb") and k > CAP else float("inf")
            trial.set_user_attr("violation", violation)
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        count += 1
    return best, spent, count


def optimum(samples=200_000):
    rng = np.random.default_rng(7)
    best = math.inf
    for focused in (False, True):
        for _ in range(samples):
            p = {"direction_scope": "one layer", "direction_layer": int(rng.integers(11, 25)) if not focused else 12}
            for c in ("attn", "mlp"):
                p[f"{c}.strength"] = rng.uniform(0.8, 1.5)
                p[f"{c}.peak_layer"] = rng.uniform(0.6, 1.0) if not (focused and c == "mlp") else rng.uniform(0.6, 0.66)
                p[f"{c}.edge_strength"] = rng.uniform(0, 1)
                p[f"{c}.reach"] = rng.uniform(1 / 27, 0.6)
            r, kl = noiseless(p)
            if r <= CAP and kl < best:
                best = kl
    return best


def main():
    seeds = range(int(sys.argv[1]), int(sys.argv[2]))
    started = time.perf_counter()
    opt = optimum(50_000)
    print(f"noiseless optimum (one layer, refusals <= {CAP}): KL {opt:.4f}")
    arms = sys.argv[3].split(",") if len(sys.argv) > 3 else ["exact", "top5-inf", "top5-lb"]
    results = {arm: [] for arm in ["full", *arms]}
    for seed in seeds:
        best_full, budget, _ = run("full", seed)
        results["full"].append((best_full, 200))
        for arm in arms:
            if arm.startswith("full-"):
                best, _, count = run("full", seed, trials=int(arm.split("-")[1]))
            else:
                best, _, count = run(arm, seed, budget=budget)
            results[arm].append((best, count))
    for arm, found in results.items():
        counts = [count for _, count in found]
        one_layer = sum(best is not None and best[1]["direction_scope"] == "one layer" for best, _ in found)
        scored = [best[0] / opt - 1 if best else math.inf for best, _ in found]
        clean = [noiseless(best[1]) if best else (N, math.inf) for best, _ in found]
        near = sum(r <= CAP + 3 and kl <= 1.05 * opt for r, kl in clean)
        print(f"{arm:9s} trials {np.mean(counts):6.1f}  best is one-layer {one_layer}/{len(found)}  "
              f"noiselessly within 5% KL and <= {CAP + 3} refusals {near}/{len(found)}  "
              f"median scored-KL regret {100 * np.median(scored):5.1f}%  median noiseless refusals "
              f"{np.median([r for r, _ in clean]):.0f}")
    import json
    rows = []
    for arm, found in results.items():
        for seed, (best, count) in zip(seeds, found):
            r, kl = noiseless(best[1]) if best else (N, None)
            rows.append(dict(arm=arm, seed=seed, trials=count, one_layer=bool(best and best[1]["direction_scope"] == "one layer"),
                             scored_kl=best[0] if best else None, clean_refusals=r, clean_kl=kl, optimum=opt))
    json.dump(rows, open(f"budget_{seeds.start}_{seeds.stop}_{len(arms)}.json", "w"))
    print(f"took {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
