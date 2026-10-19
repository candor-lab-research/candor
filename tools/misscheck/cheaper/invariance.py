"""Check that pruning a trial once it is surely in TPE's 'above' group leaves Optuna 4.9's TPE
proposing exactly the same trials, for candor's two objectives and for a constrained single
objective. A synthetic scorer stands in for the model: per-prompt refusals and KL from the params.
"""

import math
import sys
import time
import warnings

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.simplefilter("ignore")

N, BASE, TARGET, CHUNK, LAYERS = 100, 99, 0.01, 16, 28
DIFFICULTY = np.random.default_rng(123).standard_normal(N)
ORDER = np.argsort(-DIFFICULTY)  # hardest first


def suggest(trial):
    trial.suggest_categorical("direction_scope", ["one layer", "per layer"])
    trial.suggest_int("direction_layer", 11, 24)
    for c in ("attn", "mlp"):
        trial.suggest_float(f"{c}.strength", 0.8, 1.5)
        trial.suggest_float(f"{c}.peak_layer", 0.6, 1.0)
        trial.suggest_float(f"{c}.edge_strength", 0.0, 1.0)
        trial.suggest_float(f"{c}.reach", 1 / 27, 0.6)


def truth(p, number):
    """A made-up landscape: per-prompt refusals (in ORDER) and KL."""
    mass = sum(p[f"{c}.strength"] * (0.3 + p[f"{c}.reach"]) * (0.5 + 0.5 * p[f"{c}.edge_strength"]) for c in ("attn", "mlp"))
    if p["direction_scope"] == "one layer":
        fit = math.exp(-((p["direction_layer"] - 12) / 2.5) ** 2) * math.exp(-((p["mlp.peak_layer"] - 0.62) / 0.08) ** 2)
    else:
        fit = 0.7 * math.exp(-((p["mlp.peak_layer"] - 0.95) / 0.1) ** 2)
    theta = 2.2 - 2.5 * fit * mass
    noise = np.random.default_rng(number).standard_normal(N) * 0.4
    refused = (DIFFICULTY + noise) > theta
    kl = 0.02 * mass + 0.03 * (1 - fit) * mass + 0.002 * np.random.default_rng(number + 10**6).random()
    return refused[ORDER], kl


def objectives(r, kl):
    share = r / BASE
    return (kl if kl >= TARGET else share * TARGET, share)


def run(mode, prune, g=25, trials=200, cap=5):
    """mode 'mo': candor's two objectives; 'cap': min KL subject to refusals <= cap."""
    if mode == "mo":
        sampler = optuna.samplers.TPESampler(n_startup_trials=60, n_ei_candidates=128, multivariate=True, seed=0)
        study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
    else:
        def constraints(t):
            return (t.user_attrs["violation"],)
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=60, n_ei_candidates=128, multivariate=True, seed=0, constraints_func=constraints
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)
    done = []  # fully scored: (refusals, kl)
    judged = 0
    params = []
    for _ in range(trials):
        trial = study.ask()
        suggest(trial)
        params.append(dict(trial.params))
        seq, kl = truth(trial.params, trial.number)
        stopped = None
        for n in range(0, N, CHUNK):
            k = int(seq[:n].sum())
            if prune and surely_above(mode, done, k, kl, g, cap):
                stopped = n
                break
        if stopped is None:
            r = int(seq.sum())
            judged += N
            done.append((r, kl))
            if mode == "mo":
                study.tell(trial, objectives(r, kl))
            else:
                trial.set_user_attr("violation", (r - cap) / BASE)
                study.tell(trial, kl)
        else:
            judged += stopped
            if mode == "cap":
                trial.set_user_attr("violation", float("inf"))
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    return params, judged, study


def surely_above(mode, done, k, kl, g, cap):
    if mode == "mo":
        lo1, lo2 = objectives(k, kl)
        beaters = 0
        for r, y in done:
            o1, o2 = objectives(r, y)
            beaters += o1 <= lo1 and o2 <= lo2 and (o1 < lo1 or o2 < lo2)
        return beaters >= g
    if k > cap:
        return sum(r < k for r, _ in done) >= g
    return sum(r <= cap and y < kl for r, y in done) >= g


def main():
    for mode in sys.argv[1:] or ["mo", "cap"]:
        started = time.perf_counter()
        full, judged_full, _ = run(mode, prune=False)
        pruned, judged_pruned, study = run(mode, prune=True)
        same = sum(a == b for a, b in zip(full, pruned))
        first_diff = next((i for i, (a, b) in enumerate(zip(full, pruned)) if a != b), None)
        n_pruned = len(study.get_trials(states=(optuna.trial.TrialState.PRUNED,)))
        print(f"{mode}: identical params in {same}/{len(full)} trials (first difference: {first_diff}); "
              f"pruned {n_pruned}; prompts judged {judged_pruned} vs {judged_full} "
              f"({100 * (1 - judged_pruned / judged_full):.1f}% fewer); {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
