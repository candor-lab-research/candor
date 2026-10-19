"""Two objectives (candor's) plus a refusal cap as an Optuna constraint: pruning a trial once it is
surely infeasible and >= G fully scored trials have fewer refusals must leave TPE's proposals unchanged."""
import sys
import optuna
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import invariance as inv

CAP = 5


def run(prune, g):
    sampler = optuna.samplers.TPESampler(n_startup_trials=60, n_ei_candidates=128, multivariate=True, seed=0,
                                         constraints_func=lambda t: (t.user_attrs["violation"],))
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
    done, params, judged = [], [], 0
    for _ in range(200):
        trial = study.ask()
        inv.suggest(trial)
        params.append(dict(trial.params))
        seq, kl = inv.truth(trial.params, trial.number)
        stop = None
        for n in range(16, inv.N, 16):
            k = int(seq[:n].sum())
            if prune and k > CAP and sum(r < k for r in done) >= g:
                stop = (n, k)
                break
        if stop is None:
            r = int(seq.sum())
            judged += inv.N
            done.append(r)
            trial.set_user_attr("violation", (r - CAP) / inv.BASE)
            study.tell(trial, inv.objectives(r, kl))
        else:
            n, k = stop
            judged += n
            trial.set_user_attr("violation", (k * inv.N / n - CAP) / inv.BASE)
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    return params, judged


full, jf = run(False, 0)
for g in (1, 5, 20, 25):
    p, jp = run(True, g)
    first = next((i for i, (a, b) in enumerate(zip(full, p)) if a != b), None)
    print(f"two objectives + cap, G={g}: first differing trial {first}; prompts judged {jp} vs {jf}")
