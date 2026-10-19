"""Which earlier trial each of TPE's proposals was drawn around, replaying candor's history in Optuna 4.9."""
import sys, time
import numpy as np
import optuna
from optuna.samplers._tpe import probability_distributions as pdist
from optuna.samplers._tpe.sampler import TPESampler, _split_trials, default_gamma
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from load import candor
from replay import DIST, ONE, study_of

record = {}
_sample = pdist._MixtureOfProductDistribution.sample


def sample(self, rng, batch_size):
    state = rng.get_state()
    record.setdefault("active", []).append(rng.choice(len(self.weights), p=self.weights, size=batch_size))
    rng.set_state(state)
    return _sample(self, rng, batch_size)


_compare = TPESampler._compare.__func__


def compare(cls, samples, acq):
    record["best"] = int(np.argmax(acq))
    return _compare(cls, samples, acq)


pdist._MixtureOfProductDistribution.sample = sample
TPESampler._compare = classmethod(compare)

if __name__ == "__main__":
    t0 = time.time()
    c = candor().set_index("trial", drop=False)
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    tally = {}
    for n in range(60, 200, 10):
        study = study_of(c[c.trial <= n])
        trials = study.get_trials(deepcopy=False)
        below, _ = _split_trials(study, trials, default_gamma(n), False)
        for i in range(k):
            record.clear()
            study.sampler = optuna.samplers.TPESampler(n_startup_trials=1, n_ei_candidates=128, multivariate=True, seed=7 + 1000 * n + i)
            trial = study.ask()
            for name, dist in DIST.items():
                trial._suggest(name, dist)
            if trial.params["direction_scope"] != ONE:
                continue
            component = record["active"][0][record["best"]]
            if component == len(below):
                source = "prior"
            else:
                src = c.loc[below[component].number + 1]
                source = f"{'one' if src.scope == 'one_layer' else 'per'}-layer ({'<=10' if src.ref <= 10 else '>10'} refusals)"
            late = trial.params["mlp.peak_layer"] - trial.params["mlp.reach"] > 0.5
            key = (n // 50 * 50, source)
            t = tally.setdefault(key, [0, 0])
            t[0] += 1
            t[1] += late
    print("one-layer proposals by the earlier trial they were drawn around (trials-so-far bucket, source): count, of which MLP band starts past mid-depth")
    for key in sorted(tally):
        print(key, tally[key])
    print(f"took {time.time() - t0:.0f}s")
