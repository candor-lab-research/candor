"""Replay candor's search history through Optuna 4.9's TPE to see what it would propose next."""
import sys, time, warnings
import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.samplers._tpe.sampler import _split_complete_trials
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from load import candor, KCOLS

optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.simplefilter("ignore")
ONE, PER = "one layer", "per layer"
DIST = {"direction_scope": CategoricalDistribution([ONE, PER]), "direction_layer": IntDistribution(11, 24)}
for comp in ("attn", "mlp"):
    DIST[f"{comp}.strength"] = FloatDistribution(0.8, 1.5)
    DIST[f"{comp}.peak_layer"] = FloatDistribution(0.6, 1.0)
    DIST[f"{comp}.edge_strength"] = FloatDistribution(0.0, 1.0)
    DIST[f"{comp}.reach"] = FloatDistribution(1 / 27, 0.6)


def frozen(row):
    params = {"direction_scope": ONE if row.scope == "one_layer" else PER, "direction_layer": int(row.direction_layer)}
    for col in KCOLS:
        params[col] = float(row[col])
    refused = row.ref / 99
    values = [row.kl if row.kl >= 0.01 else refused * 0.01, refused]
    return optuna.trial.create_trial(params=params, distributions=DIST, values=values)


def study_of(rows):
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.add_trials([frozen(r) for _, r in rows.iterrows()])
    return study


def good_region(p):
    """Heretic's good one-layer shape: both bands from layer <= 0.3 depth to the last layer, strong attention."""
    ok = p["direction_scope"] == ONE and 12 <= p["direction_layer"] <= 15
    for comp in ("attn", "mlp"):
        start = p[f"{comp}.peak_layer"] - p[f"{comp}.reach"]
        end = p[f"{comp}.peak_layer"] + p[f"{comp}.reach"]
        ok &= start <= 0.3 and end >= 0.95
    return ok and p["attn.strength"] >= 1.1 and p["mlp.strength"] >= 0.9


def propose(study, k, seed0=1000):
    out = []
    for i in range(k):
        study.sampler = optuna.samplers.TPESampler(n_startup_trials=1, n_ei_candidates=128, multivariate=True, seed=seed0 + i)
        trial = study.ask()
        for name, dist in DIST.items():
            trial._suggest(name, dist)
        out.append(dict(trial.params))
    return out


def summary(props):
    one = [p for p in props if p["direction_scope"] == ONE]
    def med(key):
        return np.median([p[key] for p in one]) if one else float("nan")
    mlp_start = np.median([p["mlp.peak_layer"] - p["mlp.reach"] for p in one]) if one else float("nan")
    return (f"share one={len(one)/len(props):.2f}  one-layer: mlp.peak med={med('mlp.peak_layer'):.2f} "
            f"mlp.reach med={med('mlp.reach'):.2f} mlp start med={mlp_start:.2f} "
            f"attn.peak med={med('attn.peak_layer'):.2f} attn.reach med={med('attn.reach'):.2f}  "
            f"in heretic-good region: {sum(map(good_region, props))}/{len(props)}")


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    c = candor()
    t0 = time.time()
    for n in (60, 80, 100, 130, 160, 199):
        hist = c[c.trial <= n]
        study = study_of(hist)
        trials = study.get_trials(deepcopy=False)
        below, _ = _split_complete_trials(trials, study, min(int(np.ceil(0.1 * n)), 25))
        below_ids = [t.number + 1 for t in below]
        b = hist[hist.trial.isin(below_ids)]
        print(f"after {n} trials: below set {len(b)} = {sum(b.scope=='one_layer')} one-layer "
              f"(refusals {sorted(b[b.scope=='one_layer'].ref.tolist())}, mlp.peak {sorted(b[b.scope=='one_layer']['mlp.peak_layer'].round(2).tolist())}) "
              f"+ {sum(b.scope=='per_layer')} per-layer (refusals {sorted(b[b.scope=='per_layer'].ref.tolist())})")
        print("   actual history  :", summary(propose(study, k)))
        only_one = study_of(hist[hist.scope == "one_layer"])
        print("   one-layer only  :", summary(propose(only_one, k)))
    print(f"took {time.time()-t0:.0f}s")
