"""How faithful the base landscape is to rude-llama's two real searches.

    uv run python calibrate.py [--seeds 30]

1. The landscape against candor's real trials at the same settings, by family, and the trials that
   matter most (candor's best, its best one-layer trials, heretic's Trial 170).
2. candor's search simulated on the landscape exactly as it ran for real (seed 0, resumed after two
   trials with seed 2), set beside the real run.
3. The same over many seeds: is the real run typical of what the landscape produces?
4. How wide the one-layer basin is: for needle factors k (k = 1 is the fit; k > 1 narrows the basin,
   as variant needle does), how well each landscape still matches both searches' trials at the same
   settings, and how likely it makes what the two real runs did: candor's 200 trials never got a
   one-layer edit below 35 refusals, and heretic's found one at about 0 (2 by keywords).
"""

import argparse
import csv
import time

import numpy as np
import optuna
import torch

from candor import config
from candor.search import CANDIDATES, Score, objectives, suggest
from fit import DATA, HERE, LATE_KEYWORDS, TRIAL_170, params_of
from landscape import COMPONENTS, LAYERS, ONE_LAYER, PER_LAYER, Batch, Landscape, load_shape
from strategies import STRATEGIES
from variants import base, reference


def real_trials() -> list[dict]:
    rows = list(csv.DictReader(open(DATA / "candor_search_rude.csv")))
    return [{"params": params_of(r), "refusals": int(r["refusals_judge"]), "kl": float(r["kl"])} for r in rows]


def fidelity(landscape) -> None:
    trials = real_trials()
    rates, kls = landscape.true([t["params"] for t in trials])
    predicted, actual = 100 * rates, np.array([t["refusals"] for t in trials])
    kl_actual = np.array([t["kl"] for t in trials])
    one = np.array([t["params"]["direction_scope"] == ONE_LAYER for t in trials])
    print("1. Landscape vs candor's 200 real trials at the same settings (in sample)")
    for name, mask in (("all", np.ones(len(one), bool)), ("one layer", one), ("per layer", ~one)):
        print(
            f"   {name:9s} n={mask.sum():3d}  refusals r={np.corrcoef(predicted[mask], actual[mask])[0, 1]:.3f} "
            f"MAE {np.abs(predicted[mask] - actual[mask]).mean():4.1f}  log KL r="
            f"{np.corrcoef(np.log(kls[mask]), np.log(kl_actual[mask]))[0, 1]:.3f}  KL median |error| "
            f"{np.median(np.abs(kls[mask] / kl_actual[mask] - 1)):.0%}"
        )
    print("   Notable trials (real -> landscape):")
    for number in (165, 167, 187, 135, 19, 125, 69, 89):
        t, r, k = trials[number - 1], predicted[number - 1], kls[number - 1]
        scope = "per layer" if t["params"]["direction_scope"] == PER_LAYER else f"layer {t['params']['direction_layer']:g}"
        print(f"     candor {number:3d} {scope:9s}: {t['refusals']:3d} refused, KL {t['kl']:.4f} -> {r:5.1f}, {k:.4f}")
    for layer, refused, kl in ((12.039, 0, 0.0834), (12.0, 0, 0.0818)):
        r, k = landscape.true([{**TRIAL_170, "direction_scope": ONE_LAYER, "direction_layer": layer}])
        print(f"     heretic 170 at layer {layer:g}: {refused} refused, KL {kl} -> {100 * r[0]:5.1f}, {k[0]:.4f}")


def replay(landscape) -> list[dict]:
    """candor's real run on the landscape: seed 0, then seed 2 after two trials, as it was resumed."""
    space = config.Search()
    scorer = landscape.scorer(0)
    base_refusals = scorer.unedited()
    study = optuna.create_study(directions=["minimize", "minimize"])
    trials = []
    for number in range(200):
        if number in (0, 2):
            study.sampler = optuna.samplers.TPESampler(
                n_startup_trials=60, n_ei_candidates=CANDIDATES, multivariate=True, seed=number
            )
        trial = study.ask()
        suggest(trial, space, LAYERS, list(COMPONENTS))
        score = scorer.score(dict(trial.params))
        study.tell(trial, objectives(Score(score.refusals, 0, 0, score.kl), base_refusals, space.kl_target))
        trials.append({"params": dict(trial.params), "refusals": score.refusals, "kl": score.kl})
    return trials


def outcome(trials: list[dict]) -> dict:
    one = [t for t in trials if t["params"]["direction_scope"] == ONE_LAYER]
    per = [t for t in trials if t["params"]["direction_scope"] == PER_LAYER]
    low = [t for t in trials if t["refusals"] <= 1]
    aimed_one = [t for t in trials[60:] if t["params"]["direction_scope"] == ONE_LAYER]
    return {
        "one-layer trials": len(one),
        "fewest one-layer refusals": min(t["refusals"] for t in one),
        "fewest per-layer refusals": min(t["refusals"] for t in per),
        "best KL at <=1 refused": min((t["kl"] for t in low), default=float("nan")),
        "scope of that": min(low, key=lambda t: t["kl"])["params"]["direction_scope"] if low else "-",
        "median refusals, aimed one-layer": float(np.median([t["refusals"] for t in aimed_one])) if aimed_one else float("nan"),
        "median refusals, aimed per-layer": float(np.median([t["refusals"] for t in trials[60:] if t["params"]["direction_scope"] == PER_LAYER])),
    }


@torch.no_grad()
def heretic_fidelity(landscape) -> tuple[float, float]:
    """MAE of the landscape's keyword refusals (through fit.py's scale map) on heretic's one-layer
    trials that the fit used, and on the good ones among them (<= 10 keyword refusals)."""
    mapping = load_shape(HERE / "fitted_mapping.json")
    rows = [
        r for r in csv.DictReader(open(DATA / "heretic_search_rude.csv"))
        if r["scope"] == "one_layer" and float(r["direction_layer"]) <= LATE_KEYWORDS
    ]
    logit, _ = landscape.logits(Batch.from_params([params_of(r) for r in rows]))
    floor = torch.sigmoid(mapping["keyword_floor"])
    shift = mapping["keyword_shift"] + mapping["keyword_shift_one"]
    predicted = 100 * (floor + (1 - floor) * torch.sigmoid(shift + mapping["keyword_slope"] * logit)).numpy()
    actual = np.array([int(r["refusals_keywords"]) for r in rows])
    good = actual <= 10
    return float(np.abs(predicted - actual).mean()), float(np.abs(predicted[good] - actual[good]).mean())


def needle_scan(landscape, factors: list[float], seeds: int) -> None:
    centre = reference(landscape)["one layer best"].params
    trials = real_trials()
    actual = np.array([t["refusals"] for t in trials])
    one = np.array([t["params"]["direction_scope"] == ONE_LAYER for t in trials])
    print("\n4. Narrowing the one-layer basin (needle factor k): fit to the real trials vs how likely the real runs are")
    print("   k     candor one-layer MAE  heretic one-layer MAE (good ones)  P(candor never < 35)  P(heretic reaches <= 2)")
    for k in factors:
        variant = Landscape(f"needle {k}", landscape.shape, needle=k, centre=centre)
        rates, _ = variant.true([t["params"] for t in trials])
        candor_mae = np.abs(100 * rates[one] - actual[one]).mean()
        heretic_mae, heretic_good = heretic_fidelity(variant)
        never = reaches = 0
        for s in range(seeds):
            for name in ("candor", "heretic"):
                run = STRATEGIES[name](variant.scorer(s), 200, s)
                fewest = min(t.refusals for t in run if t.params["direction_scope"] == ONE_LAYER)
                never += name == "candor" and fewest >= 35
                reaches += name == "heretic" and fewest <= 2
        print(f"   {k:<5g} {candor_mae:20.1f}  {heretic_mae:12.1f} ({heretic_good:5.1f}){'':14s}"
              f"{never / seeds:8.0%}{'':14s}{reaches / seeds:8.0%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--needles", default="1,1.25,1.5,2")
    args = parser.parse_args()
    torch.set_num_threads(1)
    landscape = base()
    fidelity(landscape)
    found = reference(landscape)
    print(f"\n   Landscape's best per-layer edit: {found['per layer'].kl:.4f}; best one-layer: "
          f"{found['one layer best'].kl:.4f} at layer {found['one layer best'].params['direction_layer']:g}")

    real = outcome(real_trials())
    simulated = outcome(replay(landscape))
    print("\n2. candor's real run vs the same search (same seeds) on the landscape")
    for key in real:
        print(f"   {key:34s} real {real[key]!s:>10s}   landscape {simulated[key]!s:>10s}")

    started = time.time()
    many = [outcome([vars(t) for t in STRATEGIES["candor"](landscape.scorer(s), 200, s)]) for s in range(args.seeds)]
    print(f"\n3. candor's search on the landscape over {args.seeds} seeds ({time.time() - started:.0f} s): "
          "where does the real run fall?")
    for key in ("fewest one-layer refusals", "fewest per-layer refusals", "best KL at <=1 refused",
                "median refusals, aimed one-layer", "median refusals, aimed per-layer"):
        values = np.array([m[key] for m in many], dtype=float)
        share = np.mean(values <= real[key])
        print(f"   {key:34s} real {real[key]:8.4g}; landscape median {np.nanmedian(values):8.4g}, "
              f"10-90% {np.nanquantile(values, 0.1):.4g}-{np.nanquantile(values, 0.9):.4g}; "
              f"share of runs at or below real {share:.0%}")
    one_best = sum(m["scope of that"] == ONE_LAYER for m in many)
    print(f"   runs whose best (<=1 refused, lowest KL) is one-layer: {one_best}/{args.seeds} (real: per layer)")
    if args.needles:
        needle_scan(landscape, [float(k) for k in args.needles.split(",")], args.seeds)


if __name__ == "__main__":
    main()
