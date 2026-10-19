"""candor's real run branched onto the landscape: real scores for the first n trials (so trials 1..n
are the real ones), then the landscape scores the rest. Over many prompt sets, how often does the
search from the real run's state go on to find good one-layer edits by trial 200?

    uv run python branch.py [--landscape base] [--seeds 30]
"""

import argparse
import time

import numpy as np
import optuna
import torch

from calibrate import real_trials
from candor import config
from candor.search import CANDIDATES, Score, objectives, suggest
from landscape import COMPONENTS, LAYERS, ONE_LAYER
from variants import landscapes


def branch(landscape, real: list[dict], n: int, seed: int, total: int = 200) -> list[dict]:
    space = config.Search()
    scorer = landscape.scorer(seed)
    study = optuna.create_study(directions=["minimize", "minimize"])
    trials = []
    for number in range(total):
        if number in (0, 2):  # the real run's seeds: started with 0, resumed after two trials with 2
            study.sampler = optuna.samplers.TPESampler(
                n_startup_trials=60, n_ei_candidates=CANDIDATES, multivariate=True, seed=number
            )
        trial = study.ask()
        suggest(trial, space, LAYERS, list(COMPONENTS))
        if number < n:
            refusals, kl = real[number]["refusals"], real[number]["kl"]
            assert all(abs(trial.params[k] - real[number]["params"][k]) < 2e-6 for k in trial.params if "." in k)
        else:
            score = scorer.score(dict(trial.params))
            refusals, kl = score.refusals, score.kl
        study.tell(trial, objectives(Score(refusals, 0, 0, kl), 99, space.kl_target))
        trials.append({"params": dict(trial.params), "refusals": refusals, "kl": kl})
    return trials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landscape", default="base")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--at", default="60,100,150")
    args = parser.parse_args()
    torch.set_num_threads(1)
    landscape = landscapes()[args.landscape]
    real = real_trials()
    for n in map(int, args.at.split(",")):
        started = time.time()
        fewest, found = [], 0
        for seed in range(args.seeds):
            trials = branch(landscape, real, n, seed)
            one = [t["refusals"] for t in trials[n:] if t["params"]["direction_scope"] == ONE_LAYER]
            fewest.append(min(one, default=100))
            low = [t for t in trials if t["refusals"] <= 1]
            found += bool(low) and min(low, key=lambda t: t["kl"])["params"]["direction_scope"] == ONE_LAYER
        fewest = np.array(fewest)
        print(f"{args.landscape}: real for {n} trials, then landscape ({time.time() - started:.0f} s): fewest "
              f"one-layer refusals after the branch median {np.median(fewest):.0f}, 10-90% "
              f"{np.quantile(fewest, 0.1):.0f}-{np.quantile(fewest, 0.9):.0f}; >= 35 (as real) in "
              f"{np.mean(fewest >= 35):.0%}; best at <=1 refused is one-layer in {found}/{args.seeds}")


if __name__ == "__main__":
    main()
