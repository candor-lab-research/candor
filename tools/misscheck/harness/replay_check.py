"""Replay candor's real search with its real scores, to check that the baseline strategy is candor's
search exactly: given the scores the real trials got, it must draw the very same 200 trials.

    uv run python replay_check.py

A resumed search reseeds its sampler with seed + trials so far (search.Search._study), so at a
mismatch this looks for the reseed that explains it, and reports it.
"""

import csv
import time

import optuna

from candor import config
from candor.search import ONE_LAYER, PER_LAYER, Score, objectives, suggest
from fit import DATA
from strategies import _tpe

TOLERANCE = 2e-6  # the CSV keeps 6 decimals


def matches(params: dict, row: dict) -> bool:
    scope = ONE_LAYER if row["scope"] == "one_layer" else PER_LAYER
    return (
        params["direction_scope"] == scope
        and params["direction_layer"] == int(row["direction_layer"])
        and all(abs(params[k] - float(row[k])) < TOLERANCE for k in params if "." in k)
    )


def main() -> None:
    rows = list(csv.DictReader(open(DATA / "candor_search_rude.csv")))
    space = config.Search()
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=_tpe(0, space.startup_trials))
    started, reseeds = time.time(), []
    for row in rows:
        trial = study.ask()
        suggest(trial, space, 28, ["attn", "mlp"])
        if not matches(trial.params, row):
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            done = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
            for seed in [*range(done, done + 20), *range(done)]:
                study.sampler = _tpe(seed, space.startup_trials)
                trial = study.ask()
                suggest(trial, space, 28, ["attn", "mlp"])
                if matches(trial.params, row):
                    reseeds.append((int(row["trial"]), seed))
                    break
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            else:
                raise SystemExit(f"no seed explains trial {row['trial']}: the baseline is not candor's search")
        score = Score(int(row["refusals_judge"]), 0, 0, float(row["kl"]))
        study.tell(trial, objectives(score, 99, space.kl_target))
    print(f"All {len(rows)} trials reproduced in {time.time() - started:.1f} s; reseeds (trial, seed): {reseeds}")


if __name__ == "__main__":
    main()
