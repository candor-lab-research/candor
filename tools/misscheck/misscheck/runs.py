"""Run candor's search on the scenario landscapes many times and save every trial.

Each design drives candor's own code (a snapshot of HEAD in candor-head/): search.suggest and
search.spread draw the trials, search.violation scores the cap, and the sampler is
search.Search._studies' TPESampler (20 startup trials per study, CANDIDATES, multivariate,
constraints). The trial loop is search.Search's: studies take turns, the one with fewest finished
trials going next; a trial is judged CHUNK prompts at a time and stops once it has refused more than
max(cap, the STOP_AFTER-th fewest refusals of its study's fully scored trials); its KL is always read.

Designs:
  narrow  candor as built, strength 0.8-1.5, one study per kind of edit.
  wide    the same with strength 0.5-3.0.
  wide40  wide with 40 spread points per kind instead of 20.
  D1      wide, with position as one more parameter (categorical) in each kind's study. The spread
          start is candor's (20 Sobol points over the other settings); the sampler draws its
          positions at random, as it does any setting a queued trial leaves open.
  D2      wide, four studies (kind x position), each candor's study with its own 20-point spread.
  D3      D1 with the spread split evenly: each kind's study queues candor's spread of 10 points at
          think start and another 10 at answer start.
  D3x     D3 with 20 spread points at each position (40 per kind).
  D4      four studies like D2 with 10 spread points each.
  grow    narrow, but from a kind's 60th trial and every 30th after, if its best trial within the cap
          presses an end of the strength range (within candor's EDGE of it), or none is within the
          cap, that end moves out to 0.5 or 3.0: a new study of the kind holds its trials so far and
          queues 10 spread points over the wider range. Once only per end.
  D3grow  D3 starting at 0.8-1.5 and growing as grow does (5 of the 10 new spread points at each
          position).
  ...log  the design with strength drawn on a log scale (widelog, D1log, D3log). The only change to
          candor's code: suggest draws each strength with log=True, and spread's strengths are mapped
          from even shares of the range to even shares of its logarithm.

Runs are anytime: candor's turn-taking means the first n trials of a longer run are the run of n.

    uv run python runs.py --scenarios base,rand1 --designs narrow,wide --seeds 60 --jobs 10
"""

import argparse
import math
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import optuna
import torch

from candor import config
from candor.search import (
    CANDIDATES, CHUNK, EDGE, ONE_LAYER, PER_LAYER, STOP_AFTER, Score, _names, _ranges, _violation, spread, suggest,
    violation,
)
from landscape import COMPONENTS, LAYERS
from scenarios import ANSWER, NARROW, THINK, WIDE, scenarios

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)

HERE = Path(__file__).parent
RUNS = HERE / "runs"
PROMPTS = 100
MAX_TRIALS = 600
POSITIONS = (THINK, ANSWER)
DESIGNS = {
    # name: (strength range, studies as (kind, position or None), position a parameter, spread per position)
    "narrow": (NARROW, [(ONE_LAYER, None), (PER_LAYER, None)], False, None),
    "wide": (WIDE, [(ONE_LAYER, None), (PER_LAYER, None)], False, None),
    "wide40": (WIDE, [(ONE_LAYER, None), (PER_LAYER, None)], False, None),
    "D1": (WIDE, [(ONE_LAYER, None), (PER_LAYER, None)], True, None),
    "D2": (WIDE, [(k, p) for k in (ONE_LAYER, PER_LAYER) for p in POSITIONS], False, 20),
    "D3": (WIDE, [(ONE_LAYER, None), (PER_LAYER, None)], True, 10),
    "D3x": (WIDE, [(ONE_LAYER, None), (PER_LAYER, None)], True, 20),
    "D4": (WIDE, [(k, p) for k in (ONE_LAYER, PER_LAYER) for p in POSITIONS], False, 10),
}
DESIGNS["grow"] = DESIGNS["narrow"]
DESIGNS["D3grow"] = (NARROW, [(ONE_LAYER, None), (PER_LAYER, None)], True, 10)
for _name in ("wide", "D1", "D3"):
    DESIGNS[f"{_name}log"] = DESIGNS[_name]


def suggest_log(trial, space, layers, components, scope) -> None:
    """candor's search.suggest, with each strength drawn on a log scale."""
    if scope == ONE_LAYER:
        trial.suggest_int("direction_layer", *space.direction_layers(layers))
    ranges = _ranges(space, layers)
    for name in _names(PER_LAYER, components):
        setting = name.split(".")[1]
        trial.suggest_float(name, *ranges[setting], log=setting == "strength")


def spread_log(scope, space, layers, components, count, seed) -> list[dict]:
    """candor's search.spread, its strengths moved from even shares of the range to even shares of
    the range's logarithm."""
    low, high = space.strength
    points = spread(scope, space, layers, components, count, seed)
    for params in points:
        for name in params:
            if name.endswith(".strength"):
                params[name] = low * (high / low) ** ((params[name] - low) / (high - low))
    return points


def _sampler(space, start: int, seed: int) -> optuna.samplers.TPESampler:
    return optuna.samplers.TPESampler(
        n_startup_trials=start, n_ei_candidates=CANDIDATES, multivariate=True, seed=seed, constraints_func=_violation
    )


def pressed(rows: list[tuple], kind: str, space) -> tuple[bool, bool]:
    """Whether the kind's best trial within the cap presses the strength range's floor and ceiling;
    with none within the cap, the ceiling (the edits are too weak)."""
    low, high = space.strength
    within = [r for r in rows if r[0] == (kind == ONE_LAYER) and r[2] == PROMPTS and r[3] <= space.max_refusals]
    if not within:
        return False, True
    best = min(within, key=lambda r: (r[4], r[3]))
    strengths = best[8:10]
    margin = EDGE * (high - low)
    return any(v - low <= margin for v in strengths), any(high - v <= margin for v in strengths)


def regrown(study: optuna.Study, kind: str, space, components: list[str], seed: int, positions: bool) -> optuna.Study:
    """A new study of the kind over the space's (wider) ranges, holding the old study's trials, with 10
    spread points over the new ranges queued (5 at each position when position is a parameter)."""
    probe = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0))
    trial = probe.ask()
    suggest(trial, space, LAYERS, components, kind)
    if positions:
        trial.suggest_categorical("position", list(POSITIONS))
    distributions = trial.distributions
    new = optuna.create_study(direction="minimize")
    for old in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        new.add_trial(optuna.trial.create_trial(
            params=old.params, distributions=distributions, value=old.value, user_attrs=old.user_attrs,
            system_attrs={"constraints": (old.user_attrs["violation"],)},
        ))
    if positions:
        for j, p in enumerate(POSITIONS):
            for params in spread(kind, space, LAYERS, components, 5, seed + j):
                new.enqueue_trial({**params, "position": p})
    else:
        for params in spread(kind, space, LAYERS, components, 10, seed):
            new.enqueue_trial(params)
    new.sampler = _sampler(space, space.startup_per_scope, seed)
    return new


def measure(scorer, params: dict, stop_above: int | None) -> tuple[int, int, float, float, float]:
    """candor's _Scorer.score on a landscape: refusals and prompts judged (stopping as it does),
    measured KL, and the edit's true refusals (of 100) and KL."""
    rate, kl = scorer.landscape.true([params])
    noise = scorer._edit_noise(params)
    refused = scorer._uniform(noise) < rate[0]
    measured_kl = float(kl[0] * math.exp(scorer.kl_noise * noise.standard_normal()))
    refusals = judged = 0
    for start in range(0, PROMPTS, CHUNK):
        chunk = refused[start : start + CHUNK]
        refusals += int(chunk.sum())
        judged += len(chunk)
        if stop_above is not None and refusals > stop_above:
            break
    return refusals, judged, measured_kl, 100 * float(rate[0]), float(kl[0])


def run(task: tuple[str, str, int]) -> str:
    scenario_name, design, seed = task
    path = RUNS / scenario_name / design / f"{seed}.npz"
    if path.exists():
        return f"{scenario_name}/{design}/{seed}: kept"
    torch.set_num_threads(1)
    started = time.time()
    scenario = scenarios()[scenario_name]
    strength, keys, position_param, per_position = DESIGNS[design]
    space = config.Search(strength=strength, startup_per_scope=40 if design == "wide40" else 20)
    cap = space.max_refusals
    components = list(COMPONENTS)
    scorers = {p: land.scorer(seed) for p, land in scenario.positions.items()}
    thinking = None not in scorers
    if not thinking:
        assert not position_param and all(p is None for _, p in keys), "a design with positions needs a thinking scenario"
    elif not position_param and any(p is None for _, p in keys):
        raise SystemExit(f"{design} has no positions; {scenario_name} is a thinking scenario")

    log = design.endswith("log")
    draw, start_points = (suggest_log, spread_log) if log else (suggest, spread)
    studies, done, full = {}, {}, {}
    for i, (kind, position) in enumerate(keys):
        study = optuna.create_study(direction="minimize")
        start = space.startup_per_scope
        if per_position is None:
            for params in start_points(kind, space, LAYERS, components, start, 1000 * seed + i):
                study.enqueue_trial(params)
        elif position is not None:
            start = per_position
            for params in start_points(kind, space, LAYERS, components, per_position, 1000 * seed + i):
                study.enqueue_trial(params)
        else:
            start = per_position * len(POSITIONS)
            for j, p in enumerate(POSITIONS):
                for params in start_points(kind, space, LAYERS, components, per_position, 1000 * seed + 10 * j + i):
                    study.enqueue_trial({**params, "position": p})
        study.sampler = optuna.samplers.TPESampler(
            n_startup_trials=start,
            n_ei_candidates=CANDIDATES,
            multivariate=True,
            seed=1000 * seed + 100 + i,
            constraints_func=_violation,
        )
        studies[(kind, position)] = study
        done[(kind, position)] = 0
        full[(kind, position)] = []

    rows = []
    spaces = {key: space for key in keys}
    for _ in range(MAX_TRIALS):
        key = min(keys, key=lambda k: (done[k], keys.index(k)))
        kind, fixed = key
        if design.endswith("grow") and done[key] >= 60 and done[key] % 30 == 0:
            floor, ceiling = pressed(rows, kind, spaces[key])
            low, high = spaces[key].strength
            wider = (WIDE[0] if floor else low, WIDE[1] if ceiling else high)
            if wider != (low, high):
                spaces[key] = config.Search(strength=wider)
                studies[key] = regrown(
                    studies[key], kind, spaces[key], components, 1000 * seed + 200 + done[key], position_param
                )
        space = spaces[key]
        study = studies[key]
        trial = study.ask()
        draw(trial, space, LAYERS, components, kind)
        if position_param:
            trial.suggest_categorical("position", list(POSITIONS))
        params = dict(trial.params)
        position = params.get("position", fixed)
        refused_full = sorted(full[key])
        stop_above = max(cap, refused_full[STOP_AFTER - 1]) if space.stop_early and len(refused_full) >= STOP_AFTER else None
        edit = {"direction_scope": kind, **params, "position": position}
        refusals, judged, kl, true_refusals, true_kl = measure(scorers[position], edit, stop_above)
        trial.set_user_attr("violation", violation(Score(refusals, 0, 0, kl, judged), PROMPTS, cap))
        study.tell(trial, kl)
        done[key] += 1
        if judged == PROMPTS:
            full[key].append(refusals)
        rows.append((
            kind == ONE_LAYER, -1 if position is None else POSITIONS.index(position), judged, refusals, kl,
            true_refusals, true_kl, params.get("direction_layer", -1), params["attn.strength"], params["mlp.strength"],
        ))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, trials=np.array(rows, dtype=float), seconds=time.time() - started)
    return f"{scenario_name}/{design}/{seed}: {time.time() - started:.1f} s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--designs", required=True)
    parser.add_argument("--seeds", type=int, default=60)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()
    tasks = [
        (s, d, seed)
        for seed in range(args.first_seed, args.first_seed + args.seeds)
        for s in args.scenarios.split(",")
        for d in args.designs.split(",")
    ]
    started = time.time()
    with Pool(args.jobs) as pool:
        for i, line in enumerate(pool.imap_unordered(run, tasks), 1):
            if i % 20 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} runs, {time.time() - started:.0f} s ({line})", flush=True)


if __name__ == "__main__":
    main()
