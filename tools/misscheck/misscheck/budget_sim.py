"""Does search.max_budget_hits make candor's search miss the best edit more often? Runs candor's search
(snapshot candor-8dfc0f5/src, the commit that added the setting) on the scenario landscapes with budget
hits (budget.py), and saves every trial.

The search is candor's own: Sim is search.Search with only its scorer swapped for a landscape. Its real
_studies (and so _judge_again, the switch-over when the setting changes), _ready, _widen, _next and
_done run unchanged on an in-memory Optuna storage; _trial is candor's line for line, scoring the edit
on the landscape. A trial judges CHUNK prompts at a time and stops on refusals as candor does, and with
the limit set also once its budget hits pass it.

Arms (paired: the same scenario and seed give the same landscape draws, spread and sampler seeds):
  off     no limit; the pick is filtered by measured budget hits <= 8 at the end.
  on      search.max_budget_hits = 8 from the first trial.
  switch  off for the first 137 trials, then 8, as K2-Horizon-7B's search was: the search stops and
          starts again with the new setting, so _studies judges its trials again.
  on100   the limit at 100, which no edit can pass: must reproduce off trial for trial.

Designs (the strength range the search starts with; widest_strength is candor's 0.5-3.0):
  D3grow  0.8-1.5, widening as candor does (a thinking scenario: position is a parameter).
  D3      0.5-3.0 from the start.
  grow    0.8-1.5, widening, for a scenario that doesn't think (no position).

    PYTHONPATH=candor-8dfc0f5/src:. uv run python budget_sim.py run --scenarios a-same --designs D3grow \\
        --arms off,on,switch --seeds 60 --jobs 10
    ... budget_sim.py check            # the sanity checks
    ... budget_sim.py calib --scenarios a-same:D3grow,base:grow --intercept -5.118 --seeds 6
                                        # budget hits in the first 137 trials against K2's
"""

import argparse
import math
import os
import time
import warnings
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import torch
from candor import config, search
from candor.search import (
    CHUNK,
    ONE_LAYER,
    POSITIONS,
    STOP_AFTER,
    Score,
    _completed,
    _over_caps,
    budget_violation,
    suggest,
    violation,
)
from optuna.storages import InMemoryStorage

import budget
from landscape import COMPONENTS, LAYERS, Batch
from scenarios import NARROW, WIDE, scenarios

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)

HERE = Path(__file__).parent
RUNS = Path(os.environ.get("BUDGET_RUNS", HERE / "runs" / "budget"))
SCRATCH = Path(
    os.environ.get("BUDGET_SCRATCH", Path(os.environ.get("TMPDIR", "/tmp")) / "budget-sim-scratch")
)
PROMPTS = 100
TRIALS = 600
SWITCH = 137
LIMIT = 8
ARMS = {"off": (None, None), "on": (LIMIT, None), "switch": (None, LIMIT), "on100": (100, None)}
DESIGNS = {"D3grow": NARROW, "D3": WIDE, "grow": NARROW}
COLUMNS = (
    "one",
    "position",
    "judged",
    "refusals",
    "kl",
    "true_refusals",
    "true_kl",
    "direction_layer",
    "attn.strength",
    "mlp.strength",
    "forced",
    "true_hits",
    "number",
)

# Each Sim's storage, by its folder: candor's _storage opens a journal file there.
STORAGES: dict[Path, InMemoryStorage] = {}
search._storage = lambda folder: STORAGES[folder]


class Scorer:
    """Scores an edit on the scenario's landscape at its position, as runs.measure does, with budget hits
    drawn beside it, judging CHUNK prompts at a time and stopping as candor's _Scorer.score does."""

    def __init__(self, scenario, landscape_budget: budget.Budget, seed: int):
        self.scorers = {p: land.scorer(seed) for p, land in scenario.positions.items()}
        self.budget = landscape_budget
        self.draws = budget.Draws(seed, PROMPTS)
        self.rows = []

    def score(self, scope: str, params: dict, stop_above: int | None, forced_above: int | None) -> Score:
        position = params.get("position")
        edit = {"direction_scope": scope, **params, "position": position}
        scorer = self.scorers[position]
        batch = Batch.from_params([edit])
        with torch.no_grad():
            logit, log_kl = scorer.landscape.logits(batch)
            rate, kl = float(torch.sigmoid(logit)[0]), float(torch.exp(log_kl)[0])
            q = float(torch.sigmoid(self.budget.logit(batch, position, logit, log_kl))[0])
        noise = scorer._edit_noise(edit)
        refused = scorer._uniform(noise) < rate
        measured_kl = float(kl * math.exp(scorer.kl_noise * noise.standard_normal()))
        forced_on = self.draws.forced(edit, q)
        refusals = judged = forced = 0
        for start in range(0, PROMPTS, CHUNK):
            refusals += int(refused[start : start + CHUNK].sum())
            forced += int(forced_on[start : start + CHUNK].sum())
            judged += len(refused[start : start + CHUNK])
            if (stop_above is not None and refusals > stop_above) or (
                forced_above is not None and forced > forced_above
            ):
                break
        self.rows.append(
            (
                scope == ONE_LAYER,
                -1 if position is None else POSITIONS.index(position),
                judged,
                refusals,
                measured_kl,
                100 * rate,
                kl,
                params.get("direction_layer", -1),
                params["attn.strength"],
                params["mlp.strength"],
                forced,
                100 * q,
            )
        )
        return Score(refusals, 0, 0, measured_kl, judged, 0.0, forced)


class Sim(search.Search):
    """candor's Search on a landscape: everything but _trial's scorer is candor's."""

    def __init__(self, space: config.Search, thinking: bool, folder: Path):
        writers = [SimpleNamespace(component=c) for c in COMPONENTS]
        self.run = SimpleNamespace(
            settings=SimpleNamespace(search=space),
            shape=(None, LAYERS, None, writers),
            report=lambda line: None,
        )
        self.folder = folder
        self.positions = POSITIONS if thinking else ()
        self.thinks = thinking

    def _check(self, inputs, known=False, scoring=False) -> None:
        pass

    def _trial(self, studies, scope, scorer: Scorer, adapters) -> optuna.trial.FrozenTrial:
        """candor's Search._trial, scoring on the landscape."""
        run, space = self.run, self.run.settings.search
        _, layers, _, writers = run.shape
        study = studies[scope]
        numbers = [
            t.user_attrs.get("number", 0) for s in studies.values() for t in s.get_trials(deepcopy=False)
        ]
        number = max(numbers, default=0) + 1
        trial = study.ask()
        trial.set_user_attr("number", number)
        suggest(
            trial, self._space(study), layers, sorted({w.component for w in writers}), scope, self.positions
        )
        cap, hits = space.max_refusals, space.max_budget_hits
        full = sorted(_over_caps(t) for t in _completed(study) if not t.user_attrs["stopped"])
        stop_above = (
            max(cap, math.floor(full[STOP_AFTER - 1] + cap + 0.5))
            if space.stop_early and len(full) >= STOP_AFTER
            else None
        )
        score = scorer.score(scope, trial.params, stop_above, hits if space.stop_early else None)
        for key, value in asdict(score).items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("stopped", score.judged < PROMPTS)
        trial.set_user_attr("violation", violation(score, PROMPTS, cap))
        if hits is not None:
            trial.set_user_attr("budget violation", budget_violation(score, PROMPTS, hits))
        trial.set_user_attr("seconds", 0.0)
        return study.tell(trial, score.kl)


def simulate(
    name: str,
    variant: str,
    design: str,
    arm: str,
    seed: int,
    intercept: float = budget.A,
    trials: int = TRIALS,
) -> np.ndarray:
    """One search's trials (COLUMNS), in the order they were drawn."""
    torch.set_num_threads(1)
    scenario = scenarios()[name]
    thinking = None not in scenario.positions
    assert thinking == (design != "grow"), f"{design} on {name}"
    start, then = ARMS[arm]
    space = config.Search(strength=DESIGNS[design], seed=seed, max_budget_hits=start)
    folder = SCRATCH / f"{os.getpid()}"
    folder.parent.mkdir(parents=True, exist_ok=True)
    STORAGES[folder] = InMemoryStorage()
    sim = Sim(space, thinking, folder)
    scorer = Scorer(scenario, budget.budget_for(name, variant, intercept), seed)
    studies = sim._studies({}, PROMPTS)
    while sim._done(studies) < trials:
        if (
            then is not None
            and sim._done(studies) == SWITCH
            and sim.run.settings.search.max_budget_hits != then
        ):
            # The search stops and is started again with the limit set, as K2's was.
            sim.run.settings = SimpleNamespace(search=space.model_copy(update={"max_budget_hits": then}))
            studies = sim._studies({}, PROMPTS)
        scope = sim._next(studies)
        sim._widen(studies, scope)
        trial = sim._trial(studies, scope, scorer, None)
        scorer.rows[-1] = (*scorer.rows[-1], trial.user_attrs["number"])
    del STORAGES[folder]
    return np.array(scorer.rows, dtype=float)


def path_of(name: str, variant: str, design: str, arm: str, seed: int) -> Path:
    return RUNS / variant / name / design / arm / f"{seed}.npz"


def run(task: tuple) -> str:
    name, variant, design, arm, seed = task
    path = path_of(*task)
    if path.exists():
        return f"{name}/{variant}/{design}/{arm}/{seed}: kept"
    started = time.time()
    trials = simulate(name, variant, design, arm, seed, budget.INTERCEPT)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, trials=trials, seconds=time.time() - started)
    return f"{name}/{variant}/{design}/{arm}/{seed}: {time.time() - started:.1f} s"


def load(name: str, variant: str, design: str, arm: str) -> dict[int, np.ndarray]:
    folder = RUNS / variant / name / design / arm
    return {int(p.stem): np.load(p)["trials"] for p in folder.glob("*.npz")}


# ---- checks and calibration


def _calib_one(task) -> np.ndarray:
    name, design, seed, intercept, spread, rho = task
    budget.S, budget.RHO = spread, rho
    return simulate(name, "base", design, "off", seed, intercept, trials=SWITCH)


def calib(
    pairs: list[tuple[str, str]], seeds: int, intercept: float, spread: float, rho: float, jobs: int
) -> None:
    """Budget hits in the off arm's first 137 trials, against K2's: the share of trials scored in full over
    8, the mean, the near misses (9-14), none at all, and the rank correlation with KL and refusals."""
    import pandas as pd

    k2 = pd.read_csv(HERE / "k2" / "trials.csv")
    full = k2[k2.judged == 100]

    def line(label, forced, kl, refusals, strength):
        def rho(other):
            return np.corrcoef(pd.Series(forced).rank(), pd.Series(other).rank())[0, 1]

        print(
            f"  {label:26s} n {len(forced):6d}  over 8 {np.mean(forced > 8):5.1%}  "
            f"mean {np.mean(forced):5.2f}  "
            f"9-14 {np.mean((forced >= 9) & (forced <= 14)):5.1%}  none {np.mean(forced == 0):5.1%}  "
            f"rho(KL) {rho(kl):+.2f}  rho(refusals) {rho(refusals):+.2f}  rho(strength) {rho(strength):+.2f}"
        )

    print(
        f"Budget hits of trials scored in full, first {SWITCH} trials, "
        f"intercept {intercept:g}, S {spread:g}, "
        f"rho {rho:g}. rho(...) is Spearman's, strength the larger of attn's and mlp's."
    )
    strength = np.maximum(full["attn.strength"].values, full["mlp.strength"].values)
    line("K2-Horizon-7B", full.forced.values, full.kl.values, full.refusals.values, strength)
    tasks = [(n, d, s, intercept, spread, rho) for n, d in pairs for s in range(seeds)]
    with Pool(jobs) as pool:
        results = pool.map(_calib_one, tasks)
    pooled = []
    for name, design in pairs:
        t = np.concatenate(
            [r for (n, d, *_), r in zip(tasks, results, strict=True) if (n, d) == (name, design)]
        )
        t = t[t[:, 2] == PROMPTS]
        pooled.append(t)
        line(f"{name} ({design})", t[:, 10], t[:, 4], t[:, 3], np.maximum(t[:, 8], t[:, 9]))
    t = np.concatenate(pooled)
    line("all", t[:, 10], t[:, 4], t[:, 3], np.maximum(t[:, 8], t[:, 9]))
    over = [np.mean(p[:, 10] > 8) for p in pooled]
    means = [np.mean(p[:, 10]) for p in pooled]
    near = [np.mean((p[:, 10] >= 9) & (p[:, 10] <= 14)) for p in pooled]
    print(
        f"  {'median of the scenarios':26s}           over 8 {np.median(over):5.1%}  "
        f"mean {np.median(means):5.2f}  "
        f"9-14 {np.median(near):5.1%}"
    )


def check(jobs: int) -> None:
    """With the limit at 100, on reproduces off trial for trial; switch matches off for its first 137
    trials; and budget hits leave every trial's refusals and KL as they were."""
    cases = [("a-same", "D3grow", s) for s in range(2)] + [("fb-answer", "D3", 0), ("rand3", "grow", 0)]
    with Pool(jobs) as pool:
        offs = pool.starmap(simulate, [(n, "base", d, "off", s, budget.INTERCEPT) for n, d, s in cases])
        highs = pool.starmap(simulate, [(n, "base", d, "on100", s, budget.INTERCEPT) for n, d, s in cases])
        switches = pool.starmap(
            simulate, [(n, "base", d, "switch", s, budget.INTERCEPT) for n, d, s in cases]
        )
    for (name, design, seed), off, high, switch in zip(cases, offs, highs, switches, strict=True):
        print(
            f"{name}/{design}/{seed}: on100 = off for all {len(off)} trials: {np.array_equal(off, high)}; "
            f"switch = off for the first {SWITCH}: {np.array_equal(off[:SWITCH], switch[:SWITCH])}, "
            f"and differs after: {not np.array_equal(off[SWITCH:], switch[SWITCH:])}"
        )
    # The same edits scored with no budget hits at all (a budget landscape whose q is 0) and with them.
    scenario = scenarios()["a-same"]
    with_hits = Scorer(scenario, budget.budget_for("a-same", "base", budget.INTERCEPT), 3)
    without = Scorer(scenario, budget.Budget(1.0, budget.fields("a-same", list(scenario.positions)), -1e9), 3)
    rng = np.random.default_rng(0)
    same = True
    for _ in range(300):
        params = {
            f"{c}.{s}": float(rng.uniform(*budget.RANGES[s])) for c in COMPONENTS for s in budget.RANGES
        }
        params |= {"direction_layer": int(rng.integers(11, 25)), "position": POSITIONS[int(rng.integers(2))]}
        a = with_hits.score(ONE_LAYER, params, None, None)
        b = without.score(ONE_LAYER, params, None, None)
        same &= (a.refusals, a.kl, a.judged) == (b.refusals, b.kl, b.judged) and b.forced == 0
    forced = np.array([r[10] for r in with_hits.rows])
    print(
        f"300 random edits scored with and without budget hits: refusals and KL the same: {same}; "
        f"budget hits with them: mean {forced.mean():.2f}, over 8 {np.mean(forced > 8):.0%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "check", "calib"])
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--designs", default="")
    parser.add_argument("--arms", default="off,on,switch")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--seeds", type=int, default=60)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--intercept", type=float, default=None)
    parser.add_argument("--spread", type=float, default=budget.S, help="S, for calib")
    parser.add_argument("--rho", type=float, default=budget.RHO, help="RHO, for calib")
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()
    names = [n for n in args.scenarios.split(",") if n]
    if args.command == "check":
        check(args.jobs)
        return
    if args.command == "calib":
        pairs = [tuple(pair.split(":")) for pair in names]
        intercept = budget.INTERCEPT if args.intercept is None else args.intercept
        calib(pairs, args.seeds, intercept, args.spread, args.rho, args.jobs)
        return
    tasks = [
        (name, args.variant, design, arm, seed)
        for seed in range(args.first_seed, args.first_seed + args.seeds)
        for name in names
        for design in args.designs.split(",")
        for arm in args.arms.split(",")
        if (design == "grow") == (None in scenarios()[name].positions)
    ]
    started = time.time()
    with Pool(args.jobs) as pool:
        for i, line in enumerate(pool.imap_unordered(run, tasks), 1):
            if i % 25 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} runs, {time.time() - started:.0f} s ({line})", flush=True)


if __name__ == "__main__":
    main()
