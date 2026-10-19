"""Search designs run against the surrogate landscapes, many seeds each.

Every design gets the same trial budget and returns all the (scope, refusals, KL) it scored. The
miss metric is computed afterwards against optima.json.
"""

import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna

from surrogate import BOX, LAYERS, NAMES, landscapes

optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
BASE = 99
KL_TARGET = 0.01
CAP = 2
CANDIDATES = 128
LAST = 27
REACH = (1 / LAST, 0.6)


class Landscape:
    def __init__(self, scopes):
        self.scopes = scopes

    def __call__(self, scope: str, params: dict) -> tuple[int, float]:
        u = []
        if scope == "one":
            u.append((params["direction_layer"] - LAYERS[0]) / (LAYERS[1] - LAYERS[0]))
        for name in NAMES:
            low, high = BOX[name.split(".")[1]]
            u.append((params[name] - low) / (high - low))
        r, k = self.scopes[scope](np.array(u))
        return int(r[0]), float(k[0])


def box(name):
    kind = name.split(".")[1]
    return REACH if kind == "reach" else BOX[kind]


def draw_kernel(trial, prefix="", narrow=None):
    params = {}
    for name in NAMES:
        low, high = narrow[name] if narrow else box(name)
        params[name] = trial.suggest_float(prefix + name, low, high)
    return params


def mo_values(r, k):
    share = r / BASE
    return (k if k >= KL_TARGET else share * KL_TARGET, share)


def violation(r, k):
    """<= 0 when feasible; otherwise refusals over the cap, then KL as a tie-break below one refusal."""
    return (r - CAP + k / (1 + k),) if r > CAP else (r - CAP - 1.0,)


def constraints(trial):
    return trial.user_attrs["violation"]


def tpe(seed, startup, constrained, group=False):
    return optuna.samplers.TPESampler(
        n_startup_trials=startup,
        n_ei_candidates=CANDIDATES,
        multivariate=True,
        group=group,
        seed=seed,
        constraints_func=constraints if constrained else None,
    )


def make_study(seed, startup, constrained, group=False):
    if constrained:
        return optuna.create_study(direction="minimize", sampler=tpe(seed, startup, True, group))
    return optuna.create_study(directions=["minimize", "minimize"], sampler=tpe(seed, startup, False, group))


def tell(study, trial, r, k, constrained, log, scope):
    trial.set_user_attr("violation", violation(r, k))
    study.tell(trial, k if constrained else mo_values(r, k))
    log.append((scope, r, k, trial.params))


# --- designs -------------------------------------------------------------------------------


def today(land, budget, seed, constrained=False, startup_share=0.3):
    """One study over both scopes with shared names (candor today)."""
    study = make_study(seed, round(startup_share * budget), constrained)
    log = []
    for _ in range(budget):
        t = study.ask()
        scope = "one" if t.suggest_categorical("direction_scope", ["one", "per"]) == "one" else "per"
        params = {"direction_layer": t.suggest_int("direction_layer", *LAYERS), **draw_kernel(t)}
        tell(study, t, *land(scope, params), constrained, log, scope)
    return log


def grouped(land, budget, seed, constrained=False):
    """One study, separate names per scope, group-decomposed multivariate TPE."""
    study = make_study(seed, round(0.3 * budget), constrained, group=True)
    log = []
    for _ in range(budget):
        t = study.ask()
        scope = t.suggest_categorical("direction_scope", ["one", "per"])
        params = draw_kernel(t, prefix=f"{scope}.")
        if scope == "one":
            params["direction_layer"] = t.suggest_int("one.direction_layer", *LAYERS)
        tell(study, t, *land(scope, params), constrained, log, scope)
    return log


def sobol_points(scope, n, seed, region=None):
    """n scrambled Sobol points over the scope's box, as trial params."""
    import torch

    names = (["direction_layer"] if scope == "one" else []) + NAMES
    u = torch.quasirandom.SobolEngine(len(names), scramble=True, seed=seed).draw(n).double().numpy()
    points = []
    for row in u:
        params = {}
        for value, name in zip(row, names):
            if name == "direction_layer":
                params[name] = int(LAYERS[0] + min(int(value * (LAYERS[1] - LAYERS[0] + 1)), LAYERS[1] - LAYERS[0]))
            else:
                low, high = region[name] if region else box(name)
                params[name] = float(low + value * (high - low))
        points.append(params)
    return points


def family(land, scope, budget, seed, constrained, startup_share=0.3, narrow=None, enqueue=(), sobol=False):
    """One study for one scope; direction_layer only drawn for 'one'."""
    startup = max(1, round(startup_share * budget))
    study = make_study(seed, startup, constrained)
    if sobol:
        enqueue = [*enqueue, *sobol_points(scope, startup, seed)]
    for params in enqueue:
        study.enqueue_trial(params)
    log = []
    for _ in range(budget):
        t = study.ask()
        params = {}
        if scope == "one":
            low, high = narrow["direction_layer"] if narrow else LAYERS
            params["direction_layer"] = t.suggest_int("direction_layer", low, high)
        params.update(draw_kernel(t, narrow=narrow))
        tell(study, t, *land(scope, params), constrained, log, scope)
    return log


def split(land, budget, seed, constrained, restarts=1, startup_share=0.3, sobol=False):
    """One study per scope, budget halved, each half split into `restarts` independent studies.
    Trials are interleaved in reality; order doesn't matter to the result here."""
    log = []
    per = budget // (2 * restarts)
    for i, scope in enumerate(("one", "per")):
        for j in range(restarts):
            log += family(land, scope, per, seed * 100 + i * 10 + j, constrained, startup_share, sobol=sobol)
    return log


def incumbent(log, scope):
    rows = [row for row in log if row[0] == scope]
    return min(rows, key=lambda row: (max(row[1] - CAP, 0), row[2]))


def narrowed(params, scope, width=0.15):
    out = {}
    for name in NAMES:
        low, high = box(name)
        span = (high - low) * width
        out[name] = (max(low, params[name] - span), min(high, params[name] + span))
    if scope == "one":
        d = params["direction_layer"]
        out["direction_layer"] = (max(LAYERS[0], d - 1), min(LAYERS[1], d + 1))
    return out


def split_refine(land, budget, seed, refine_share=0.3, restarts=1, sobol=False):
    """split + constrained, then a narrowed TPE around each scope's incumbent."""
    main = round(budget * (1 - refine_share))
    log = split(land, main, seed, True, restarts, sobol=sobol)
    rest = (budget - len(log)) // 2
    for i, scope in enumerate(("one", "per")):
        best = incumbent(log, scope)[3]
        log += family(land, scope, rest, seed * 100 + 50 + i, True, 0.0, narrowed(best, scope), [best])
    return log


def gp_constraints(trial):
    return (trial.user_attrs["refusals"] - CAP - 0.5,)


# A wider box on the faces the best trials pile up against (H2): peak floor, reach and strength.
WIDE = {name: {"strength": (0.6, 1.8), "peak_layer": (0.4, 1.0), "edge_strength": (0.0, 1.0),
               "reach": (1 / LAST, 0.8)}[name.split(".")[1]] for name in NAMES}
WIDE["direction_layer"] = LAYERS


def gp_family(land, scope, budget, seed, startup=20, narrow=None, enqueue=()):
    """GP-BO for one scope: log KL, subject to refusals <= CAP, Sobol start."""
    sampler = optuna.samplers.GPSampler(seed=seed, n_startup_trials=startup, constraints_func=gp_constraints)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    sobol = sobol_points(scope, startup, seed, narrow) if narrow is WIDE else sobol_points(scope, startup, seed)
    for params in [*enqueue, *(sobol if narrow is None or narrow is WIDE else [])]:
        study.enqueue_trial(params)
    log = []
    for _ in range(budget):
        t = study.ask()
        params = {}
        if scope == "one":
            low, high = narrow["direction_layer"] if narrow else LAYERS
            params["direction_layer"] = t.suggest_int("direction_layer", low, high)
        params.update(draw_kernel(t, narrow=narrow))
        r, k = land(scope, params)
        t.set_user_attr("refusals", r)
        study.tell(t, math.log(k))
        log.append((scope, r, k, t.params))
    return log


def split_gp(land, budget, seed):
    return [row for i, scope in enumerate(("one", "per")) for row in gp_family(land, scope, budget // 2, seed * 100 + i)]


def split_gp_refine(land, budget, seed, refine_share=0.3):
    main = round(budget * (1 - refine_share))
    log = split_gp(land, main, seed)
    rest = (budget - len(log)) // 2
    for i, scope in enumerate(("one", "per")):
        best = incumbent(log, scope)[3]
        log += gp_family(land, scope, rest, seed * 100 + 50 + i, 1, narrowed(best, scope), [best])
    return log


# Physical limits a refine box may extend to; the default box is only where exploration starts.
HARD = {"strength": (0.5, 2.0), "peak_layer": (0.3, 1.0), "edge_strength": (0.0, 1.0), "reach": (1 / LAST, 1.0)}
HARD_LAYERS = (7, 26)


def trust_box(params, scope, width=0.15):
    """A box of +-width of the default range around the incumbent, clipped only at physical limits."""
    out = {}
    for name in NAMES:
        low, high = box(name)
        hard = HARD[name.split(".")[1]]
        span = (high - low) * width
        out[name] = (max(hard[0], params[name] - span), min(hard[1], params[name] + span))
    if scope == "one":
        d = params["direction_layer"]
        out["direction_layer"] = (max(HARD_LAYERS[0], d - 1), min(HARD_LAYERS[1], d + 1))
    return out


def split_trust(land, budget, seed, refine_share=0.3, rounds=2, clip=False):
    """split + Sobol + lexicographic TPE, then `rounds` trust-region refines that follow the incumbent."""
    main = round(budget * (1 - refine_share))
    log = split(land, main, seed, True, sobol=True)
    per_round = (budget - len(log)) // (2 * rounds)
    for r in range(rounds):
        for i, scope in enumerate(("one", "per")):
            best = incumbent(log, scope)[3]
            region = narrowed(best, scope) if clip else trust_box(best, scope)
            log += family(land, scope, per_round, seed * 100 + 50 + 10 * r + i, True, 0.0, region, [best])
    return log


class Shifted:
    """The one-layer landscape with its peaks moved down by `by`, so its best lies below the default floor."""

    def __init__(self, land, by=0.1):
        self.land, self.by = land, by

    def __call__(self, scope, params):
        if scope == "one":
            params = {**params, "attn.peak_layer": params["attn.peak_layer"] + self.by,
                      "mlp.peak_layer": params["mlp.peak_layer"] + self.by}
        return self.land(scope, params)


DESIGNS = {
    "split_trust": split_trust,
    "split_trust_clipped": lambda land, b, s: split_trust(land, b, s, clip=True),
    "split_gp": split_gp,
    "split_gp_refine": split_gp_refine,
    "split_x2_sobol_refine": lambda land, b, s: split_refine(land, b, s, restarts=2, sobol=True),
    "split_sobol_refine50": lambda land, b, s: split_refine(land, b, s, refine_share=0.5, sobol=True),
    "today": lambda land, b, s: today(land, b, s),
    "today_constrained": lambda land, b, s: today(land, b, s, constrained=True),
    "grouped": lambda land, b, s: grouped(land, b, s),
    "grouped_constrained": lambda land, b, s: grouped(land, b, s, constrained=True),
    "split_mo": lambda land, b, s: split(land, b, s, False),
    "split_constrained": lambda land, b, s: split(land, b, s, True),
    "split_constrained_x2": lambda land, b, s: split(land, b, s, True, restarts=2),
    "split_constrained_refine": lambda land, b, s: split_refine(land, b, s),
    "split_constrained_sobol": lambda land, b, s: split(land, b, s, True, sobol=True),
    "split_constrained_sobol_refine": lambda land, b, s: split_refine(land, b, s, sobol=True),
    "split_mo_sobol": lambda land, b, s: split(land, b, s, False, sobol=True),
}


def summary(log, optima, landscape):
    out = {}
    for cap in (0, 2, 5):
        best = min((k for _, r, k, _ in log if r <= cap), default=math.inf)
        truth = min(optima[landscape][f"{s}@{cap}"]["kl"] for s in ("one", "per"))
        scope = min(((k, s) for s, r, k, _ in log if r <= cap), default=(math.inf, None))[1]
        out[cap] = (best / truth if math.isfinite(truth) else math.nan, scope)
    return out


def run(job):
    landscape, design, budget, seed = job
    land = Shifted(Landscape(LANDS["union"])) if landscape == "shifted" else Landscape(LANDS[landscape])
    start = time.time()
    log = DESIGNS[design](land, budget, seed)
    return landscape, design, budget, seed, summary(log, OPTIMA, landscape), time.time() - start


def init():
    global LANDS, OPTIMA
    LANDS = landscapes()
    OPTIMA = json.loads((HERE / "optima.json").read_text())
    OPTIMA["shifted"] = OPTIMA["union"]  # a translation: same best values, reachable below the floor


if __name__ == "__main__":
    import multiprocessing as mp

    designs = sys.argv[1].split(",")
    lands = sys.argv[2].split(",")
    budgets = [int(b) for b in sys.argv[3].split(",")]
    seeds = range(int(sys.argv[4]))
    out = HERE / sys.argv[5]
    jobs = [(l, d, b, s) for l in lands for d in designs for b in budgets for s in seeds]
    start = time.time()
    with mp.get_context("spawn").Pool(2, initializer=init) as pool, out.open("a") as f:
        for result in pool.imap_unordered(run, jobs):
            landscape, design, budget, seed, summ, secs = result
            f.write(json.dumps({"landscape": landscape, "design": design, "budget": budget, "seed": seed,
                                "result": {str(c): v for c, v in summ.items()}, "seconds": secs}) + "\n")
            f.flush()
    print(f"{len(jobs)} runs in {time.time() - start:.0f} s")
