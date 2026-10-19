"""Search strategies to test on a landscape.

A strategy is a function (scorer, budget, seed) -> list[Trial]: it may only call scorer.score (and
scorer.unedited), never look at the landscape's truth, and returns every trial it scored in order.
budget is a number of trials. A strategy whose first n trials are what it would do with budget n
sets anytime = True, so one long run can be judged at several budgets.

A trial scored on fewer prompts than the scorer's full set (a pruned or cheap trial) records how many
it used and costs less; only trials scored on the full set with a KL pass can be picked as the result.
"""

import math
import warnings
from dataclasses import dataclass

import optuna
import torch

from candor import config
from candor.search import CANDIDATES, ONE_LAYER, PER_LAYER, Score, objectives, suggest
from landscape import COMPONENTS, LAST, LAYERS, Scorer

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)


@dataclass
class Trial:
    params: dict
    refusals: int
    prompts: int
    kl: float
    seconds: float


def _tpe(seed: int, startup: int) -> optuna.samplers.TPESampler:
    """candor's sampler exactly (search.Search._study), which is heretic's too."""
    return optuna.samplers.TPESampler(
        n_startup_trials=startup, n_ei_candidates=CANDIDATES, multivariate=True, seed=seed
    )


def _run(scorer: Scorer, budget: int, seed: int, draw, sampler=None) -> list[Trial]:
    """Ask, score and tell as candor's search does: two objectives from candor's own objectives()."""
    space = config.Search()
    sampler = sampler or _tpe(seed, space.startup_trials)
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
    base = scorer.unedited()
    trials = []
    for _ in range(budget):
        trial = study.ask()
        draw(trial, space)
        params = dict(trial.params)
        score = scorer.score(params)
        study.tell(trial, objectives(Score(score.refusals, 0, 0, score.kl), base, space.kl_target))
        trials.append(Trial(params, score.refusals, score.prompts, score.kl, score.seconds))
    return trials


def candor_today(scorer: Scorer, budget: int, seed: int) -> list[Trial]:
    """candor's search as it is: search.suggest on a 28-layer model with attn and mlp edits, the
    config.Search defaults (60 random trials, then multivariate TPE with 128 candidates)."""
    return _run(scorer, budget, seed, lambda trial, space: suggest(trial, space, LAYERS, list(COMPONENTS)))


candor_today.anytime = True


def heretic(scorer: Scorer, budget: int, seed: int) -> list[Trial]:
    """heretic's search: candor's, but the direction layer is a float (0.4-0.9 of the depth), blending
    the two nearest layers' directions."""

    def draw(trial, space):
        trial.suggest_categorical("direction_scope", [ONE_LAYER, PER_LAYER])
        low, high = space.direction_layer
        trial.suggest_float("direction_layer", low * LAST, high * LAST)
        reach = (max(space.reach[0], 1 / LAST), space.reach[1])
        for c in COMPONENTS:
            trial.suggest_float(f"{c}.strength", *space.strength)
            trial.suggest_float(f"{c}.peak_layer", *space.peak_layer)
            trial.suggest_float(f"{c}.edge_strength", *space.edge_strength)
            trial.suggest_float(f"{c}.reach", *reach)

    return _run(scorer, budget, seed, draw)


heretic.anytime = True


def random_search(scorer: Scorer, budget: int, seed: int) -> list[Trial]:
    """candor's parameters drawn at random every trial: a floor any design should beat."""
    return _run(
        scorer, budget, seed, lambda trial, space: suggest(trial, space, LAYERS, list(COMPONENTS)),
        optuna.samplers.RandomSampler(seed=seed),
    )


random_search.anytime = True


# --- Candidate designs: one study per direction scope, KL minimised under a refusal cap ----------

CAP = 2  # refusals out of 100 an edit may keep and still count
STARTUP = 20  # Sobol trials per scope before the sampler aims
CHUNK = 8  # prompts judged at a time when a trial may be stopped early (the server's 8 slots)
SCOPES = (ONE_LAYER, PER_LAYER)
SETTINGS = ("strength", "peak_layer", "edge_strength", "reach")

# Ranges as fractions; the direction layer as a fraction of depth, drawn as whole layers.
TODAY = {"direction_layer": (0.4, 0.9), "strength": (0.8, 1.5), "peak_layer": (0.6, 1.0),
         "edge_strength": (0.0, 1.0), "reach": (1 / LAST, 0.6)}
WIDER = {**TODAY, "direction_layer": (0.3, 0.9), "peak_layer": (0.4, 1.0), "reach": (1 / LAST, 1.0)}


def _layers(box: dict) -> tuple[int, int]:
    low, high = box["direction_layer"]
    return round(low * LAST), round(high * LAST)


def _names(scope: str) -> list[str]:
    kernel = [f"{c}.{s}" for c in COMPONENTS for s in SETTINGS]
    return ["direction_layer", *kernel] if scope == ONE_LAYER else kernel


def _draw(trial, scope: str, box: dict) -> dict:
    params = {"direction_scope": scope}
    for name in _names(scope):
        if name == "direction_layer":
            params[name] = trial.suggest_int(name, *_layers(box))
        else:
            params[name] = trial.suggest_float(name, *box[name.split(".")[1]])
    return params


def _sobol(scope: str, box: dict, n: int, seed: int) -> list[dict]:
    """n scrambled Sobol points over the scope's ranges, as enqueue_trial params."""
    names = _names(scope)
    points = torch.quasirandom.SobolEngine(len(names), scramble=True, seed=seed).draw(n).double()
    low_layer, high_layer = _layers(box)
    out = []
    for row in points.tolist():
        params = {}
        for u, name in zip(row, names, strict=True):
            if name == "direction_layer":
                params[name] = min(low_layer + int(u * (high_layer - low_layer + 1)), high_layer)
            else:
                low, high = box[name.split(".")[1]]
                params[name] = low + u * (high - low)
        out.append(params)
    return out


def _constraint(trial) -> tuple[float]:
    return (trial.user_attrs["violation"],)


def _linear(refusals: float, cap: int) -> float:
    return refusals - cap - 0.5


def _logged(refusals: float, cap: int) -> float:
    """The violation on a log scale, finer near the cap than far above it."""
    return math.log((refusals + 1) / (cap + 1.5))


def _gp(seed: int):
    return optuna.samplers.GPSampler(seed=seed, n_startup_trials=STARTUP, constraints_func=_constraint)


def _tpe_capped(seed: int):
    return optuna.samplers.TPESampler(
        n_startup_trials=STARTUP, n_ei_candidates=CANDIDATES, multivariate=True, seed=seed,
        constraints_func=_constraint,
    )


@dataclass
class Measured:
    refusals: int  # on all the prompts judged
    prompts: int
    estimate: float  # refusals per 100 prompts, as the sampler is told
    first: int | None  # refusals on the first full set, if it was judged in full
    kl: float
    seconds: float


def _extra(scorer: Scorer) -> Scorer:
    """A second set of as many fresh prompts from the same model, for confirming near-feasible edits."""
    return Scorer(scorer.landscape, scorer.seed + 7919, scorer.prompts, scorer.prompt_correlation, scorer.kl_noise)


def _measure(scorer: Scorer, params: dict, fewest_full: int | None, prune: bool, cap: int,
             extend: int | None = None) -> Measured:
    """Score an edit. With prune, its harmful prompts are judged CHUNK at a time and it stops once it
    has surely refused more than the cap and more than the scope's fewest on a fully scored trial
    (refusals can only grow), so the trial with the fewest refusals is always scored in full. A
    stopped trial still gets its KL pass, and pays for one more chunk of answers already in flight.
    With extend, an edit refusing at most cap + extend of the full set is judged on a second set too,
    and is told by its rate over both."""
    full = scorer.prompts
    if prune:
        for n in range(CHUNK, full, CHUNK):
            k = scorer.score(params, prompts=n, kl_pass=False).refusals
            if k > cap and fewest_full is not None and k > fewest_full:
                charged = scorer.score(params, prompts=min(n + CHUNK, full))
                return Measured(k, n, k * full / n, None, charged.kl, charged.seconds)
    score = scorer.score(params)
    if extend is not None and score.refusals <= cap + extend:
        more = _extra(scorer).score(params, kl_pass=False)
        total = score.refusals + more.refusals
        return Measured(total, 2 * full, total / 2, score.refusals, score.kl, score.seconds + more.seconds)
    return Measured(score.refusals, full, score.refusals, score.refusals, score.kl, score.seconds)


def split_search(scorer: Scorer, budget: int, seed: int, box: dict, sampler, prune: bool = False,
                 cap: int = CAP, violation=_linear, sobol: bool = True, extend: int | None = None) -> list[Trial]:
    """One study per direction scope, taking turns trial by trial, each minimising log KL subject to
    refusals <= cap, each starting from STARTUP Sobol points (or the sampler's own random ones)."""
    studies = {}
    for i, scope in enumerate(SCOPES):
        study = optuna.create_study(direction="minimize", sampler=sampler(seed * 2 + i))
        for params in _sobol(scope, box, STARTUP, seed * 2 + i) if sobol else []:
            study.enqueue_trial(params)
        studies[scope] = study
    fewest = {scope: None for scope in SCOPES}
    trials = []
    for n in range(budget):
        scope = SCOPES[n % 2]
        study = studies[scope]
        trial = study.ask()
        params = _draw(trial, scope, box)
        m = _measure(scorer, params, fewest[scope], prune, cap, extend)
        if m.first is not None:
            fewest[scope] = m.first if fewest[scope] is None else min(fewest[scope], m.first)
        trial.set_user_attr("violation", violation(m.estimate, cap))
        study.tell(trial, math.log(max(m.kl, 1e-6)))
        trials.append(Trial(params, m.refusals, m.prompts, m.kl, m.seconds))
    return trials


def gp_today(scorer, budget, seed):
    """GP per scope in today's ranges."""
    return split_search(scorer, budget, seed, TODAY, _gp)


def gp_wider(scorer, budget, seed):
    """GP per scope with the direction floor at 0.3, the peak floor at 0.4 and reach up to 1."""
    return split_search(scorer, budget, seed, WIDER, _gp)


def gp_wider_stop(scorer, budget, seed):
    """gp_wider, stopping a trial early once it surely fails the cap."""
    return split_search(scorer, budget, seed, WIDER, _gp, prune=True)


def tpe_split(scorer, budget, seed):
    """Capped multivariate TPE per scope in today's ranges: the design with no new dependency."""
    return split_search(scorer, budget, seed, TODAY, _tpe_capped)


def tpe_wider(scorer, budget, seed):
    return split_search(scorer, budget, seed, WIDER, _tpe_capped)


def gp_cap1(scorer, budget, seed):
    """gp_today aiming at <= 1 refusal, the level a user picks at."""
    return split_search(scorer, budget, seed, TODAY, _gp, cap=1)


def gp_log(scorer, budget, seed):
    """gp_today with the cap's violation on a log scale."""
    return split_search(scorer, budget, seed, TODAY, _gp, violation=_logged)


def gp_random_start(scorer, budget, seed):
    """gp_today starting from the sampler's own random trials instead of Sobol points."""
    return split_search(scorer, budget, seed, TODAY, _gp, sobol=False)


def gp_stop_today(scorer, budget, seed):
    """gp_today, stopping a trial early once it surely fails the cap."""
    return split_search(scorer, budget, seed, TODAY, _gp, prune=True)


for _f in (gp_today, gp_wider, gp_wider_stop, tpe_split, tpe_wider, gp_cap1, gp_log, gp_random_start, gp_stop_today):
    _f.anytime = True


def tpe_cap(cap: int, box: dict = TODAY, prune: bool = False, extend: int | None = None):
    def run(scorer, budget, seed):
        return split_search(scorer, budget, seed, box, _tpe_capped, prune=prune, cap=cap, extend=extend)

    run.anytime = True
    return run


# --- Explore, then refine around each scope's best ----------------------------------------------


def _incumbent(trials: list[Trial], scope: str, cap: int, full: int) -> dict:
    """The scope's best so far: the lowest KL at <= cap refusals on all prompts, else the fewest refusals."""
    scored = [t for t in trials if t.params["direction_scope"] == scope and t.prompts == full]
    feasible = [t for t in scored if t.refusals <= cap]
    if feasible:
        return min(feasible, key=lambda t: t.kl).params
    return min(scored, key=lambda t: (t.refusals, t.kl)).params


def _around(params: dict, box: dict, width: float) -> dict:
    """A box of +-width of each range around params, inside box; the direction layer +-1."""
    out = {}
    for name in _names(params["direction_scope"]):
        if name == "direction_layer":
            low, high = _layers(box)
            layer = params[name]
            out[name] = (max(low, layer - 1), min(high, layer + 1))
        else:
            low, high = box[name.split(".")[1]]
            span = (high - low) * width
            out[name] = (max(low, params[name] - span), min(high, params[name] + span))
    return out


def _distributions(region: dict) -> dict:
    return {
        name: optuna.distributions.IntDistribution(*bounds) if name == "direction_layer"
        else optuna.distributions.FloatDistribution(*bounds)
        for name, bounds in region.items()
    }


def _inside(params: dict, region: dict) -> bool:
    return all(low <= params[name] <= high for name, (low, high) in region.items())


def refine_search(scorer: Scorer, budget: int, seed: int, box: dict, sampler, explore: float = 0.6,
                  width: float = 0.15, local: str = "tpe", cap: int = CAP) -> list[Trial]:
    """split_search for the first share of the budget, then each scope searches a narrow box around
    its best so far (TPE, which also learns from the trials already inside the box, or CMA-ES)."""
    first = 2 * round(explore * budget / 2)
    trials = split_search(scorer, first, seed, box, sampler, cap=cap)
    studies, regions = {}, {}
    for i, scope in enumerate(SCOPES):
        best = _incumbent(trials, scope, cap, scorer.prompts)
        region = _around(best, box, width)
        regions[scope] = region
        if local == "tpe":
            study = optuna.create_study(direction="minimize", sampler=_tpe_refine(seed * 2 + i + 100))
            distributions = _distributions(region)
            for t in trials:
                if t.params["direction_scope"] == scope and t.prompts == scorer.prompts and _inside(t.params, region):
                    v = _linear(t.refusals, cap)
                    study.add_trial(optuna.trial.create_trial(
                        params={k: t.params[k] for k in region}, distributions=distributions,
                        value=math.log(max(t.kl, 1e-6)), user_attrs={"violation": v},
                        system_attrs={"constraints": (v,)},
                    ))
        else:
            x0 = {k: best[k] for k in region if k != "direction_layer"}
            cma = optuna.samplers.CmaEsSampler(
                x0=x0, sigma0=width / 2, seed=seed * 2 + i + 100, n_startup_trials=0, warn_independent_sampling=False
            )
            study = optuna.create_study(direction="minimize", sampler=cma)
            region = {k: v for k, v in region.items() if k != "direction_layer"}
            regions[scope] = region
            study.set_user_attr("layer", best.get("direction_layer"))
        studies[scope] = study
    for n in range(budget - first):
        scope = SCOPES[n % 2]
        study, region = studies[scope], regions[scope]
        trial = study.ask()
        params = {"direction_scope": scope}
        for name, bounds in region.items():
            params[name] = trial.suggest_int(name, *bounds) if name == "direction_layer" else trial.suggest_float(name, *bounds)
        if local != "tpe" and scope == ONE_LAYER:
            params["direction_layer"] = study.user_attrs["layer"]
        params = {"direction_scope": scope, **{k: params[k] for k in _names(scope)}}
        m = _measure(scorer, params, None, False, cap)
        v = _linear(m.estimate, cap)
        trial.set_user_attr("violation", v)
        value = math.log(max(m.kl, 1e-6))
        study.tell(trial, value if local == "tpe" else value + max(v, 0.0))
        trials.append(Trial(params, m.refusals, m.prompts, m.kl, m.seconds))
    return trials


def _tpe_refine(seed: int):
    return optuna.samplers.TPESampler(
        n_startup_trials=5, n_ei_candidates=CANDIDATES, multivariate=True, seed=seed, constraints_func=_constraint
    )


# --- Widen a range when a scope's best presses against it ------------------------------------------

# Faces a range may be pushed past, and how far each push goes (in the setting's own units); the
# far side of each is a hard limit. edge_strength (0-1) and a peak at the last layer are limits already.
PUSH = {("peak_layer", 0): 0.1, ("reach", 1): 0.15, ("strength", 0): 0.15, ("strength", 1): 0.25,
        ("direction_layer", 0): 0.1, ("direction_layer", 1): 0.1}
HARD = {"peak_layer": (0.1, 1.0), "reach": (1 / LAST, 1.0), "strength": (0.3, 2.5), "direction_layer": (0.15, 1.0)}
EDGE = 0.05  # within this share of a range from a face counts as pressing on it
CHECKS = 50  # a scope's best is checked every this many of its trials, from 100


def _pressed(params: dict, box: dict) -> list[tuple[str, int]]:
    """The faces of box the params press on, as (setting, 0 low / 1 high), that can still be pushed."""
    out = []
    for name in _names(params["direction_scope"]):
        setting = name.split(".")[-1] if name != "direction_layer" else name
        low, high = box[setting]
        if setting == "direction_layer":
            low_layer, high_layer = _layers(box)
            value, near = params[name], 0.5
            sides = ((0, value - low_layer <= near), (1, high_layer - value <= near))
        else:
            value, span = params[name], high - low
            sides = ((0, value - low <= EDGE * span), (1, high - value <= EDGE * span))
        for side, hit in sides:
            limit = HARD.get(setting, (low, high))[side]
            if hit and (setting, side) in PUSH and box[setting][side] != limit:
                out.append((setting, side))
    return sorted(set(out))


def _pushed(box: dict, faces: list[tuple[str, int]]) -> dict:
    wider = dict(box)
    for setting, side in faces:
        low, high = wider[setting]
        step = PUSH[(setting, side)]
        wider[setting] = (max(HARD[setting][0], low - step), high) if side == 0 else (low, min(HARD[setting][1], high + step))
    return wider


def _restudy(old: list[Trial], scope: str, box: dict, seed: int, cap: int, best: dict, faces) -> optuna.Study:
    """A new study for scope over the wider box, holding every trial so far, with probes past each
    pushed face: the scope's best moved a third and two thirds of the way across the new slab."""
    study = optuna.create_study(direction="minimize", sampler=_tpe_capped(seed))
    distributions = {name: optuna.distributions.IntDistribution(*_layers(box)) if name == "direction_layer"
                     else optuna.distributions.FloatDistribution(*box[name.split(".")[1]]) for name in _names(scope)}
    for t in old:
        if t.params["direction_scope"] != scope:
            continue
        v = _linear(t.refusals * 100 / t.prompts, cap)
        study.add_trial(optuna.trial.create_trial(
            params={k: t.params[k] for k in distributions}, distributions=distributions,
            value=math.log(max(t.kl, 1e-6)), user_attrs={"violation": v}, system_attrs={"constraints": (v,)},
        ))
    for setting, side in faces:
        for share in (1 / 3, 2 / 3):
            probe = {k: best[k] for k in distributions}
            for name in distributions:
                if (name.split(".")[-1] if name != "direction_layer" else name) != setting:
                    continue
                if setting == "direction_layer":
                    low, high = _layers(box)
                    probe[name] = int(min(high, max(low, best[name] + (-1 if side == 0 else 1) * round(share * 3))))
                else:
                    low, high = box[setting]
                    probe[name] = best[name] + (-1 if side == 0 else 1) * share * PUSH[(setting, side)]
                    probe[name] = min(high, max(low, probe[name]))
            study.enqueue_trial(probe)
    return study


def expand_search(scorer: Scorer, budget: int, seed: int, box: dict = TODAY, cap: int = CAP) -> list[Trial]:
    """split_search with capped TPE, but when a scope's best (checked every CHECKS of its trials from
    100) presses on a face of its ranges, that scope's ranges are pushed past the face and its search
    goes on over the wider box with every trial so far."""
    studies, boxes = {}, {}
    for i, scope in enumerate(SCOPES):
        study = optuna.create_study(direction="minimize", sampler=_tpe_capped(seed * 2 + i))
        for params in _sobol(scope, box, STARTUP, seed * 2 + i):
            study.enqueue_trial(params)
        studies[scope], boxes[scope] = study, dict(box)
    fewest = {scope: None for scope in SCOPES}
    counts = {scope: 0 for scope in SCOPES}
    trials = []
    for n in range(budget):
        scope = SCOPES[n % 2]
        if counts[scope] >= 100 and counts[scope] % CHECKS == 0:
            best = _incumbent(trials, scope, cap, scorer.prompts)
            faces = _pressed(best, boxes[scope])
            if faces:
                boxes[scope] = _pushed(boxes[scope], faces)
                studies[scope] = _restudy(trials, scope, boxes[scope], seed * 2 + 1000 + n, cap, best, faces)
        study = studies[scope]
        trial = study.ask()
        params = _draw(trial, scope, boxes[scope])
        m = _measure(scorer, params, fewest[scope], False, cap)
        if m.first is not None:
            fewest[scope] = m.first if fewest[scope] is None else min(fewest[scope], m.first)
        trial.set_user_attr("violation", _linear(m.estimate, cap))
        study.tell(trial, math.log(max(m.kl, 1e-6)))
        trials.append(Trial(params, m.refusals, m.prompts, m.kl, m.seconds))
        counts[scope] += 1
    return trials


expand_search.anytime = True


def tpe_refine(scorer, budget, seed):
    """TPE per scope for 60% of the budget, then TPE in a +-15% box around each scope's best."""
    return refine_search(scorer, budget, seed, TODAY, _tpe_capped)


def tpe_cma(scorer, budget, seed):
    """TPE per scope for 60% of the budget, then CMA-ES from each scope's best (cap as a penalty)."""
    return refine_search(scorer, budget, seed, TODAY, _tpe_capped, local="cma")


def tpe_refine_cap1(scorer, budget, seed):
    return refine_search(scorer, budget, seed, TODAY, _tpe_capped, cap=1)

STRATEGIES = {
    "candor": candor_today, "heretic": heretic, "random": random_search,
    "gp": gp_today, "gpwide": gp_wider, "gpstop": gp_wider_stop, "tpesplit": tpe_split, "tpewide": tpe_wider,
    "gpcap1": gp_cap1, "gplog": gp_log, "gprandom": gp_random_start, "gpstoptoday": gp_stop_today,
    "tpecap1": tpe_cap(1), "tpecap0": tpe_cap(0), "tperefine": tpe_refine, "tpecma": tpe_cma,
    "tperefine1": tpe_refine_cap1,
    "tpeexpand": expand_search,
    "tpestop": tpe_cap(CAP, prune=True), "tpeextend": tpe_cap(CAP, extend=2), "tpestaged": tpe_cap(CAP, prune=True, extend=2),
}
