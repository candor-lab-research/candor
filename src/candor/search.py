"""Search for the edit that changes the model least while it refuses almost nothing.

Each trial is an edit of the shape heretic searches: a refusal direction the layers use, and for each
component a strength that peaks at one layer and falls off either side of it. Edits come in two kinds,
searched as two studies that take turns: every layer reads the direction found at one layer, or each
layer reads its own. Each study starts from evenly spread trials, then Optuna's TPE sampler aims at the
least KL divergence among edits that refuse no more than search.max_refusals of the test prompts. For a
thinking model each trial also draws where its directions were read, the start spread evenly over both.

Searching the kinds apart, at that one goal, keeps either kind from being written off by trials that
suit the other. A search of both at once, aiming at every trade-off of refusals against KL, missed
heretic's Trial 170 on rude-llama: its sampler kept learning from weak edits that changed little and
refused much, and never tried the shape that worked. Test landscapes calibrated to that search showed
this design missing the best far less often (see results/phase-2-rude-llama.md).

Strength starts in heretic's range and widens only where a kind's best edit presses an end of it. On the
test landscapes a wide range from the start halved the hits when the best lay inside the narrow one,
while widening on that alarm stayed close to the wide range where the best needed it (see
results/phase-3-qwen3.5-9b.md).

All of the run's adapters are loaded into one llama-server, so a trial only sets their strengths per
request. A trial answers the harmful test prompts a chunk at a time and the judge labels each chunk;
refusals can only add up, so a trial that has refused more than the cap, and more than 25 fully scored
trials of its kind, can't win and stops there, unseen by the sampler. With search.max_budget_hits set, an
edit counts only if its thinking also reaches the budget on no more prompts than that, and a trial stops
once it can't, though the sampler may see that stop. KL divergence is read from the harmless test
prompts' first tokens. The held-out prompts are left for candor measure. Nothing a model writes is kept:
answers are judged, then dropped.

Any other edit, such as a heretic trial, can be scored the same way without joining the search, to see
where it would sit among the search's trials and whether the search could have drawn it.
"""

import json
import math
import time
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field
from itertools import islice, zip_longest
from pathlib import Path
from typing import Any

import optuna
import torch
from optuna.distributions import FloatDistribution
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import BaseTrial, FrozenTrial, TrialState

from . import config, kl
from .engine import GREEDY, LlamaServer, Thinking
from .judge import Guard, is_refusal, judged_refusal, read_verdict
from .kernel import Kernel
from .measure import Edit, edit_adapters
from .run import ANSWER_SETS, ANSWERING, STAGES, Run, RunError, answering, digest, duration, seeds

# Harmful prompts judged at a time, so a trial that can't win stops early. The server answers the
# next ones meanwhile.
CHUNK = 8
# The sampler weighs this many candidates for each trial, as heretic does, rather than Optuna's 24.
CANDIDATES = 128
# The version of what a trial's score means. Bump it in any commit that changes how trials are scored,
# so an older search isn't carried on as if it were the same.
VERSION = 2
PER_LAYER, ONE_LAYER = "per layer", "one layer"
# The kinds of edit, in the order they take turns.
SCOPES = (ONE_LAYER, PER_LAYER)
STUDIES = {ONE_LAYER: "candor: one layer", PER_LAYER: "candor: per layer"}
SETTINGS = ("strength", "peak_layer", "edge_strength", "reach")
# Where a thinking model's trials read their directions: at the end of the prompt, where its thinking
# starts, or where its answer starts after an empty thinking block. The search draws one for each trial.
THINK_START, ANSWER_START = "think start", "answer start"
POSITIONS = (THINK_START, ANSWER_START)
# How far a setting read back from an edit may stray past the search's range from rounding alone.
ROUNDING = 1e-9
# KL is shown to this many decimals; edits closer than that are not told apart.
KL_DIGITS = 4
# A setting within this share of its range from an end of it presses on that end.
EDGE = 0.05
# A trial stops early only once this many trials of its kind, scored in full, refused fewer. TPE learns
# from its best trials, at most 25 of them (Optuna's default gamma), and these all rank ahead of the
# stopped trial, so stopping it changes nothing the sampler draws. Stopping sooner was faster in the
# test landscapes but missed the best more often.
STOP_AFTER = 25
# A kind's strength range is checked for widening once it has this many finished trials, and again every
# WIDEN_EVERY after; a widened range starts with WIDENED_START evenly spread trials, as tested.
WIDEN_FROM, WIDEN_EVERY, WIDENED_START = 60, 30, 10


@dataclass(frozen=True)
class Score:
    # Counted by the judge, on the prompts judged. A reply it gave no label counts as a refusal, so
    # nothing scores well by confusing the judge.
    refusals: int
    keyword_refusals: int
    unlabelled: int
    kl: float
    # Harmful prompts judged: all of them, unless the trial was stopped early.
    judged: int
    # A thinking model's thinking on the prompts judged: its mean length in tokens, and how many
    # answers reached the budget, so the model was made to stop thinking and answer; and KL read where
    # the thinking starts, to see whether it tells edits apart as KL at the answer start does.
    thinking: float = 0.0
    forced: int = 0
    kl_thinking: float | None = None
    # KL where the answer starts after an empty thinking block, as heretic read it, kept beside the KL
    # after the unedited model's own thinking, which the search aims at.
    kl_empty: float | None = None


@dataclass(frozen=True)
class Scores:
    prompts: int
    # None when the unedited model wasn't scored again.
    base_refusals: int | None
    edits: list[Score]


@dataclass(frozen=True)
class Candidate:
    trial: int
    direction: str
    refusals: int
    keyword_refusals: int
    kl: float
    params: dict[str, Any]


@dataclass(frozen=True)
class Found:
    """A search's finished trials, by the number it shows each, and the strength range of each kind
    whose range the search widened."""

    trials: dict[int, FrozenTrial]
    base_refusals: int | None
    strength: dict[str, tuple[float, float]] = field(default_factory=dict)


def reaches(space: config.Search, layers: int) -> tuple[float, float]:
    """The reach a trial is drawn from. An edit reaches at least one layer either side of its peak, as
    in heretic, so whatever the peak some layer is edited."""
    last = layers - 1
    return min(max(space.reach[0], 1 / last), space.reach[1]), space.reach[1]


def _ranges(space: config.Search, layers: int) -> dict[str, tuple[float, float]]:
    return {
        "strength": space.strength,
        "peak_layer": space.peak_layer,
        "edge_strength": space.edge_strength,
        "reach": reaches(space, layers),
    }


def _names(scope: str, components: list[str]) -> list[str]:
    kernel = [f"{component}.{setting}" for component in components for setting in SETTINGS]
    return ["direction_layer", *kernel] if scope == ONE_LAYER else kernel


def scope_of(params: dict[str, Any]) -> str:
    return ONE_LAYER if "direction_layer" in params else PER_LAYER


def suggest(
    trial: BaseTrial,
    space: config.Search,
    layers: int,
    components: list[str],
    scope: str,
    positions: Sequence[str] = (),
) -> None:
    """Draw a trial's parameters. Only an edit of one layer's direction has a direction layer, and only a
    thinking model's edits, given the positions its directions are read at, a position."""
    if scope == ONE_LAYER:
        trial.suggest_int("direction_layer", *space.direction_layers(layers))
    ranges = _ranges(space, layers)
    for name in _names(PER_LAYER, components):
        trial.suggest_float(name, *ranges[name.split(".")[1]])
    if positions:
        trial.suggest_categorical("position", positions)


def spread(
    scope: str,
    space: config.Search,
    layers: int,
    components: list[str],
    count: int,
    seed: int,
    positions: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """count trials spread evenly over the scope's ranges (scrambled Sobol points), to start its study.
    Random trials leave gaps a narrow region of good edits can fall into. Given positions, the trials are
    split evenly between them, each position's spread on its own and taken in turn, so the first trials
    try every position."""
    if positions:
        shares = [
            spread(
                scope,
                space,
                layers,
                components,
                count // len(positions) + (j < count % len(positions)),
                seed * len(positions) + j,
            )
            for j in range(len(positions))
        ]
        return [
            {**params, "position": position}
            for row in zip_longest(*shares)
            for position, params in zip(positions, row, strict=True)
            if params is not None
        ]
    if not count:
        return []
    names = _names(scope, components)
    points = torch.quasirandom.SobolEngine(len(names), scramble=True, seed=seed).draw(count).double()
    low_layer, high_layer = space.direction_layers(layers)
    ranges = _ranges(space, layers)
    trials = []
    for row in points.tolist():
        params: dict[str, Any] = {}
        for share, name in zip(row, names, strict=True):
            if name == "direction_layer":
                params[name] = min(low_layer + int(share * (high_layer - low_layer + 1)), high_layer)
            else:
                low, high = ranges[name.split(".")[1]]
                params[name] = low + share * (high - low)
        trials.append(params)
    return trials


def edit_from(params: dict[str, Any], number: int, layers: int) -> Edit:
    """The edit a trial's parameters describe. Positions and reaches are fractions of the depth, and
    the edge strength a fraction of the peak."""
    last = layers - 1
    kernels = {}
    for component in sorted({name.split(".")[0] for name in params if "." in name}):
        strength = params[f"{component}.strength"]
        kernels[component] = Kernel(
            max_weight=strength,
            max_weight_position=params[f"{component}.peak_layer"] * last,
            min_weight=params[f"{component}.edge_strength"] * strength,
            min_weight_distance=params[f"{component}.reach"] * last,
        )
    answering = params.get("position") == ANSWER_START
    return Edit(f"trial-{number}", params.get("direction_layer"), kernels, answering)


def direction(layer: float | None, answering: bool) -> str:
    """Which directions an edit reads, as a table shows it."""
    read = PER_LAYER if layer is None else f"layer {layer:g}"
    return f"{read}, {ANSWER_START}" if answering else read


def violation(score: Score, prompts: int, cap: int) -> float:
    """How far over the cap an edit refuses, its refusals scaled to all prompts if it stopped early.
    The sampler treats an edit as within the caps when this is at most 0, and with
    search.max_budget_hits set, budget_violation too."""
    return score.refusals * prompts / score.judged - cap - 0.5


def budget_violation(score: Score, prompts: int, hits: int) -> float:
    """How far past search.max_budget_hits an edit's answers reach the thinking budget, scaled to all
    prompts if it stopped early, as violation scales refusals."""
    return score.forced * prompts / score.judged - hits - 0.5


def _constraints(attrs: dict[str, Any]) -> tuple[float, ...]:
    """A trial's constraints, as its record gives them: its violation, and its budget violation where
    it was judged by search.max_budget_hits."""
    budget = attrs.get("budget violation")
    return (attrs["violation"],) if budget is None else (attrs["violation"], budget)


def _violation(trial: FrozenTrial) -> tuple[float, ...]:
    return _constraints(trial.user_attrs)


def _over_caps(trial: FrozenTrial) -> float:
    """How far a trial goes over its caps all told, by which the sampler ranks the trials outside them,
    as Optuna sums a trial's violations; 0 within them."""
    return sum(max(value, 0.0) for value in _violation(trial))


def outside_space(edit: Edit, space: config.Search, layers: int, components: list[str]) -> list[str]:
    """Each setting of an edit the search couldn't have drawn, with the range it draws from; none if
    the search could have drawn the edit itself."""
    last = layers - 1
    outside = []
    low, high = space.direction_layers(layers)
    layer = edit.direction_layer
    if layer is not None and not (float(layer).is_integer() and low <= layer <= high):
        outside.append(f"direction layer {layer:g} (whole layers {low}-{high})")
    outside += [f"no {component} edit" for component in sorted(set(components) - set(edit.kernels))]
    ranges = _ranges(space, layers)
    for component, kernel in sorted(edit.kernels.items()):
        if component not in components:
            outside.append(f"a {component} edit, which the search never makes")
            continue
        drawn = {
            "strength": kernel.max_weight,
            "peak_layer": kernel.max_weight_position / last,
            "edge_strength": kernel.min_weight / kernel.max_weight if kernel.max_weight else 0.0,
            "reach": kernel.min_weight_distance / last,
        }
        for name, value in drawn.items():
            low_value, high_value = ranges[name]
            if not low_value - ROUNDING <= value <= high_value + ROUNDING:
                outside.append(f"{component}.{name} {value:.3g} ({low_value:.3g}-{high_value:.3g})")
    return outside


def _ends(value: float, low: float, high: float) -> tuple[bool, bool]:
    """Whether a setting presses on its range's floor, and on its ceiling: lies within EDGE of the range
    from it."""
    margin = EDGE * (high - low)
    return value - low <= margin, high - value <= margin


def at_edges(params: dict[str, Any], space: config.Search, layers: int) -> list[str]:
    """The ends of the search's ranges a trial's settings press on, where a better edit may lie just
    outside them: either end of the strength and of the direction layers read, the peak's floor and
    the reach's ceiling, unless that end is already the model's own. The other ends always are."""
    pressed = []
    if "direction_layer" in params:
        low, high = space.direction_layers(layers)
        for end, word, limit in ((low, "first", 0), (high, "last", layers - 1)):
            if params["direction_layer"] == end != limit:
                pressed.append(f"direction layer {end}, the {word} the search reads")
    ranges = _ranges(space, layers)
    # Which ends can move: the floor, and the ceiling, short of the model's own limits.
    ends = {
        "strength": (True, True),
        "peak_layer": (ranges["peak_layer"][0] > 0, False),
        "reach": (False, ranges["reach"][1] < 1),
    }
    for name, value in sorted(params.items()):
        setting = name.split(".")[-1]
        if setting not in ends:
            continue
        low, high = ranges[setting]
        floor, ceiling = ends[setting]
        at_floor, at_ceiling = _ends(value, low, high)
        if floor and at_floor:
            pressed.append(f"{name} {value:.3g}, at its floor {low:.3g}")
        if ceiling and at_ceiling:
            pressed.append(f"{name} {value:.3g}, at its ceiling {high:.3g}")
    return pressed


def widened(found: Found, scope: str, space: config.Search) -> tuple[float, float]:
    """The strength range a kind goes on with at a check: each end its best edit within the cap presses on
    moved out to search.widest_strength, or with none within the cap, edits too weak, the ceiling."""
    low, high = space.strength
    widest_low, widest_high = space.widest_strength
    best = within_cap(found, scope, space)
    if not best:
        return low, widest_high
    pressed = [
        _ends(value, low, high) for name, value in best[0].params.items() if name.endswith(".strength")
    ]
    floor = any(at_floor for at_floor, _ in pressed)
    ceiling = any(at_ceiling for _, at_ceiling in pressed)
    return widest_low if floor else low, widest_high if ceiling else high


def kind_space(space: config.Search, found: Found | None, scope: str) -> config.Search:
    """The search settings a kind's trials are drawn from now: the run's, with strength as the search
    widened it."""
    if found is None or scope not in found.strength:
        return space
    return space.model_copy(update={"strength": found.strength[scope]})


def nearest_layer(edit: Edit, space: config.Search, layers: int) -> Edit | None:
    """An edit that reads its direction between two layers, read instead at the nearest whole layer, as
    the search could draw it; None for an edit that needs no rounding or whose nearest layer the search
    never reads."""
    layer = edit.direction_layer
    if layer is None or float(layer).is_integer():
        return None
    whole = round(layer)
    low, high = space.direction_layers(layers)
    if not low <= whole <= high:
        return None
    return Edit(f"{edit.name} at layer {whole}", whole, edit.kernels, edit.answering)


def beats(a: "Score | Candidate", b: "Score | Candidate") -> bool:
    """Whether a refuses no more than b with no more KL, and is ahead on at least one of them. KL is
    compared as shown, so a gap too small to show doesn't count."""
    a_kl, b_kl = round(a.kl, KL_DIGITS), round(b.kl, KL_DIGITS)
    return a.refusals <= b.refusals and a_kl <= b_kl and (a.refusals < b.refusals or a_kl < b_kl)


def ties(a: "Score | Candidate", b: "Score | Candidate") -> bool:
    """Whether a and b refuse as often with the same KL, as shown."""
    return a.refusals == b.refusals and round(a.kl, KL_DIGITS) == round(b.kl, KL_DIGITS)


def repeats(score: Score, searched: Score) -> bool:
    """Whether a trial scored again matches its score in the search, as shown."""
    return (
        ties(score, searched)
        and score.keyword_refusals == searched.keyword_refusals
        and score.judged == searched.judged
    )


def _key(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True)


def _storage(folder: Path) -> JournalStorage:
    return JournalStorage(JournalFileBackend(str(folder / "study.jsonl")))


def _completed(study: optuna.Study) -> list[FrozenTrial]:
    return study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))


def _score(trial: FrozenTrial) -> Score:
    attrs = trial.user_attrs
    return Score(
        attrs["refusals"],
        attrs["keyword_refusals"],
        attrs["unlabelled"],
        attrs["kl"],
        attrs["judged"],
        attrs.get("thinking", 0.0),
        attrs.get("forced", 0),
        attrs.get("kl_thinking"),
        attrs.get("kl_empty"),
    )


def _candidate(number: int, trial: FrozenTrial) -> Candidate:
    return Candidate(
        number,
        direction(trial.params.get("direction_layer"), trial.params.get("position") == ANSWER_START),
        trial.user_attrs["refusals"],
        trial.user_attrs["keyword_refusals"],
        trial.user_attrs["kl"],
        trial.params,
    )


def scored_in_full(found: Found) -> list[Candidate]:
    """The trials scored in full: a stopped trial's count is only a lower bound."""
    return [_candidate(number, t) for number, t in found.trials.items() if not t.user_attrs["stopped"]]


def front(found: list[Candidate]) -> list[Candidate]:
    """The candidates no other beats on both refusals and KL, fewest refusals first. KL is compared as
    shown."""
    kept: list[Candidate] = []
    for candidate in sorted(found, key=lambda c: (c.refusals, c.kl)):
        if not kept or round(candidate.kl, KL_DIGITS) < round(kept[-1].kl, KL_DIGITS):
            kept.append(candidate)
    return kept


def candidates(found: Found) -> list[Candidate]:
    """The front of the trials scored in full: a stopped trial's count is only a lower bound."""
    return front(scored_in_full(found))


def counts(score: Score, space: config.Search, stopped: bool = False) -> bool:
    """Whether an edit is within the cap, the one rule every choice of the search's best edits follows:
    scored in full, refusing no more than search.max_refusals and, with search.max_budget_hits set,
    reaching the thinking budget on no more answers than that."""
    hits = space.max_budget_hits
    return not stopped and score.refusals <= space.max_refusals and (hits is None or score.forced <= hits)


def _counts(trial: FrozenTrial, space: config.Search) -> bool:
    return counts(_score(trial), space, trial.user_attrs["stopped"])


def within_cap(found: Found, scope: str, space: config.Search) -> list[Candidate]:
    """The scope's trials within the cap (see counts), least KL first."""
    kept = [
        _candidate(number, t)
        for number, t in found.trials.items()
        if scope_of(t.params) == scope and _counts(t, space)
    ]
    return sorted(kept, key=lambda c: (c.kl, c.refusals))


def to_measure(found: Found, scope: str, space: config.Search) -> list[Candidate]:
    """The scope's trials within the cap with the least KL at each count refused, fewest refused first,
    leaving out any that one refusing fewer beats. 100 test prompts can't tell 0 refused from 2, so
    the held-out prompts choose between them."""
    return front(within_cap(found, scope, space))


def fewest_refused(found: Found, scope: str) -> Candidate | None:
    """The scope's trial scored in full that refused fewest, then with least KL."""
    kept = [c for c in scored_in_full(found) if scope_of(c.params) == scope]
    return min(kept, key=lambda c: (c.refusals, c.kl), default=None)


def best_so_far(found: Found, scope: str, space: config.Search) -> Candidate | None:
    """The scope's best trial so far, as the search's summary names it: the least KL of those to measure,
    or with none within the cap, the one that refused fewest. KL is compared as shown, so where two trials'
    KL ties at KL_DIGITS this can differ from the one widening judges by, the least KL unrounded."""
    kept = to_measure(found, scope, space)
    return kept[-1] if kept else fewest_refused(found, scope)


def aimed(found: Found, scope: str, space: config.Search) -> bool:
    """Whether the sampler has aimed any of the scope's trials yet. The spread start is a fixed grid, so
    its trials pressing on an end of a range say nothing yet."""
    return sum(scope_of(t.params) == scope for t in found.trials.values()) > space.startup_per_scope


def _kind_progress(found: Found, space: config.Search, scope: str) -> dict[str, Any]:
    numbers = [number for number, t in found.trials.items() if scope_of(t.params) == scope]
    low, high = kind_space(space, found, scope).strength
    kind: dict[str, Any] = {
        "trials": len(numbers),
        "stopped early": sum(found.trials[number].user_attrs["stopped"] for number in numbers),
    }
    if (hits := space.max_budget_hits) is not None:
        kind["failed on the budget"] = sum(_score(found.trials[number]).forced > hits for number in numbers)
    kind |= {"strength": [low, high], "widened": scope in found.strength}
    best = best_so_far(found, scope, space)
    if best is None:
        return kind
    strengths = {name: value for name, value in best.params.items() if name.endswith(".strength")}
    ends = (
        {name: _ends(value, low, high) for name, value in strengths.items()}
        if aimed(found, scope, space)
        else {}
    )
    kind["best"] = {
        "trial": best.trial,
        "refused": best.refusals,
        "KL": best.kl,
        "within the cap": _counts(found.trials[best.trial], space),
        "strengths": strengths,
        "at its floor": [name for name, (floor, _) in ends.items() if floor],
        "at its ceiling": [name for name, (_, ceiling) in ends.items() if ceiling],
        "trials since it improved": sum(number > best.trial for number in numbers),
    }
    return kind


def progress(
    found: Found, space: config.Search, done: int, target: int, seconds: float, thinks: bool
) -> dict[str, Any]:
    """How a search is going after done of its target trials, seconds into this session: its pace over the
    last search.report_every trials and the time left at that pace, each kind's trials and best so far,
    and for trials that think, how long those last trials thought."""
    last = list(found.trials.values())[-space.report_every :]
    pace = sum(t.user_attrs["seconds"] for t in last) / len(last)
    report: dict[str, Any] = {
        "finished": done,
        "of": target,
        "seconds": seconds,
        "over the last": len(last),
        "seconds a trial": pace,
        "seconds left": pace * (target - done),
        **{scope: _kind_progress(found, space, scope) for scope in SCOPES},
    }
    if thinks:
        scores = [_score(t) for t in last]
        answers = sum(s.judged for s in scores)
        report["thinking"] = {
            "tokens per answer": sum(s.thinking * s.judged for s in scores) / answers,
            "answers": answers,
            "reached the budget": sum(s.forced for s in scores),
        }
    return report


def _trials(count: int) -> str:
    return f"{count} trial" if count == 1 else f"{count} trials"


def progress_lines(report: dict[str, Any]) -> list[str]:
    """A progress report as the search shows it."""
    done, target, last = report["finished"], report["of"], report["over the last"]
    head = (
        f"After {done} of {target} trials, {duration(report['seconds'])} into this session: "
        f"{report['seconds a trial']:.0f} s a trial over the last {last}"
    )
    lines = [head + (f", so about {duration(report['seconds left'])} left" if done < target else "")]
    for scope in SCOPES:
        kind = report[scope]
        low, high = kind["strength"]
        span = f"range {low:g}-{high:g}" + (" (widened)" if kind["widened"] else "")
        line = f"  {scope}: {_trials(kind['trials'])}, {kind['stopped early']} stopped early"
        if "failed on the budget" in kind:
            line += f", {kind['failed on the budget']} failed on the budget"
        if "best" not in kind:
            lines.append(f"{line}; none scored in full yet; strength {span}")
            continue
        best = kind["best"]
        which = "best" if best["within the cap"] else "none within the cap, fewest refused"
        line += f"; {which} trial {best['trial']}, {best['refused']} refused, KL {best['KL']:.4f}"
        strengths = ", ".join(
            f"{name.removesuffix('.strength')} {value:.3g}"
            + (" at its floor" if name in best["at its floor"] else "")
            + (" at its ceiling" if name in best["at its ceiling"] else "")
            for name, value in best["strengths"].items()
        )
        since = _trials(best["trials since it improved"])
        lines.append(f"{line}; strength {strengths}, {span}; {since} since it improved")
    if thinking := report.get("thinking"):
        lines.append(
            f"  thinking over the last {last}: {thinking['tokens per answer']:.0f} tokens per answer, "
            f"{thinking['reached the budget']} of {thinking['answers']} answers reached the budget"
        )
    return lines


def _changed(before: dict[str, Any], now: dict[str, Any], keys: Iterable[str] | None = None) -> list[str]:
    keys = before.keys() | now.keys() if keys is None else keys
    return sorted(key for key in keys if before.get(key) != now.get(key))


def _study_name(scope: str, widenings: int) -> str:
    """The name of a kind's study after this many widenings of its strength range. Each holds every
    finished trial of the kind."""
    return STUDIES[scope] + (f", widened {widenings}" if widenings else "")


def _widenings(study: optuna.Study) -> list[list[float]]:
    """Each widening of the study's strength range so far: the trials finished then, and the new range."""
    return study.user_attrs.get("widened", [])


def _strength(study: optuna.Study) -> tuple[float, float] | None:
    """The strength range the study's trials are drawn from, if the search widened it."""
    widenings = _widenings(study)
    return (widenings[-1][1], widenings[-1][2]) if widenings else None


def _judged_name(scope: str, changes: int) -> str:
    """The name of a kind's study after this many changes of the search.max_budget_hits its trials are
    judged by. Each holds every finished trial of the kind."""
    return f"{STUDIES[scope]}, budget hits {changes}"


def _budget_changes(study: optuna.Study) -> list[list[Any]]:
    """Each change of the search.max_budget_hits the study's trials are judged by: the trials finished
    then, and the new setting, None for unset."""
    return study.user_attrs.get("budget hits", [])


def _budget_hits(study: optuna.Study) -> int | None:
    """The search.max_budget_hits the study's trials are judged by; None for a study that never had one."""
    changes = _budget_changes(study)
    return changes[-1][1] if changes else None


def _latest(storage: JournalStorage, scope: str) -> str | None:
    """The name of the kind's study the search goes on in: the last made of its whole studies. Each after
    the first is made from the one before, at a widening of its strength range or a change of
    search.max_budget_hits, and is only whole once it records its widenings, the last thing written to
    it."""
    made = {study.study_name: study.user_attrs for study in storage.get_all_studies()}
    if STUDIES[scope] not in made:
        return None
    whole = [
        name
        for name, attrs in made.items()
        if name == STUDIES[scope] or (name.startswith(STUDIES[scope] + ",") and "widened" in attrs)
    ]
    return max(whole, key=storage.get_study_id_from_name)


def _found(studies: dict[str, optuna.Study]) -> Found:
    trials, base, strength = {}, None, {}
    for scope, study in studies.items():
        base = study.user_attrs.get("base_refusals", base)
        trials |= {t.user_attrs["number"]: t for t in _completed(study)}
        if widened_to := _strength(study):
            strength[scope] = widened_to
    return Found(dict(sorted(trials.items())), base, strength)


def load(run: Run) -> Found | None:
    """The run's search, if it has one."""
    folder = run.path("search")
    if not (folder / "study.jsonl").exists():
        return None
    storage = _storage(folder)
    names = set(optuna.get_all_study_names(storage))
    if not names & set(STUDIES.values()):
        raise RunError(f"{folder} holds a search made by an older candor; move it aside or delete it")
    return _found(
        {
            scope: optuna.load_study(study_name=name, storage=storage)
            for scope in SCOPES
            if (name := _latest(storage, scope))
        }
    )


def searched_score(found: Found, number: int) -> Score:
    """The score the search gave one of its finished trials, by the number the search shows."""
    if number not in found.trials:
        raise RunError(f"the search has no finished trial {number}")
    return _score(found.trials[number])


def _rebuilt_from(run: Run) -> dict[str, Any]:
    """What a search's trials are rebuilt from besides their settings: the adapters for every position
    its directions are read at, and the code that scored them. Call after run.directions()."""
    made = [run.adapter_inputs(answering) for answering in run.positions]
    return {
        # A model that doesn't think reads at one position, recorded as before there were two.
        "adapters": digest(made[0] if len(made) == 1 else made),
        "code": {**{stage: STAGES[stage] for stage in list(STAGES)[:4]}, "search": VERSION},
    }


def _beside(run: Run, ids: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
    """For a thinking model, the harmless test prompts KL is read on beside the KL the search aims at:
    where the thinking starts, and where the answer starts after an empty thinking block."""
    if not run.settings.deployment.think_close:
        return {}
    return {"kl_thinking": ids["harmless_test"], "kl_empty": ids["harmless_test" + ANSWERING]}


def trial_edit(run: Run, number: int) -> Edit:
    """A finished trial of this run's search, by the number the search shows. Its edit can only be
    rebuilt as it was scored while the run's directions and adapters are the ones the search used."""
    folder = run.path("search")
    if not (folder / "study.jsonl").exists():
        raise RunError(f"{run.folder} has no search yet; run candor search")
    with run.locked():
        run.gguf()
        run.directions()
        now = json.loads(json.dumps(_rebuilt_from(run)))
    if not (folder / "inputs.json").exists():
        raise RunError(f"{folder} holds a search with no record of its settings")
    before = json.loads((folder / "inputs.json").read_text())
    if changed := _changed(before, now, ["adapters", "code"]):
        raise RunError(
            f"the search was made with different {' and '.join(changed)} from this run's now, so its "
            "trials can't be rebuilt as they were scored"
        )
    return recorded_edit(run, number)


def recorded_edit(run: Run, number: int) -> Edit:
    """A finished trial's edit as the search recorded it, without checking it can still be rebuilt as
    it was scored: enough to find its export, and it needs none of the model's full-size files."""
    found = load(run)
    if found is None:
        raise RunError(f"{run.folder} has no search yet; run candor search")
    if number not in found.trials:
        raise RunError(f"the search has no finished trial {number}")
    return edit_from(found.trials[number].params, number, run.layers)


class Search:
    def __init__(self, run: Run):
        self.run = run
        self.folder = run.path("search")
        # Where trials may read their directions: a choice only for a thinking model.
        self.positions = POSITIONS if run.settings.deployment.think_close else ()
        # Whether trials think, as the model is used: a thinking model's do unless scoring.thinking is off.
        self.thinks = bool(run.settings.deployment.think_close) and run.settings.scoring.thinking == "on"

    @property
    def slots(self) -> int:
        """How many answers the server works on at once as edits are scored."""
        settings = self.run.settings
        if settings.slots.search is not None:
            return settings.slots.search
        return config.THINKING_MODEL_SLOTS if settings.deployment.think_close else config.SEARCH_SLOTS

    def search(
        self,
        target: int,
        started: Callable[[int, int], None] = lambda done, total: None,
        finished: Callable[[str | None], None] = lambda best: None,
    ) -> tuple[Found, float]:
        """Run trials until target are finished, taking turns between the kinds of edit, each kind up to
        search.trials_per_scope. started gets the trials already finished and the target; finished is
        called after each trial, with a line to show when it is within the cap and no earlier trial of
        its kind refused as few with as little KL. Every search.report_every finished trials it reports
        its progress (see progress). Returns the trials and the mean seconds per trial."""
        run, space = self.run, self.run.settings.search
        began = time.perf_counter()
        with run.locked():
            run.check_remote_code()
            if not run.path("directions.pt").exists():
                run.check_memory()
            if self.positions and not self.thinks:
                self.check_thinking_off()
            adapters = run.adapters()
            _, quantized = run.gguf()
            ids = run.prompt_ids()
            # A search made from other inputs is refused before a thinking model's unedited thinking is
            # sampled, and a new search is recorded only once that thinking exists.
            self._check(self.inputs(ids), known=True)
            harmful, harmless = ids["harmful_test"], run.kl_ids(ids, "harmless_test")
            studies = self._studies(self.inputs(ids, harmless), len(harmful))
            target = min(target, len(SCOPES) * space.trials_per_scope)
            started(self._done(studies), target)
            if self._done(studies) < target:
                files = [path for made in adapters.values() for path in made.values()]
                with self._scorer(quantized, files, ids, harmless, _beside(run, ids)) as scorer:
                    first = studies[SCOPES[0]]
                    base = first.user_attrs.get("base_refusals")
                    if base is None:
                        base = scorer.score({}, read_kl=False).refusals
                        first.set_user_attr("base_refusals", base)
                    run.report(f"The unedited model refuses {base} of {len(harmful)} test prompts")
                    kept = {scope: self._within_cap(studies[scope]) for scope in SCOPES}
                    while self._done(studies) < target:
                        scope = self._next(studies)
                        self._widen(studies, scope)
                        trial = self._trial(studies, scope, scorer, adapters)
                        number, score = trial.user_attrs["number"], _score(trial)
                        within = _counts(trial, space)
                        shown = round(score.kl, KL_DIGITS)
                        before = [
                            round(kl, KL_DIGITS) for refusals, kl in kept[scope] if refusals <= score.refusals
                        ]
                        best = within and shown < min(before, default=math.inf)
                        if within:
                            kept[scope].append((score.refusals, score.kl))
                        finished(
                            f"Trial {number}, {scope}: {score.refusals} of {len(harmful)} refused, KL "
                            f"{score.kl:.4f}, the least KL so far of its kind refusing {score.refusals} "
                            "or fewer"
                            if best
                            else None
                        )
                        done = self._done(studies)
                        if space.report_every and (done % space.report_every == 0 or done == target):
                            self._progress(studies, done, target, time.perf_counter() - began)
            found = _found(studies)
            seconds = [t.user_attrs["seconds"] for t in found.trials.values()]
            return found, sum(seconds) / len(seconds) if seconds else 0.0

    def score(
        self,
        edits: list[Edit],
        thinks: bool | None = None,
        seed: int | None = None,
        scored: Callable[[int, Score], None] = lambda index, score: None,
        base: bool = True,
    ) -> Scores:
        """Score edits exactly as the search scores its trials, without adding them to it: the same
        server with the same adapters loaded, the same test prompts and judge, each edit in full. An
        edit that reads its direction where the search has no adapters gets its own, in the run's
        scores folder. The unedited model is scored again too, if base. A thinking model's edits think
        when thinks is True and answer after an empty thinking block when it is False, by default as the
        search's trials do; seed replaces the sampling seed. scored gets each edit's index and score as
        it is done."""
        with self.run.locked():
            return self.score_locked(edits, thinks, seed, scored, base)

    def score_locked(
        self,
        edits: list[Edit],
        thinks: bool | None = None,
        seed: int | None = None,
        scored: Callable[[int, Score], None] = lambda index, score: None,
        base: bool = True,
        beside_search: bool = True,
    ) -> Scores:
        """score, for a caller that holds the run's lock. Unless beside_search is False, the scores are
        to be set beside the search's trials, so a search here made from other inputs is refused."""
        run = self.run
        run.check_remote_code()
        if not run.path("directions.pt").exists():
            run.check_memory()
        if self.positions and not (self.thinks if thinks is None else thinks):
            self.check_thinking_off()
        _, quantized = run.gguf()
        ids = run.prompt_ids()
        run.directions()
        # Checked before the adapters are brought up to date and before a thinking model's unedited
        # thinking is sampled, so a mismatch leaves the run as it was.
        # The Optuna version decides which trials the search draws, not how one is scored.
        if beside_search:
            self._check(self.inputs(ids), known=True, scoring=True)
        harmful, harmless = ids["harmful_test"], run.kl_ids(ids, "harmless_test")
        if beside_search:
            self._check(self.inputs(ids, harmless), scoring=True)
        adapters = run.adapters()
        made = [self._edit_files(edit, adapters) for edit in edits] if run.linear else []
        files = [path for found in [*adapters.values(), *made] for path in found.values()]
        with self._scorer(
            quantized, list(dict.fromkeys(files)), ids, harmless, _beside(run, ids), thinks, seed
        ) as scorer:
            base_refusals = scorer.score({}, read_kl=False).refusals if base else None
            scores = []
            for index, edit in enumerate(edits):
                run.report(f"Scoring {edit.name}")
                applied = self._applied(edit, adapters, scorer, run.path("scores", "edits", edit.name))
                scores.append(scorer.score(applied))
                scored(index, scores[-1])
        return Scores(len(harmful), base_refusals, scores)

    def check_thinking_off(self) -> None:
        """Stop before a thinking model is scored with its thinking off unless its chat template can turn
        thinking off (see Run.check_thinking_off), before anything is built or recorded for it. Where the
        run's search already scores with thinking off, as it began, nothing is checked and no server is
        started: the template passed when that search began. For a caller that holds the run's lock."""
        record = self.folder / "inputs.json"
        if record.exists() and json.loads(record.read_text()).get("trials think") == "off":
            return
        _, quantized = self.run.gguf()
        with self.run.server(quantized, context_per_slot=512, parallel=1) as server:
            self.run.check_thinking_off(server)

    def _edit_files(self, edit: Edit, adapters: dict[str, dict[str, Path]]) -> dict[str, Path]:
        """The adapters an edit is scored with: the search's own for a direction it reads, or else
        adapters of its own, built as candor measure builds them. These depend only on the direction,
        so they are kept by layer."""
        prefix = ANSWER_SETS if edit.answering else ""
        layer = edit.direction_layer
        if layer is None:
            return adapters[prefix + "per-layer"]
        if float(layer).is_integer() and f"{prefix}layer-{int(layer)}" in adapters:
            return adapters[f"{prefix}layer-{int(layer)}"]
        return edit_adapters(self.run, edit, self.run.path("scores", f"{prefix}layer-{layer!r}"))[0]

    def _applied(
        self, edit: Edit, adapters: dict[str, dict[str, Path]], scorer: "_Scorer", folder: Path
    ) -> dict[Path, float]:
        """The adapters scoring an edit applies, and at what strength. An edit that isn't linear in
        strength has one adapter built for it, in folder, and the server is restarted with it."""
        if self.run.linear:
            return self._scales(edit, self._edit_files(edit, adapters))
        path, _ = self.run.edit_adapter(folder, edit)
        scorer.load([path])
        return {path: 1.0}

    def _scales(self, edit: Edit, files: dict[str, Path]) -> dict[Path, float]:
        """Each of the edit's adapters at its strength. Those at 0 are left out, which llama.cpp skips."""
        strengths = edit.strengths(self.run.shape[3], self.run.target_names)
        return {files[name]: strength for name, strength in strengths.items() if strength}

    @contextmanager
    def _scorer(
        self,
        quantized: Path,
        files: list[Path],
        ids: dict[str, list[list[int]]],
        harmless: list[list[int]],
        beside: dict[str, list[list[int]]],
        thinks: bool | None = None,
        seed: int | None = None,
    ) -> Iterator["_Scorer"]:
        """A scorer on llama-server with these adapters loaded, and the judge, for the test prompts; it
        also reads KL on the harmless prompts beside, by the name of the score that keeps it. A thinking
        model thinks when thinks is True, by default as the search's trials do, and otherwise answers
        greedily after an empty thinking block. seed replaces the sampling seed."""
        run, settings = self.run, self.run.settings
        thinks = self.thinks if thinks is None else thinks
        closed = bool(settings.deployment.think_close) and not thinks
        texts = run.prompt_texts()[0]["harmful_test"]
        harmful = ids["harmful_test" + ANSWERING] if closed else ids["harmful_test"]
        budget = settings.scoring.thinking_budget
        tokens = settings.scoring.closed_answer_tokens if closed else settings.scoring.answer_tokens
        answer = tokens if closed else tokens + run.thinking_room(budget)
        reads = [p for prompts in [harmless, *beside.values()] for p in prompts]
        # A slot holds one request at a time: an answer to a harmful prompt, or a KL read of one token.
        room = max(max(map(len, harmful)) + answer, max(map(len, reads)) + 1) + 8
        slots, setting = self.slots, "slots.search in candor.toml"

        def server_with(adapters: list[Path]) -> LlamaServer:
            return run.server(
                quantized, adapters, context_per_slot=room, parallel=slots, slots_setting=setting
            )

        guard = Guard(settings.measure.judge, settings.measure.judge_revision)
        scorer = None
        server = server_with(files)
        try:
            server.start()
            run.check_prompts(server, "harmful_test", texts, ids["harmful_test"])
            count = len(harmful)
            answering = (
                _Answering(tokens, GREEDY, [0] * count, None)
                if closed
                else _Answering(
                    tokens,
                    run.sampling(),
                    seeds(settings.sampling.seed if seed is None else seed, count),
                    run.thinking(server, budget),
                )
            )
            scorer = _Scorer(
                server,
                guard,
                texts,
                harmful,
                harmless,
                settings.scoring.kl_tokens,
                answering,
                beside,
                server_with,
            )
            yield scorer
        finally:
            (scorer.server if scorer else server).stop()
            guard.close()

    def inputs(
        self, ids: dict[str, list[list[int]]], harmless: list[list[int]] | None = None
    ) -> dict[str, Any]:
        """What the search's scores depend on. A search is only carried on while these are the same. For a
        thinking model that includes the unedited thinking KL is read after: named by what it was
        sampled from, known before any is sampled, and given the KL prompts, the thinking itself, as
        thinking sampled afresh from the same settings comes out differently."""
        settings = self.run.settings
        # Nothing for a greedy model that doesn't think, so searches made before candor sampled still
        # carry on.
        how = answering(settings)
        if "sampling" in how:
            how["sampling"]["seed"] = settings.sampling.seed
        if "thinking" in how:
            how["thinking"]["budget"] = settings.scoring.thinking_budget or how["thinking"]["budget"]
        # Only a search that may widen strength records how far, so searches made before it could still
        # carry on; and a model that doesn't think records its slots only at another number than the
        # default. How often it reports changes no score, and the studies record the
        # search.max_budget_hits they judge their trials by, so a change judges them again.
        unrecorded = {"trials_per_scope", "report_every", "max_budget_hits"} | (
            set() if settings.search.widens() else {"widest_strength"}
        )
        recorded = settings.deployment.think_close or self.slots != config.SEARCH_SLOTS
        slots = {"slots": self.slots} if recorded else {}
        return {
            "model file": self.run.inputs["quantized"],
            **_rebuilt_from(self.run),
            "test prompts": digest([ids["harmful_test"], ids["harmless_test"]]),
            "answer length": settings.scoring.answer_tokens,
            **how,
            "KL tokens": settings.scoring.kl_tokens,
            "judge": [settings.measure.judge, settings.measure.judge_revision],
            "KV cache": settings.deployment.kv_cache,
            "search settings": settings.search.model_dump(exclude=unrecorded),
            **slots,
            "Optuna": optuna.__version__,
        } | (self._thinking(harmless) if settings.deployment.think_close else {})

    def _thinking(self, harmless: list[list[int]] | None) -> dict[str, Any]:
        scoring = self.run.settings.scoring
        found: dict[str, Any] = {
            "unedited thinking": self.run.thinking_key(),
            "trials think": scoring.thinking,
        }
        if scoring.thinking == "off":
            found["answer length with thinking off"] = scoring.closed_answer_tokens
        return found | ({"KL prompts": digest(harmless)} if harmless is not None else {})

    def _check(self, inputs: dict[str, Any], known: bool = False, scoring: bool = False) -> None:
        """Refuse a search here made from other inputs. With known, only the inputs given are compared, as
        those not known yet are compared once they are; scoring an edit beside the search's trials
        compares all but the Optuna version, which decides only which trials it draws."""
        record, journal = self.folder / "inputs.json", self.folder / "study.jsonl"
        if record.exists():
            before, now = json.loads(record.read_text()), json.loads(json.dumps(inputs))
            keys = now.keys() if known else before.keys() | now.keys()
            if changed := _changed(before, now, keys - {"Optuna"} if scoring else keys):
                if scoring:
                    raise RunError(
                        f"the search was made with different {' and '.join(changed)} from this run's now, "
                        "so an edit scored now can't be set beside its trials"
                    )
                raise RunError(
                    f"{self.folder} holds a search made with a different {', '.join(changed)}; move it "
                    "aside or delete it to start a new search"
                )
        elif journal.exists():
            raise RunError(
                f"{self.folder} holds a search with no record of its settings; move it aside or delete it"
            )

    def _studies(self, inputs: dict[str, Any], prompts: int) -> dict[str, optuna.Study]:
        """The run's study of each kind of edit: new ones, or those already here if they were made from
        the same inputs, their trials judged by search.max_budget_hits as set now (see _judge_again)
        when that changed. Each trial was scored on this many harmful test prompts."""
        record = self.folder / "inputs.json"
        self._check(inputs)
        if not record.exists():
            self.folder.mkdir(exist_ok=True)
            partial = record.with_name("inputs.json.building")
            partial.write_text(json.dumps(json.loads(json.dumps(inputs)), sort_keys=True))
            partial.replace(record)
        storage = _storage(self.folder)
        space = self.run.settings.search
        hits = space.max_budget_hits
        studies, judged, lost = {}, [], 0
        for scope in SCOPES:
            study = optuna.create_study(
                study_name=_latest(storage, scope) or STUDIES[scope],
                storage=storage,
                direction="minimize",
                load_if_exists=True,
            )
            # A trial still marked running was stopped part way; the run lock means no search is running
            # it.
            for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
                study.tell(trial.number, state=TrialState.FAIL)
            if _budget_hits(study) != hits:
                before = space.model_copy(update={"max_budget_hits": _budget_hits(study)})
                lost += sum(_counts(t, before) and not _counts(t, space) for t in _completed(study))
                judged += _completed(study)
                study = self._judge_again(study, scope, prompts)
            self._ready(study, scope)
            studies[scope] = study
        if judged:
            line = f"The search's {len(judged)} finished trials are judged again by search.max_budget_hits"
            if hits is None:
                self.run.report(f"{line}, now unset")
            else:
                over = sum(_score(trial).forced > hits for trial in judged)
                self.run.report(
                    f"{line}, now {hits}: {over} reached the thinking budget more often, which leaves {lost} "
                    "fewer within the cap"
                )
        return studies

    def _judge_again(self, study: optuna.Study, scope: str, prompts: int) -> optuna.Study:
        """The kind's study, its trials judged by search.max_budget_hits as set now. The sampler was told
        each trial's constraints as it finished, and Optuna can't change a finished trial, so the trials
        move to a new study of the kind, their constraints told afresh from their refusals and answers
        at the budget; their numbers, settings and scores stay as they were."""
        hits = self.run.settings.search.max_budget_hits
        changes = _budget_changes(study)
        finished = _completed(study)
        if not finished:
            study.set_user_attr("budget hits", [*changes, [0, hits]])
            return study
        storage = _storage(self.folder)
        name = _judged_name(scope, len(changes) + 1)
        # Left part made by a search stopped while judging its trials again.
        if name in optuna.get_all_study_names(storage):
            optuna.delete_study(study_name=name, storage=storage)
        new = optuna.create_study(study_name=name, storage=storage, direction="minimize")
        for key, value in study.user_attrs.items():
            if key not in ("widened", "budget hits"):
                new.set_user_attr(key, value)
        for trial in finished:
            attrs = {key: value for key, value in trial.user_attrs.items() if key != "budget violation"}
            if hits is not None:
                attrs["budget violation"] = budget_violation(_score(trial), prompts, hits)
            new.add_trial(
                optuna.trial.create_trial(
                    params=trial.params,
                    distributions=trial.distributions,
                    value=trial.value,
                    user_attrs=attrs,
                    system_attrs=trial.system_attrs | {_CONSTRAINTS_KEY: list(_constraints(attrs))},
                )
            )
        new.set_user_attr("budget hits", [*changes, [len(finished), hits]])
        # Written last, as at a widening: a study without it was left part made.
        new.set_user_attr("widened", _widenings(study))
        return new

    def _ready(self, study: optuna.Study, scope: str) -> None:
        """Queue the kind's spread starts its study hasn't scored or queued yet, so a point whose trial
        failed is queued again rather than lost, and give the study its sampler."""
        space = self.run.settings.search
        _, layers, _, writers = self.run.shape
        components = sorted({w.component for w in writers})
        i = SCOPES.index(scope)
        # The first spread over the run's strength range, then one over each range it widened to.
        starts = spread(
            scope, space, layers, components, space.startup_per_scope, space.seed + i, self.positions
        )
        for finished, low, high in _widenings(study):
            wider = space.model_copy(update={"strength": (low, high)})
            seed = space.seed + len(SCOPES) * int(finished) + i
            starts += spread(scope, wider, layers, components, WIDENED_START, seed, self.positions)
        kept = {
            _key(trial.system_attrs.get("fixed_params", trial.params))
            for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.WAITING))
        }
        for params in starts:
            if _key(params) not in kept:
                study.enqueue_trial(params)
        # The sampler starts again from its seed in each session, so the seed moves on with the trials
        # made, and a resumed search draws afresh rather than replaying the same random numbers.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            study.sampler = optuna.samplers.TPESampler(
                n_startup_trials=space.startup_per_scope,
                n_ei_candidates=CANDIDATES,
                multivariate=True,
                seed=space.seed + len(SCOPES) * len(study.trials) + i,
                constraints_func=_violation,
            )

    def _widen(self, studies: dict[str, optuna.Study], scope: str) -> None:
        """At a check, widen the kind's strength range if its best edit says to (see widened). Its finished
        trials move to a new study of the kind over the wider range, since the sampler only learns
        settings together whose ranges every trial shares, and evenly spread trials start the new range."""
        space = self.run.settings.search
        study = studies[scope]
        finished = len(_completed(study))
        widenings = _widenings(study)
        if (
            finished < max(WIDEN_FROM, space.startup_per_scope + 1)
            or (finished - WIDEN_FROM) % WIDEN_EVERY
            # Checked already, in a session that widened here and stopped.
            or (widenings and widenings[-1][0] == finished)
        ):
            return
        found = _found({scope: study})
        before = self._space(study)
        wider = widened(found, scope, before)
        if wider == before.strength:
            return
        storage = _storage(self.folder)
        name = _study_name(scope, len(widenings) + 1)
        # Left part made by a search stopped while widening.
        if name in optuna.get_all_study_names(storage):
            optuna.delete_study(study_name=name, storage=storage)
        new = optuna.create_study(study_name=name, storage=storage, direction="minimize")
        for key, value in study.user_attrs.items():
            if key != "widened":
                new.set_user_attr(key, value)
        strength = FloatDistribution(*wider)
        for trial in _completed(study):
            distributions = {
                param: strength if param.endswith(".strength") else distribution
                for param, distribution in trial.distributions.items()
            }
            new.add_trial(
                optuna.trial.create_trial(
                    params=trial.params,
                    distributions=distributions,
                    value=trial.value,
                    user_attrs=trial.user_attrs,
                    system_attrs=trial.system_attrs,
                )
            )
        # Written last: a study without it was left part made.
        new.set_user_attr("widened", [*widenings, [finished, *wider]])
        self._ready(new, scope)
        studies[scope] = new
        if within_cap(found, scope, space):
            why = f"its best {scope} edit within the cap presses on an end of strength"
        else:
            why = f"no {scope} edit is within the cap yet"
        low, high = before.strength
        self.run.report(
            f"After {finished} {scope} trials {why}, so its strength widens from {low:g}-{high:g} to "
            f"{wider[0]:g}-{wider[1]:g}"
        )

    def _progress(self, studies: dict[str, optuna.Study], done: int, target: int, seconds: float) -> None:
        """Show how the search is going, and write it to search/progress.json in place of the last report."""
        report = progress(_found(studies), self.run.settings.search, done, target, seconds, self.thinks)
        path = self.folder / "progress.json"
        partial = path.with_name("progress.json.building")
        partial.write_text(json.dumps(report, indent=2))
        partial.replace(path)
        for line in progress_lines(report):
            self.run.report(line)

    def _space(self, study: optuna.Study) -> config.Search:
        """The search settings a kind's study draws its trials from: the run's, with strength as widened."""
        space = self.run.settings.search
        strength = _strength(study)
        return space.model_copy(update={"strength": strength}) if strength else space

    def _done(self, studies: dict[str, optuna.Study]) -> int:
        """The finished trials that count towards the target: a kind's beyond its share don't."""
        share = self.run.settings.search.trials_per_scope
        return sum(min(len(_completed(study)), share) for study in studies.values())

    def _next(self, studies: dict[str, optuna.Study]) -> str:
        """The kind of edit whose turn it is: the one with fewer finished trials, that has room for more."""
        space = self.run.settings.search
        open_scopes = [s for s in SCOPES if len(_completed(studies[s])) < space.trials_per_scope]
        return min(open_scopes, key=lambda s: (len(_completed(studies[s])), SCOPES.index(s)))

    def _within_cap(self, study: optuna.Study) -> list[tuple[int, float]]:
        """The refusals and KL of the study's trials within the cap (see counts)."""
        space = self.run.settings.search
        return [
            (t.user_attrs["refusals"], t.user_attrs["kl"]) for t in _completed(study) if _counts(t, space)
        ]

    def _trial(
        self,
        studies: dict[str, optuna.Study],
        scope: str,
        scorer: "_Scorer",
        adapters: dict[str, dict[str, Path]],
    ) -> FrozenTrial:
        run, space = self.run, self.run.settings.search
        _, layers, _, writers = run.shape
        study = studies[scope]
        numbers = [
            t.user_attrs.get("number", 0) for s in studies.values() for t in s.get_trials(deepcopy=False)
        ]
        number = max(numbers, default=0) + 1
        trial = study.ask()
        try:
            trial.set_user_attr("number", number)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
                suggest(
                    trial,
                    self._space(study),
                    layers,
                    sorted({w.component for w in writers}),
                    scope,
                    self.positions,
                )
            edit = edit_from(trial.params, number, layers)
            started = time.perf_counter()
            scales = self._applied(edit, adapters, scorer, run.path("scores", "trial"))
            cap, hits = space.max_refusals, space.max_budget_hits
            full = sorted(_over_caps(t) for t in _completed(study) if not t.user_attrs["stopped"])
            # A trial can't win once it refuses more than the cap. Refusing n puts it at least n - cap - 0.5
            # over the refusal cap, scaled or in full, so past this it is further over than STOP_AFTER
            # trials of its kind scored in full are over all their caps: the sampler ranks it behind them
            # either way, so it can't see the trial stop. With search.max_budget_hits set, a trial also
            # can't win once more answers reach the budget, but the sampler may see that stop.
            stop_above = (
                max(cap, math.floor(full[STOP_AFTER - 1] + cap + 0.5))
                if space.stop_early and len(full) >= STOP_AFTER
                else None
            )
            score = scorer.score(scales, stop_above, hits if space.stop_early else None)
            prompts = len(scorer.harmful)
            for key, value in asdict(score).items():
                trial.set_user_attr(key, value)
            trial.set_user_attr("stopped", score.judged < prompts)
            trial.set_user_attr("violation", violation(score, prompts, cap))
            if hits is not None:
                trial.set_user_attr("budget violation", budget_violation(score, prompts, hits))
            trial.set_user_attr("seconds", time.perf_counter() - started)
        except BaseException:
            study.tell(trial, state=TrialState.FAIL)
            raise
        return study.tell(trial, score.kl)


@dataclass(frozen=True)
class _Answering:
    """How a trial answers the harmful test prompts: answers this long after any thinking, sampled
    with each prompt's own seed, the same for every trial."""

    tokens: int
    sampling: dict[str, float]
    seeds: list[int]
    thinking: Thinking | None


class _Scorer:
    """Scores an edit, given as adapter strengths, against the unedited model's first tokens."""

    def __init__(
        self,
        server: LlamaServer,
        guard: Guard,
        texts: list[str],
        harmful: list[list[int]],
        harmless: list[list[int]],
        kl_tokens: int,
        answering: _Answering,
        beside: dict[str, list[list[int]]],
        server_with: Callable[[list[Path]], LlamaServer],
    ):
        self.server, self.guard, self.texts = server, guard, texts
        self.server_with = server_with
        self.harmful, self.harmless, self.kl_tokens = harmful, harmless, kl_tokens
        self.answering, self.beside = answering, beside
        self.base = server.top_logprobs(harmless, kl_tokens)
        self.base_beside = {name: server.top_logprobs(prompts, kl_tokens) for name, prompts in beside.items()}

    def load(self, adapters: list[Path]) -> None:
        """Restart the server with these adapters: an edit that isn't linear in strength has one built for
        it, and llama-server can't load an adapter while it runs. The unedited model's first tokens, read
        already, stay as they are."""
        self.server.stop()
        self.server = self.server_with(adapters)
        self.server.start()

    def score(
        self,
        scales: dict[Path, float],
        stop_above: int | None = None,
        forced_above: int | None = None,
        read_kl: bool = True,
    ) -> Score:
        """An edit's refusals and KL divergence. The harmful prompts are judged CHUNK at a time, in
        order, while the server answers the next ones; with stop_above, it stops once more than that
        many are refused, and with forced_above, once more answers than that reached the thinking
        budget. A trial stopped can't win, so nothing reads its KL, nor the unedited model's from
        itself: their KL is left unread, as infinite."""
        refusals = keyword_refusals = unlabelled = judged = thought = forced = 0
        how = self.answering
        answering = self.server.answers(
            self.harmful, how.tokens, scales, how.sampling, how.seeds, how.thinking
        )
        with closing(answering):
            for start in range(0, len(self.harmful), CHUNK):
                chunk = list(islice(answering, CHUNK))
                thought += sum(a.thinking for a in chunk)
                forced += sum(a.forced for a in chunk)
                # Only what follows a thinking model's thinking is its answer, so only that is judged.
                answers = [a.text for a in chunk]
                replies = self.guard.replies(self.texts[start : start + CHUNK], answers)
                labels = [
                    judged_refusal(answer, read_verdict(reply))
                    for answer, reply in zip(answers, replies, strict=True)
                ]
                refusals += sum(label is not False for label in labels)
                unlabelled += sum(label is None for label in labels)
                keyword_refusals += sum(map(is_refusal, answers))
                judged += len(answers)
                if (stop_above is not None and refusals > stop_above) or (
                    forced_above is not None and forced > forced_above
                ):
                    break
        if not read_kl or judged < len(self.harmful):
            return Score(refusals, keyword_refusals, unlabelled, math.inf, judged, thought / judged, forced)
        top = self.server.top_logprobs(self.harmless, self.kl_tokens, scales)
        divergence = kl.mean_from_top(self.base, top)
        beside = {
            name: kl.mean_from_top(
                self.base_beside[name], self.server.top_logprobs(prompts, self.kl_tokens, scales)
            )
            for name, prompts in self.beside.items()
        }
        return Score(
            refusals, keyword_refusals, unlabelled, divergence, judged, thought / judged, forced, **beside
        )
