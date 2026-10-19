"""Whether a thinking model's search may score its trials with thinking off.

A trial scored as the model is used thinks to its budget on every test prompt, many times the cost of
answering after an empty thinking block. The cheap score only helps if it ranks edits as thinking does,
and no published work shows that it does: the gap between the two changes size and even sign by model
and by edit. So candor proxy scores edits both ways on the search's own test prompts and compares the
order. The edits are the best a short search scored with thinking off found, as random edits mostly
change nothing (on Qwen3.5-9B 11 of 12 refused 97-99 of 100) and so can't show an order; or, once the
search may score thinking off, the trials named, so the ones it suggests are scored as used. Its scores
are kept in the run's proxy folder, so a stopped check carries on.
"""

import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from .files import write_atomic
from .measure import Edit
from .run import Run, RunError
from .search import POSITIONS, SCOPES, Score, Search, edit_from, load, scope_of, scored_in_full

# The search's best trials taken for each kind and position, and the strengths each is tried at as
# multiples of its own: thinking may need a stronger edit than the cheap score shows.
BEST = 3
FACTORS = (1.0, 1.4)
# The cheap score is trusted when it ranks the edits' refusals as thinking does at least this well
# (Spearman's rank correlation).
AGREEMENT = 0.7
# Thinking needs a stronger edit than the cheap score shows when, among the edits the cheap score says
# nearly work (refusing at most this share of the prompts), thinking refuses more than this by the
# median: a search scored with thinking off would settle on edits too weak when thinking.
NEARLY_WORKS, OFFSET = 0.25, 1
MODES = {"off": False, "on": True}


@dataclass(frozen=True)
class Check:
    edits: list[Edit]
    # The search trial each edit comes from, and its strength as a multiple of that trial's, once held
    # within search.widest_strength: the least and most over its strengths.
    trials: list[int]
    factors: list[tuple[float, float]]
    off: list[Score]
    on: list[Score]
    base: dict[str, int]
    # The edit scored again thinking, with another seed, and that score.
    repeated: int
    repeat: Score

    @property
    def agreement(self) -> float:
        return spearman([s.refusals for s in self.off], [s.refusals for s in self.on])

    @property
    def offset(self) -> float | None:
        """How many more prompts thinking refuses, by the median, among the edits the cheap score says
        nearly work; None if it says none do."""
        near = [
            on.refusals - off.refusals
            for off, on in zip(self.off, self.on, strict=True)
            if off.refusals <= NEARLY_WORKS * off.judged
        ]
        return statistics.median(near) if near else None

    def verdict(self) -> str:
        """ "holds" when the cheap score ranks edits as thinking does and isn't more lenient where it
        matters; "offset" when it ranks them alike but thinking needs stronger edits; "fails" when it
        doesn't rank them alike, or no edit came near working to tell."""
        if not self.agreement >= AGREEMENT or self.offset is None:
            return "fails"
        return "offset" if self.offset > OFFSET else "holds"


def edits(run: Run, chosen: Sequence[int] = ()) -> tuple[list[Edit], list[int], list[tuple[float, float]]]:
    """The edits checked: from the run's search, scored with thinking off, the trials chosen, or else
    the BEST trials scored in full of each kind and position, fewest refusals first and then least KL;
    each as found and at the other FACTORS of its strength, held within search.widest_strength. Each
    position's own best are taken, as the cheap score may favour directions read where the answer
    starts."""
    record = run.path("search", "inputs.json")
    if not record.exists() or json.loads(record.read_text()).get("trials think") != "off":
        raise RunError(
            "candor proxy checks the edits a search scored with thinking off finds: set "
            'scoring.thinking = "off" and run candor search with --trials 150 first'
        )
    found = load(run)
    full = scored_in_full(found) if found else []
    if chosen:
        by_trial = {c.trial: c for c in full}
        missing = [str(trial) for trial in dict.fromkeys(chosen) if trial not in by_trial]
        if missing:
            raise RunError(f"not trials the search scored in full: {', '.join(missing)}")
        picked = [by_trial[trial] for trial in dict.fromkeys(chosen)]
    else:
        picked = []
        for scope in SCOPES:
            for position in POSITIONS:
                group = [
                    c for c in full if scope_of(c.params) == scope and c.params.get("position") == position
                ]
                picked += sorted(group, key=lambda c: (c.refusals, c.kl))[:BEST]
    _, layers, _, _ = run.shape
    low, high = run.settings.search.widest_strength
    made, trials, factors = [], [], []
    for best in picked:
        for factor in FACTORS:
            params = {
                name: min(max(value * factor, low), high) if name.endswith(".strength") else value
                for name, value in best.params.items()
            }
            made.append(replace(edit_from(params, best.trial, layers), name=f"edit-{len(made) + 1}"))
            trials.append(best.trial)
            held = [params[name] / value for name, value in best.params.items() if name.endswith(".strength")]
            factors.append((min(held), max(held)))
    if not made:
        raise RunError("the search has no trials scored in full yet")
    return made, trials, factors


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman's rank correlation, tied values sharing their mean rank; nan if either is constant."""
    ra, rb = _ranks(a), _ranks(b)
    if len(set(ra)) < 2 or len(set(rb)) < 2:
        return math.nan
    return statistics.correlation(ra, rb)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for i in order[start : end + 1]:
            ranks[i] = (start + end) / 2
        start = end + 1
    return ranks


def check(run: Run, chosen: Sequence[int] = ()) -> Check:
    """Score the edits with thinking off, then thinking as the model is used, then the edit refusing
    fewest thinking again with another seed, to see how far sampling alone moves a score."""
    if not run.settings.deployment.think_close:
        raise RunError("candor proxy is for a model that thinks; this one doesn't")
    search = Search(run)
    made, trials, factors = edits(run, chosen)
    with run.locked():
        ids = run.prompt_ids()
        run.directions()
        inputs = _inputs(search, ids, run.kl_ids(ids, "harmless_test"), trials)
        # Trials named are kept apart, so checking them never replaces the check of the best.
        name = "trials.json" if chosen else "scores.json"
        saved = _load(run, inputs, name)
        # Scores read on another machine can differ slightly, so each one the check was scored on is kept.
        if (here := run.machine()) not in saved.setdefault("machines", []):
            saved["machines"].append(here)
        for mode, thinks in MODES.items():
            todo = [i for i in range(len(made)) if str(i) not in saved[mode]]
            if not todo and mode in saved["base"]:
                continue
            run.report(f"Scoring {len(todo)} edits with thinking {mode}")

            def keep(index: int, score: Score, mode: str = mode, todo: list[int] = todo) -> None:
                saved[mode][str(todo[index])] = asdict(score)
                _save(run, saved, name)

            scores = search.score_locked(
                [made[i] for i in todo],
                thinks,
                scored=keep,
                base=mode not in saved["base"],
                beside_search=False,
            )
            if scores.base_refusals is not None:
                saved["base"][mode] = scores.base_refusals
                _save(run, saved, name)
        on = [Score(**saved["on"][str(i)]) for i in range(len(made))]
        repeated = min(range(len(made)), key=lambda i: (on[i].refusals, on[i].kl))
        if not saved["repeat"]:
            run.report(f"Scoring {made[repeated].name} again with another seed")
            again = search.score_locked(
                [made[repeated]], True, seed=run.settings.sampling.seed + 1, base=False, beside_search=False
            ).edits[0]
            saved["repeat"] = {str(repeated): asdict(again)}
            _save(run, saved, name)
    off = [Score(**saved["off"][str(i)]) for i in range(len(made))]
    index, repeat = next(iter(saved["repeat"].items()))
    return Check(made, trials, factors, off, on, saved["base"], int(index), Score(**repeat))


def _inputs(
    search: Search, ids: dict[str, list[list[int]]], harmless: list[list[int]], trials: list[int]
) -> dict[str, Any]:
    """What the check's scores depend on: the search's inputs, but scored both ways whichever way the
    search scores, and the trials the edits come from."""
    settings = search.run.settings
    found = search.inputs(ids, harmless)
    for key in ("trials think", "answer length with thinking off", "Optuna"):
        found.pop(key, None)
    return found | {
        "answer length with thinking off": settings.scoring.closed_answer_tokens,
        "edits": [trials, list(FACTORS)],
    }


def _load(run: Run, inputs: dict[str, Any], name: str) -> dict[str, Any]:
    path = run.path("proxy", name)
    empty = {"inputs": json.loads(json.dumps(inputs)), "base": {}, "off": {}, "on": {}, "repeat": {}}
    if not path.exists():
        return empty
    saved = json.loads(path.read_text())
    if saved["inputs"] == empty["inputs"]:
        return saved
    run.report("Scoring the check afresh: what its scores depend on has changed")
    return empty


def _save(run: Run, saved: dict[str, Any], name: str) -> None:
    path = run.path("proxy", name)
    path.parent.mkdir(exist_ok=True)
    write_atomic(path, json.dumps(saved, sort_keys=True))
