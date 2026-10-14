"""Whether a thinking model's search may score its trials with thinking off.

A trial scored as the model is used thinks to its budget on every test prompt, many times the cost of
answering after an empty thinking block. The cheap score only helps if it ranks edits as thinking does,
and no published work shows that it does: the gap between the two changes size and even sign by model
and by edit. So candor proxy scores a spread of edits both ways on the search's own test prompts and
compares the order. Its scores are kept in the run's proxy folder, so a stopped check carries on.
"""

import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from .measure import Edit
from .run import Run, RunError
from .search import POSITIONS, SCOPES, Score, Search, edit_from, spread

# Shapes of edit tried for each kind and position, and the strengths each is tried at as multiples of
# its own: thinking may need a stronger edit than the cheap score shows, or a weaker one.
SHAPES = 2
FACTORS = (0.7, 1.0, 1.6)
# The cheap score is trusted when it ranks the edits' refusals as thinking does at least this well
# (Spearman's rank correlation).
AGREEMENT = 0.7
# Thinking needs a stronger edit than the cheap score shows when, among the edits the cheap score says
# nearly work (refusing at most this share of the prompts), thinking refuses more than this by the
# median: a search scored with thinking off would settle on edits too weak when thinking.
NEARLY_WORKS, OFFSET = 0.25, 1
# The shapes are spread from the search's seed plus this, beyond any seed it draws its starts from.
SEED_OFFSET = 10_000
MODES = {"off": False, "on": True}


@dataclass(frozen=True)
class Check:
    edits: list[Edit]
    # Each edit's strength as a multiple of its shape's.
    factors: list[float]
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


def edits(run: Run) -> tuple[list[Edit], list[float]]:
    """The edits checked: for each kind and position, SHAPES shapes spread evenly over the search's
    ranges, each at every strength in FACTORS, held within search.widest_strength."""
    space = run.settings.search
    _, layers, _, writers = run.shape
    components = sorted({writer.component for writer in writers})
    low, high = space.widest_strength
    made, factors = [], []
    for scope in SCOPES:
        # Drawn with a seed of their own, so they aren't the search's first trials.
        count = SHAPES * len(POSITIONS)
        shapes = spread(scope, space, layers, components, count, space.seed + SEED_OFFSET, POSITIONS)
        for params in shapes:
            for factor in FACTORS:
                scaled = {
                    name: min(max(value * factor, low), high) if name.endswith(".strength") else value
                    for name, value in params.items()
                }
                made.append(replace(edit_from(scaled, 0, layers), name=f"edit-{len(made) + 1}"))
                factors.append(factor)
    return made, factors


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


def check(run: Run) -> Check:
    """Score the edits with thinking off, then thinking as the model is used, then the edit refusing
    fewest thinking again with another seed, to see how far sampling alone moves a score."""
    if not run.settings.deployment.think_close:
        raise RunError("candor proxy is for a model that thinks; this one doesn't")
    search = Search(run)
    made, factors = edits(run)
    with run.locked():
        ids = run.prompt_ids()
        run.directions()
        inputs = _inputs(search, ids, run.kl_ids(ids, "harmless_test"))
        saved = _load(run, inputs)
        for mode, thinks in MODES.items():
            todo = [i for i in range(len(made)) if str(i) not in saved[mode]]
            if not todo and mode in saved["base"]:
                continue
            run.report(f"Scoring {len(todo)} edits with thinking {mode}")

            def keep(index: int, score: Score, mode: str = mode, todo: list[int] = todo) -> None:
                saved[mode][str(todo[index])] = asdict(score)
                _save(run, saved)

            scores = search.score_locked(
                [made[i] for i in todo],
                thinks,
                scored=keep,
                base=mode not in saved["base"],
                beside_search=False,
            )
            if scores.base_refusals is not None:
                saved["base"][mode] = scores.base_refusals
                _save(run, saved)
        on = [Score(**saved["on"][str(i)]) for i in range(len(made))]
        repeated = min(range(len(made)), key=lambda i: (on[i].refusals, on[i].kl))
        if not saved["repeat"]:
            run.report(f"Scoring {made[repeated].name} again with another seed")
            again = search.score_locked(
                [made[repeated]], True, seed=run.settings.sampling.seed + 1, base=False, beside_search=False
            ).edits[0]
            saved["repeat"] = {str(repeated): asdict(again)}
            _save(run, saved)
    off = [Score(**saved["off"][str(i)]) for i in range(len(made))]
    index, repeat = next(iter(saved["repeat"].items()))
    return Check(made, factors, off, on, saved["base"], int(index), Score(**repeat))


def _inputs(search: Search, ids: dict[str, list[list[int]]], harmless: list[list[int]]) -> dict[str, Any]:
    """What the check's scores depend on: the search's inputs, but scored both ways whichever way the
    search scores, with the edits drawn as they are now."""
    settings = search.run.settings
    found = search.inputs(ids, harmless)
    for key in ("trials think", "answer length with thinking off", "Optuna"):
        found.pop(key, None)
    return found | {
        "answer length with thinking off": settings.scoring.closed_answer_tokens,
        "edits": [SHAPES, list(FACTORS), SEED_OFFSET],
    }


def _load(run: Run, inputs: dict[str, Any]) -> dict[str, Any]:
    path = run.path("proxy", "scores.json")
    empty = {"inputs": json.loads(json.dumps(inputs)), "base": {}, "off": {}, "on": {}, "repeat": {}}
    if not path.exists():
        return empty
    saved = json.loads(path.read_text())
    if saved["inputs"] == empty["inputs"]:
        return saved
    run.report("Scoring the check afresh: what its scores depend on has changed")
    return empty


def _save(run: Run, saved: dict[str, Any]) -> None:
    path = run.path("proxy", "scores.json")
    path.parent.mkdir(exist_ok=True)
    partial = path.with_name("scores.json.building")
    partial.write_text(json.dumps(saved, sort_keys=True))
    partial.replace(path)
