"""Search for the edit that changes the model least while it refuses almost nothing.

Each trial is an edit of the shape heretic searches: a refusal direction the layers use, and for each
component a strength that peaks at one layer and falls off either side of it. Edits come in two kinds,
searched as two studies that take turns: every layer reads the direction found at one layer, or each
layer reads its own. Each study starts from evenly spread trials, then Optuna's TPE sampler aims at the
least KL divergence among edits that refuse no more than search.max_refusals of the test prompts.

Searching the kinds apart, at that one goal, keeps either kind from being written off by trials that
suit the other. A search of both at once, aiming at every trade-off of refusals against KL, missed
heretic's Trial 170 on rude-llama: its sampler kept learning from weak edits that changed little and
refused much, and never tried the shape that worked. Test landscapes calibrated to that search showed
this design missing the best far less often (see results/phase-2-rude-llama.md).

All of the run's adapters are loaded into one llama-server, so a trial only sets their strengths per
request. A trial answers the harmful test prompts a chunk at a time and the judge labels each chunk;
refusals can only add up, so a trial that has refused more than the cap, and more than 25 fully scored
trials of its kind, can't win and stops there, unseen by the sampler. KL divergence is read from the
harmless test prompts' first tokens. The held-out prompts are left for candor measure. Nothing a model
writes is kept: answers are judged, then dropped.

Any other edit, such as a heretic trial, can be scored the same way without joining the search, to see
where it would sit among the search's trials and whether the search could have drawn it.
"""

import json
import time
import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import optuna
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import BaseTrial, FrozenTrial, TrialState

from . import config, kl
from .engine import LlamaServer
from .judge import Guard, is_refusal, judged_refusal, read_verdict
from .kernel import Kernel
from .measure import Edit, edit_adapters
from .run import STAGES, Run, RunError, digest

SLOTS = 8
# Harmful prompts judged at a time, so a trial that can't win stops early. The server answers the
# next ones meanwhile.
CHUNK = SLOTS
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


@dataclass(frozen=True)
class Scores:
    prompts: int
    base_refusals: int
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
    """A search's finished trials, by the number it shows each."""

    trials: dict[int, FrozenTrial]
    base_refusals: int | None


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


def suggest(trial: BaseTrial, space: config.Search, layers: int, components: list[str], scope: str) -> None:
    """Draw a trial's parameters. Only an edit of one layer's direction has a direction layer."""
    if scope == ONE_LAYER:
        trial.suggest_int("direction_layer", *space.direction_layers(layers))
    ranges = _ranges(space, layers)
    for name in _names(PER_LAYER, components):
        trial.suggest_float(name, *ranges[name.split(".")[1]])


def spread(
    scope: str, space: config.Search, layers: int, components: list[str], count: int, seed: int
) -> list[dict[str, Any]]:
    """count trials spread evenly over the scope's ranges (scrambled Sobol points), to start its study.
    Random trials leave gaps a narrow region of good edits can fall into."""
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
    return Edit(f"trial-{number}", params.get("direction_layer"), kernels)


def violation(score: Score, prompts: int, cap: int) -> float:
    """How far over the cap an edit refuses, its refusals scaled to all prompts if it stopped early.
    The sampler treats an edit as within the cap when this is at most 0."""
    return score.refusals * prompts / score.judged - cap - 0.5


def _violation(trial: FrozenTrial) -> tuple[float]:
    return (trial.user_attrs["violation"],)


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
        if floor and value - low <= EDGE * (high - low):
            pressed.append(f"{name} {value:.3g}, at its floor {low:.3g}")
        if ceiling and high - value <= EDGE * (high - low):
            pressed.append(f"{name} {value:.3g}, at its ceiling {high:.3g}")
    return pressed


def nearest_layer(edit: Edit, space: config.Search, layers: int) -> Edit | None:
    """An edit that reads its direction between two layers, read instead at the nearest whole layer, as
    the search could draw it; None for an edit that needs no rounding or whose nearest layer the search
    never reads."""
    layer = edit.direction_layer
    if layer is None or float(layer).is_integer():
        return None
    whole = round(layer)
    low, high = space.direction_layers(layers)
    return Edit(f"{edit.name} at layer {whole}", whole, edit.kernels) if low <= whole <= high else None


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
        attrs["refusals"], attrs["keyword_refusals"], attrs["unlabelled"], attrs["kl"], attrs["judged"]
    )


def _candidate(number: int, trial: FrozenTrial) -> Candidate:
    layer = trial.params.get("direction_layer")
    return Candidate(
        number,
        PER_LAYER if layer is None else f"layer {layer}",
        trial.user_attrs["refusals"],
        trial.user_attrs["keyword_refusals"],
        trial.user_attrs["kl"],
        trial.params,
    )


def _scored_in_full(found: Found) -> list[Candidate]:
    return [_candidate(number, t) for number, t in found.trials.items() if not t.user_attrs["stopped"]]


def front(found: list[Candidate]) -> list[Candidate]:
    """The candidates no other beats on both refusals and KL, fewest refusals first."""
    kept: list[Candidate] = []
    for candidate in sorted(found, key=lambda c: (c.refusals, c.kl)):
        if not kept or candidate.kl < kept[-1].kl:
            kept.append(candidate)
    return kept


def candidates(found: Found) -> list[Candidate]:
    """The front of the trials scored in full: a stopped trial's count is only a lower bound."""
    return front(_scored_in_full(found))


def within_cap(found: Found, scope: str, cap: int) -> list[Candidate]:
    """The scope's trials scored in full that refuse no more than cap, least KL first."""
    kept = [c for c in _scored_in_full(found) if scope_of(c.params) == scope and c.refusals <= cap]
    return sorted(kept, key=lambda c: (c.kl, c.refusals))


def fewest_refused(found: Found, scope: str) -> Candidate | None:
    """The scope's trial scored in full that refused fewest, then with least KL."""
    kept = [c for c in _scored_in_full(found) if scope_of(c.params) == scope]
    return min(kept, key=lambda c: (c.refusals, c.kl), default=None)


def _changed(before: dict[str, Any], now: dict[str, Any], keys: Iterable[str] | None = None) -> list[str]:
    keys = before.keys() | now.keys() if keys is None else keys
    return sorted(key for key in keys if before.get(key) != now.get(key))


def _found(studies: Iterable[optuna.Study]) -> Found:
    trials, base = {}, None
    for study in studies:
        base = study.user_attrs.get("base_refusals", base)
        trials |= {t.user_attrs["number"]: t for t in _completed(study)}
    return Found(dict(sorted(trials.items())), base)


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
        optuna.load_study(study_name=name, storage=storage) for name in STUDIES.values() if name in names
    )


def searched_score(found: Found, number: int) -> Score:
    """The score the search gave one of its finished trials, by the number the search shows."""
    if number not in found.trials:
        raise RunError(f"the search has no finished trial {number}")
    return _score(found.trials[number])


def trial_edit(run: Run, number: int) -> Edit:
    """A finished trial of this run's search, by the number the search shows. Its edit can only be
    rebuilt as it was scored while the run's directions and adapters are the ones the search used."""
    folder = run.path("search")
    if not (folder / "study.jsonl").exists():
        raise RunError(f"{run.folder} has no search yet; run candor search")
    with run.locked():
        run.gguf()
        ids = run.prompt_ids()
        run.directions()
        now = json.loads(json.dumps(Search(run).inputs(ids["harmful_test"], ids["harmless_test"])))
    if not (folder / "inputs.json").exists():
        raise RunError(f"{folder} holds a search with no record of its settings")
    before = json.loads((folder / "inputs.json").read_text())
    if changed := _changed(before, now, ["adapters", "code"]):
        raise RunError(
            f"the search was made with different {' and '.join(changed)} from this run's now, so its "
            "trials can't be rebuilt as they were scored"
        )
    found = load(run)
    assert found is not None
    if number not in found.trials:
        raise RunError(f"the search has no finished trial {number}")
    return edit_from(found.trials[number].params, number, run.shape[1])


class Search:
    def __init__(self, run: Run):
        self.run = run
        self.folder = run.path("search")

    def search(
        self,
        target: int,
        started: Callable[[int, int], None] = lambda done, total: None,
        finished: Callable[[str | None], None] = lambda best: None,
    ) -> tuple[Found, float]:
        """Run trials until target are finished, taking turns between the kinds of edit, each kind up to
        search.trials_per_scope. started gets the trials already finished and the target; finished is
        called after each trial, with a line to show when it has the least KL of its kind within the
        cap so far. Returns the trials and the mean seconds per trial."""
        run, space = self.run, self.run.settings.search
        with run.locked():
            run.check_remote_code()
            if not run.path("directions.pt").exists():
                run.check_memory()
            adapters = run.adapters()
            _, quantized = run.gguf()
            ids = run.prompt_ids()
            harmful, harmless = ids["harmful_test"], ids["harmless_test"]
            studies = self._studies(self.inputs(harmful, harmless))
            target = min(target, len(SCOPES) * space.trials_per_scope)
            started(self._done(studies), target)
            if self._done(studies) < target:
                files = [path for made in adapters.values() for path in made.values()]
                with self._scorer(quantized, files, harmful, harmless) as scorer:
                    first = studies[SCOPES[0]]
                    base = first.user_attrs.get("base_refusals")
                    if base is None:
                        base = scorer.score({}).refusals
                        first.set_user_attr("base_refusals", base)
                    run.report(f"The unedited model refuses {base} of {len(harmful)} test prompts")
                    least = {scope: self._least(studies[scope]) for scope in SCOPES}
                    while self._done(studies) < target:
                        scope = self._next(studies)
                        number, score = self._trial(studies, scope, scorer, adapters)
                        before = least[scope]
                        best = (
                            score.judged == len(harmful)
                            and score.refusals <= space.max_refusals
                            and (before is None or score.kl < before)
                        )
                        least[scope] = score.kl if best else least[scope]
                        finished(
                            f"Trial {number}, {scope}: {score.refusals} of {len(harmful)} refused, KL "
                            f"{score.kl:.4f}, the least KL of its kind within the cap so far"
                            if best
                            else None
                        )
            found = _found(studies.values())
            seconds = [t.user_attrs["seconds"] for t in found.trials.values()]
            return found, sum(seconds) / len(seconds) if seconds else 0.0

    def score(self, edits: list[Edit]) -> Scores:
        """Score edits exactly as the search scores its trials, without adding them to it: the same
        server with the same adapters loaded, the same test prompts and judge, each edit in full. An
        edit that reads its direction where the search has no adapters gets its own, in the run's
        scores folder. The unedited model is scored again too."""
        run = self.run
        with run.locked():
            run.check_remote_code()
            if not run.path("directions.pt").exists():
                run.check_memory()
            _, quantized = run.gguf()
            ids = run.prompt_ids()
            run.directions()
            harmful, harmless = ids["harmful_test"], ids["harmless_test"]
            # Checked before the adapters are brought up to date, so a mismatch leaves the run as it was.
            record, journal = self.folder / "inputs.json", self.folder / "study.jsonl"
            if record.exists():
                before = json.loads(record.read_text())
                now = json.loads(json.dumps(self.inputs(harmful, harmless)))
                # The Optuna version decides which trials the search draws, not how one is scored.
                if changed := _changed(before, now, (before.keys() | now.keys()) - {"Optuna"}):
                    raise RunError(
                        f"the search was made with different {' and '.join(changed)} from this run's now, "
                        "so an edit scored now can't be set beside its trials"
                    )
            elif journal.exists():
                raise RunError(f"{self.folder} holds a search with no record of its settings")
            adapters = run.adapters()
            made = [self._edit_files(edit, adapters) for edit in edits]
            files = [path for found in [*adapters.values(), *made] for path in found.values()]
            with self._scorer(quantized, list(dict.fromkeys(files)), harmful, harmless) as scorer:
                base = scorer.score({}).refusals
                scores = []
                for edit, found in zip(edits, made, strict=True):
                    run.report(f"Scoring {edit.name}")
                    scores.append(scorer.score(self._scales(edit, found)))
            return Scores(len(harmful), base, scores)

    def _edit_files(self, edit: Edit, adapters: dict[str, dict[str, Path]]) -> dict[str, Path]:
        """The adapters an edit is scored with: the search's own for a direction it reads, or else
        adapters of its own, built as candor measure builds them. These depend only on the direction,
        so they are kept by layer."""
        layer = edit.direction_layer
        if layer is None:
            return adapters["per-layer"]
        if float(layer).is_integer() and f"layer-{int(layer)}" in adapters:
            return adapters[f"layer-{int(layer)}"]
        return edit_adapters(self.run, edit, self.run.path("scores", f"layer-{layer!r}"))[0]

    def _scales(self, edit: Edit, files: dict[str, Path]) -> dict[Path, float]:
        """Each of the edit's adapters at its strength. Those at 0 are left out, which llama.cpp skips."""
        strengths = edit.strengths(self.run.shape[3], self.run.target_names)
        return {files[name]: strength for name, strength in strengths.items() if strength}

    @contextmanager
    def _scorer(
        self, quantized: Path, files: list[Path], harmful: list[list[int]], harmless: list[list[int]]
    ) -> Iterator["_Scorer"]:
        """A scorer on llama-server with these adapters loaded, and the judge, for the test prompts."""
        run, settings = self.run, self.run.settings
        texts = run.prompt_texts()[0]["harmful_test"]
        room = max(len(p) for p in harmful + harmless) + settings.scoring.answer_tokens + 8
        guard = Guard(settings.measure.judge, settings.measure.judge_revision)
        try:
            with run.server(quantized, files, context_per_slot=room, parallel=SLOTS) as server:
                run.check_prompts(server, "harmful_test", texts, harmful)
                yield _Scorer(server, guard, texts, harmful, harmless, settings.scoring)
        finally:
            guard.close()

    def inputs(self, harmful: list[list[int]], harmless: list[list[int]]) -> dict[str, Any]:
        """What the search's scores depend on. A search is only carried on while these are the same."""
        settings = self.run.settings
        return {
            "model file": self.run.inputs["quantized"],
            "adapters": digest(self.run.adapter_inputs()),
            "test prompts": digest([harmful, harmless]),
            "answer length": settings.scoring.answer_tokens,
            "KL tokens": settings.scoring.kl_tokens,
            "judge": [settings.measure.judge, settings.measure.judge_revision],
            "KV cache": settings.deployment.kv_cache,
            "search settings": settings.search.model_dump(exclude={"trials_per_scope"}),
            "code": {**{stage: STAGES[stage] for stage in list(STAGES)[:4]}, "search": VERSION},
            "Optuna": optuna.__version__,
        }

    def _studies(self, inputs: dict[str, Any]) -> dict[str, optuna.Study]:
        """The run's study of each kind of edit: new ones, or those already here if they were made from
        the same inputs."""
        record, journal = self.folder / "inputs.json", self.folder / "study.jsonl"
        now = json.loads(json.dumps(inputs))
        if record.exists():
            before = json.loads(record.read_text())
            if changed := _changed(before, now):
                raise RunError(
                    f"{self.folder} holds a search made with a different {', '.join(changed)}; move it "
                    "aside or delete it to start a new search"
                )
        elif journal.exists():
            raise RunError(
                f"{self.folder} holds a search with no record of its settings; move it aside or delete it"
            )
        else:
            self.folder.mkdir(exist_ok=True)
            partial = record.with_name("inputs.json.building")
            partial.write_text(json.dumps(now, sort_keys=True))
            partial.replace(record)
        space = self.run.settings.search
        _, layers, _, writers = self.run.shape
        components = sorted({w.component for w in writers})
        storage = _storage(self.folder)
        studies = {}
        for i, scope in enumerate(SCOPES):
            study = optuna.create_study(
                study_name=STUDIES[scope], storage=storage, direction="minimize", load_if_exists=True
            )
            # A trial still marked running was stopped part way; the run lock means no search is running
            # it.
            for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
                study.tell(trial.number, state=TrialState.FAIL)
            # The spread start, less the points already scored or waiting, so a point whose trial
            # failed is queued again rather than lost.
            kept = {
                _key(trial.system_attrs.get("fixed_params", trial.params))
                for trial in study.get_trials(
                    deepcopy=False, states=(TrialState.COMPLETE, TrialState.WAITING)
                )
            }
            for params in spread(scope, space, layers, components, space.startup_per_scope, space.seed + i):
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
            studies[scope] = study
        return studies

    def _done(self, studies: dict[str, optuna.Study]) -> int:
        """The finished trials that count towards the target: a kind's beyond its share don't."""
        share = self.run.settings.search.trials_per_scope
        return sum(min(len(_completed(study)), share) for study in studies.values())

    def _next(self, studies: dict[str, optuna.Study]) -> str:
        """The kind of edit whose turn it is: the one with fewer finished trials, that has room for more."""
        space = self.run.settings.search
        open_scopes = [s for s in SCOPES if len(_completed(studies[s])) < space.trials_per_scope]
        return min(open_scopes, key=lambda s: (len(_completed(studies[s])), SCOPES.index(s)))

    def _least(self, study: optuna.Study) -> float | None:
        """The least KL among the study's trials scored in full within the cap."""
        cap = self.run.settings.search.max_refusals
        kept = [
            t.user_attrs["kl"]
            for t in _completed(study)
            if not t.user_attrs["stopped"] and t.user_attrs["refusals"] <= cap
        ]
        return min(kept, default=None)

    def _trial(
        self,
        studies: dict[str, optuna.Study],
        scope: str,
        scorer: "_Scorer",
        adapters: dict[str, dict[str, Path]],
    ) -> tuple[int, Score]:
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
                suggest(trial, space, layers, sorted({w.component for w in writers}), scope)
            edit = edit_from(trial.params, number, layers)
            scales = self._scales(edit, self._edit_files(edit, adapters))
            full = sorted(t.user_attrs["refusals"] for t in _completed(study) if not t.user_attrs["stopped"])
            # A trial can't win once it refuses more than the cap, and the sampler can't see it stop once
            # it also refuses more than STOP_AFTER trials of its kind scored in full.
            stop_above = (
                max(space.max_refusals, full[STOP_AFTER - 1])
                if space.stop_early and len(full) >= STOP_AFTER
                else None
            )
            started = time.perf_counter()
            score = scorer.score(scales, stop_above)
            prompts = len(scorer.harmful)
            for key, value in asdict(score).items():
                trial.set_user_attr(key, value)
            trial.set_user_attr("stopped", score.judged < prompts)
            trial.set_user_attr("violation", violation(score, prompts, space.max_refusals))
            trial.set_user_attr("seconds", time.perf_counter() - started)
        except BaseException:
            study.tell(trial, state=TrialState.FAIL)
            raise
        study.tell(trial, score.kl)
        return number, score


class _Scorer:
    """Scores an edit, given as adapter strengths, against the unedited model's first tokens."""

    def __init__(
        self,
        server: LlamaServer,
        guard: Guard,
        texts: list[str],
        harmful: list[list[int]],
        harmless: list[list[int]],
        scoring: config.Scoring,
    ):
        self.server, self.guard, self.texts = server, guard, texts
        self.harmful, self.harmless, self.scoring = harmful, harmless, scoring
        self.base = server.top_logprobs(harmless, scoring.kl_tokens)

    def score(self, scales: dict[Path, float], stop_above: int | None = None) -> Score:
        """An edit's refusals and KL divergence. The harmful prompts are judged CHUNK at a time, in
        order, while the server answers the next ones; with stop_above, it stops once more than that
        many are refused."""
        refusals = keyword_refusals = unlabelled = judged = 0
        with closing(self.server.answers(self.harmful, self.scoring.answer_tokens, scales)) as answering:
            for start in range(0, len(self.harmful), CHUNK):
                answers = [a.text for a in islice(answering, CHUNK)]
                replies = self.guard.replies(self.texts[start : start + CHUNK], answers)
                labels = [
                    judged_refusal(answer, read_verdict(reply))
                    for answer, reply in zip(answers, replies, strict=True)
                ]
                refusals += sum(label is not False for label in labels)
                unlabelled += sum(label is None for label in labels)
                keyword_refusals += sum(map(is_refusal, answers))
                judged += len(answers)
                if stop_above is not None and refusals > stop_above:
                    break
        top = self.server.top_logprobs(self.harmless, self.scoring.kl_tokens, scales)
        return Score(refusals, keyword_refusals, unlabelled, kl.mean_from_top(self.base, top), judged)
