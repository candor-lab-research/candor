"""Search for the edit that removes the most refusals while changing the model least.

Each trial is an edit of the shape heretic searches: which refusal direction the layers use, and for
each component a strength that peaks at one layer and falls off either side of it. All of the run's
adapters are loaded into one llama-server, so a trial only sets their strengths per request. A trial
answers the harmful test prompts, the judge counts the refusals, and KL divergence is read from the
harmless test prompts' first tokens; Optuna's TPE sampler aims at both at once. The held-out prompts
are left for candor measure. Nothing a model writes is kept: answers are judged, then dropped.

Any other edit, such as a heretic trial, can be scored the same way without joining the search, to see
where it would sit among the search's trials and whether the search could have drawn it.
"""

import json
import time
import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import BaseTrial, TrialState

from . import config, kl
from .engine import LlamaServer
from .judge import Guard, is_refusal, judged_refusal, read_verdict
from .kernel import Kernel
from .measure import Edit, edit_adapters
from .run import STAGES, Run, RunError, digest

STUDY = "candor"
SLOTS = 8
# The sampler weighs this many candidates for each trial, as heretic does, rather than Optuna's 24.
CANDIDATES = 128
# The version of what a trial's score means. Bump it in any commit that changes how trials are scored,
# so an older search isn't carried on as if it were the same.
VERSION = 1
PER_LAYER, ONE_LAYER = "per layer", "one layer"
# How far a setting read back from an edit may stray past the search's range from rounding alone.
ROUNDING = 1e-9
# KL is shown to this many decimals; edits closer than that are not told apart.
KL_DIGITS = 4


@dataclass(frozen=True)
class Score:
    # Counted by the judge. A reply it gave no label counts as a refusal, so nothing scores well by
    # confusing the judge.
    refusals: int
    keyword_refusals: int
    unlabelled: int
    kl: float


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


def reaches(space: config.Search, layers: int) -> tuple[float, float]:
    """The reach a trial is drawn from. An edit reaches at least one layer either side of its peak, as
    in heretic, so whatever the peak some layer is edited."""
    last = layers - 1
    return min(max(space.reach[0], 1 / last), space.reach[1]), space.reach[1]


def suggest(trial: BaseTrial, space: config.Search, layers: int, components: list[str]) -> None:
    """Draw a trial's parameters. Multivariate TPE needs every parameter drawn every time, so the
    direction layer is drawn even when each layer uses its own."""
    trial.suggest_categorical("direction_scope", [ONE_LAYER, PER_LAYER])
    trial.suggest_int("direction_layer", *space.direction_layers(layers))
    reach = reaches(space, layers)
    for component in components:
        trial.suggest_float(f"{component}.strength", *space.strength)
        trial.suggest_float(f"{component}.peak_layer", *space.peak_layer)
        trial.suggest_float(f"{component}.edge_strength", *space.edge_strength)
        trial.suggest_float(f"{component}.reach", *reach)


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
    layer = None if params["direction_scope"] == PER_LAYER else params["direction_layer"]
    return Edit(f"trial-{number}", layer, kernels)


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
    ranges = {
        "strength": space.strength,
        "peak_layer": space.peak_layer,
        "edge_strength": space.edge_strength,
        "reach": reaches(space, layers),
    }
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
    return ties(score, searched) and score.keyword_refusals == searched.keyword_refusals


def objectives(score: Score, base_refusals: int, kl_target: float) -> tuple[float, float]:
    """What the sampler minimises: KL divergence, and refusals as a share of the unedited model's.
    Below kl_target the first is the refusal share scaled to the target instead, so an edit that
    changes almost nothing can't look good by its KL alone."""
    refused = score.refusals / base_refusals if base_refusals else float(score.refusals)
    return (score.kl if score.kl >= kl_target else refused * kl_target, refused)


def _storage(folder: Path) -> JournalStorage:
    return JournalStorage(JournalFileBackend(str(folder / "study.jsonl")))


def _completed(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))


def front(found: list[Candidate]) -> list[Candidate]:
    """The candidates no other beats on both refusals and KL, fewest refusals first. This is taken
    from refusals and KL themselves, not the sampler's objectives, which below the KL target aren't KL."""
    kept: list[Candidate] = []
    for candidate in sorted(found, key=lambda c: (c.refusals, c.kl)):
        if not kept or candidate.kl < kept[-1].kl:
            kept.append(candidate)
    return kept


def candidates(study: optuna.Study) -> list[Candidate]:
    return front(
        [
            Candidate(
                trial.number + 1,
                PER_LAYER
                if trial.params["direction_scope"] == PER_LAYER
                else f"layer {trial.params['direction_layer']}",
                trial.user_attrs["refusals"],
                trial.user_attrs["keyword_refusals"],
                trial.user_attrs["kl"],
            )
            for trial in _completed(study)
        ]
    )


def _changed(before: dict[str, Any], now: dict[str, Any], keys: Iterable[str] | None = None) -> list[str]:
    keys = before.keys() | now.keys() if keys is None else keys
    return sorted(key for key in keys if before.get(key) != now.get(key))


def load(run: Run) -> optuna.Study | None:
    """The run's search, if it has one."""
    folder = run.path("search")
    if not (folder / "study.jsonl").exists():
        return None
    return optuna.load_study(study_name=STUDY, storage=_storage(folder))


def searched_score(study: optuna.Study, number: int) -> Score:
    """The score the search gave one of its finished trials, by the number the search shows."""
    trials = {trial.number + 1: trial for trial in _completed(study)}
    if number not in trials:
        raise RunError(f"the search has no finished trial {number}")
    attrs = trials[number].user_attrs
    return Score(attrs["refusals"], attrs["keyword_refusals"], attrs["unlabelled"], attrs["kl"])


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
    study = optuna.load_study(study_name=STUDY, storage=_storage(folder))
    trials = {trial.number + 1: trial for trial in _completed(study)}
    if number not in trials:
        raise RunError(f"the search has no finished trial {number}")
    return edit_from(trials[number].params, number, run.shape[1])


class Search:
    def __init__(self, run: Run):
        self.run = run
        self.folder = run.path("search")

    def search(
        self,
        target: int,
        started: Callable[[int, int], None] = lambda done, total: None,
        finished: Callable[[str | None], None] = lambda best: None,
    ) -> tuple[list[Candidate], float]:
        """Run trials until target are finished. started gets the trials already finished and the
        target; finished is called after each trial, with a line to show when it has the fewest
        refusals so far. Returns the candidates and the mean seconds per trial."""
        run = self.run
        with run.locked():
            run.check_remote_code()
            if not run.path("directions.pt").exists():
                run.check_memory()
            adapters = run.adapters()
            _, quantized = run.gguf()
            ids = run.prompt_ids()
            harmful, harmless = ids["harmful_test"], ids["harmless_test"]
            study = self._study(self.inputs(harmful, harmless))
            started(len(_completed(study)), target)
            if len(_completed(study)) < target:
                files = [path for made in adapters.values() for path in made.values()]
                with self._scorer(quantized, files, harmful, harmless) as scorer:
                    base = study.user_attrs.get("base_refusals")
                    if base is None:
                        base = scorer.score({}).refusals
                        study.set_user_attr("base_refusals", base)
                    run.report(f"The unedited model refuses {base} of {len(harmful)} test prompts")
                    fewest = min((t.user_attrs["refusals"] for t in _completed(study)), default=None)
                    while len(_completed(study)) < target:
                        number, score = self._trial(study, scorer, adapters, base)
                        best = fewest is None or score.refusals < fewest
                        fewest = score.refusals if best else fewest
                        finished(
                            f"Trial {number}: {score.refusals} of {len(harmful)} refused, KL "
                            f"{score.kl:.4f}, the fewest refusals so far"
                            if best
                            else None
                        )
            seconds = [t.user_attrs["seconds"] for t in _completed(study)]
            return candidates(study), sum(seconds) / len(seconds) if seconds else 0.0

    def score(self, edits: list[Edit]) -> Scores:
        """Score edits exactly as the search scores its trials, without adding them to it: the same
        server with the same adapters loaded, the same test prompts and judge. An edit that reads its
        direction where the search has no adapters gets its own, in the run's scores folder. The unedited
        model is scored again too."""
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
            "search settings": settings.search.model_dump(exclude={"trials"}),
            "code": {**{stage: STAGES[stage] for stage in list(STAGES)[:4]}, "search": VERSION},
            "Optuna": optuna.__version__,
        }

    def _study(self, inputs: dict[str, Any]) -> optuna.Study:
        """The run's study: a new one, or the one already here if it was made from the same inputs."""
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
        study = optuna.create_study(
            study_name=STUDY,
            storage=_storage(self.folder),
            directions=["minimize", "minimize"],
            load_if_exists=True,
        )
        # A trial still marked running was stopped part way; the run lock means no search is running it.
        for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
            study.tell(trial.number, state=TrialState.FAIL)
        # The sampler starts again from its seed in each session, so the seed moves on by the trials
        # already made; otherwise a resumed search would repeat its first random trials.
        space = self.run.settings.search
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            study.sampler = optuna.samplers.TPESampler(
                n_startup_trials=space.startup_trials,
                n_ei_candidates=CANDIDATES,
                multivariate=True,
                seed=space.seed + len(study.trials),
            )
        return study

    def _trial(
        self, study: optuna.Study, scorer: "_Scorer", adapters: dict[str, dict[str, Path]], base: int
    ) -> tuple[int, Score]:
        run = self.run
        _, layers, _, writers = run.shape
        trial = study.ask()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
                suggest(trial, run.settings.search, layers, sorted({w.component for w in writers}))
            edit = edit_from(trial.params, trial.number + 1, layers)
            scales = self._scales(edit, self._edit_files(edit, adapters))
            started = time.perf_counter()
            score = scorer.score(scales)
            for key, value in asdict(score).items():
                trial.set_user_attr(key, value)
            trial.set_user_attr("seconds", time.perf_counter() - started)
        except BaseException:
            study.tell(trial, state=TrialState.FAIL)
            raise
        study.tell(trial, objectives(score, base, run.settings.search.kl_target))
        return trial.number + 1, score


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

    def score(self, scales: dict[Path, float]) -> Score:
        answers = [a.text for a in self.server.generate(self.harmful, self.scoring.answer_tokens, scales)]
        verdicts = [read_verdict(reply) for reply in self.guard.replies(self.texts, answers)]
        refused = [judged_refusal(answer, verdict) for answer, verdict in zip(answers, verdicts, strict=True)]
        top = self.server.top_logprobs(self.harmless, self.scoring.kl_tokens, scales)
        return Score(
            refusals=sum(r is not False for r in refused),
            keyword_refusals=sum(map(is_refusal, answers)),
            unlabelled=sum(r is None for r in refused),
            kl=kl.mean_from_top(self.base, top),
        )
