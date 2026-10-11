"""Search for the edit that removes the most refusals while changing the model least.

Each trial is an edit of the shape heretic searches: which refusal direction the layers use, and for
each component a strength that peaks at one layer and falls off either side of it. All of the run's
adapters are loaded into one llama-server, so a trial only sets their strengths per request. A trial
answers the harmful test prompts, the judge counts the refusals, and KL divergence is read from the
harmless test prompts' first tokens; Optuna's TPE sampler aims at both at once. The held-out prompts
are left for candor measure. Nothing a model writes is kept: answers are judged, then dropped.
"""

import json
import time
import warnings
from collections.abc import Callable, Iterable
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
from .measure import Edit
from .run import STAGES, Run, RunError, digest

STUDY = "candor"
SLOTS = 8
# The sampler weighs this many candidates for each trial, as heretic does, rather than Optuna's 24.
CANDIDATES = 128
# The version of what a trial's score means. Bump it in any commit that changes how trials are scored,
# so an older search isn't carried on as if it were the same.
VERSION = 1
PER_LAYER, ONE_LAYER = "per layer", "one layer"


@dataclass(frozen=True)
class Score:
    # Counted by the judge. A reply it gave no label counts as a refusal, so nothing scores well by
    # confusing the judge.
    refusals: int
    keyword_refusals: int
    unlabelled: int
    kl: float


@dataclass(frozen=True)
class Candidate:
    trial: int
    direction: str
    refusals: int
    keyword_refusals: int
    kl: float


def suggest(trial: BaseTrial, space: config.Search, layers: int, components: list[str]) -> None:
    """Draw a trial's parameters. Multivariate TPE needs every parameter drawn every time, so the
    direction layer is drawn even when each layer uses its own."""
    last = layers - 1
    trial.suggest_categorical("direction_scope", [ONE_LAYER, PER_LAYER])
    low, high = space.direction_layer
    trial.suggest_int("direction_layer", round(low * last), round(high * last))
    # An edit reaches at least one layer either side of its peak, as in heretic, so whatever the peak
    # some layer is edited.
    reach = (min(max(space.reach[0], 1 / last), space.reach[1]), space.reach[1])
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
        run, settings = self.run, self.run.settings
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
                texts = run.prompt_texts()[0]["harmful_test"]
                files = [path for made in adapters.values() for path in made.values()]
                room = max(len(p) for p in harmful + harmless) + settings.scoring.answer_tokens + 8
                guard = Guard(settings.measure.judge, settings.measure.judge_revision)
                try:
                    with run.server(quantized, files, context_per_slot=room, parallel=SLOTS) as server:
                        run.check_prompts(server, "harmful_test", texts, harmful)
                        scorer = _Scorer(server, guard, texts, harmful, harmless, settings.scoring)
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
                finally:
                    guard.close()
            seconds = [t.user_attrs["seconds"] for t in _completed(study)]
            return candidates(study), sum(seconds) / len(seconds) if seconds else 0.0

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
            files = adapters["per-layer" if edit.direction_layer is None else f"layer-{edit.direction_layer}"]
            strengths = edit.strengths(writers, run.target_names)
            started = time.perf_counter()
            score = scorer.score({files[name]: s for name, s in strengths.items() if s})
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
