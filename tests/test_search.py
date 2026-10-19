import dataclasses
import filecmp
import hashlib
import itertools
import json
import math
import re
import shutil
from pathlib import Path

import optuna
import pytest
import torch
from optuna.distributions import BaseDistribution, FloatDistribution, IntDistribution
from optuna.samplers._tpe.sampler import default_gamma
from optuna.study import StudyDirection
from optuna.trial import FixedTrial, TrialState
from typer.testing import CliRunner

from candor import cli, config, proxy, run, search
from candor.cli import app
from candor.engine import GREEDY, Answer, LlamaServer
from candor.judge import is_refusal
from candor.kernel import Kernel
from candor.measure import Edit, Measurement, edit_adapters
from candor.run import RunError
from candor.search import Score, Search
from candor.targets import Writer

from .helpers import flat, on_disk, recorded_syncs, small, write_journal
from .tiny_gguf import END_THINK
from .tiny_run import (
    ALWAYS_THINKING_TEMPLATE,
    ANSWERED,
    MESSAGE_WHEN_THINKING_TEMPLATE,
    PLANNED_DIRECTIONS,
    READ_DIRECTIONS,
    REFUSED,
    FakeGuard,
    fake_prompts,
    fixed_directions,
    thinking,
    thinks_through,
)

# How a seeded search drew, recorded and told the sampler its trials, as candor did before
# search.max_budget_hits existed.
SEEDED = Path(__file__).parent / "records_seeded_search.json"
CHANGES = [
    (lambda s, m: setattr(s.deployment, "quantization", "Q8_0"), "model file"),
    (lambda s, m: setattr(s.edit, "row_norm", "none"), "adapters"),
    (lambda s, m: setattr(s.prompts.harmful_test, "split", "test[:99]"), "test prompts"),
    (lambda s, m: setattr(s.scoring, "answer_tokens", 8), "answer length"),
    (lambda s, m: setattr(s.scoring, "kl_tokens", 5), "KL tokens"),
    (lambda s, m: setattr(s.measure, "judge_revision", "another"), "judge"),
    (lambda s, m: setattr(s.deployment, "kv_cache", "q8_0"), "KV cache"),
    (lambda s, m: setattr(s.sampling, "temperature", 0.7), "sampling"),
    (lambda s, m: setattr(s.search, "strength", (0.9, 1.4)), "search settings"),
    (lambda s, m: setattr(s.search, "widest_strength", (0.5, 2.0)), "search settings"),
    (lambda s, m: setattr(s.slots, "search", 3), "slots"),
    (lambda s, m: m.setattr(search, "VERSION", search.VERSION + 1), "candor code"),
    (lambda s, m: m.setattr(search, "EDITS", search.EDITS + 1), "candor code"),
    (lambda s, m: m.setattr(optuna, "__version__", "9.9.9"), "Optuna"),
]


def studies(run) -> dict[str, optuna.Study]:
    storage = search._storage(run.path("search"))
    return {
        scope: optuna.load_study(study_name=search.STUDIES[scope], storage=storage) for scope in search.SCOPES
    }


def states(run) -> dict[str, list[TrialState]]:
    return {scope: [trial.state for trial in study.get_trials()] for scope, study in studies(run).items()}


def finished(run) -> list[optuna.trial.FrozenTrial]:
    """Every finished trial, by the number the search shows."""
    found = search.load(run)
    assert found is not None
    return list(found.trials.values())


def recording_answers(monkeypatch) -> list[str]:
    """Every answer the server gives, so tests can check none of it is kept."""
    answers = []
    real = LlamaServer.answers

    def recording(self, *args, **kwargs):
        for answer in real(self, *args, **kwargs):
            answers.append(answer.text)
            yield answer

    monkeypatch.setattr(LlamaServer, "answers", recording)
    return answers


def test_a_search_runs_its_trials_and_keeps_only_numbers(tiny, monkeypatch):
    small(tiny)
    answers = recording_answers(monkeypatch)
    found, seconds = Search(tiny).search(3)
    assert states(tiny) == {
        search.ONE_LAYER: [TrialState.COMPLETE] * 2,
        search.PER_LAYER: [TrialState.COMPLETE],
    }
    assert studies(tiny)[search.ONE_LAYER].user_attrs["base_refusals"] == found.base_refusals == 1
    assert list(found.trials) == [1, 2, 3]
    assert [search.scope_of(t.params) for t in found.trials.values()] == [
        search.ONE_LAYER,
        search.PER_LAYER,
        search.ONE_LAYER,
    ]
    assert all(c.refusals == 1 and c.kl >= 0 for c in search.candidates(found))
    assert seconds > 0
    kept = "".join(p.read_text() for p in tiny.path("search").iterdir() if p.is_file())
    assert len(answers) == 4 * 3
    assert all(len(a.strip()) < 4 or a.strip() not in kept for a in answers)


def test_each_kind_gets_its_share_of_trials_and_no_more(tiny):
    small(tiny, trials=1)
    found, _ = Search(tiny).search(5)
    assert sorted(search.scope_of(t.params) for t in found.trials.values()) == sorted(search.SCOPES)


def test_a_stopped_search_carries_on_without_scoring_anything_twice(tiny, monkeypatch):
    small(tiny)
    Search(tiny).search(2)
    answers = recording_answers(monkeypatch)
    Search(tiny).search(3)
    assert len(answers) == 3
    assert len(finished(tiny)) == 3


def test_a_trial_stopped_part_way_is_marked_failed_and_the_search_carries_on(tiny, monkeypatch):
    small(tiny)
    real = search._Scorer.score
    calls = []

    def stopping(self, scales, stop_above=None, forced_above=None, read_kl=True):
        calls.append(scales)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return real(self, scales, stop_above, forced_above, read_kl)

    monkeypatch.setattr(search._Scorer, "score", stopping)
    with pytest.raises(KeyboardInterrupt):
        Search(tiny).search(3)
    assert states(tiny) == {search.ONE_LAYER: [TrialState.COMPLETE], search.PER_LAYER: [TrialState.FAIL]}
    studies(tiny)[search.ONE_LAYER].ask()
    monkeypatch.setattr(search._Scorer, "score", real)
    Search(tiny).search(3)
    assert states(tiny) == {
        search.ONE_LAYER: [TrialState.COMPLETE, TrialState.FAIL, TrialState.COMPLETE],
        search.PER_LAYER: [TrialState.FAIL, TrialState.COMPLETE],
    }
    found = search.load(tiny)
    assert found is not None and list(found.trials) == [1, 3, 4]


@pytest.mark.parametrize(("change", "named"), CHANGES)
def test_a_search_isnt_carried_on_or_scored_beside_once_anything_its_scores_depend_on_changes(
    tiny, monkeypatch, change, named
):
    small(tiny)
    Search(tiny).search(1)
    # What the search was scored with is left as it was by a refusal.
    scored_with = model_files(tiny)
    change(tiny.settings, monkeypatch)
    edits = [Edit("each", None, KERNELS)]
    if named == "Optuna":
        # Scoring an edit asks Optuna for nothing.
        assert len(Search(tiny).score(edits).edits) == 1
    else:
        with pytest.raises(RunError, match=f"the search was made with different {named} from this run's now"):
            Search(tiny).score(edits)
        assert model_files(tiny) == scored_with
    with pytest.raises(RunError, match=f"holds a search made with a different {named}; move it aside"):
        Search(tiny).search(2)
    assert model_files(tiny) == scored_with


def test_candor_score_refused_for_another_model_file_leaves_the_run_as_it_was(tiny):
    folder = cli_run(tiny)
    runner = CliRunner(env={"COLUMNS": "200"})
    assert runner.invoke(app, ["search", str(folder)]).exit_code == 0
    scored_with = model_files(run.Run.open(folder))
    path = folder / config.FILE
    path.write_text(path.read_text().replace('quantization = "Q4_K_M"', 'quantization = "Q8_0"'))
    refused = runner.invoke(app, ["score", str(folder), "--trial", "1"])
    assert refused.exit_code == cli.FAILED_CHECK and "different model file" in refused.output
    assert model_files(run.Run.open(folder)) == scored_with


def test_a_search_refused_for_other_prompts_reads_no_directions_first(tiny, monkeypatch):
    # The run's own directions stage, its residuals made up.
    monkeypatch.setattr(run.Run, "directions", READ_DIRECTIONS)
    monkeypatch.setattr(run.Run, "planned_directions", PLANNED_DIRECTIONS)
    read = []

    def read_directions(self, stale):
        read.append(len(stale))
        for path, inputs in stale.values():
            partial = path.with_name(f"{path.name}.building")
            torch.save(fixed_directions(self), partial)
            self.commit(partial, path, inputs)

    monkeypatch.setattr(run.Run, "_read_directions", read_directions)
    small(tiny)
    Search(tiny).search(1)
    assert read == [1]
    kept = tiny.path("directions.pt").read_bytes()
    read.clear()
    tiny.settings.prompts.harmful.split = "train[:399]"
    with pytest.raises(RunError, match="different adapters"):
        Search(tiny).search(2)
    with pytest.raises(RunError, match="different adapters"):
        Search(tiny).score([Edit("each", None, KERNELS)])
    # Nor before a trial is rebuilt, to measure or export.
    with pytest.raises(RunError, match="can't be rebuilt as they were scored"):
        search.trial_edit(tiny, 1)
    assert read == [] and tiny.path("directions.pt").read_bytes() == kept


def model_files(run) -> dict[str, str]:
    """The run's GGUFs and adapters, by path, as digests of their bytes."""
    found = [p for folder in ("gguf", "adapters") for p in run.path(folder).rglob("*") if p.is_file()]
    return {str(p.relative_to(run.folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in found}


def test_each_trial_records_its_machine_and_a_search_moved_to_another_says_so(tiny, monkeypatch):
    small(tiny, trials=2)
    Search(tiny).search(2)
    assert all(t.user_attrs["machine"] == tiny.machine() for t in finished(tiny))
    said = []
    tiny.report = said.append
    a40 = {"platform": "linux-cuda", "gpus": ["NVIDIA A40"]}
    monkeypatch.setattr(run.Run, "machine", lambda self: a40)
    Search(tiny).search(3)
    [moved] = [line for line in said if "ran on" in line]
    assert "and this session's run on NVIDIA A40 (linux-cuda)" in moved
    assert max(finished(tiny), key=lambda t: t.user_attrs["number"]).user_attrs["machine"] == a40
    said.clear()
    Search(tiny).search(4)
    assert not [line for line in said if "ran on" in line]


def test_the_unedited_models_first_tokens_are_read_once_a_search_until_the_machine_changes(tiny, monkeypatch):
    small(tiny, trials=2)
    Search(tiny).search(1)
    read, real = [], LlamaServer.top_logprobs

    def top(self, prompts, k, scales=None, done=None):
        read.append(bool(scales))
        return real(self, prompts, k, scales, done)

    monkeypatch.setattr(LlamaServer, "top_logprobs", top)
    Search(tiny).search(2)
    assert read == [True]
    read.clear()
    here = tiny.machine()
    monkeypatch.setattr(run.Run, "machine", lambda self: {"platform": "linux-cuda", "gpus": ["NVIDIA A40"]})
    Search(tiny).search(3)
    assert read == [False, True]
    # Back on the first machine, its own are still kept.
    read.clear()
    monkeypatch.setattr(run.Run, "machine", lambda self: here)
    Search(tiny).search(4)
    assert read == [True]


def test_a_search_session_reads_only_the_harmful_test_prompts_from_their_dataset(tiny, monkeypatch):
    small(tiny)
    Search(tiny).search(1)
    loaded, real = [], run.load_prompts

    def load(spec):
        loaded.append(spec)
        return real(spec)

    monkeypatch.setattr(run, "load_prompts", load)
    Search(tiny).search(2)
    assert loaded == [tiny.settings.prompts.harmful_test]
    assert run.load_prompts(tiny.settings.prompts.harmful_test) == tiny.prompt_texts()[0]["harmful_test"]


def test_more_trials_carry_on_the_same_search(tiny):
    small(tiny, trials=1)
    Search(tiny).search(2)
    tiny.settings.search.trials_per_scope = 2
    Search(tiny).search(4)
    assert len(finished(tiny)) == 4


def test_a_searchs_record_and_progress_are_on_the_disk_once_written(tiny, monkeypatch):
    small(tiny, trials=1)
    tiny.settings.search.report_every = 1
    synced = recorded_syncs(monkeypatch)
    Search(tiny).search(1)
    assert {"inputs.json", "progress.json"} <= {name for name, _ in synced}


def test_a_search_without_its_record_is_refused(tiny):
    small(tiny)
    Search(tiny).search(1)
    tiny.path("search", "inputs.json").unlink()
    with pytest.raises(RunError, match="no record of its settings"):
        Search(tiny).search(2)


def test_a_search_made_by_an_older_candor_is_named(tiny):
    folder = tiny.path("search")
    folder.mkdir()
    optuna.create_study(study_name="candor", storage=search._storage(folder))
    with pytest.raises(RunError, match="holds a search made by an older candor; move it aside"):
        search.load(tiny)


def test_replies_the_judge_didnt_label_count_as_refusals(tiny, monkeypatch):
    small(tiny)
    monkeypatch.setattr(FakeGuard, "replies", lambda self, prompts, answers, **kwargs: ["?"] * len(answers))
    Search(tiny).search(1)
    [trial] = finished(tiny)
    assert trial.user_attrs["refusals"] == trial.user_attrs["unlabelled"] == 3


def test_trial_parameters_become_the_edit_heretic_would_make():
    params = {
        "direction_layer": 15,
        "attn.strength": 1.2,
        "attn.peak_layer": 0.75,
        "attn.edge_strength": 0.5,
        "attn.reach": 0.25,
        "mlp.strength": 1.0,
        "mlp.peak_layer": 1.0,
        "mlp.edge_strength": 0.0,
        "mlp.reach": 0.5,
    }
    trial = FixedTrial(params)
    search.suggest(trial, config.Search(), 29, ["attn", "mlp"], search.ONE_LAYER)
    assert trial.params == params
    edit = search.edit_from(params, 7, 29)
    assert edit.name == "trial-7" and edit.direction_layer == 15
    assert edit.kernels == {"attn": Kernel(1.2, 21.0, 0.6, 7.0), "mlp": Kernel(1.0, 28.0, 0.0, 14.0)}
    del params["direction_layer"]
    trial = FixedTrial(params)
    search.suggest(trial, config.Search(), 29, ["attn", "mlp"], search.PER_LAYER)
    assert trial.params == params
    assert search.edit_from(params, 7, 29).direction_layer is None


def test_only_an_edit_of_one_layers_direction_draws_a_direction_layer():
    for scope, drawn in ((search.ONE_LAYER, True), (search.PER_LAYER, False)):
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        trial = study.ask()
        search.suggest(trial, config.Search(), 28, ["attn", "mlp"], scope)
        assert ("direction_layer" in trial.params) is drawn
        assert len(trial.params) == 8 + drawn
        assert search.scope_of(trial.params) == scope


def test_an_edit_over_the_cap_is_told_by_how_far_its_refusals_scaled_to_every_prompt_go_over():
    assert search.violation(Score(2, 0, 0, 0.1, 100), 100, 2) == -0.5
    assert search.violation(Score(3, 0, 0, 0.1, 100), 100, 2) == 0.5
    assert search.violation(Score(3, 0, 0, 0.1, 8), 100, 2) == pytest.approx(3 * 100 / 8 - 2.5)
    assert search.violation(Score(0, 0, 0, 0.0, 100), 100, 0) == -0.5


def test_each_kind_starts_from_points_spread_evenly_over_its_ranges():
    space = config.Search()
    ranges = search._ranges(space, 28)
    for scope in search.SCOPES:
        points = search.spread(scope, space, 28, ["attn", "mlp"], 20, 7)
        assert points == search.spread(scope, space, 28, ["attn", "mlp"], 20, 7)
        assert points != search.spread(scope, space, 28, ["attn", "mlp"], 20, 8)
        assert len(points) == 20 and all(search.scope_of(p) == scope for p in points)
        for name in points[0]:
            values = [p[name] for p in points]
            if name == "direction_layer":
                assert all(isinstance(v, int) and 11 <= v <= 24 for v in values)
                assert len(set(values)) >= 12
                continue
            low, high = ranges[name.split(".")[1]]
            assert all(low <= v <= high for v in values)
            # Twenty evenly spread points leave no stretch as wide as a fifth of a range empty.
            edges = [low, *sorted(values), high]
            assert max(b - a for a, b in itertools.pairwise(edges)) < 0.2 * (high - low)


def test_each_study_starts_with_its_spread_points(tiny):
    small(tiny, trials=3)
    tiny.settings.search.startup_per_scope = 2
    found, _ = Search(tiny).search(4)
    _, layers, _, writers = tiny.shape
    components = sorted({w.component for w in writers})
    for i, scope in enumerate(search.SCOPES):
        drawn = [t.params for t in found.trials.values() if search.scope_of(t.params) == scope]
        assert drawn == search.spread(scope, tiny.settings.search, layers, components, 2, i)


def test_each_trial_is_scored_with_its_own_directions_adapters(tiny, monkeypatch):
    small(tiny, trials=3)
    scored = []
    real = search._Scorer.score

    def recording(self, scales, stop_above=None, forced_above=None, read_kl=True):
        scored.append({path.parent.name for path in scales})
        return real(self, scales, stop_above, forced_above, read_kl)

    monkeypatch.setattr(search._Scorer, "score", recording)
    Search(tiny).search(6)
    expected = [
        {"per-layer" if edit.direction_layer is None else f"layer-{edit.direction_layer}"}
        for edit in (search.trial_edit(tiny, number) for number in range(1, 7))
    ]
    assert scored[1:] == expected
    assert len({frozenset(s) for s in expected}) > 1


def test_a_search_trial_is_measured_with_the_very_adapters_it_was_scored_with(tiny):
    small(tiny)
    Search(tiny).search(2)
    edit = search.trial_edit(tiny, 1)
    Measurement(tiny, edit, "trial-1", limit=1).measure()
    scored = tiny.path(
        "adapters", "per-layer" if edit.direction_layer is None else f"layer-{edit.direction_layer}"
    )
    measured = tiny.path("measurements", "trial-1", "adapters")
    names = sorted(p.name for p in scored.iterdir())
    assert names == sorted(p.name for p in measured.iterdir())
    assert all(filecmp.cmp(scored / name, measured / name, shallow=False) for name in names)
    record = json.loads(tiny.path("measurements", "trial-1", "maths.json.inputs.json").read_text())
    expected = edit.strengths(tiny.shape[3], tiny.target_names)
    assert record["edit"]["strengths"] == pytest.approx(expected)


def test_a_trial_that_doesnt_exist_is_named(tiny):
    with pytest.raises(RunError, match="has no search yet"):
        search.trial_edit(tiny, 1)
    with pytest.raises(RunError, match="has no search yet"):
        search.recorded_edit(tiny, 1)
    small(tiny)
    Search(tiny).search(1)
    with pytest.raises(RunError, match="the search has no finished trial 9"):
        search.trial_edit(tiny, 9)
    with pytest.raises(RunError, match="the search has no finished trial 9"):
        search.recorded_edit(tiny, 9)


def test_a_trials_recorded_edit_is_its_edit_and_needs_no_gguf(tiny):
    small(tiny)
    Search(tiny).search(2)
    edit = search.trial_edit(tiny, 2)
    for leftover in tiny.path("gguf").iterdir():
        leftover.unlink()
    assert search.recorded_edit(tiny, 2).record() == edit.record()
    assert not any(tiny.path("gguf").iterdir())


def test_search_and_measure_a_trial_from_the_command_line(tiny):
    small(tiny)
    folder = on_disk(tiny)
    runner = CliRunner(env={"COLUMNS": "200"})
    searched = runner.invoke(app, ["search", str(folder), "--trials", "2"])
    assert searched.exit_code == 0, searched.output
    shown = flat(searched.output)
    assert "Fully scored trials no other beats on both refusals and KL" in shown
    assert "so the 2 left of 4 would take about" in shown
    assert "Trial 1, one layer: 1 of 3 refused, KL" in shown
    assert "Trial 2, per layer: 1 of 3 refused, KL" in shown
    assert "The least KL of each kind at each count within the cap (2 refused)" in shown
    # Output going to a file, as a pod's log, shows no bar, so each tenth is a line.
    assert "Trials: 1 of 2" in shown and "Trials: 2 of 2" in shown
    finished_search = runner.invoke(app, ["search", str(folder)])
    assert finished_search.exit_code == 0, finished_search.output
    assert "Trials: 3 of 4" in finished_search.output and "Trials: 1 of 4" not in finished_search.output
    commands = [line.split() for line in finished_search.output.splitlines() if "candor measure" in line]
    found = search.load(run.Run.open(folder))
    assert found is not None
    expected = [c.trial for scope in search.SCOPES for c in search.to_measure(found, scope, config.Search())]
    assert [int(words[-1]) for words in commands] == expected and len(expected) >= 2
    measured = runner.invoke(app, ["measure", str(folder), "--trial", "1", "--limit", "1"])
    assert measured.exit_code == 0, measured.output
    assert "candor measure: trial-1-first-1" in measured.output


def test_test_prompts_that_changed_since_prompts_json_stop_the_search(tiny, monkeypatch):
    small(tiny)
    tiny.prompt_ids()

    def changed(spec):
        prompts = fake_prompts(spec)
        return [f"{prompts[0]} changed", *prompts[1:]] if spec.split == "test[:100]" else prompts

    monkeypatch.setattr(run, "load_prompts", changed)
    with pytest.raises(RunError, match=r"harmful_test prompts have changed since prompts\.json was built"):
        Search(tiny).search(1)


def test_every_edit_the_search_draws_changes_some_layer_of_each_component():
    layers = 28
    writers = [Writer(f"{c}.{i}", i, c) for i in range(layers) for c in ("attn", "mlp")]
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for scope in search.SCOPES:
        drawn = [*search.spread(scope, config.Search(), layers, ["attn", "mlp"], 20, 0)]
        for _ in range(300):
            trial = study.ask()
            search.suggest(trial, config.Search(), layers, ["attn", "mlp"], scope)
            drawn.append(trial.params)
            study.tell(trial, 0.0)
        for params in drawn:
            strengths = search.edit_from(params, 1, layers).strengths(
                writers, {w.name: w.name for w in writers}
            )
            assert all(
                any(strengths[w.name] > 0 for w in writers if w.component == c) for c in ("attn", "mlp")
            )


def test_a_resumed_search_doesnt_repeat_its_trials(tiny):
    small(tiny, trials=4)
    Search(tiny).search(4)
    Search(tiny).search(8)
    params = [json.dumps(trial.params, sort_keys=True) for trial in finished(tiny)]
    assert len(set(params)) == 8


def test_the_table_is_the_front_of_refusals_and_kl_themselves():
    def made(name, refusals, kl):
        return search.Candidate(name, search.PER_LAYER, refusals, refusals, kl, {})

    lowest_kl, best_twenty, worse_twenty, fewest = (
        made(1, 40, 0.0001),
        made(2, 20, 0.004),
        made(3, 20, 0.009),
        made(4, 5, 0.3),
    )
    found = search.front([lowest_kl, best_twenty, worse_twenty, fewest])
    assert found == [fewest, best_twenty, lowest_kl]


def test_a_trial_isnt_rebuilt_from_other_directions_or_adapters(tiny):
    small(tiny)
    Search(tiny).search(1)
    tiny.settings.edit.row_norm = "none"
    with pytest.raises(RunError, match="made with different adapters from this run's now"):
        search.trial_edit(tiny, 1)


def test_a_trial_is_rebuilt_from_a_search_whose_scores_meant_something_else(tiny, monkeypatch):
    small(tiny)
    Search(tiny).search(1)
    assert not search.older(tiny.folder)
    # As a search recorded before rebuilding a trial's edit had a version of its own, by candor whose
    # scores meant something else.
    record = tiny.path("search", "inputs.json")
    made = json.loads(record.read_text())
    del made["code"]["edits"]
    record.write_text(json.dumps(made))
    monkeypatch.setattr(search, "VERSION", search.VERSION + 1)
    found = search.load(tiny)
    assert found is not None
    number = next(iter(found.trials))
    assert search.older(tiny.folder)
    assert search.trial_edit(tiny, number) == search.recorded_edit(tiny, number)
    monkeypatch.setattr(search, "EDITS", search.EDITS + 1)
    with pytest.raises(RunError, match="made with different candor code from this run's now, so its trials"):
        search.trial_edit(tiny, number)


def silent(monkeypatch):
    """The model says nothing at all, which the fake judge doesn't call a refusal."""
    generate, answers = LlamaServer.generate, LlamaServer.answers

    def nothing(self, *args, **kwargs):
        return [Answer("", 0, True) for _ in generate(self, *args, **kwargs)]

    def nothing_each(self, *args, **kwargs):
        for _ in answers(self, *args, **kwargs):
            yield Answer("", 0, True)

    monkeypatch.setattr(LlamaServer, "generate", nothing)
    monkeypatch.setattr(LlamaServer, "answers", nothing_each)


def test_an_edit_that_says_nothing_counts_as_refusing_everything(tiny, monkeypatch):
    small(tiny)
    silent(monkeypatch)
    Search(tiny).search(1)
    [trial] = finished(tiny)
    assert trial.user_attrs["refusals"] == 3 and trial.user_attrs["unlabelled"] == 0
    results = Measurement(tiny, None, "silent").measure()
    assert results["harmful_held_out"]["judge_refused"] == [True, True, True]


def test_each_trial_is_scored_with_its_strengths_on_the_test_prompts(tiny, monkeypatch):
    small(tiny)
    ids = tiny.prompt_ids()
    asked = {"answers": [], "top_logprobs": [], "scales": []}
    answers = []
    real_answers, real_top = LlamaServer.answers, LlamaServer.top_logprobs

    def answering(self, prompts, tokens, scales=None, *args):
        asked["answers"].append(prompts)
        asked["scales"].append(dict(scales or {}))
        found = list(real_answers(self, prompts, tokens, scales, *args))
        answers.append([a.text for a in found])
        yield from found

    def top_logprobs(self, prompts, *args, **kwargs):
        asked["top_logprobs"].append(prompts)
        return real_top(self, prompts, *args, **kwargs)

    monkeypatch.setattr(LlamaServer, "answers", answering)
    monkeypatch.setattr(LlamaServer, "top_logprobs", top_logprobs)
    _, seconds = Search(tiny).search(3)
    assert all(prompts == ids["harmful_test"] for prompts in asked["answers"])
    assert all(prompts == ids["harmless_test"] for prompts in asked["top_logprobs"])
    adapters = tiny.adapters()
    trials = finished(tiny)
    assert asked["scales"][0] == {}
    for number, (trial, scales, said) in enumerate(
        zip(trials, asked["scales"][1:], answers[1:], strict=True), 1
    ):
        edit = search.trial_edit(tiny, number)
        files = adapters["per-layer" if edit.direction_layer is None else f"layer-{edit.direction_layer}"]
        strengths = edit.strengths(tiny.shape[3], tiny.target_names)
        assert scales == {files[name]: s for name, s in strengths.items() if s}
        assert trial.user_attrs["kl"] > 0
        assert trial.user_attrs["keyword_refusals"] == sum(map(is_refusal, said))
    assert seconds == pytest.approx(sum(t.user_attrs["seconds"] for t in trials) / 3)


def test_the_sampler_is_told_each_trials_kl_and_how_far_it_goes_over_the_cap(tiny):
    small(tiny)
    tiny.settings.search.max_refusals = 0
    Search(tiny).search(2)
    assert [s.direction for s in studies(tiny).values()] == [StudyDirection.MINIMIZE] * 2
    for trial in finished(tiny):
        assert trial.values == [trial.user_attrs["kl"]]
        assert trial.user_attrs["refusals"] == 1 and trial.user_attrs["judged"] == 3
        assert trial.user_attrs["violation"] == 0.5
        assert list(trial.system_attrs["constraints"]) == [0.5]


def scripted(monkeypatch, scores: list[Score], budget: list | None = None):
    """The scorer gives these scores in turn, the unedited model's first, and records each stop_above,
    and in budget each forced_above."""
    asked = []
    queue = iter(scores)

    def score(self, scales, stop_above=None, forced_above=None, read_kl=True):
        asked.append(stop_above)
        if budget is not None:
            budget.append(forced_above)
        return next(queue)

    monkeypatch.setattr(search._Scorer, "score", score)
    return asked


def test_a_trial_is_announced_when_none_of_its_kind_refused_as_few_with_as_little_kl(tiny, monkeypatch):
    small(tiny, trials=4)
    scripted(
        monkeypatch,
        [
            Score(3, 0, 0, 0.0, 3),  # the unedited model
            Score(1, 0, 0, 0.5, 3),  # 1, one layer: the first within the cap
            Score(0, 0, 0, 0.1, 1),  # 2, per layer: stopped early, so not scored in full
            Score(0, 0, 0, 0.6, 3),  # 3, one layer: more KL, but the first to refuse none
            Score(2, 0, 0, 0.2, 3),  # 4, per layer: the first of its kind scored in full
            Score(0, 0, 0, 0.55, 3),  # 5, one layer, in a new session: less KL than trial 3
            Score(0, 0, 0, 0.1, 1),  # 6, per layer: stopped early again
            Score(2, 0, 0, 0.49996, 3),  # 7, one layer: trial 1's KL as shown, refusing more
            Score(1, 0, 0, 0.15, 3),  # 8, per layer: the first to refuse 1 or fewer in full
        ],
    )
    said = []
    Search(tiny).search(4, finished=said.append)
    Search(tiny).search(8, finished=said.append)
    assert [line and line.split(":")[0] for line in said] == [
        "Trial 1, one layer",
        None,
        "Trial 3, one layer",
        "Trial 4, per layer",
        "Trial 5, one layer",
        None,
        None,
        "Trial 8, per layer",
    ]
    assert said[0] == (
        "Trial 1, one layer: 1 of 3 refused, KL 0.5000, the least KL so far of its kind refusing 1 or fewer"
    )


def test_a_trial_stops_once_it_refuses_more_than_the_cap_and_enough_trials_of_its_kind(tiny, monkeypatch):
    small(tiny, trials=3)
    monkeypatch.setattr(search, "STOP_AFTER", 1)
    asked = scripted(
        monkeypatch,
        [
            Score(3, 0, 0, 0.0, 3),
            Score(1, 0, 0, 0.5, 3),  # 1, one layer
            Score(3, 0, 0, 0.1, 3),  # 2, per layer
            Score(3, 0, 0, 0.01, 1),  # 3, one layer, stopped early with the least KL
            Score(0, 0, 0, 0.2, 3),  # 4, per layer
            Score(1, 0, 0, 0.4, 3),  # 5, one layer
        ],
    )
    Search(tiny).search(5)
    # Nothing to stop the first trial of each kind; then the more of the cap (2) and the kind's fewest.
    assert asked == [None, None, None, 2, 3, 2]
    trials = finished(tiny)
    assert [t.user_attrs["stopped"] for t in trials] == [False, False, True, False, False]
    assert trials[2].user_attrs["violation"] == pytest.approx(3 * 3 / 1 - 2.5)
    found = search.load(tiny)
    assert found is not None
    assert 3 not in [c.trial for c in search.candidates(found)]
    assert [c.trial for c in search.within_cap(found, search.ONE_LAYER, tiny.settings.search)] == [5, 1]
    assert [c.trial for c in search.within_cap(found, search.PER_LAYER, tiny.settings.search)] == [4]
    tiny.settings.search.stop_early = False
    tiny.path("search").rename(tiny.path("search-before"))
    asked = scripted(monkeypatch, [Score(3, 0, 0, 0.0, 3)] + [Score(1, 0, 0, 0.5, 3)] * 4)
    Search(tiny).search(4)
    assert asked == [None] * 5


def stop_free_draws(refusals_and_kl):
    """A scorer whose scores follow from each edit's strengths alone, and which stops as the real one
    does: a stopped trial is judged in part and its KL never read."""

    def score(self, scales, stop_above=None, forced_above=None, read_kl=True):
        if not scales:
            return Score(3, 0, 0, math.inf, 3)
        refusals, kl = refusals_and_kl(scales)
        if stop_above is not None and refusals > stop_above and stop_above + 1 < 3:
            return Score(stop_above + 1, 0, 0, math.inf, stop_above + 1)
        return Score(refusals, 0, 0, kl, 3)

    return score


@pytest.mark.parametrize("cap", [0, 1])
def test_stopping_a_trial_changes_no_draw_the_sampler_makes(tiny, monkeypatch, cap):
    # Optuna ranks trials over the caps by how far over they are, never by KL: so the trials it draws
    # are the same whether the ones that can't win are stopped or scored in full.
    def made_up(scales):
        key = hashlib.sha256(
            repr(sorted((p.name, round(s, 12)) for p, s in scales.items())).encode()
        ).digest()
        return key[0] % 4, (key[1] + 1) / 2560

    monkeypatch.setattr(search, "STOP_AFTER", 1)
    monkeypatch.setattr(search._Scorer, "score", stop_free_draws(made_up))
    small(tiny, trials=10)
    tiny.settings.search.startup_per_scope = 2
    tiny.settings.search.max_refusals = cap
    drawn = {}
    for stop in (True, False):
        tiny.settings.search.stop_early = stop
        Search(tiny).search(20)
        drawn[stop] = [(t.params, t.user_attrs["stopped"]) for t in finished(tiny)]
        tiny.path("search").rename(tiny.path(f"search-{stop}"))
    assert [params for params, _ in drawn[True]] == [params for params, _ in drawn[False]]
    assert any(stopped for _, stopped in drawn[True]) and not any(stopped for _, stopped in drawn[False])


@pytest.mark.parametrize(
    ("stop_after", "expected"), [(1, [None, None, None, 3, 3, 2]), (2, [None] * 5 + [3])]
)
def test_a_trial_stops_above_the_stop_after_fewest_refusals_of_its_kind(
    tiny, monkeypatch, stop_after, expected
):
    small(tiny, trials=3)
    tiny.settings.search.max_refusals = 0
    monkeypatch.setattr(search, "STOP_AFTER", stop_after)
    asked = scripted(
        monkeypatch,
        [
            Score(3, 0, 0, 0.0, 3),
            Score(3, 0, 0, 0.5, 3),  # 1, one layer
            Score(3, 0, 0, 0.5, 3),  # 2, per layer
            Score(2, 0, 0, 0.4, 3),  # 3, one layer
            Score(3, 0, 0, 0.5, 3),  # 4, per layer
            Score(1, 0, 0, 0.3, 3),  # 5, one layer: its kind has refused 3 and 2 in full
        ],
    )
    Search(tiny).search(5)
    assert asked == expected


def test_by_default_a_trial_stops_only_once_25_of_its_kind_refused_fewer_in_full():
    # TPE's good trials number at most 25, so a trial behind 25 of its kind is never one of them.
    assert search.STOP_AFTER == 25
    assert default_gamma(10_000) == 25


def test_scoring_stops_at_the_end_of_the_chunk_that_goes_over(tiny, monkeypatch):
    monkeypatch.setattr(search, "CHUNK", 2)
    labels = []

    asked = []

    def replies(self, prompts, answers, **kwargs):
        asked.append(len(answers))
        return [REFUSED if labels.pop(0) else ANSWERED for _ in answers]

    monkeypatch.setattr(FakeGuard, "replies", replies)
    read = []
    real = LlamaServer.top_logprobs
    monkeypatch.setattr(
        LlamaServer, "top_logprobs", lambda *args, **kwargs: read.append(1) or real(*args, **kwargs)
    )
    with Search(tiny)._scorer(*scorer_inputs(tiny)) as scorer:
        labels[:], asked[:] = [True, False, True], []
        whole = scorer.score({}, stop_above=2)
        assert (whole.refusals, whole.judged, asked) == (2, 3, [2, 1])
        labels[:], asked[:], read[:] = [True, True, True], [], []
        stopped = scorer.score({}, stop_above=1)
        assert (stopped.refusals, stopped.judged, asked) == (2, 2, [2])
        # A trial stopped on its refusals can't win, so its KL is never read.
        assert math.isinf(stopped.kl) and stopped.kl_thinking is None and whole.kl == 0.0 and not read
        labels[:] = [True, True, True]
        assert math.isinf(scorer.score({}, read_kl=False).kl) and not read
        # One stopped on its thinking reads it: it could count once search.max_budget_hits rises.
        labels[:] = [False, False, False]
        budget = scorer.score({}, forced_above=-1)
        assert budget.judged == 2 and budget.kl == 0.0 and read
        labels[:], asked[:] = [True, True, True], []
        assert scorer.score({}).judged == 3 and asked == [2, 1]
        labels[:], asked[:] = [True, True, False], []
        level = scorer.score({}, stop_above=2)
        assert (level.refusals, level.judged, asked) == (2, 3, [2, 1])


def scorer_inputs(tiny):
    ids = tiny.prompt_ids()
    _, quantized = tiny.gguf()
    return quantized, [], ids, ids["harmless_test"], {}


def test_the_search_shows_time_left_from_every_trial_of_the_session(tiny, monkeypatch):
    small(tiny)
    folder = on_disk(tiny)
    made = []
    real = cli.Progress

    def recording(*args, **kwargs):
        made.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "Progress", recording)
    result = CliRunner().invoke(app, ["search", str(folder), "--trials", "1"])
    assert result.exit_code == 0, result.output
    assert made[0]["speed_estimate_period"] >= 3600


def test_keywords_are_counted_on_each_trials_answers(tiny, monkeypatch):
    small(tiny)
    real = LlamaServer.answers

    def one_sorry(self, *args, **kwargs):
        for i, answer in enumerate(real(self, *args, **kwargs)):
            yield Answer("Here it is." if i else "I'm sorry.", answer.tokens, True)

    monkeypatch.setattr(LlamaServer, "answers", one_sorry)
    Search(tiny).search(2)
    assert [t.user_attrs["keyword_refusals"] for t in finished(tiny)] == [1, 1]


def test_the_best_trial_pressing_on_an_end_of_a_range_is_flagged():
    space = config.Search()
    inside = {
        "direction_layer": 15,
        "attn.strength": 1.2,
        "attn.peak_layer": 0.8,
        "attn.edge_strength": 0.0,
        "attn.reach": 0.3,
        "mlp.strength": 1.2,
        "mlp.peak_layer": 1.0,
        "mlp.edge_strength": 1.0,
        "mlp.reach": 0.04,
    }
    # The edge strength's ends, the peak's ceiling and the reach's floor are the model's own ends.
    assert search.at_edges(inside, space, 28) == []
    pressed = inside | {
        "direction_layer": 11,
        "attn.strength": 1.49,
        "attn.peak_layer": 0.61,
        "mlp.strength": 0.82,
        "mlp.reach": 0.59,
    }
    assert search.at_edges(pressed, space, 28) == [
        "direction layer 11, the first the search reads",
        "attn.peak_layer 0.61, at its floor 0.6",
        "attn.strength 1.49, at its ceiling 1.5",
        "mlp.reach 0.59, at its ceiling 0.6",
        "mlp.strength 0.82, at its floor 0.8",
    ]
    assert search.at_edges(inside | {"direction_layer": 24}, space, 28) == [
        "direction layer 24, the last the search reads"
    ]
    per_layer = {k: v for k, v in pressed.items() if k != "direction_layer"}
    assert "direction layer" not in " ".join(search.at_edges(per_layer, space, 28))


# An edit that changes both layers of the tiny model, inside the search's space there.
KERNELS = {"attn": Kernel(1.2, 0.8, 0.6, 0.6), "mlp": Kernel(1.0, 1.0, 0.5, 0.6)}
# heretic's Trial 170 for rude-llama, 28 layers.
RUDE = {"attn": Kernel(1.33, 16.5, 1.05, 13.5), "mlp": Kernel(1.26, 16.6, 0.57, 12.8)}


def scored_folders(monkeypatch, root) -> list[tuple[set[str], dict]]:
    """The adapter folders, relative to root, and the scales of every edit scored."""
    used = []
    real = search._Scorer.score

    def recording(self, scales, **kwargs):
        used.append(({str(path.parent.relative_to(root)) for path in scales}, scales))
        return real(self, scales, **kwargs)

    monkeypatch.setattr(search._Scorer, "score", recording)
    return used


def test_a_trial_scored_again_scores_as_it_did_in_the_search_and_isnt_added_to_it(tiny):
    small(tiny)
    Search(tiny).search(3)
    record = tiny.path("search", "study.jsonl").read_bytes()
    scored = Search(tiny).score([search.trial_edit(tiny, number) for number in (1, 2, 3)])
    found = search.load(tiny)
    assert found is not None
    assert scored.prompts == 3
    assert scored.base_refusals == found.base_refusals
    assert scored.edits == [search.searched_score(found, number) for number in (1, 2, 3)]
    assert tiny.path("search", "study.jsonl").read_bytes() == record


def test_an_edit_between_layers_gets_its_own_adapters_and_the_nearest_layer_the_searchs(tiny, monkeypatch):
    small(tiny)
    Search(tiny).search(1)
    edit = Edit("heretic-7", 0.25, KERNELS)
    nearest = search.nearest_layer(edit, tiny.settings.search, 2)
    assert nearest is not None and nearest == Edit("heretic-7 at layer 0", 0, KERNELS)
    used = scored_folders(monkeypatch, tiny.folder)
    Search(tiny).score([edit, nearest, Edit("each", None, KERNELS)])
    assert [folders for folders, _ in used] == [
        set(),
        {"scores/layer-0.25"},
        {"adapters/layer-0"},
        {"adapters/per-layer"},
    ]
    strengths = edit.strengths(tiny.shape[3], tiny.target_names)
    for _, scales in used[1:]:
        assert {path.name: s for path, s in scales.items()} == {
            f"{name}.gguf": s for name, s in strengths.items() if s
        }
    Measurement(tiny, edit, "heretic-7", limit=1).measure()
    measured = tiny.path("measurements", "heretic-7", "adapters")
    own = tiny.path("scores", "layer-0.25")
    names = sorted(p.name for p in measured.glob("*.gguf"))
    assert names == sorted(p.name for p in own.glob("*.gguf"))
    assert all(filecmp.cmp(measured / name, own / name, shallow=False) for name in names)


def test_adapters_built_for_one_direction_are_built_again_for_another(tiny):
    folder = tiny.path("measurements", "e", "adapters")
    edit_adapters(tiny, Edit("e", 0.25, KERNELS), folder)
    edit_adapters(tiny, Edit("e", 0.75, KERNELS), folder)
    fresh, _ = edit_adapters(tiny, Edit("e", 0.75, KERNELS), tiny.path("measurements", "fresh", "adapters"))
    assert all(filecmp.cmp(path, folder / path.name, shallow=False) for path in fresh.values())


def test_an_edit_at_a_layer_the_search_never_reads_gets_its_own_adapters(tiny, monkeypatch):
    tiny.settings.search.direction_layer = (1.0, 1.0)
    used = scored_folders(monkeypatch, tiny.folder)
    Search(tiny).score([Edit("low", 0, KERNELS), Edit("high", 1.0, KERNELS)])
    assert [folders for folders, _ in used[1:]] == [{"scores/layer-0"}, {"adapters/layer-1"}]


@pytest.mark.parametrize("layer", [1.5, -0.5])
def test_a_direction_layer_outside_the_model_isnt_scored(tiny, layer):
    with pytest.raises(RunError, match=f"far reads its direction at layer {layer:g}; this model has 2"):
        Search(tiny).score([Edit("far", layer, KERNELS)])


def test_scoring_loads_every_adapter_the_search_loads_once_each(tiny, monkeypatch):
    small(tiny)
    Search(tiny).search(1)
    loaded, real = [], LlamaServer.__init__

    def recording(self, *args, **kwargs):
        real(self, *args, **kwargs)
        loaded.append(list(self.adapters))

    monkeypatch.setattr(LlamaServer, "__init__", recording)
    Search(tiny).score([Edit("a", 0.25, KERNELS), Edit("b", 0.25, KERNELS), Edit("c", 0, KERNELS)])
    [files] = loaded
    searched = {path.resolve() for made in tiny.adapters().values() for path in made.values()}
    assert len(files) == len(set(files))
    assert set(files) == searched | {
        path.resolve() for path in tiny.path("scores", "layer-0.25").glob("*.gguf")
    }


def test_scoring_keeps_no_answers(tiny, monkeypatch):
    answers = recording_answers(monkeypatch)
    Search(tiny).score([Edit("heretic-7", 0.25, KERNELS)])
    files = [p for p in tiny.path("scores").rglob("*") if p.is_file()]
    assert len(answers) == 2 * 3
    assert files and all(p.suffix == ".gguf" or p.name.endswith(".inputs.json") for p in files)


def test_scoring_without_a_search_starts_none(tiny):
    scored = Search(tiny).score([Edit("each", None, KERNELS)])
    assert len(scored.edits) == 1
    assert not tiny.path("search").exists()


def test_a_search_without_its_record_isnt_set_beside_anything(tiny):
    small(tiny)
    Search(tiny).search(1)
    tiny.path("search", "inputs.json").unlink()
    with pytest.raises(RunError, match="holds a search with no record of its settings"):
        Search(tiny).score([Edit("each", None, KERNELS)])
    with pytest.raises(RunError, match="holds a search with no record of its settings"):
        search.trial_edit(tiny, 1)


def test_scoring_waits_for_no_other_command(tiny):
    with tiny.locked(), pytest.raises(RunError, match="another candor command is using"):
        Search(tiny).score([Edit("each", None, KERNELS)])


def test_scoring_closes_the_judge_even_when_it_stops(tiny, monkeypatch):
    closed = []
    monkeypatch.setattr(FakeGuard, "close", lambda self: closed.append(self))
    Search(tiny).score([Edit("each", None, KERNELS)])

    def broken(self, *args, **kwargs):
        raise RuntimeError("stopped")

    monkeypatch.setattr(LlamaServer, "answers", broken)
    with pytest.raises(RuntimeError, match="stopped"):
        Search(tiny).score([Edit("each", None, KERNELS)])
    assert len(closed) == 2


def test_an_edit_drawn_by_the_search_is_inside_its_space():
    params = {
        "direction_layer": 15,
        "attn.strength": 1.2,
        "attn.peak_layer": 0.75,
        "attn.edge_strength": 0.5,
        "attn.reach": 0.25,
        "mlp.strength": 0.8,
        "mlp.peak_layer": 1.0,
        "mlp.edge_strength": 1.0,
        "mlp.reach": 0.6,
    }
    per_layer = {k: v for k, v in params.items() if k != "direction_layer"}
    for drawn in (params, per_layer):
        edit = search.edit_from(drawn, 1, 29)
        assert edit.direction_layer == drawn.get("direction_layer")
        assert search.outside_space(edit, config.Search(), 29, ["attn", "mlp"]) == []


def test_a_drawn_setting_on_the_edge_of_its_range_isnt_pushed_outside_by_rounding():
    space = config.Search(edge_strength=(0.7, 1.0), peak_layer=(0.6, 0.8))
    params = {
        "attn.strength": 0.8,
        "attn.peak_layer": 0.8,
        "attn.edge_strength": 0.7,
        "attn.reach": 0.6,
    }
    assert search.outside_space(search.edit_from(params, 1, 4), space, 4, ["attn"]) == []


def test_the_first_and_last_layers_the_search_reads_are_inside_its_space():
    space = config.Search()
    low, high = space.direction_layers(28)
    assert (low, high) == (11, 24)
    for layer in (low, high):
        assert search.outside_space(Edit("e", layer, RUDE), space, 28, ["attn", "mlp"]) == []
    for layer in (low - 1, high + 1):
        assert search.outside_space(Edit("e", layer, RUDE), space, 28, ["attn", "mlp"]) == [
            f"direction layer {layer} (whole layers 11-24)"
        ]


def test_heretic_trial_170_at_layer_12_is_inside_the_search_space_for_rude_llama():
    edit = Edit("heretic-170", 12.04, RUDE)
    nearest = search.nearest_layer(edit, config.Search(), 28)
    assert nearest is not None and nearest == Edit("heretic-170 at layer 12", 12, RUDE)
    assert search.outside_space(nearest, config.Search(), 28, ["attn", "mlp"]) == []
    assert search.outside_space(edit, config.Search(), 28, ["attn", "mlp"]) == [
        "direction layer 12.04 (whole layers 11-24)"
    ]


def test_each_setting_the_search_couldnt_draw_is_named_with_its_range():
    kernels = {
        "attn": Kernel(1.6, 0.5 * 27, 1.0, 0.7 * 27),
        "moe": Kernel(1.0, 20.0, 0.5, 5.0),
    }
    outside = search.outside_space(Edit("e", 25, kernels), config.Search(), 28, ["attn", "mlp"])
    assert outside == [
        "direction layer 25 (whole layers 11-24)",
        "no mlp edit",
        "attn.strength 1.6 (0.8-1.5)",
        "attn.peak_layer 0.5 (0.6-1)",
        "attn.reach 0.7 (0.037-0.6)",
        "a moe edit, which the search never makes",
    ]
    stronger_edge = {"attn": Kernel(1.0, 20.0, 1.2, 5.0), "mlp": Kernel(1.0, 20.0, 0.5, 5.0)}
    assert search.outside_space(Edit("e", None, stronger_edge), config.Search(), 28, ["attn", "mlp"]) == [
        "attn.edge_strength 1.2 (0-1)"
    ]


@pytest.mark.parametrize(("layer", "nearest"), [(12.04, 12), (12.6, 13), (10.6, 11), (24.4, 24)])
def test_the_nearest_layer_rounds_either_way(layer, nearest):
    found = search.nearest_layer(Edit("e", layer, RUDE), config.Search(), 28)
    assert found is not None and found.direction_layer == nearest


@pytest.mark.parametrize("layer", [None, 12, 12.0, 3.4, 25.4])
def test_only_an_edit_between_layers_the_search_reads_gets_a_nearest_layer(layer):
    assert search.nearest_layer(Edit("e", layer, KERNELS), config.Search(), 28) is None


def test_an_edit_beats_another_only_if_no_worse_on_both_and_better_on_one_as_shown():
    def score(refusals, kl, keywords=0, judged=100):
        return Score(refusals, keywords, 0, kl, judged)

    assert search.beats(score(1, 0.1), score(2, 0.1))
    assert search.beats(score(2, 0.09), score(2, 0.1))
    assert not search.beats(score(2, 0.1), score(2, 0.1))
    assert not search.beats(score(1, 0.2), score(2, 0.1))
    assert not search.beats(score(2, 0.1), score(1, 0.2))
    assert not search.beats(score(1, 0.101109), score(1, 0.101112))
    assert search.ties(score(1, 0.101109), score(1, 0.101112))
    assert not search.ties(score(1, 0.1011), score(2, 0.1011))
    assert search.beats(search.Candidate(5, "per layer", 0, 4, 0.1, {}), score(1, 0.1))


def test_a_trial_scored_again_repeats_its_score_within_noise():
    def score(refusals, kl, keywords=0, judged=100):
        return Score(refusals, keywords, 0, kl, judged)

    # Batching noise alone moves a refusal or so, and KL by a few percent.
    assert search.differences(score(1, 0.1, 3), score(2, 0.102, 4)) == []
    assert search.differences(score(1, 0.1, 3), score(3, 0.1, 3)) == ["refused 1, not 3"]
    assert search.differences(score(1, 0.1, 3), score(1, 0.1, 5)) == ["refused by keywords 3, not 5"]
    assert search.differences(score(1, 0.1, 3), score(1, 0.104, 3)) == ["KL 0.1000, not 0.1040"]
    assert search.differences(score(1, 0.1, 3), score(1, 0.1, 3, judged=8)) == ["judged 100, not 8"]
    # KL too small to show apart is the same.
    assert search.differences(score(1, 0.00012), score(1, 0.00009)) == []


def test_the_searched_score_is_the_trials_own():
    study = optuna.create_study()
    trial = study.ask()
    attrs = {"number": 4, "refusals": 3, "keyword_refusals": 2, "unlabelled": 1, "kl": 0.5, "judged": 8}
    for key, value in attrs.items():
        trial.set_user_attr(key, value)
    study.tell(trial, 0.5)
    found = search._found({search.PER_LAYER: study})
    assert search.searched_score(found, 4) == Score(3, 2, 1, 0.5, 8)
    with pytest.raises(RunError, match="the search has no finished trial 1"):
        search.searched_score(found, 1)


def cli_run(tiny, trials=2):
    small(tiny, trials=trials)
    return on_disk(tiny)


def test_score_a_trial_and_a_heretic_trial_from_the_command_line(tiny):
    folder = cli_run(tiny)
    journal = write_journal(tiny.folder.parent / "heretic.jsonl", 0.25)
    runner = CliRunner(env={"COLUMNS": "200"})
    early = runner.invoke(app, ["score", str(folder), "--heretic", str(journal), "--trial", "1"])
    assert early.exit_code == 0, early.output
    assert "heretic-1 at layer 0" in early.output
    assert "The search has no trials yet to set these beside." in flat(early.output)
    missing = runner.invoke(app, ["score", str(folder), "--trial", "1"])
    assert missing.exit_code == cli.FAILED_CHECK
    assert "has no search yet" in missing.output

    assert runner.invoke(app, ["search", str(folder)]).exit_code == 0
    again = runner.invoke(app, ["score", str(folder), "--trial", "2"])
    assert again.exit_code == 0, again.output
    shown = flat(again.output)
    assert "Scored as the search scores its trials, on its 3 test prompts:" in shown
    assert "trial-2 scores as it did in the search" in shown
    # Trial 2 scored again shows scoring repeats, so the unedited model isn't scored again.
    assert "The unedited model refuses 1 of 3 in the search." in shown

    heretic = runner.invoke(app, ["score", str(folder), "--heretic", str(journal), "--trial", "1"])
    assert heretic.exit_code == 0, heretic.output
    shown = flat(heretic.output)
    found = search.load(run.Run.open(folder))
    assert found is not None
    best = search.candidates(found)[0].trial
    assert f"trial-{best} scores as it did in the search" in shown
    assert re.search(r"heretic-1 +layer 0\.25 ", heretic.output)
    assert re.search(r"heretic-1 at layer 0 +layer 0 ", heretic.output)
    assert "The search couldn't have drawn it: direction layer 0.25 (whole layers 0-1)." in shown
    assert "The search could have drawn it." in shown
    assert shown.count("on both refusals and KL") == 2


def test_the_command_line_says_which_trials_beat_an_edit_and_which_it_beats(tiny, monkeypatch):
    folder = cli_run(tiny, trials=3)
    runner = CliRunner(env={"COLUMNS": "200"})
    assert runner.invoke(app, ["search", str(folder)]).exit_code == 0
    found = search.load(run.Run.open(folder))
    assert found is not None
    front = search.candidates(found)
    best = search.searched_score(found, front[0].trial)
    journal = write_journal(tiny.folder.parent / "heretic.jsonl", 1)
    args = ["score", str(folder), "--heretic", str(journal), "--trial", "1"]
    trials = cli._trials([c.trial for c in front])

    def scoring(*scores):
        monkeypatch.setattr(Search, "score", lambda self, edits, base=True: search.Scores(3, 1, list(scores)))
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        return flat(result.output)

    worse = Score(best.refusals + 1, 0, 0, max(c.kl for c in front) + 1, 3)
    shown = scoring(worse, best)
    assert f"trial-{front[0].trial} scores as it did in the search" in shown
    assert f"heretic-1: the search's best {trials} beat" in shown
    assert f"{best.refusals}, {best.keyword_refusals}, {best.kl:.4f}" in shown

    shown = scoring(worse, Score(best.refusals, best.keyword_refusals, 0, best.kl * 1.1 + 0.01, 3))
    assert f"trial-{front[0].trial} scores differently from the search, by more than noise: KL" in shown
    assert "so the verdicts below don't hold. heretic-1: the search's best" in shown

    shown = scoring(Score(0, 0, 0, 0.0, 3), best)
    none = "heretic-1: no trial of the search beats it on both refusals and KL"
    assert f"{none}; it beats the search's best {trials} on both." in shown

    shown = scoring(best, best)
    assert f"{none}; it ties with trial {front[0].trial}." in shown

    monkeypatch.setattr(search, "candidates", lambda study: [])
    shown = scoring(worse)
    assert "The search has no trial scored in full yet to set these beside." in shown
    assert "beats it" not in shown


def made(number, refusals, kl, layer=None, judged=3):
    """A finished trial as the search records it, with a direction layer for one of one layer."""
    params = {"attn.strength": 1.0} | ({} if layer is None else {"direction_layer": layer})
    distributions: dict[str, BaseDistribution] = {
        k: IntDistribution(0, 30) if k == "direction_layer" else FloatDistribution(0, 2) for k in params
    }
    attrs = {
        "number": number,
        "refusals": refusals,
        "keyword_refusals": 0,
        "unlabelled": 0,
        "kl": kl,
        "judged": judged,
        "stopped": judged < 3,
    }
    return optuna.trial.create_trial(params=params, distributions=distributions, value=kl, user_attrs=attrs)


def test_within_cap_keeps_the_cap_and_puts_the_least_kl_first():
    trials = [made(1, 0, 0.3, 5), made(2, 2, 0.1, 5), made(3, 3, 0.05, 5), made(4, 0, 0.01)]
    found = search.Found({t.user_attrs["number"]: t for t in trials}, 1)
    assert [c.trial for c in search.within_cap(found, search.ONE_LAYER, config.Search())] == [2, 1]
    fewest = search.fewest_refused(found, search.ONE_LAYER)
    assert fewest is not None and fewest.trial == 1
    assert search.fewest_refused(search.Found({}, 1), search.PER_LAYER) is None


def test_to_measure_keeps_the_least_kl_at_each_count_unless_one_refusing_fewer_has_less():
    # As on rude-llama: none refused at a little more KL than the least within the cap.
    trials = [
        made(535, 0, 0.0778, 12),
        made(231, 1, 0.0833, 14),
        made(305, 2, 0.0760, 14),
        made(327, 2, 0.0764, 14),
        made(131, 3, 0.0661, 14),
        made(448, 2, 0.0831),
        made(462, 1, 0.0874),
        made(324, 0, 0.0914),
    ]
    found = search.Found({t.user_attrs["number"]: t for t in trials}, 99)
    space = config.Search()
    assert [c.trial for c in search.to_measure(found, search.ONE_LAYER, space)] == [535, 305]
    assert [c.trial for c in search.to_measure(found, search.PER_LAYER, space)] == [324, 462, 448]
    none = config.Search(max_refusals=0)
    assert [c.trial for c in search.to_measure(found, search.ONE_LAYER, none)] == [535]
    assert search.to_measure(search.Found({1: made(1, 1, 0.1)}, 99), search.PER_LAYER, none) == []
    close = search.Found({1: made(1, 0, 0.07604, 12), 2: made(2, 2, 0.07601, 12)}, 99)
    assert [c.trial for c in search.to_measure(close, search.ONE_LAYER, space)] == [1]  # the same KL as shown


def test_a_stopped_trial_is_never_a_candidate():
    found = search.Found({1: made(1, 1, 0.3), 2: made(2, 3, 0.01, judged=1)}, 1)
    assert [c.trial for c in search.candidates(found)] == [1]
    assert search.fewest_refused(found, search.PER_LAYER).trial == 1  # pyright: ignore[reportOptionalMemberAccess]


def test_a_new_session_announces_only_less_kl_than_the_last_sessions(tiny, monkeypatch):
    small(tiny, trials=3)
    scripted(
        monkeypatch,
        [
            Score(3, 0, 0, 0.0, 3),
            Score(1, 0, 0, 0.5, 3),  # 1, one layer
            Score(3, 0, 0, 0.1, 3),  # 2, per layer: over the cap
            Score(2, 0, 0, 0.3, 3),  # 3, one layer: at the cap, the least KL
            Score(3, 0, 0, 0.2, 3),  # 4, per layer: over the cap, in a new session
            Score(2, 0, 0, 0.4, 3),  # 5, one layer: more KL than trial 3
        ],
    )
    said = []
    Search(tiny).search(3, finished=said.append)
    Search(tiny).search(5, finished=said.append)
    assert [line and line.split(":")[0] for line in said] == [
        "Trial 1, one layer",
        None,
        "Trial 3, one layer",
        None,
        None,
    ]


def test_the_spread_start_reads_the_first_and_last_layers():
    for seed in range(5):
        points = search.spread(search.ONE_LAYER, config.Search(), 28, ["attn"], 20, seed)
        assert {11, 24} <= {p["direction_layer"] for p in points}


def test_a_spread_trial_that_failed_is_queued_again(tiny, monkeypatch):
    small(tiny, trials=3)
    tiny.settings.search.startup_per_scope = 2
    real = search._Scorer.score
    calls = []

    def stopping(self, scales, stop_above=None, forced_above=None, read_kl=True):
        calls.append(scales)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return real(self, scales, stop_above, forced_above, read_kl)

    monkeypatch.setattr(search._Scorer, "score", stopping)
    with pytest.raises(KeyboardInterrupt):
        Search(tiny).search(6)
    monkeypatch.setattr(search._Scorer, "score", real)
    Search(tiny).search(6)
    _, layers, _, writers = tiny.shape
    components = sorted({w.component for w in writers})
    for i, scope in enumerate(search.SCOPES):
        drawn = [search._key(t.params) for t in finished(tiny) if search.scope_of(t.params) == scope]
        spread = search.spread(scope, tiny.settings.search, layers, components, 2, i)
        assert {search._key(p) for p in spread} <= set(drawn)


def test_more_harmless_than_harmful_test_prompts(tiny, monkeypatch):
    small(tiny)

    def prompts(spec):
        found = fake_prompts(spec)
        extra = "harmless" in spec.dataset and spec.split.startswith("test")
        return [*found, f"one more {spec.dataset}"] if extra else found

    monkeypatch.setattr(run, "load_prompts", prompts)
    found, _ = Search(tiny).search(2)
    assert [(t.user_attrs["stopped"], t.user_attrs["violation"]) for t in found.trials.values()] == [
        (False, -1.5)
    ] * 2


def test_settings_near_but_not_at_an_end_arent_flagged():
    inside = {
        "direction_layer": 12,
        "attn.strength": 1.45,
        "attn.peak_layer": 0.63,
        "attn.edge_strength": 0.5,
        "attn.reach": 0.3,
    }
    assert search.at_edges(inside, config.Search(), 28) == []


def test_an_end_short_of_how_far_a_setting_can_go_is_flagged():
    space = config.Search(peak_layer=(0.5, 0.9), edge_strength=(0.2, 0.8), reach=(0.1, 0.6))
    low = {"attn.strength": 1.2, "attn.peak_layer": 0.5, "attn.edge_strength": 0.2, "attn.reach": 0.1}
    high = {"attn.strength": 1.2, "attn.peak_layer": 0.9, "attn.edge_strength": 0.8, "attn.reach": 0.6}
    assert search.at_edges(low, space, 28) == [
        "attn.edge_strength 0.2, at its floor 0.2",
        "attn.peak_layer 0.5, at its floor 0.5",
        "attn.reach 0.1, at its floor 0.1",
    ]
    assert search.at_edges(high, space, 28) == [
        "attn.edge_strength 0.8, at its ceiling 0.8",
        "attn.peak_layer 0.9, at its ceiling 0.9",
        "attn.reach 0.6, at its ceiling 0.6",
    ]


def test_an_end_that_is_already_the_models_own_isnt_flagged():
    space = config.Search(direction_layer=(0.0, 1.0), peak_layer=(0.0, 1.0), reach=(0.01, 1.0))
    pressed = {"direction_layer": 0, "attn.strength": 1.2, "attn.peak_layer": 0.0, "attn.reach": 1.0}
    assert search.at_edges(pressed, space, 28) == []
    assert search.at_edges(pressed | {"direction_layer": 27}, space, 28) == []


def test_a_search_holding_only_one_kinds_study_still_loads(tiny):
    folder = tiny.path("search")
    folder.mkdir()
    optuna.create_study(study_name=search.STUDIES[search.ONE_LAYER], storage=search._storage(folder))
    found = search.load(tiny)
    assert found is not None and found.trials == {}


def test_the_report_names_each_kinds_best_at_each_count_its_edges_and_what_to_measure(tiny, monkeypatch):
    small(tiny, trials=3)
    tiny.settings.search.strength = (1.0, 1.0)  # every trial presses on the strength's ends
    folder = on_disk(tiny)
    scripted(
        monkeypatch,
        [
            Score(3, 0, 0, 0.0, 3),  # the unedited model
            Score(0, 0, 0, 0.30, 3),  # 1, one layer: refuses none
            Score(1, 0, 0, 0.25, 3),  # 2, per layer
            Score(2, 0, 0, 0.20, 3),  # 3, one layer: less KL, refusing more
            Score(3, 0, 0, 0.10, 3),  # 4, per layer: over the cap
            Score(1, 0, 0, 0.35, 3),  # 5, one layer: trial 1 beats it
            Score(2, 0, 0, 0.26, 3),  # 6, per layer: trial 2 beats it
        ],
    )
    result = CliRunner(env={"COLUMNS": "200"}).invoke(app, ["search", str(folder)])
    assert result.exit_code == 0, result.output
    shown = flat(result.output)
    found = search.load(run.Run.open(folder))
    assert found is not None
    assert "Fully scored trials no other beats on both refusals and KL" in shown
    for c in search.candidates(found):
        row = rf"^ *{c.trial} +{c.direction} +{c.refusals} +{c.keyword_refusals} +{c.kl:.4f} *$"
        assert re.search(row, result.output, re.M)
    picks = {scope: search.to_measure(found, scope, config.Search()) for scope in search.SCOPES}
    assert {scope: [c.trial for c in kept] for scope, kept in picks.items()} == {
        search.ONE_LAYER: [1, 3],
        search.PER_LAYER: [2],
    }
    for scope, kept in picks.items():
        for c in kept:
            assert f"{scope}: trial {c.trial}, {c.direction}, {c.refusals} refused, KL {c.kl:.4f}" in shown
    assert "one layer: trial 5" not in shown and "per layer: trial 6" not in shown
    commands = [line.split() for line in result.output.splitlines() if "candor measure" in line]
    assert [int(words[-1]) for words in commands] == [1, 3, 2]
    assert shown.count("It presses on the edge of the search's ranges (") == 3
    assert shown.count("attn.strength 1, at its floor 1; attn.strength 1, at its ceiling 1") == 3


def test_the_edge_alarm_waits_for_the_trials_the_search_aims(tiny):
    small(tiny)
    tiny.settings.search.startup_per_scope = 2
    tiny.settings.search.strength = (1.0, 1.0)
    folder = on_disk(tiny)
    result = CliRunner(env={"COLUMNS": "200"}).invoke(app, ["search", str(folder)])
    assert result.exit_code == 0, result.output
    assert "It presses on the edge" not in flat(result.output)


def test_with_nothing_within_the_cap_the_fewest_refused_of_each_kind_is_reported(tiny):
    small(tiny)
    tiny.settings.search.max_refusals = 0
    folder = on_disk(tiny)
    result = CliRunner(env={"COLUMNS": "200"}).invoke(app, ["search", str(folder)])
    assert result.exit_code == 0, result.output
    shown = flat(result.output)
    found = search.load(run.Run.open(folder))
    assert found is not None
    expected = []
    for scope in search.SCOPES:
        fewest = search.fewest_refused(found, scope)
        assert fewest is not None
        assert f"{scope}: none refused 0 or fewer; the fewest refused: trial {fewest.trial}," in shown
        expected.append(fewest.trial)
    commands = [line.split() for line in result.output.splitlines() if "candor measure" in line]
    assert [int(words[-1]) for words in commands] == expected


def test_scoring_a_trial_stopped_early_says_so(tiny, monkeypatch):
    folder = cli_run(tiny, trials=1)
    scripted(monkeypatch, [Score(1, 0, 0, 0.0, 3), Score(3, 0, 0, math.inf, 1), Score(1, 0, 0, 0.2, 3)])
    runner = CliRunner(env={"COLUMNS": "200"})
    assert runner.invoke(app, ["search", str(folder)]).exit_code == 0
    based = []
    monkeypatch.setattr(
        Search,
        "score",
        lambda self, edits, base=True: based.append(base) or search.Scores(3, 1, [Score(3, 0, 0, 0.1, 3)]),
    )
    result = runner.invoke(app, ["score", str(folder), "--trial", "1"])
    assert result.exit_code == 0, result.output
    shown = flat(result.output)
    assert "trial-1 was stopped early in the search, after 1 prompts" in shown
    # With no full score to repeat, the unedited model is scored again as the check.
    assert based == [True] and "The unedited model refuses 1 of 3 (1 in the search)." in shown
    assert "scores differently" not in shown
    # Its KL in the search was never read, so the table doesn't show one.
    assert "3 of 1, stopped early" in shown and "inf" not in shown
    # Said too when no trial of the search was scored in full.
    monkeypatch.setattr(search, "candidates", lambda found: [])
    shown = flat(runner.invoke(app, ["score", str(folder), "--trial", "1"]).output)
    assert "trial-1 was stopped early in the search" in shown
    assert "The search has no trial scored in full yet to set these beside." in shown


def test_the_search_scores_a_thinking_model_at_its_own_budget_with_the_same_seeds(tiny, monkeypatch):
    thinks_through(monkeypatch)
    small(thinking(tiny, budget=8))
    tiny.settings.scoring.thinking_budget = 6
    asked, first_tokens = [], []
    real, real_top = LlamaServer.answers, LlamaServer.top_logprobs

    def recording(self, *args, **kwargs):
        asked.append(args)
        yield from real(self, *args, **kwargs)

    def reading(self, prompts, *args, **kwargs):
        first_tokens.append(prompts)
        return real_top(self, prompts, *args, **kwargs)

    monkeypatch.setattr(LlamaServer, "answers", recording)
    monkeypatch.setattr(LlamaServer, "top_logprobs", reading)
    Search(tiny).search(2)
    assert {args[5].budget for args in asked} == {6} and all(args[4] == asked[0][4] for args in asked)
    assert all(args[3]["temperature"] == 1.0 for args in asked)
    ids = tiny.prompt_ids()
    # The unedited model's first tokens: after its own thinking, where it starts, after an empty block.
    assert first_tokens[:3] == [
        tiny.kl_ids(ids, "harmless_test"),
        ids["harmless_test"],
        ids["harmless_test" + run.ANSWERING],
    ]
    inputs = json.loads(tiny.path("search", "inputs.json").read_text())
    assert inputs["thinking"]["budget"] == 6 and inputs["sampling"]["temperature"] == 1.0
    scores = [search._score(trial) for trial in finished(tiny)]
    assert sum(score.thinking for score in scores) > 0 and all(0 <= s.forced <= s.judged for s in scores)
    assert all(score.kl_thinking is not None and score.kl_empty is not None for score in scores)
    folder = on_disk(tiny)
    shown = flat(
        CliRunner(env={"COLUMNS": "200"}).invoke(app, ["search", str(folder), "--trials", "1"]).output
    )
    assert (
        "tokens per answer on average across trials" in shown and "reached the search's budget of 6" in shown
    )
    for where in (
        "where the answer starts after the unedited model's own thinking, which the search aims at",
        "after an empty thinking block",
        "where the thinking starts",
    ):
        assert where in shown


def test_a_thinking_search_draws_where_each_trial_reads_its_directions(tiny, monkeypatch):
    thinks_through(monkeypatch)
    small(thinking(tiny), trials=3)
    Search(tiny).search(6)
    drawn = [trial.params["position"] for trial in finished(tiny)]
    assert len(drawn) == 6 and set(drawn) <= set(search.POSITIONS)
    for scope in search.SCOPES:
        start = search.spread(scope, config.Search(), 28, ["attn", "mlp"], 20, 7, search.POSITIONS)
        assert [p["position"] for p in start] == list(search.POSITIONS) * 10


@pytest.mark.parametrize("count", [20, 10, 5, 1])
def test_a_thinking_models_spread_is_split_evenly_each_position_spread_on_its_own(count):
    space = config.Search()
    for scope in search.SCOPES:
        start = search.spread(scope, space, 28, ["attn", "mlp"], count, 7, search.POSITIONS)
        # Taken in turn, so the first trials of a pilot try both positions.
        assert [p["position"] for p in start] == [search.POSITIONS[i % 2] for i in range(count)]
        for j, position in enumerate(search.POSITIONS):
            own = [p for p in start if p["position"] == position]
            own = [{k: v for k, v in p.items() if k != "position"} for p in own]
            assert own == search.spread(scope, space, 28, ["attn", "mlp"], len(own), 7 * 2 + j)


def test_a_trial_reading_where_the_answer_starts_uses_the_adapters_read_there(tiny, monkeypatch):
    thinks_through(monkeypatch)
    thinking(tiny)
    adapters = tiny.adapters()
    params = {f"{c}.{s}": 0.5 for c in ("attn", "mlp") for s in search.SETTINGS}
    for position, prefix in ((search.THINK_START, ""), (search.ANSWER_START, run.ANSWER_SETS)):
        per_layer = search.edit_from({**params, "position": position}, 1, 2)
        one = search.edit_from({**params, "position": position, "direction_layer": 1}, 2, 2)
        assert per_layer.answering == one.answering == (position == search.ANSWER_START)
        found = Search(tiny)
        assert found._edit_files(per_layer, adapters) == adapters[prefix + "per-layer"]
        assert found._edit_files(one, adapters) == adapters[prefix + "layer-1"]
    assert search.direction(None, True) == "per layer, answer start"
    assert search.direction(12, False) == "layer 12"


def test_a_model_that_doesnt_think_is_searched_and_recorded_as_before(tiny):
    tiny.gguf()
    tiny.directions()
    # Its adapters are recorded as they were before directions were read at two places.
    assert search._rebuilt_from(tiny)["adapters"] == run.digest(tiny.adapter_inputs())
    assert Search(tiny).positions == ()
    trial = optuna.create_study().ask()
    search.suggest(trial, config.Search(), 4, ["attn"], search.PER_LAYER)
    assert "position" not in trial.params
    start = search.spread(search.ONE_LAYER, config.Search(), 4, ["attn"], 5, 0)
    assert all("position" not in params for params in start)
    assert search.edit_from(start[0], 1, 4).answering is False


def test_a_search_after_other_unedited_thinking_is_refused_before_any_is_sampled(tiny, monkeypatch):
    thinks_through(monkeypatch)
    small(thinking(tiny, budget=8))
    Search(tiny).search(1)
    sampled = []
    real = LlamaServer.thoughts
    monkeypatch.setattr(LlamaServer, "thoughts", lambda *a, **k: sampled.append(1) or real(*a, **k))
    tiny.settings.deployment.thinking_budget = 9
    # Refused on the budget it thinks to, before the unedited thinking is even named.
    with pytest.raises(RunError, match="a different thinking; move it aside"):
        Search(tiny).search(2)
    with pytest.raises(RunError, match="made with different thinking from this run's now"):
        Search(tiny).score([search.edit_from(finished(tiny)[0].params, 1, 2)])
    assert sampled == []
    tiny.settings.deployment.thinking_budget = 8
    # Or on the unedited thinking alone, as with it sampled at other slots.
    tiny.settings.slots.thinking = 3
    with pytest.raises(RunError, match="unedited thinking"):
        Search(tiny).search(2)
    assert sampled == []
    tiny.settings.slots.thinking = config.Slots().thinking
    Search(tiny).search(2)
    assert sampled == [] and len(finished(tiny)) == 2


def test_a_search_isnt_carried_on_after_thinking_sampled_afresh_from_the_same_settings(tiny, monkeypatch):
    thinks_through(monkeypatch)
    small(thinking(tiny, budget=8))
    Search(tiny).search(1)
    key = tiny.thinking_key()
    tiny.path("private", "thinking", f"{key}.json").unlink()
    tiny.path("thinking", f"{key}.json").unlink()

    def otherwise(self, prompts, sampling, seeds, thinking, done=None):
        return [[*self.tokenize("w7"), *self.tokenize(thinking.end)]] * len(prompts)

    monkeypatch.setattr(LlamaServer, "thoughts", otherwise)
    with pytest.raises(RunError, match="different KL prompts"):
        Search(tiny).search(2)
    with pytest.raises(RunError, match="different KL prompts"):
        Search(tiny).score([search.edit_from(finished(tiny)[0].params, 1, 2)])
    assert len(finished(tiny)) == 1


def test_a_search_whose_unedited_thinking_fails_leaves_nothing_to_move_aside(tiny, monkeypatch):
    monkeypatch.setattr(LlamaServer, "thoughts", lambda self, prompts, *a, **k: [None] * len(prompts))
    small(thinking(tiny, budget=8))
    with pytest.raises(RunError, match=r"raise deployment\.thinking_budget"):
        Search(tiny).search(1)
    assert not tiny.path("search", "inputs.json").exists() and not tiny.path("search", "study.jsonl").exists()
    thinks_through(monkeypatch)
    tiny.settings.deployment.thinking_budget = 9
    Search(tiny).search(1)
    assert len(finished(tiny)) == 1


@pytest.mark.parametrize("thinks", ["on", "off"])
def test_kl_after_long_unedited_thinking_fits_the_scorers_context(tiny, monkeypatch, thinks):
    # The deployment's budget, which the unedited model thinks to, is longer than the search's.
    # Long enough to need more than the room for a harmful prompt's answer, short enough for the tiny
    # model's 256 tokens.
    thinks_through(monkeypatch, words=15)
    small(thinking(tiny, budget=64))
    tiny.settings.scoring.thinking_budget = 6
    tiny.settings.scoring.thinking = thinks
    tiny.settings.scoring.closed_answer_tokens = 4
    asked = []
    real = run.Run.server
    monkeypatch.setattr(run.Run, "server", lambda self, *a, **k: asked.append(k) or real(self, *a, **k))
    Search(tiny).search(1)
    ids = tiny.prompt_ids()
    longest = max(len(p) for p in tiny.kl_ids(ids, "harmless_test"))
    if thinks == "on":
        answering = (
            max(map(len, ids["harmful_test"])) + tiny.settings.scoring.answer_tokens + tiny.thinking_room(6)
        )
    else:
        # Answers after an empty thinking block, as long as scoring.closed_answer_tokens.
        answering = max(map(len, ids["harmful_test" + run.ANSWERING])) + 4
    assert longest > answering + 8
    # llama-server pads a small model's slots, so the room asked for is checked rather than the answers.
    # A slot holds an answer or a KL read at a time, so it needs room for the larger, not both.
    assert asked[-1]["context_per_slot"] == max(answering, longest + 1) + 8


def test_the_unedited_thinking_is_sampled_as_deployed_each_prompt_with_its_seed(tiny, monkeypatch):
    asked = []
    thinks_through(monkeypatch, asked=asked)
    thinking(tiny, budget=8)
    tiny.settings.scoring.thinking_budget = 4
    tiny.settings.sampling.seed = 5
    said = []
    tiny.report = said.append
    tiny.thinking_traces()
    ids = tiny.prompt_ids()
    assert [prompts for prompts, *_ in asked] == [ids[name] for name in run.KL_SETS]
    for prompts, sampling, drawn, how in asked:
        assert sampling == tiny.sampling() and sampling["temperature"] == 1.0
        assert drawn == run.seeds(5, len(prompts))
        assert how.budget == 8
    assert [line for line in said if line.startswith("Thought through")] == [
        f"Thought through {name.replace('_', ' ')} prompts: {count} of 3"
        for name in run.KL_SETS
        for count in (1, 2, 3)
    ]


def test_the_unedited_model_thinks_slots_thinking_at_once_recorded_as_before_at_8(tiny, monkeypatch):
    thinks_through(monkeypatch)
    thinking(tiny)
    started = []
    real = LlamaServer.start

    def start(self, *args, **kwargs):
        started.append((self.parallel, self.slots_setting))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(LlamaServer, "start", start)
    for slots in (8, 3):
        tiny.settings.slots.thinking = slots
        tiny.thinking_traces()
        assert started[-1] == (slots, "slots.thinking in candor.toml")
        record = run.record_of(tiny.path("thinking", f"{tiny.thinking_key()}.json"))
        assert json.loads(record.read_text())["slots"] == slots
    # Thinking sampled at other slots is kept apart, so going back finds the first again.
    assert len(list(tiny.path("thinking").glob("*.json.inputs.json"))) == 2


def test_run_reports_how_many_prompts_the_unedited_model_thought_through(tiny, monkeypatch):
    thinks_through(monkeypatch)
    done = thinking(tiny).prepare()
    assert done["of 3 harmless test prompts thought through"] == 2
    assert done["of 3 harmless held out prompts thought through"] == 2
    assert done["direction sets"] == 2 * len(tiny.direction_sets(tiny.directions(), tiny.shape[1]))


def test_a_trial_is_rebuilt_without_thinking_again(tiny, monkeypatch):
    thinks_through(monkeypatch)
    small(thinking(tiny))
    Search(tiny).search(1)
    monkeypatch.setattr(run.Run, "thinking_traces", lambda self: pytest.fail("thought again"))
    [trial] = finished(tiny)
    edit = search.trial_edit(tiny, trial.user_attrs["number"])
    assert edit.answering == (trial.params["position"] == search.ANSWER_START)


def test_a_sampled_search_isnt_carried_on_with_another_seed_and_a_greedy_one_is(tiny):
    small(tiny)
    tiny.settings.sampling.seed = 3
    Search(tiny).search(1)  # greedy: the seed changes nothing, so it isn't recorded
    tiny.settings.sampling.seed = 4
    Search(tiny).search(2)
    shutil.rmtree(tiny.path("search"))
    tiny.settings.sampling.temperature = 0.7
    Search(tiny).search(1)
    tiny.settings.sampling.seed = 5
    with pytest.raises(RunError, match="made with a different sampling"):
        Search(tiny).search(2)


def widening_soon(monkeypatch):
    """Check a kind's strength range from its 2nd finished trial and every 2nd after, starting a widened
    range with 2 spread points, so a tiny search widens."""
    monkeypatch.setattr(search, "WIDEN_FROM", 2)
    monkeypatch.setattr(search, "WIDEN_EVERY", 2)
    monkeypatch.setattr(search, "WIDENED_START", 2)


def refusing(monkeypatch, refusals=3, stop_at=None):
    """Every edit refuses this many of the 3 test prompts, with KL growing call by call; the stop_at-th
    call, counting the unedited model's, is stopped part way."""
    calls = []

    def score(self, scales, stop_above=None, forced_above=None, read_kl=True):
        calls.append(scales)
        if len(calls) == stop_at:
            raise KeyboardInterrupt
        return Score(refusals, 0, 0, 0.01 * len(calls), 3)

    monkeypatch.setattr(search._Scorer, "score", score)


def latest(run, scope) -> optuna.Study:
    storage = search._storage(run.path("search"))
    name = search._latest(storage, scope)
    assert name is not None
    return optuna.load_study(study_name=name, storage=storage)


def kind_spread(run, scope, strength, count, seed):
    space = run.settings.search.model_copy(update={"strength": strength})
    _, layers, _, writers = run.shape
    return search.spread(scope, space, layers, sorted({w.component for w in writers}), count, seed)


def test_with_no_edit_within_the_cap_a_kinds_strength_ceiling_widens_and_its_trials_go_on(tiny, monkeypatch):
    widening_soon(monkeypatch)
    refusing(monkeypatch)
    small(tiny, trials=6)
    said = []
    tiny.report = said.append
    found, _ = Search(tiny).search(12)
    storage = search._storage(tiny.path("search"))
    names = set(optuna.get_all_study_names(storage))
    seed = tiny.settings.search.seed
    for i, scope in enumerate(search.SCOPES):
        first = optuna.load_study(study_name=search.STUDIES[scope], storage=storage)
        study = latest(tiny, scope)
        # Checked again at 4, where the ceiling is already at its widest.
        assert study.study_name == search._study_name(scope, 1) and search._study_name(scope, 2) not in names
        assert study.user_attrs["widened"] == [[2, 0.8, 3.0]]
        old = first.get_trials(states=(TrialState.COMPLETE,))
        new = study.get_trials(states=(TrialState.COMPLETE,))
        assert len(old) == 2 and len(new) == 6
        for a, b in zip(old, new, strict=False):
            assert (a.params, a.user_attrs, a.system_attrs, a.value) == (
                b.params,
                b.user_attrs,
                b.system_attrs,
                b.value,
            )
        # The wider range starts with its own spread, then the sampler draws from it.
        assert [t.params for t in new[2:4]] == kind_spread(tiny, scope, (0.8, 3.0), 2, seed + 2 * 2 + i)
        for trial in new:
            for name, distribution in trial.distributions.items():
                if name.endswith(".strength"):
                    assert distribution == FloatDistribution(0.8, 3.0)
        # Every trial shares strength's range, so the sampler learns it with the other settings.
        joint = optuna.search_space.IntersectionSearchSpace().calculate(study)
        assert {"attn.strength", "mlp.strength"} <= set(joint)
    assert latest(tiny, search.ONE_LAYER).user_attrs["base_refusals"] == 3
    assert sorted(found.trials) == list(range(1, 13))
    loaded = search.load(tiny)
    assert loaded is not None
    assert found.strength == loaded.strength == {scope: (0.8, 3.0) for scope in search.SCOPES}
    for number in (1, 12):
        search.trial_edit(tiny, number)
    assert [line for line in said if "widens" in line] == [
        "After 2 one layer trials no one layer edit is within the cap yet, so its strength widens from "
        "0.8-1.5 to 0.8-3",
        "After 2 per layer trials no per layer edit is within the cap yet, so its strength widens from "
        "0.8-1.5 to 0.8-3",
    ]


def test_a_search_stopped_after_widening_goes_on_in_the_wider_study_without_checking_again(tiny, monkeypatch):
    widening_soon(monkeypatch)
    small(tiny, trials=6)
    checked = []
    real = search.widened

    def checking(found, scope, space):
        checked.append((scope, len(found.trials)))
        return real(found, scope, space)

    monkeypatch.setattr(search, "widened", checking)
    # The unedited model, trials 1-4, then the first of the wider range is stopped part way.
    refusing(monkeypatch, stop_at=6)
    with pytest.raises(KeyboardInterrupt):
        Search(tiny).search(12)
    assert checked == [(search.ONE_LAYER, 2)]
    refusing(monkeypatch)
    Search(tiny).search(12)
    assert checked == [
        (search.ONE_LAYER, 2),
        (search.PER_LAYER, 2),
        (search.ONE_LAYER, 4),
        (search.PER_LAYER, 4),
    ]
    study = latest(tiny, search.ONE_LAYER)
    assert study.user_attrs["widened"] == [[2, 0.8, 3.0]]
    drawn = {search._key(t.params) for t in study.get_trials(states=(TrialState.COMPLETE,))}
    start = kind_spread(tiny, search.ONE_LAYER, (0.8, 3.0), 2, tiny.settings.search.seed + 4)
    assert {search._key(params) for params in start} <= drawn
    assert len(drawn) == 6


def test_a_widened_study_left_part_made_is_ignored_and_made_again(tiny, monkeypatch):
    widening_soon(monkeypatch)
    refusing(monkeypatch)
    small(tiny, trials=6)
    Search(tiny).search(4)
    storage = search._storage(tiny.path("search"))
    part = optuna.create_study(study_name=search._study_name(search.ONE_LAYER, 1), storage=storage)
    part.set_user_attr("base_refusals", 3)
    found = search.load(tiny)
    assert found is not None and len(found.trials) == 4 and found.strength == {}
    Search(tiny).search(12)
    study = latest(tiny, search.ONE_LAYER)
    assert study.study_name == search._study_name(search.ONE_LAYER, 1)
    assert study.user_attrs["widened"] == [[2, 0.8, 3.0]]
    assert len(study.get_trials(states=(TrialState.COMPLETE,))) == 6
    assert sorted(finished(tiny), key=lambda t: t.user_attrs["number"])[-1].user_attrs["number"] == 12


def test_a_search_that_may_not_widen_never_does_and_records_nothing_of_it(tiny, monkeypatch):
    widening_soon(monkeypatch)
    refusing(monkeypatch)
    small(tiny, trials=6)
    tiny.settings.search.widest_strength = tiny.settings.search.strength
    found, _ = Search(tiny).search(12)
    assert found.strength == {}
    assert sorted(optuna.get_all_study_names(search._storage(tiny.path("search")))) == sorted(
        search.STUDIES.values()
    )
    recorded = json.loads(tiny.path("search", "inputs.json").read_text())["search settings"]
    assert "widest_strength" not in recorded


def test_a_search_that_may_widen_records_how_far(tiny):
    small(tiny)
    Search(tiny).search(1)
    recorded = json.loads(tiny.path("search", "inputs.json").read_text())["search settings"]
    assert recorded["widest_strength"] == [0.5, 3.0]


def at_strengths(number, refusals, kl, attn, mlp, judged=3):
    """A per layer trial with these strengths, as the search records it."""
    params = {"attn.strength": attn, "mlp.strength": mlp}
    return optuna.trial.create_trial(
        params=params,
        distributions={name: FloatDistribution(0, 5) for name in params},
        value=kl,
        user_attrs=made(number, refusals, kl, judged=judged).user_attrs,
    )


@pytest.mark.parametrize(
    ("trials", "strength"),
    [
        ([at_strengths(1, 0, 0.1, 1.49, 1.0)], (0.8, 3.0)),
        ([at_strengths(1, 2, 0.1, 1.0, 0.83)], (0.5, 1.5)),
        ([at_strengths(1, 0, 0.1, 1.47, 0.81)], (0.5, 3.0)),
        ([at_strengths(1, 0, 0.1, 1.0, 1.2)], (0.8, 1.5)),
        # Only the best within the cap counts.
        ([at_strengths(1, 0, 0.1, 1.0, 1.2), at_strengths(2, 0, 0.2, 1.5, 0.8)], (0.8, 1.5)),
        ([at_strengths(1, 3, 0.05, 1.5, 0.8), at_strengths(2, 1, 0.2, 1.0, 1.2)], (0.8, 1.5)),
        # Just further than EDGE of the range from either end.
        ([at_strengths(1, 0, 0.1, 1.46, 0.84)], (0.8, 1.5)),
        # None within the cap: the edits are too weak.
        ([at_strengths(1, 3, 0.1, 1.0, 1.0)], (0.8, 3.0)),
        ([at_strengths(1, 0, 0.1, 1.0, 1.0, judged=1)], (0.8, 3.0)),
        ([], (0.8, 3.0)),
    ],
)
def test_strength_widens_at_each_end_the_best_edit_within_the_cap_presses(trials, strength):
    found = search.Found({t.user_attrs["number"]: t for t in trials}, 99)
    assert search.widened(found, search.PER_LAYER, config.Search()) == strength


def test_an_end_of_strength_already_at_its_widest_stays():
    space = config.Search(strength=(0.5, 1.5), widest_strength=(0.5, 3.0))
    found = search.Found({1: at_strengths(1, 0, 0.1, 0.51, 1.0)}, 99)
    assert search.widened(found, search.PER_LAYER, space) == (0.5, 1.5)
    assert not config.Search(widest_strength=(0.8, 1.5)).widens() and config.Search().widens()


def test_each_kind_is_judged_by_its_own_widened_strength():
    space = config.Search()
    found = search.Found({}, 1, {search.ONE_LAYER: (0.8, 3.0)})
    assert search.kind_space(space, found, search.ONE_LAYER).strength == (0.8, 3.0)
    assert search.kind_space(space, found, search.PER_LAYER) is space
    assert search.kind_space(space, None, search.ONE_LAYER) is space
    edit = Edit("strong", 12, {"attn": Kernel(2.0, 20, 1.0, 5)})
    assert search.outside_space(edit, space, 28, ["attn"]) == ["attn.strength 2 (0.8-1.5)"]
    assert search.outside_space(edit, search.kind_space(space, found, search.ONE_LAYER), 28, ["attn"]) == []


def test_the_report_shows_a_widened_strength_and_judges_its_edges_by_it(tiny, monkeypatch):
    widening_soon(monkeypatch)
    refusing(monkeypatch)
    small(tiny, trials=4)
    tiny.settings.search.strength = (1.0, 1.0)  # every trial at 1, on both ends until the ceiling widens
    tiny.settings.search.widest_strength = (1.0, 3.0)
    folder = on_disk(tiny)
    result = CliRunner(env={"COLUMNS": "200"}).invoke(app, ["search", str(folder)])
    assert result.exit_code == 0, result.output
    shown = flat(result.output)
    assert "After 2 one layer trials no one layer edit is within the cap yet" in shown
    assert shown.count("the search widened strength to 1-3") == 2
    # The fewest refused of each kind is its first trial, at strength 1: on the floor, far from 3.
    assert shown.count("attn.strength 1, at its floor 1") == 2
    assert "at its ceiling 1" not in shown


def test_score_says_whether_the_search_could_have_drawn_an_edit_from_its_widened_strength(tiny, monkeypatch):
    widening_soon(monkeypatch)
    refusing(monkeypatch)
    folder = cli_run(tiny, trials=4)
    journal = write_journal(tiny.folder.parent / "heretic.jsonl", 0.25)
    journal.write_text(journal.read_text().replace('"max_weight": 1.2', '"max_weight": 2.0'))
    runner = CliRunner(env={"COLUMNS": "200"})
    # Two trials of each kind: not yet checked for widening.
    assert runner.invoke(app, ["search", str(folder), "--trials", "4"]).exit_code == 0
    before = runner.invoke(app, ["score", str(folder), "--heretic", str(journal), "--trial", "1"])
    assert before.exit_code == 0, before.output
    assert "heretic-1 at layer 0: " in flat(before.output)
    assert "couldn't have drawn it: attn.strength 2 (0.8-1.5)." in flat(before.output)
    assert runner.invoke(app, ["search", str(folder)]).exit_code == 0
    after = runner.invoke(app, ["score", str(folder), "--heretic", str(journal), "--trial", "1"])
    assert after.exit_code == 0, after.output
    assert "attn.strength" not in flat(after.output)


def test_strength_isnt_checked_for_widening_until_the_start_spread_is_done(tiny, monkeypatch):
    widening_soon(monkeypatch)
    refusing(monkeypatch)
    small(tiny, trials=6)
    tiny.settings.search.startup_per_scope = 3
    checked = []
    real = search.widened

    def checking(found, scope, space):
        checked.append((scope, len(found.trials)))
        return real(found, scope, space)

    monkeypatch.setattr(search, "widened", checking)
    Search(tiny).search(12)
    assert checked == [(search.ONE_LAYER, 4), (search.PER_LAYER, 4)]


def test_a_search_killed_while_widening_loses_and_repeats_no_trial(tiny, monkeypatch):
    widening_soon(monkeypatch)
    refusing(monkeypatch)
    small(tiny, trials=6)
    real = optuna.Study.add_trial
    copied = []

    def dying(self, trial):
        if self.study_name.endswith("widened 1") and trial.state == TrialState.COMPLETE:
            copied.append(trial)
            if len(copied) == 2:
                raise KeyboardInterrupt
        real(self, trial)

    monkeypatch.setattr(optuna.Study, "add_trial", dying)
    with pytest.raises(KeyboardInterrupt):
        Search(tiny).search(12)
    monkeypatch.setattr(optuna.Study, "add_trial", real)
    Search(tiny).search(12)
    found = search.load(tiny)
    assert found is not None and sorted(found.trials) == list(range(1, 13))
    for scope in search.SCOPES:
        study = latest(tiny, scope)
        assert study.user_attrs["widened"] == [[2, 0.8, 3.0]]
        done = [search._key(t.params) for t in study.get_trials(states=(TrialState.COMPLETE,))]
        assert len(done) == len(set(done)) == 6
        waiting = study.get_trials(states=(TrialState.WAITING,))
        assert not {search._key(t.system_attrs["fixed_params"]) for t in waiting} & set(done)


def test_a_thinking_search_with_thinking_off_answers_greedily_after_an_empty_block(tiny, monkeypatch):
    thinks_through(monkeypatch)
    small(thinking(tiny, budget=8))
    tiny.settings.scoring.thinking = "off"
    tiny.settings.scoring.closed_answer_tokens = 12
    asked, first_tokens = [], []
    real, real_top = LlamaServer.answers, LlamaServer.top_logprobs

    def recording(self, *args, **kwargs):
        asked.append(args)
        yield from real(self, *args, **kwargs)

    def reading(self, prompts, *args, **kwargs):
        first_tokens.append(prompts)
        return real_top(self, prompts, *args, **kwargs)

    monkeypatch.setattr(LlamaServer, "answers", recording)
    monkeypatch.setattr(LlamaServer, "top_logprobs", reading)
    Search(tiny).search(2)
    ids = tiny.prompt_ids()
    assert all(args[0] == ids["harmful_test" + run.ANSWERING] for args in asked)
    assert all(args[1] == 12 and args[3] == GREEDY and args[5] is None for args in asked)
    # KL is still read where the answer starts after the unedited model's own thinking.
    assert first_tokens[0] == tiny.kl_ids(ids, "harmless_test")
    scores = [search._score(trial) for trial in finished(tiny)]
    assert all(s.thinking == 0 and s.forced == 0 and s.kl_thinking is not None for s in scores)
    inputs = json.loads(tiny.path("search", "inputs.json").read_text())
    assert inputs["trials think"] == "off" and inputs["answer length with thinking off"] == 12
    # A search scored with thinking off isn't carried on as one that thinks.
    tiny.settings.scoring.thinking = "on"
    with pytest.raises(RunError, match="trials think"):
        Search(tiny).search(4)


def test_a_model_whose_template_cant_turn_thinking_off_is_scored_thinking_and_never_with_it_off(
    tiny, monkeypatch
):
    thinks_through(monkeypatch)
    small(thinking(tiny, budget=8, template=ALWAYS_THINKING_TEMPLATE, close=END_THINK))
    tiny.settings.scoring.thinking = "off"
    with pytest.raises(RunError, match="can't turn its thinking off"):
        Search(tiny).search(2)
    # Refused before anything is sampled or recorded, so a search scored thinking can start.
    assert not tiny.path("search").exists() and not tiny.path("thinking").exists()
    with pytest.raises(RunError, match="can't turn its thinking off"):
        Search(tiny).score([Edit("e", None, {"mlp": Kernel(1.0, 1.0, 0.0, 0.5)})], thinks=False)
    # candor proxy needs a search scored with thinking off, which there can't be.
    with pytest.raises(RunError, match="run candor search with --trials 150 first"):
        proxy.check(tiny)
    tiny.settings.scoring.thinking = "on"
    Search(tiny).search(2)
    assert sum(search._score(trial).thinking for trial in finished(tiny)) > 0


def test_a_model_whose_template_changes_more_than_the_close_is_never_scored_with_thinking_off(
    tiny, monkeypatch
):
    thinks_through(monkeypatch)
    small(thinking(tiny, budget=8, template=MESSAGE_WHEN_THINKING_TEMPLATE))
    tiny.settings.scoring.thinking = "off"
    with pytest.raises(RunError, match="changes more than the end of a prompt when it turns thinking off"):
        Search(tiny).search(2)
    assert not tiny.path("search").exists()


def test_a_search_already_scored_with_thinking_off_starts_no_server_to_check_it(tiny, monkeypatch):
    thinks_through(monkeypatch)
    small(thinking(tiny, budget=8))
    tiny.settings.scoring.thinking = "off"
    tiny.settings.scoring.closed_answer_tokens = 12
    started = []
    real = run.Run.server

    def server(self, *args, **kwargs):
        started.append(kwargs.get("parallel"))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(run.Run, "server", server)
    Search(tiny).search(2)
    # A new search checks once that the template turns thinking off, with a server of one slot.
    assert started[0] == 1
    started.clear()
    Search(tiny).search(2)
    assert started == []
    Search(tiny).score([Edit("e", None, {"mlp": Kernel(1.0, 1.0, 0.0, 0.5)})])
    assert started == [config.THINKING_MODEL_SLOTS]


@pytest.mark.parametrize("thinks", [False, True])
@pytest.mark.parametrize("slots", [None, 3])
def test_scoring_answers_a_thinking_models_trials_many_at_once_unless_set(tiny, monkeypatch, thinks, slots):
    thinks_through(monkeypatch)
    started = []
    real = LlamaServer.start

    def start(self, *args, **kwargs):
        started.append((self.parallel, self.slots_setting))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(LlamaServer, "start", start)
    small(thinking(tiny) if thinks else tiny)
    tiny.settings.slots.search = slots
    Search(tiny).search(1)
    used = slots or (config.THINKING_MODEL_SLOTS if thinks else config.SEARCH_SLOTS)
    # The scorer's server is the last one the search starts.
    assert started[-1] == (used, "slots.search in candor.toml")
    recorded = json.loads(tiny.path("search", "inputs.json").read_text())
    # A model that doesn't think records its slots only where they aren't the number used before.
    assert recorded.get("slots") == (used if thinks or slots else None)


@pytest.mark.parametrize("thinks", [False, True])
def test_a_search_carries_on_with_its_slots_set_to_the_number_it_used_unset(tiny, monkeypatch, thinks):
    thinks_through(monkeypatch)
    small(thinking(tiny) if thinks else tiny)
    Search(tiny).search(1)
    tiny.settings.slots.search = config.THINKING_MODEL_SLOTS if thinks else config.SEARCH_SLOTS
    Search(tiny).search(2)
    assert len(finished(tiny)) == 2
    tiny.settings.slots.search = 3
    with pytest.raises(RunError, match="holds a search made with a different slots"):
        Search(tiny).search(3)


def test_a_full_edits_trials_each_restart_the_server_with_an_adapter_built_for_them(tiny, monkeypatch):
    small(tiny, trials=2)
    tiny.settings.edit.row_norm = "full"
    started, running, beside = [], set(), []
    real_start, real_stop, real_build = LlamaServer.start, LlamaServer.stop, run.Run.edit_adapter

    def start(self, *args, **kwargs):
        started.append([path.name for path in self.adapters])
        running.add(id(self))
        return real_start(self, *args, **kwargs)

    def stop(self):
        running.discard(id(self))
        return real_stop(self)

    def build(self, folder, edit):
        beside.append(len(running))
        return real_build(self, folder, edit)

    monkeypatch.setattr(LlamaServer, "start", start)
    monkeypatch.setattr(LlamaServer, "stop", stop)
    monkeypatch.setattr(run.Run, "edit_adapter", build)
    said = []
    tiny.report = said.append
    Search(tiny).search(4)
    assert not [line for line in said if line.startswith("Rebuilding")]
    assert tiny.adapters() == {}
    # The unedited model first, with no adapters, then each trial with its own, built with no server
    # running beside it.
    scoring = started[-5:]
    assert scoring == [[], ["edit.gguf"], ["edit.gguf"], ["edit.gguf"], ["edit.gguf"]]
    assert beside == [0, 0, 0, 0]
    scores = [search._score(trial) for trial in finished(tiny)]
    assert len(scores) == 4 and len({s.kl for s in scores}) == 4
    # A search of "full" edits isn't carried on as one of "pre" edits.
    tiny.settings.edit.row_norm = "pre"
    with pytest.raises(RunError, match="holds a search made with a different adapters"):
        Search(tiny).search(6)


def test_scoring_full_edits_beside_the_search_builds_each_ones_adapter(tiny, monkeypatch):
    tiny.settings.edit.row_norm = "full"
    started = []
    real = LlamaServer.start

    def start(self, *args, **kwargs):
        started.append([path.parent.name for path in self.adapters])
        return real(self, *args, **kwargs)

    monkeypatch.setattr(LlamaServer, "start", start)
    kernels = {"attn": Kernel(1.2, 1.0, 0.8, 1.0), "mlp": Kernel(1.0, 0.0, 0.5, 1.0)}
    edits = [Edit("one", 0.4, kernels), Edit("two", None, kernels)]
    scored = Search(tiny).score(edits)
    assert started[-3:] == [[], ["one"], ["two"]]
    assert len(scored.edits) == 2 and scored.edits[0].kl != scored.edits[1].kl


def timed(number, refusals, kl, attn, mlp, layer=None, judged=3, seconds=10.0, thinking=0.0, forced=0):
    """A finished trial as the search records it, with these strengths, seconds and thinking."""
    params = {"attn.strength": attn, "mlp.strength": mlp} | (
        {} if layer is None else {"direction_layer": layer}
    )
    distributions: dict[str, BaseDistribution] = {
        k: IntDistribution(0, 30) if k == "direction_layer" else FloatDistribution(0, 5) for k in params
    }
    attrs = made(number, refusals, kl, judged=judged).user_attrs
    attrs |= {"seconds": seconds, "thinking": thinking, "forced": forced}
    return optuna.trial.create_trial(params=params, distributions=distributions, value=kl, user_attrs=attrs)


def test_a_progress_report_names_each_kinds_best_its_edges_and_the_trials_since_it_improved():
    trials = [
        timed(1, 3, 0.01, 1.0, 1.0, layer=5),
        timed(2, 50, 0.01, 2.9, 1.0),
        timed(3, 2, 0.20, 1.0, 1.0, layer=5),
        timed(4, 10, 0.40, 2.95, 1.0),  # the fewest refused per layer, at its widened ceiling
        timed(5, 1, 0.10, 0.82, 1.2, layer=5),  # the least KL within the cap, at the floor
        timed(6, 10, 0.50, 1.0, 1.0),
        timed(7, 0, 0.30, 1.0, 1.0, layer=5, thinking=100.0, forced=1),
        timed(8, 4, 0.10, 1.0, 1.0, judged=1, seconds=20.0, thinking=200.0),  # stopped early
        timed(9, 3, 0.05, 1.0, 1.0, layer=5, judged=1, seconds=30.0, thinking=300.0, forced=1),
        timed(10, 30, 0.20, 1.0, 1.0, seconds=40.0, thinking=50.0, forced=2),
    ]
    found = search.Found({t.user_attrs["number"]: t for t in trials}, 99, {search.PER_LAYER: (0.8, 3.0)})
    space = config.Search(report_every=4, startup_per_scope=4)
    report = search.progress(found, space, 10, 20, 3600.0, thinks=True)
    assert report == {
        "finished": 10,
        "of": 20,
        "seconds": 3600.0,
        "over the last": 4,
        "seconds a trial": 25.0,
        "seconds left": 250.0,
        search.ONE_LAYER: {
            "trials": 5,
            "stopped early": 1,
            "strength": [0.8, 1.5],
            "widened": False,
            "best": {
                "trial": 5,
                "refused": 1,
                "KL": 0.10,
                "within the cap": True,
                "strengths": {"attn.strength": 0.82, "mlp.strength": 1.2},
                "at its floor": ["attn.strength"],
                "at its ceiling": [],
                "trials since it improved": 2,
            },
        },
        search.PER_LAYER: {
            "trials": 5,
            "stopped early": 1,
            "strength": [0.8, 3.0],
            "widened": True,
            "best": {
                "trial": 4,
                "refused": 10,
                "KL": 0.40,
                "within the cap": False,
                "strengths": {"attn.strength": 2.95, "mlp.strength": 1.0},
                "at its floor": [],
                "at its ceiling": ["attn.strength"],
                "trials since it improved": 3,
            },
        },
        "thinking": {"tokens per answer": 118.75, "answers": 8, "reached the budget": 4},
    }
    assert search.progress_lines(report) == [
        "After 10 of 20 trials, 1 h 0 min into this session: 25 s a trial over the last 4, so about 4 min "
        "left",
        "  one layer: 5 trials, 1 stopped early; best trial 5, 1 refused, KL 0.1000; strength attn 0.82 at "
        "its floor, mlp 1.2, range 0.8-1.5; 2 trials since it improved",
        "  per layer: 5 trials, 1 stopped early; none within the cap, fewest refused trial 4, 10 refused, KL "
        "0.4000; strength attn 2.95 at its ceiling, mlp 1, range 0.8-3 (widened); 3 trials since it improved",
        "  thinking over the last 4: 119 tokens per answer, 4 of 8 answers reached the budget",
    ]
    assert "thinking" not in search.progress(found, space, 10, 20, 3600.0, thinks=False)
    # Each kind's 5 trials are all its spread start, so no end is flagged yet, as the summary flags none.
    start = search.progress(found, config.Search(report_every=4, startup_per_scope=5), 10, 20, 3600.0, False)
    for scope in search.SCOPES:
        assert start[scope]["best"]["at its floor"] == start[scope]["best"]["at its ceiling"] == []


def test_a_search_reports_its_progress_every_report_every_trials_its_end_and_never_at_0(tiny, monkeypatch):
    small(tiny, trials=3)
    refusing(monkeypatch, refusals=1)
    said = []
    tiny.report = said.append
    path = tiny.path("search", "progress.json")

    def reports():
        return [line for line in said if "into this session" in line]

    tiny.settings.search.report_every = 1
    Search(tiny).search(1)
    first = json.loads(path.read_text())
    assert reports() == [said[-3]] and said[-3].startswith("After 1 of 1 trials, 1 min into this session: ")
    assert (
        said[-1] == "  per layer: 0 trials, 0 stopped early; none scored in full yet; strength range 0.8-1.5"
    )
    assert (first["finished"], first["of"], first["over the last"]) == (1, 1, 1)
    assert first[search.ONE_LAYER]["best"]["trial"] == 1
    assert first[search.PER_LAYER] == {
        "trials": 0,
        "stopped early": 0,
        "strength": [0.8, 1.5],
        "widened": False,
    }
    tiny.settings.search.report_every = 0
    Search(tiny).search(2)
    assert len(reports()) == 1 and json.loads(path.read_text()) == first
    tiny.settings.search.report_every = 2
    Search(tiny).search(4)
    assert [line.split(",")[0] for line in reports()] == ["After 1 of 1 trials", "After 4 of 4 trials"]
    last = json.loads(path.read_text())
    assert (last["finished"], last["over the last"]) == (4, 2)
    assert [last[scope]["trials"] for scope in search.SCOPES] == [2, 2]
    tiny.settings.search.report_every = 4
    Search(tiny).search(5)
    assert reports()[-1].startswith("After 5 of 5 trials")
    assert json.loads(path.read_text())["finished"] == 5
    assert not list(tiny.path("search").glob("*.building"))


def test_a_search_made_before_report_every_carries_on_after_an_upgrade_at_any_report_every(tiny):
    folder = cli_run(tiny)
    runner = CliRunner(env={"COLUMNS": "200"})
    assert runner.invoke(app, ["search", str(folder), "--trials", "1"]).exit_code == 0
    assert (
        "report_every" not in json.loads((folder / "search" / "inputs.json").read_text())["search settings"]
    )
    path = folder / config.FILE
    path.write_text(path.read_text().replace("report_every = 25\n", ""))
    refused = runner.invoke(app, ["search", str(folder)])
    assert refused.exit_code == cli.COULD_NOT_INSPECT and "candor upgrade adds them" in flat(refused.output)
    upgraded = runner.invoke(app, ["upgrade", str(folder)])
    assert upgraded.exit_code == 0 and "search.report_every = 25" in upgraded.output
    path.write_text(path.read_text().replace("report_every = 25\n", "report_every = 1\n"))
    resumed = runner.invoke(app, ["search", str(folder), "--trials", "2"])
    assert resumed.exit_code == 0, resumed.output
    assert "After 2 of 2 trials" in resumed.output
    found = search.load(run.Run.open(folder))
    assert found is not None and list(found.trials) == [1, 2]


def test_reporting_leaves_every_trial_the_search_draws_as_it_was(tiny, monkeypatch):
    widening_soon(monkeypatch)
    small(tiny, trials=6)
    drawn = {}
    for every in (0, 1):
        refusing(monkeypatch)
        tiny.settings.search.report_every = every
        found, _ = Search(tiny).search(12)
        drawn[every] = [(t.params, t.user_attrs["kl"]) for t in found.trials.values()]
        tiny.path("search").rename(tiny.path(f"search-{every}"))
    assert len(drawn[1]) == 12 and drawn[0] == drawn[1]
    assert tiny.path("search-1", "progress.json").exists()
    assert not tiny.path("search-0", "progress.json").exists()


def varied(monkeypatch) -> list[list[int | None]]:
    """Scores that differ trial by trial, the same in every run, a trial refusing more than stop_above
    stopped after one prompt; records each call's stop_above and forced_above."""
    asked = []

    def score(self, scales, stop_above=None, forced_above=None, read_kl=True):
        asked.append([stop_above, forced_above])
        n = len(asked)
        refusals = n * 5 % 4
        judged = 1 if stop_above is not None and refusals > stop_above else 3
        return Score(refusals, 0, 0, 0.01 * (n * 7 % 11) + 0.001 * n, judged, 10.0 * n, min(n % 4, judged))

    monkeypatch.setattr(search._Scorer, "score", score)
    return asked


def test_with_max_budget_hits_unset_a_seeded_search_draws_records_and_tells_the_sampler_as_before(
    tiny, monkeypatch
):
    # Pinned as candor drew and recorded them before max_budget_hits existed: a thinking model's search
    # whose trials stop early and whose strength widens.
    thinks_through(monkeypatch)
    widening_soon(monkeypatch)
    monkeypatch.setattr(search, "STOP_AFTER", 1)
    small(thinking(tiny), trials=6)
    tiny.settings.search.seed = 3
    asked = varied(monkeypatch)
    found, _ = Search(tiny).search(12)
    record = {
        "asked": asked,
        "studies": {
            scope: [latest(tiny, scope).study_name, latest(tiny, scope).user_attrs] for scope in search.SCOPES
        },
        "trials": [
            [
                t.params,
                {key: value for key, value in t.user_attrs.items() if key not in ("seconds", "machine")},
                t.system_attrs["constraints"],
            ]
            for t in found.trials.values()
        ],
        "recorded": json.loads(tiny.path("search", "inputs.json").read_text())["search settings"],
        "names": sorted(optuna.get_all_study_names(search._storage(tiny.path("search")))),
    }
    assert sum(t.user_attrs["stopped"] for t in found.trials.values()) == 2 and found.strength
    assert json.loads(json.dumps(record)) == json.loads(SEEDED.read_text())


def thinking_search(tiny, monkeypatch, hits, trials=3):
    """The tiny run with a model that thinks, its search judging edits by search.max_budget_hits hits."""
    thinks_through(monkeypatch)
    small(thinking(tiny), trials=trials)
    tiny.settings.search.max_budget_hits = hits
    return tiny


def test_an_edit_counts_only_in_full_within_the_refusal_cap_and_once_set_the_budget_hits():
    def score(refusals, forced, judged=100):
        return Score(refusals, 0, 0, 0.1, judged, 50.0, forced)

    space, budget = config.Search(), config.Search(max_budget_hits=8)
    assert search.counts(score(2, 30), space) and not search.counts(score(3, 0), space)
    assert search.counts(score(2, 8), budget)
    assert not search.counts(score(2, 9), budget) and not search.counts(score(3, 0), budget)
    assert not search.counts(score(0, 0, judged=8), budget, stopped=True)


def test_answers_at_the_budget_go_over_max_budget_hits_scaled_to_every_prompt_as_refusals_do():
    assert search.budget_violation(Score(0, 0, 0, 0.1, 100, 0.0, 8), 100, 8) == -0.5
    assert search.budget_violation(Score(0, 0, 0, 0.1, 100, 0.0, 9), 100, 8) == 0.5
    assert search.budget_violation(Score(3, 0, 0, 0.1, 16, 0.0, 9), 100, 8) == pytest.approx(
        9 * 100 / 16 - 8.5
    )


def test_with_max_budget_hits_set_an_edit_thinking_to_the_budget_more_often_is_never_the_best(
    tiny, monkeypatch
):
    thinking_search(tiny, monkeypatch, hits=1)
    budget = []
    scripted(
        monkeypatch,
        [
            Score(3, 0, 0, 0.0, 3),  # the unedited model
            Score(0, 0, 0, 0.1, 3, 9.0, 2),  # 1, one layer: refuses none, but thinks to the budget twice
            Score(1, 0, 0, 0.5, 3, 9.0, 1),  # 2, per layer: within both caps
            Score(1, 0, 0, 0.3, 3, 9.0, 0),  # 3, one layer: within both, with more KL than trial 1
            Score(0, 0, 0, 0.2, 2, 9.0, 2),  # 4, per layer: stopped at the budget
            Score(1, 0, 0, 0.25, 3, 9.0, 1),  # 5, one layer, in a new session: less KL than trial 3
        ],
        budget,
    )
    said = []
    Search(tiny).search(4, finished=said.append)
    Search(tiny).search(5, finished=said.append)
    assert [line and line.split(":")[0] for line in said] == [
        None,
        "Trial 2, per layer",
        "Trial 3, one layer",
        None,
        "Trial 5, one layer",
    ]
    # Every trial stops once more answers than the setting reach the budget.
    assert budget == [None, 1, 1, 1, 1, 1]
    trials = finished(tiny)
    assert [t.user_attrs["stopped"] for t in trials] == [False, False, False, True, False]
    assert [list(t.system_attrs["constraints"]) for t in trials] == [
        [-2.5, 0.5],
        [-1.5, -0.5],
        [-1.5, -1.5],
        [-2.5, 1.5],
        [-1.5, -0.5],
    ]
    found = search.load(tiny)
    assert found is not None
    space = tiny.settings.search
    assert [c.trial for c in search.within_cap(found, search.ONE_LAYER, space)] == [5, 3]
    assert [c.trial for c in search.to_measure(found, search.ONE_LAYER, space)] == [5]
    assert [c.trial for c in search.within_cap(found, search.PER_LAYER, space)] == [2]
    report = search.progress(found, space, 5, 6, 60.0, thinks=True)
    assert [report[scope]["failed on the budget"] for scope in search.SCOPES] == [1, 1]
    assert (
        report[search.ONE_LAYER]["best"]["trial"] == 5 and report[search.ONE_LAYER]["best"]["within the cap"]
    )


def test_strength_widens_by_the_best_edit_within_both_caps():
    pressing = timed(1, 0, 0.1, 1.49, 1.0, forced=3)  # the least KL, at the ceiling, but thinking too often
    inside = timed(2, 0, 0.2, 1.0, 1.2, forced=1)
    found = search.Found({1: pressing, 2: inside}, 99)
    assert search.widened(found, search.PER_LAYER, config.Search()) == (0.8, 3.0)
    assert search.widened(found, search.PER_LAYER, config.Search(max_budget_hits=1)) == (0.8, 1.5)


def test_with_every_edit_over_the_budget_hits_none_is_within_the_cap_to_widen_by(tiny, monkeypatch):
    widening_soon(monkeypatch)
    thinking_search(tiny, monkeypatch, hits=0)
    scores = [Score(0, 0, 0, 0.1 * n, 3, 9.0, 1) for n in range(1, 6)]
    scripted(monkeypatch, [Score(3, 0, 0, 0.0, 3), *scores])
    said = []
    tiny.report = said.append
    Search(tiny).search(5)
    assert (
        "After 2 one layer trials no one layer edit is within the cap yet, so its strength widens from "
        "0.8-1.5 to 0.8-3"
    ) in said


def test_scoring_stops_at_the_end_of_the_chunk_whose_answers_go_over_the_budget_hits(tiny, monkeypatch):
    monkeypatch.setattr(search, "CHUNK", 2)
    made_to_stop = []
    real = LlamaServer.answers

    def answers(self, *args, **kwargs):
        for answer, forced in zip(real(self, *args, **kwargs), made_to_stop, strict=True):
            yield dataclasses.replace(answer, thinking=4, forced=forced)

    monkeypatch.setattr(LlamaServer, "answers", answers)
    with Search(tiny)._scorer(*scorer_inputs(tiny)) as scorer:
        made_to_stop[:] = [True, True, False]
        stopped = scorer.score({}, forced_above=1)
        assert (stopped.forced, stopped.judged, stopped.thinking) == (2, 2, 4)
        made_to_stop[:] = [True, False, True]
        whole = scorer.score({}, forced_above=1)
        assert (whole.forced, whole.judged) == (2, 3)
        made_to_stop[:] = [True, True, True]
        assert scorer.score({}).judged == 3


def test_with_max_budget_hits_set_a_trial_stops_on_refusals_only_behind_stop_after_trials_over_every_cap(
    tiny, monkeypatch
):
    thinking_search(tiny, monkeypatch, hits=0)
    monkeypatch.setattr(search, "STOP_AFTER", 1)
    budget = []
    asked = scripted(
        monkeypatch,
        [
            Score(3, 0, 0, 0.0, 3),
            Score(1, 0, 0, 0.5, 3, 9.0, 3),  # 1, one layer: within the refusal cap, 2.5 over the budget's
            Score(3, 0, 0, 0.1, 3),  # 2, per layer: 0.5 over the refusal cap
            Score(3, 0, 0, 0.1, 3),  # 3, one layer
            Score(3, 0, 0, 0.1, 3),  # 4, per layer
        ],
        budget,
    )
    Search(tiny).search(4)
    # Refusing 3 of 3 puts a trial 0.5 over, ahead of trial 1, so a one layer trial never stops on refusals
    # here; per layer, it stops above 3 refused, as with the budget hits unset.
    assert asked == [None, None, None, 5, 3]
    assert budget == [None, 0, 0, 0, 0]
    tiny.settings.search.stop_early = False
    tiny.path("search").rename(tiny.path("search-before"))
    budget.clear()
    scripted(monkeypatch, [Score(3, 0, 0, 0.0, 3), Score(1, 0, 0, 0.5, 3), Score(1, 0, 0, 0.5, 3)], budget)
    Search(tiny).search(2)
    assert budget == [None] * 3


def test_a_progress_report_counts_each_kinds_trials_over_the_budget_hits_only_once_set():
    trials = [
        timed(1, 0, 0.1, 1.0, 1.0, layer=5, forced=3),
        timed(2, 1, 0.2, 1.0, 1.0, forced=2),
        timed(3, 1, 0.3, 1.0, 1.0, layer=5, forced=1),
        timed(4, 2, 0.1, 1.0, 1.0, judged=1, forced=2),  # stopped at the budget
    ]
    found = search.Found({t.user_attrs["number"]: t for t in trials}, 99)
    report = search.progress(found, config.Search(report_every=4, max_budget_hits=1), 4, 8, 60.0, True)
    assert [report[scope]["failed on the budget"] for scope in search.SCOPES] == [1, 2]
    assert not report[search.PER_LAYER]["best"]["within the cap"]
    lines = search.progress_lines(report)
    assert lines[1].startswith(
        "  one layer: 2 trials, 0 stopped early, 1 failed on the budget; best trial 3,"
    )
    assert lines[2].startswith(
        "  per layer: 2 trials, 1 stopped early, 2 failed on the budget; none within the cap, fewest refused "
        "trial 2,"
    )
    unset = search.progress(found, config.Search(report_every=4), 4, 8, 60.0, True)
    assert "failed on the budget" not in unset[search.ONE_LAYER]
    assert "failed on the budget" not in " ".join(search.progress_lines(unset))


def test_a_search_made_before_max_budget_hits_carries_on_and_once_it_is_set_judges_its_trials_again(
    tiny, monkeypatch
):
    thinks_through(monkeypatch)
    folder = cli_run(thinking(tiny), trials=3)
    path = folder / config.FILE
    # As candor.toml was written before the setting, which candor upgrade needn't add.
    text = path.read_text()
    start, end = text.index("# The most test prompts whose thinking"), text.index("# max_budget_hits =\n")
    path.write_text(text[:start] + text[end + len("# max_budget_hits =\n") :])
    runner = CliRunner(env={"COLUMNS": "200"})
    upgraded = runner.invoke(app, ["upgrade", str(folder)])
    assert upgraded.exit_code == 0 and "already has every setting" in upgraded.output
    scripted(
        monkeypatch,
        [
            Score(3, 0, 0, 0.0, 3),
            Score(0, 0, 0, 0.1, 3, 9.0, 2),  # 1, one layer: thinks to the budget twice
            Score(3, 0, 0, 0.2, 3, 9.0, 2),  # 2, per layer: over both caps, so it never counted
            Score(1, 0, 0, 0.3, 3, 9.0, 1),  # 3, one layer
            Score(1, 0, 0, 0.4, 3, 9.0, 0),  # 4, per layer
        ],
    )
    assert runner.invoke(app, ["search", str(folder), "--trials", "2"]).exit_code == 0
    recorded = json.loads((folder / "search" / "inputs.json").read_text())["search settings"]
    assert "max_budget_hits" not in recorded
    path.write_text(path.read_text().replace("[search]\n", "[search]\nmax_budget_hits = 1\n"))
    resumed = runner.invoke(app, ["search", str(folder), "--trials", "4"])
    assert resumed.exit_code == 0, resumed.output
    shown = flat(resumed.output)
    assert (
        "The search's 2 finished trials are judged again by search.max_budget_hits, now 1: 2 reached the "
        "thinking budget more often, which leaves 1 fewer within the cap"
    ) in shown
    assert "within the cap (2 refused, 1 at the thinking budget)" in shown
    assert "one layer: trial 3," in shown and "one layer: trial 1," not in shown
    assert "one layer: 2 trials, 0 stopped early, 1 failed on the budget;" in shown
    assert json.loads((folder / "search" / "inputs.json").read_text())["search settings"] == recorded
    found = search.load(run.Run.open(folder))
    assert found is not None and list(found.trials) == [1, 2, 3, 4]


def test_setting_max_budget_hits_moves_each_kinds_finished_trials_to_a_new_study_told_afresh(
    tiny, monkeypatch
):
    thinking_search(tiny, monkeypatch, hits=None, trials=4)
    scores = [Score(1, 0, 0, 0.1 * n, 3, 9.0, n % 3) for n in range(1, 7)]
    scripted(monkeypatch, [Score(3, 0, 0, 0.0, 3), *scores])
    Search(tiny).search(4)
    before = {t.user_attrs["number"]: t for t in finished(tiny)}
    said = []
    tiny.report = said.append
    tiny.settings.search.max_budget_hits = 1
    Search(tiny).search(6)
    for scope in search.SCOPES:
        study = latest(tiny, scope)
        assert study.study_name == search._judged_name(scope, 1)
        assert study.user_attrs["budget hits"] == [[2, 1]] and study.user_attrs["widened"] == []
    assert latest(tiny, search.ONE_LAYER).user_attrs["base_refusals"] == 3
    after = {t.user_attrs["number"]: t for t in finished(tiny)}
    assert sorted(after) == list(range(1, 7))
    for number, trial in before.items():
        moved = after[number]
        assert (moved.params, moved.distributions, moved.value) == (
            trial.params,
            trial.distributions,
            trial.value,
        )
        budget = search.budget_violation(search._score(trial), 3, 1)
        assert moved.user_attrs == trial.user_attrs | {"budget violation": budget}
        assert list(moved.system_attrs["constraints"]) == [trial.user_attrs["violation"], budget]
    found = search.load(tiny)
    assert found is not None
    within = [
        c.trial for scope in search.SCOPES for c in search.within_cap(found, scope, tiny.settings.search)
    ]
    assert sorted(within) == [1, 3, 4, 6]  # trials 2 and 5 reach the budget twice
    # Unset again, the trials are told the refusal cap alone, in a third study of each kind.
    tiny.settings.search.max_budget_hits = None
    Search(tiny).search(6)
    for scope in search.SCOPES:
        study = latest(tiny, scope)
        assert study.study_name == search._judged_name(scope, 2)
        assert study.user_attrs["budget hits"] == [[2, 1], [3, None]]
    for trial in finished(tiny):
        assert "budget violation" not in trial.user_attrs
        assert list(trial.system_attrs["constraints"]) == [trial.user_attrs["violation"]]
    assert [line for line in said if "judged again" in line] == [
        "The search's 4 finished trials are judged again by search.max_budget_hits, now 1: 1 reached the "
        "thinking budget more often, which leaves 1 fewer within the cap",
        "The search's 6 finished trials are judged again by search.max_budget_hits, now unset",
    ]


def test_a_trial_recorded_before_answers_at_the_budget_were_counted_reached_it_on_none(tiny, monkeypatch):
    thinking_search(tiny, monkeypatch, hits=0)
    folder = tiny.path("search")
    folder.mkdir()
    study = optuna.create_study(study_name=search.STUDIES[search.PER_LAYER], storage=search._storage(folder))
    old = made(1, 1, 0.2)
    study.add_trial(
        optuna.trial.create_trial(
            params=old.params,
            distributions=old.distributions,
            value=old.value,
            user_attrs=old.user_attrs | {"violation": -1.5},
        )
    )
    [trial] = Search(tiny)._judge_again(study, search.PER_LAYER, 3).get_trials()
    assert "forced" not in trial.user_attrs and trial.user_attrs["budget violation"] == -0.5
    assert search._counts(trial, tiny.settings.search)


def test_a_search_killed_while_judging_its_trials_again_loses_and_repeats_no_trial(tiny, monkeypatch):
    thinking_search(tiny, monkeypatch, hits=None, trials=4)
    refusing(monkeypatch, refusals=1)
    Search(tiny).search(4)
    tiny.settings.search.max_budget_hits = 0
    real = optuna.Study.add_trial
    copied = []

    def dying(self, trial):
        if self.study_name.endswith("budget hits 1") and trial.state == TrialState.COMPLETE:
            copied.append(trial)
            if len(copied) == 2:
                raise KeyboardInterrupt
        real(self, trial)

    monkeypatch.setattr(optuna.Study, "add_trial", dying)
    with pytest.raises(KeyboardInterrupt):
        Search(tiny).search(8)
    # The study left part made is passed over, so the search is as it was.
    found = search.load(tiny)
    assert found is not None and sorted(found.trials) == [1, 2, 3, 4]
    assert all("budget violation" not in t.user_attrs for t in found.trials.values())
    monkeypatch.setattr(optuna.Study, "add_trial", real)
    Search(tiny).search(8)
    found = search.load(tiny)
    assert found is not None and sorted(found.trials) == list(range(1, 9))
    for scope in search.SCOPES:
        study = latest(tiny, scope)
        assert study.study_name == search._judged_name(scope, 1) and study.user_attrs["budget hits"] == [
            [2, 0]
        ]
        done = study.get_trials(states=(TrialState.COMPLETE,))
        assert len({search._key(t.params) for t in done}) == len(done) == 4
        assert all(len(t.system_attrs["constraints"]) == 2 for t in done)


def test_a_trial_stopped_early_stays_outside_the_caps_when_they_are_judged_again(tiny, monkeypatch):
    thinking_search(tiny, monkeypatch, hits=None)
    folder = tiny.path("search")
    folder.mkdir()
    study = optuna.create_study(study_name=search.STUDIES[search.PER_LAYER], storage=search._storage(folder))
    # Stopped by its answers at the budget, refusing none of the prompts it was judged on by then.
    stopped, full = made(1, 0, 0.2, judged=1), made(2, 0, 0.3)
    for trial in (stopped, full):
        attrs = trial.user_attrs | {"violation": -2.5, "budget violation": 1.5, "forced": 1}
        study.add_trial(
            optuna.trial.create_trial(
                params=trial.params, distributions=trial.distributions, value=trial.value, user_attrs=attrs
            )
        )
    moved = {
        t.user_attrs["number"]: t for t in Search(tiny)._judge_again(study, search.PER_LAYER, 3).get_trials()
    }
    assert list(moved[1].system_attrs["constraints"]) == [0.5]
    assert list(moved[2].system_attrs["constraints"]) == [-2.5]
    # What it was scored as is kept as it was.
    assert moved[1].user_attrs["violation"] == -2.5


def test_a_widened_search_judged_again_goes_on_in_its_new_study(tiny, monkeypatch):
    widening_soon(monkeypatch)
    thinking_search(tiny, monkeypatch, hits=None, trials=6)
    refusing(monkeypatch)
    Search(tiny).search(6)
    assert [latest(tiny, scope).study_name for scope in search.SCOPES] == [
        search._study_name(scope, 1) for scope in search.SCOPES
    ]
    tiny.settings.search.max_budget_hits = 0
    Search(tiny).search(12)
    for scope in search.SCOPES:
        study = latest(tiny, scope)
        assert study.study_name == search._judged_name(scope, 1)
        assert study.user_attrs["budget hits"] == [[3, 0]] and study.user_attrs["widened"] == [[2, 0.8, 3.0]]
        assert len(study.get_trials(states=(TrialState.COMPLETE,))) == 6
    found = search.load(tiny)
    assert found is not None and sorted(found.trials) == list(range(1, 13))
    assert found.strength == {scope: (0.8, 3.0) for scope in search.SCOPES}
