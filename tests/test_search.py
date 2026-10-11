import filecmp
import json

import optuna
import pytest
from optuna.trial import FixedTrial, TrialState
from typer.testing import CliRunner

from candor import cli, config, run, search
from candor.cli import app
from candor.engine import Answer, LlamaServer
from candor.judge import is_refusal
from candor.kernel import Kernel
from candor.measure import Measurement
from candor.run import RunError
from candor.search import Score, Search
from candor.targets import Writer

from .tiny_run import ANSWERED, REFUSED, FakeGuard, fake_prompts


def small(run, trials=3):
    run.settings.search.trials = trials
    run.settings.search.startup_trials = 2


def study(run) -> optuna.Study:
    return optuna.load_study(study_name=search.STUDY, storage=search._storage(run.path("search")))


def states(run) -> list[TrialState]:
    return [trial.state for trial in study(run).get_trials()]


def recording_answers(monkeypatch) -> list[str]:
    """Every answer the server gives, so tests can check none of it is kept."""
    answers = []
    real = LlamaServer.generate

    def recording(self, *args, **kwargs):
        found = real(self, *args, **kwargs)
        answers.extend(a.text for a in found)
        return found

    monkeypatch.setattr(LlamaServer, "generate", recording)
    return answers


def test_a_search_runs_its_trials_and_keeps_only_numbers(tiny, monkeypatch):
    small(tiny)
    answers = recording_answers(monkeypatch)
    found, seconds = Search(tiny).search(3)
    assert states(tiny) == [TrialState.COMPLETE] * 3
    assert study(tiny).user_attrs["base_refusals"] == 1
    assert found and all(c.refusals == 1 and c.kl >= 0 for c in found)
    assert seconds > 0
    kept = "".join(p.read_text() for p in tiny.path("search").iterdir() if p.is_file())
    assert len(answers) == 4 * 3
    assert all(len(a.strip()) < 4 or a.strip() not in kept for a in answers)


def test_a_stopped_search_carries_on_without_scoring_anything_twice(tiny, monkeypatch):
    small(tiny)
    Search(tiny).search(2)
    answers = recording_answers(monkeypatch)
    Search(tiny).search(3)
    assert len(answers) == 3
    assert states(tiny) == [TrialState.COMPLETE] * 3


def test_a_trial_stopped_part_way_is_marked_failed_and_the_search_carries_on(tiny, monkeypatch):
    small(tiny)
    real = search._Scorer.score
    calls = []

    def stopping(self, scales):
        calls.append(scales)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return real(self, scales)

    monkeypatch.setattr(search._Scorer, "score", stopping)
    with pytest.raises(KeyboardInterrupt):
        Search(tiny).search(3)
    assert states(tiny) == [TrialState.COMPLETE, TrialState.FAIL]
    left_running = study(tiny)
    left_running.ask()
    monkeypatch.setattr(search._Scorer, "score", real)
    Search(tiny).search(3)
    assert states(tiny) == [TrialState.COMPLETE, TrialState.FAIL, TrialState.FAIL] + [TrialState.COMPLETE] * 2


@pytest.mark.parametrize(
    ("change", "named"),
    [
        (lambda s, m: setattr(s.deployment, "quantization", "Q8_0"), "model file"),
        (lambda s, m: setattr(s.edit, "row_norm", "none"), "adapters"),
        (lambda s, m: setattr(s.prompts.harmful_test, "split", "test[:99]"), "test prompts"),
        (lambda s, m: setattr(s.scoring, "answer_tokens", 8), "answer length"),
        (lambda s, m: setattr(s.scoring, "kl_tokens", 5), "KL tokens"),
        (lambda s, m: setattr(s.measure, "judge_revision", "another"), "judge"),
        (lambda s, m: setattr(s.deployment, "kv_cache", "q8_0"), "KV cache"),
        (lambda s, m: setattr(s.search, "strength", (0.9, 1.4)), "search settings"),
        (lambda s, m: m.setattr(search, "VERSION", search.VERSION + 1), "code"),
        (lambda s, m: m.setattr(optuna, "__version__", "9.9.9"), "Optuna"),
    ],
)
def test_a_search_isnt_carried_on_after_anything_its_scores_depend_on_changes(
    tiny, monkeypatch, change, named
):
    small(tiny)
    Search(tiny).search(1)
    change(tiny.settings, monkeypatch)
    with pytest.raises(RunError, match=f"holds a search made with a different {named}; move it aside"):
        Search(tiny).search(2)


def test_more_trials_carry_on_the_same_search(tiny):
    small(tiny, trials=1)
    Search(tiny).search(1)
    tiny.settings.search.trials = 2
    Search(tiny).search(2)
    assert states(tiny) == [TrialState.COMPLETE] * 2


def test_a_search_without_its_record_is_refused(tiny):
    small(tiny)
    Search(tiny).search(1)
    tiny.path("search", "inputs.json").unlink()
    with pytest.raises(RunError, match="no record of its settings"):
        Search(tiny).search(2)


def test_replies_the_judge_didnt_label_count_as_refusals(tiny, monkeypatch):
    small(tiny)
    monkeypatch.setattr(FakeGuard, "replies", lambda self, prompts, answers, **kwargs: ["?"] * len(answers))
    Search(tiny).search(1)
    [trial] = study(tiny).get_trials()
    assert trial.user_attrs["refusals"] == trial.user_attrs["unlabelled"] == 3


def test_trial_parameters_become_the_edit_heretic_would_make():
    params = {
        "direction_scope": search.ONE_LAYER,
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
    search.suggest(FixedTrial(params), config.Search(), 29, ["attn", "mlp"])
    edit = search.edit_from(params, 7, 29)
    assert edit.name == "trial-7" and edit.direction_layer == 15
    assert edit.kernels == {"attn": Kernel(1.2, 21.0, 0.6, 7.0), "mlp": Kernel(1.0, 28.0, 0.0, 14.0)}
    per_layer = search.edit_from(params | {"direction_scope": search.PER_LAYER}, 7, 29)
    assert per_layer.direction_layer is None


def test_edits_that_change_almost_nothing_are_scored_by_their_refusals():
    assert search.objectives(Score(10, 12, 0, 0.05), 100, 0.01) == (0.05, 0.1)
    assert search.objectives(Score(10, 12, 0, 0.002), 100, 0.01) == pytest.approx((0.001, 0.1))
    assert search.objectives(Score(3, 3, 0, 0.2), 0, 0.01) == (0.2, 3.0)


def test_each_trial_is_scored_with_its_own_directions_adapters(tiny, monkeypatch):
    small(tiny, trials=6)
    tiny.settings.search.startup_trials = 6  # random draws, so both kinds of direction come up
    scored = []
    real = search._Scorer.score

    def recording(self, scales):
        scored.append({path.parent.name for path in scales})
        return real(self, scales)

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
    small(tiny)
    Search(tiny).search(1)
    with pytest.raises(RunError, match="the search has no finished trial 9"):
        search.trial_edit(tiny, 9)


def test_search_and_measure_a_trial_from_the_command_line(tiny):
    small(tiny, trials=4)
    config.write(tiny.folder.parent, "cli", tiny.settings, "made by a test")
    folder = tiny.folder.parent / "cli"
    runner = CliRunner(env={"COLUMNS": "100"})
    searched = runner.invoke(app, ["search", str(folder), "--trials", "2"])
    assert searched.exit_code == 0, searched.output
    assert "Trials no other beats on both refusals and KL" in searched.output
    assert "so all 4 would take about" in searched.output
    assert "Trial 1: 1 of 3 refused, KL" in searched.output
    finished = runner.invoke(app, ["search", str(folder)])
    assert finished.exit_code == 0, finished.output
    assert f"uv run candor measure {folder} --trial" in finished.output
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
    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(seed=0), directions=["minimize", "minimize"]
    )
    for _ in range(300):
        trial = study.ask()
        search.suggest(trial, config.Search(), layers, ["attn", "mlp"])
        strengths = search.edit_from(trial.params, 1, layers).strengths(
            writers, {w.name: w.name for w in writers}
        )
        assert all(any(strengths[w.name] > 0 for w in writers if w.component == c) for c in ("attn", "mlp"))
        study.tell(trial, (0.0, 0.0))


def test_a_resumed_search_doesnt_repeat_its_random_trials(tiny):
    small(tiny, trials=4)
    tiny.settings.search.startup_trials = 4
    Search(tiny).search(2)
    Search(tiny).search(4)
    params = [json.dumps(trial.params, sort_keys=True) for trial in study(tiny).get_trials()]
    assert len(set(params)) == 4


def test_the_table_is_the_front_of_refusals_and_kl_themselves():
    def made(name, refusals, kl):
        return search.Candidate(name, search.PER_LAYER, refusals, refusals, kl)

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


def silent(monkeypatch):
    """The model says nothing at all, which the fake judge doesn't call a refusal."""
    real = LlamaServer.generate

    def nothing(self, *args, **kwargs):
        return [Answer("", 0, True) for _ in real(self, *args, **kwargs)]

    monkeypatch.setattr(LlamaServer, "generate", nothing)


def test_an_edit_that_says_nothing_counts_as_refusing_everything(tiny, monkeypatch):
    small(tiny)
    silent(monkeypatch)
    Search(tiny).search(1)
    [trial] = study(tiny).get_trials()
    assert trial.user_attrs["refusals"] == 3 and trial.user_attrs["unlabelled"] == 0
    results = Measurement(tiny, None, "silent").measure()
    assert results["harmful_held_out"]["judge_refused"] == [True, True, True]


def test_each_trial_is_scored_with_its_strengths_on_the_test_prompts(tiny, monkeypatch):
    small(tiny)
    ids = tiny.prompt_ids()
    asked = {"generate": [], "top_logprobs": [], "scales": []}
    answers = []
    real_generate, real_top = LlamaServer.generate, LlamaServer.top_logprobs

    def generate(self, prompts, tokens, scales=None, done=None):
        asked["generate"].append(prompts)
        asked["scales"].append(dict(scales or {}))
        found = real_generate(self, prompts, tokens, scales, done)
        answers.append([a.text for a in found])
        return found

    def top_logprobs(self, prompts, *args, **kwargs):
        asked["top_logprobs"].append(prompts)
        return real_top(self, prompts, *args, **kwargs)

    monkeypatch.setattr(LlamaServer, "generate", generate)
    monkeypatch.setattr(LlamaServer, "top_logprobs", top_logprobs)
    _, seconds = Search(tiny).search(3)
    assert all(prompts == ids["harmful_test"] for prompts in asked["generate"])
    assert all(prompts == ids["harmless_test"] for prompts in asked["top_logprobs"])
    adapters = tiny.adapters()
    trials = study(tiny).get_trials()
    assert asked["scales"][0] == {}
    for trial, scales, said in zip(trials, asked["scales"][1:], answers[1:], strict=True):
        edit = search.trial_edit(tiny, trial.number + 1)
        files = adapters["per-layer" if edit.direction_layer is None else f"layer-{edit.direction_layer}"]
        strengths = edit.strengths(tiny.shape[3], tiny.target_names)
        assert scales == {files[name]: s for name, s in strengths.items() if s}
        assert trial.user_attrs["kl"] > 0
        assert trial.user_attrs["keyword_refusals"] == sum(map(is_refusal, said))
    assert seconds == pytest.approx(sum(t.user_attrs["seconds"] for t in trials) / 3)


def test_the_sampler_is_told_each_trials_objectives(tiny):
    small(tiny)
    tiny.settings.search.kl_target = 1e9  # every trial falls below it, so the refusal share stands in
    Search(tiny).search(2)
    base = study(tiny).user_attrs["base_refusals"]
    for trial in study(tiny).get_trials():
        score = Score(trial.user_attrs["refusals"], 0, 0, trial.user_attrs["kl"])
        assert trial.values == pytest.approx(list(search.objectives(score, base, 1e9)))
        assert trial.values[0] == pytest.approx(trial.values[1] * 1e9)


def test_only_a_trial_with_fewer_refusals_than_any_before_it_is_announced(tiny, monkeypatch):
    small(tiny, trials=4)
    counts = iter([3, 2, 2, 1, 1])  # the unedited model, then each trial

    def replies(self, prompts, answers, **kwargs):
        refused = next(counts)
        return [REFUSED] * refused + [ANSWERED] * (len(answers) - refused)

    monkeypatch.setattr(FakeGuard, "replies", replies)
    said = []
    Search(tiny).search(3, finished=said.append)
    Search(tiny).search(4, finished=said.append)
    assert [line and line.split(":")[0] for line in said] == ["Trial 1", None, "Trial 3", None]
    assert said[0].startswith("Trial 1: 2 of 3 refused, KL ")


def test_the_search_shows_time_left_from_every_trial_of_the_session(tiny, monkeypatch):
    small(tiny)
    config.write(tiny.folder.parent, "cli", tiny.settings, "made by a test")
    made = []
    real = cli.Progress

    def recording(*args, **kwargs):
        made.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "Progress", recording)
    result = CliRunner().invoke(app, ["search", str(tiny.folder.parent / "cli"), "--trials", "1"])
    assert result.exit_code == 0, result.output
    assert made[0]["speed_estimate_period"] >= 3600


def test_keywords_are_counted_on_each_trials_answers(tiny, monkeypatch):
    small(tiny)
    real = LlamaServer.generate

    def one_sorry(self, *args, **kwargs):
        found = real(self, *args, **kwargs)
        texts = ["I'm sorry."] + ["Here it is."] * (len(found) - 1)
        return [Answer(text, a.tokens, True) for text, a in zip(texts, found, strict=True)]

    monkeypatch.setattr(LlamaServer, "generate", one_sorry)
    Search(tiny).search(2)
    assert [t.user_attrs["keyword_refusals"] for t in study(tiny).get_trials()] == [1, 1]
