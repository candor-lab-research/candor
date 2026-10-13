import json

import pytest
from typer.testing import CliRunner

from candor import config, measure, run
from candor.cli import app
from candor.engine import LlamaServer
from candor.kernel import Kernel
from candor.measure import Edit, Measurement
from candor.run import RunError
from candor.targets import Writer

from .test_heretic import PARAMETERS
from .tiny_gguf import END_THINK, THINK, THINKING_TOKENS
from .tiny_run import FakeGuard, fake_prompts, thinking, thinks_through

EDIT = Edit("trial", 0.4, {"attn": Kernel(1.2, 1.0, 0.8, 1.0), "mlp": Kernel(1.0, 0.0, 0.5, 1.0)})


def test_a_measurement_keeps_numbers_per_prompt_and_answers_only_in_private(tiny):
    results = Measurement(tiny, None, "base").measure()
    assert {name: len(r["finished"]) for name, r in results.items()} == dict.fromkeys(measure.SETS, 3)
    harmful = results["harmful_held_out"]
    assert harmful["judge_refused"] == [True, False, False]
    assert harmful["judge_safety"] == ["Safe", "Unsafe", "Unsafe"]
    assert len(harmful["keyword_refused"]) == 3
    assert len(results["harmless_held_out"]["first_token"]) == 3
    assert len(results["maths"]["correct"]) == 3
    rows = dict(measure.summary(results))
    disagree = sum(k != j for k, j in zip(harmful["keyword_refused"], [True, False, False], strict=True))
    assert rows["Harmful, keywords disagree"] == f"{disagree}/3"
    private = tiny.path("private", "measurements", "base")
    numbers = "".join(p.read_text() for p in tiny.path("measurements", "base").glob("*.json"))
    for name in measure.SETS:
        texts = json.loads((private / f"{name}.json").read_text())
        assert len(texts["answers"]) == 3
        assert (private / f"{name}.judged.json").exists() == (name in run.HELD_OUT)
        for answer in texts["answers"]:
            assert len(answer) < 4 or json.dumps(answer)[1:-1] not in numbers


def test_an_edits_record_names_where_it_reads_its_directions_only_where_the_answer_starts():
    assert "answering" not in EDIT.record()
    assert Edit(EDIT.name, EDIT.direction_layer, EDIT.kernels, answering=True).record()["answering"] is True


def test_an_edit_read_where_the_answer_starts_gets_adapters_read_there(tiny):
    thinking(tiny)
    answering = Edit(EDIT.name, EDIT.direction_layer, EDIT.kernels, answering=True)
    files, inputs = measure.edit_adapters(tiny, answering, tiny.path("measurements", "answering", "adapters"))
    built, start = measure.edit_adapters(tiny, EDIT, tiny.path("measurements", "start", "adapters"))
    assert inputs["directions"] == tiny.inputs["answering directions"] != start["directions"]
    assert all(files[name].read_bytes() != built[name].read_bytes() for name in files)


def test_a_finished_measurement_is_reused_without_answering_again(tiny, monkeypatch):
    first = Measurement(tiny, None, "base").measure()

    def no_server(*args, **kwargs):
        raise AssertionError("started a server for a finished measurement")

    monkeypatch.setattr(LlamaServer, "start", no_server)
    assert Measurement(tiny, None, "base").measure() == first


def counting_answers(monkeypatch) -> list[int]:
    """How many prompts each call to the server's generate answers."""
    answered = []
    real = LlamaServer.generate

    def counting(self, prompts, *args, **kwargs):
        answered.append(len(prompts))
        return real(self, prompts, *args, **kwargs)

    monkeypatch.setattr(LlamaServer, "generate", counting)
    return answered


def test_a_new_judge_judges_the_saved_answers_without_answering_again(tiny, monkeypatch):
    first = Measurement(tiny, None, "base").measure()
    answered = counting_answers(monkeypatch)
    tiny.settings.measure.judge_revision = "another-commit"
    FakeGuard.judged = []
    again = Measurement(tiny, None, "base").measure()
    assert answered == [] and FakeGuard.judged == [3, 3]
    assert again["maths"] == first["maths"]
    with pytest.raises(RunError, match="judged by different judges"):
        measure.compare(first, again)


def test_a_new_way_of_counting_labels_judges_again_without_answering_again(tiny, monkeypatch):
    Measurement(tiny, None, "base").measure()
    answered = counting_answers(monkeypatch)
    monkeypatch.setattr(measure, "COUNTING", measure.COUNTING + 1)
    FakeGuard.judged = []
    messages = []
    tiny.report = messages.append
    Measurement(tiny, None, "base").measure()
    assert answered == [] and FakeGuard.judged == [3, 3]
    rebuilt = [m for m in messages if m.startswith("Rebuilding")]
    assert sorted(rebuilt) == [
        "Rebuilding measurements/base/harmful_held_out.judged.json: what it was built from has changed",
        "Rebuilding measurements/base/harmless_held_out.judged.json: what it was built from has changed",
    ]


def test_answers_survive_a_judge_that_fails_to_load(tiny, monkeypatch):
    def broken(*args):
        raise OSError("no network")

    monkeypatch.setattr(measure, "Guard", broken)
    with pytest.raises(OSError):
        Measurement(tiny, None, "base").measure()
    monkeypatch.setattr(measure, "Guard", FakeGuard)
    answered = counting_answers(monkeypatch)
    FakeGuard.judged = []
    Measurement(tiny, None, "base").measure()
    assert answered == [] and FakeGuard.judged == [3, 3]


def test_missing_saved_answers_are_answered_again(tiny, monkeypatch):
    messages = []
    tiny.report = messages.append
    Measurement(tiny, None, "base").measure()
    tiny.path("private", "measurements", "base", "maths.json").unlink()
    answered = counting_answers(monkeypatch)
    Measurement(tiny, None, "base").measure()
    assert answered == [3]
    assert "Answering the maths questions again: their saved answers are missing or changed" in messages


def test_another_quantisation_is_measured_again_and_not_compared(tiny, monkeypatch):
    q4 = Measurement(tiny, None, "base").measure()
    tiny.settings.deployment.quantization = "Q8_0"
    answered = counting_answers(monkeypatch)
    q8 = Measurement(tiny, None, "base").measure()
    assert answered == [3, 3, 3]
    with pytest.raises(RunError, match="differ in more than the edit and slots"):
        measure.compare(q4, q8)


def test_measurements_with_other_settings_are_not_compared(tiny):
    first = Measurement(tiny, None, "base").measure()
    tiny.settings.measure.answer_tokens = 8
    other = Measurement(tiny, None, "shorter").measure()
    with pytest.raises(RunError, match="held-out harmful prompts differ in more than the edit and slots"):
        measure.compare(first, other)
    tiny.settings.measure.answer_tokens = 16
    assert measure.compare(first, Measurement(tiny, None, "again", slots=2).measure())


def test_prompts_that_changed_since_prompts_json_stop_the_measurement(tiny, monkeypatch):
    tiny.prompt_ids()

    def changed(spec):
        prompts = fake_prompts(spec)
        return [f"{prompts[0]} changed", *prompts[1:]] if spec.column != "answer" else prompts

    monkeypatch.setattr(run, "load_prompts", changed)
    with pytest.raises(RunError, match=r"have changed since prompts\.json was built"):
        Measurement(tiny, None, "base").measure()


def test_an_edit_gets_its_own_adapters_and_strengths_and_moves_the_first_tokens(tiny):
    base = Measurement(tiny, None, "base").measure()
    edited = Measurement(tiny, EDIT, "trial").measure()
    adapters = sorted(p.name for p in tiny.path("measurements", "trial", "adapters").iterdir())
    assert adapters == sorted(f"{name}.gguf" for name in tiny.target_names.values())
    record = json.loads(tiny.path("measurements", "trial", "maths.json.inputs.json").read_text())
    assert record["edit"]["direction_layer"] == 0.4
    assert record["edit"]["strengths"] == {
        "blk.0.attn_output.weight": pytest.approx(0.8),
        "blk.1.attn_output.weight": pytest.approx(1.2),
        "blk.0.ffn_down.weight": pytest.approx(1.0),
        "blk.1.ffn_down.weight": pytest.approx(0.5),
    }
    rows = measure.compare(base, edited)
    assert rows[-1][0] == "Harmless KL from A" and float(rows[-1][2]) > 0
    assert float(measure.compare(base, base)[-1][2]) == 0


def test_saved_answers_are_not_judged_against_prompts_that_changed(tiny, monkeypatch):
    Measurement(tiny, None, "base").measure()

    def changed(spec):
        prompts = fake_prompts(spec)
        return [f"{prompts[0]} changed", *prompts[1:]] if spec.dataset == "walledai/StrongREJECT" else prompts

    monkeypatch.setattr(run, "load_prompts", changed)
    tiny.settings.measure.judge_revision = "another-commit"
    with pytest.raises(RunError, match="have changed since they were answered"):
        Measurement(tiny, None, "base").measure()


def test_a_limit_measures_the_first_prompts_and_knows_the_full_sizes(tiny):
    measurement = Measurement(tiny, None, "base-first-2", limit=2)
    results = measurement.measure()
    assert {name: len(r["finished"]) for name, r in results.items()} == dict.fromkeys(measure.SETS, 2)
    assert measurement.sizes == dict.fromkeys(measure.SETS, 3)


def test_measurements_of_different_prompts_are_not_compared(tiny):
    whole = Measurement(tiny, None, "base").measure()
    part = Measurement(tiny, None, "base-first-2", limit=2).measure()
    with pytest.raises(RunError, match="held-out harmful prompts differ in more than the edit and slots"):
        measure.compare(whole, part)


def test_a_direction_layer_outside_the_model_is_refused(tiny):
    edit = Edit("far", 2.5, EDIT.kernels)
    with pytest.raises(RunError, match=r"far reads its direction at layer 2\.5; this model has 2"):
        Measurement(tiny, edit, "far").measure()


def test_measure_and_compare_from_the_command_line_show_counts_only(tiny):
    config.write(tiny.folder.parent, "cli", tiny.settings, "made by a test")
    folder = tiny.folder.parent / "cli"
    runner = CliRunner(env={"COLUMNS": "250"})
    measured = runner.invoke(app, ["measure", str(folder), "--limit", "2"])
    assert measured.exit_code == 0, measured.output
    assert "candor measure: base-first-2" in measured.output
    assert "Harmful refused, judge" in measured.output
    assert "Measuring every prompt would take at most about" in measured.output
    assert "2 prompts can't keep all 8 slots busy" in measured.output
    compared = runner.invoke(app, ["compare", str(folder), "base-first-2", "base-first-2"])
    assert compared.exit_code == 0, compared.output
    assert "Harmless KL from A" in compared.output
    private = folder / "private" / "measurements" / "base-first-2"
    for name in measure.SETS:
        for answer in json.loads((private / f"{name}.json").read_text())["answers"]:
            assert len(answer.strip()) < 4 or answer.strip() not in measured.output + compared.output


def test_a_heretic_study_needs_a_trial():
    result = CliRunner().invoke(app, ["measure", "runs/none", "--heretic", "study.jsonl"])
    assert result.exit_code == 2
    assert "say which of the study's trials with --trial" in result.output


@pytest.mark.parametrize(
    ("answer", "gold", "correct"),
    [
        ("Six times seven.\nAnswer: 42", 42, True),
        ("The total is 1,000 apples.\nAnswer: **$1,000**", 1000, True),
        ("It is 3.50 dollars", 3.5, True),
        ("Answer: 41, though 42 was close", 42, False),
        ("No idea", 42, False),
        ("Answer: -2.5", -2.5, True),
    ],
)
def test_maths_answers_are_read_from_the_stated_answer_or_the_last_number(answer, gold, correct):
    assert measure.is_correct(answer, gold) is correct


def test_a_cut_off_answer_counts_only_if_it_stated_its_answer():
    assert not measure.is_correct("18 + 24 = 42. So the total is", 42, finished=False)
    assert measure.is_correct("So 42.\nAnswer: 42\nTo check, 6 times", 42, finished=False)


def test_gsm8k_answers_give_their_final_number():
    assert measure.gold_number("Work: 2 + 2 = 4\n#### 1,234") == 1234
    with pytest.raises(RunError, match="no '#### ' and number"):
        measure.gold_number("4")


def test_mcnemar_is_exact_and_symmetric():
    assert measure.mcnemar(0, 0) == 1
    assert measure.mcnemar(5, 5) == 1
    assert measure.mcnemar(10, 0) == pytest.approx(2 / 1024)
    assert measure.mcnemar(3, 9) == measure.mcnemar(9, 3) == pytest.approx(0.146, abs=0.001)


def test_strengths_follow_each_components_kernel():
    writers = [Writer("a.0.o", 0, "attn"), Writer("a.1.o", 1, "attn"), Writer("m.1.d", 1, "mlp")]
    names = {"a.0.o": "blk.0.attn", "a.1.o": "blk.1.attn", "m.1.d": "blk.1.ffn"}
    edit = Edit("e", None, {"attn": Kernel(1.5, 1.0, 1.0, 1.0)})
    assert edit.strengths(writers, names) == {"blk.0.attn": 1.0, "blk.1.attn": 1.5, "blk.1.ffn": 0.0}


def write_journal(path, settings):
    ops = [
        {"op_code": 2, "study_id": 0, "user_attr": {"settings": json.dumps(settings)}},
        {"op_code": 8, "trial_id": 0, "user_attr": {"index": 1, "direction_index": 12.5}},
        {"op_code": 8, "trial_id": 0, "user_attr": {"parameters": PARAMETERS, "refusals": 4}},
        {"op_code": 8, "trial_id": 0, "user_attr": {"base_refusals": 96, "kl_divergence": 0.03}},
    ]
    path.write_text("".join(json.dumps(op) + "\n" for op in ops))
    return path


def test_a_heretic_trial_is_read_as_an_edit(tmp_path):
    journal = write_journal(
        tmp_path / "study.jsonl", {"row_normalization": "pre", "orthogonalize_direction": True}
    )
    edit = measure.heretic_trial(journal, 1, "pre")
    assert edit.name == "heretic-1" and edit.direction_layer == 12.5
    assert edit.kernels["attn"] == Kernel(1.3, 16.5, 1.0, 13.5)
    with pytest.raises(RunError, match="has no finished trial 2"):
        measure.heretic_trial(journal, 2, "pre")


@pytest.mark.parametrize(
    ("settings", "problem"),
    [
        ({"row_normalization": "none"}, 'normalised rows as "none"'),
        ({"row_normalization": "pre", "winsorization_quantile": 0.95}, "winsorised"),
        ({"row_normalization": "pre", "orthogonalize_direction": False}, "orthogonalising"),
        ({"row_normalization": "pre"}, "orthogonalising"),
        (
            {
                "row_normalization": "pre",
                "orthogonalize_direction": True,
                "response_prefix": "\n</think>\n\n",
            },
            "response prefix",
        ),
    ],
)
def test_a_heretic_trial_built_differently_is_refused(tmp_path, settings, problem):
    journal = write_journal(tmp_path / "study.jsonl", settings)
    with pytest.raises(RunError, match=f"can't be rebuilt like for like: .*{problem}"):
        measure.heretic_trial(journal, 1, "pre")


def test_an_unreadable_journal_is_named(tmp_path):
    broken = tmp_path / "study.jsonl"
    broken.write_text("{not json\n")
    with pytest.raises(
        RunError, match=r"couldn't read .*study\.jsonl as a heretic study \(JSONDecodeError\)"
    ):
        measure.heretic_trial(broken, 1, "pre")


def recording(monkeypatch, method: str) -> list:
    """Every call to one of the server's methods: its prompts, and what it returned."""
    calls = []
    real = getattr(LlamaServer, method)

    def record(self, prompts, *args, **kwargs):
        result = real(self, prompts, *args, **kwargs)
        calls.append((prompts, result))
        return result

    monkeypatch.setattr(LlamaServer, method, record)
    return calls


def test_a_thinking_models_prompts_are_also_kept_with_their_thinking_closed(tiny):
    ids = thinking(tiny).prompt_ids()
    think, end = THINKING_TOKENS.index(THINK), THINKING_TOKENS.index(END_THINK)
    assert all(think in prompt and end not in prompt for prompt in ids["harmful_test"])
    for name in run.CLOSED_SETS:
        closed = ids[name + run.ANSWERING]
        assert [len(p) for p in closed] > [len(p) for p in ids[name]] and all(end in p for p in closed)


def after_the_end_tag(ids: dict[str, list[list[int]]], name: str) -> list[int]:
    closed = ids[name + run.ANSWERING][0]
    return closed[closed.index(THINKING_TOKENS.index(END_THINK)) + 1 :]


def test_kl_is_read_after_the_unedited_models_own_thinking_where_it_ended_by_itself(tiny, monkeypatch):
    thought = thinks_through(monkeypatch)
    ids = thinking(tiny).prompt_ids()
    tiny.thinking_traces()
    for name in run.KL_SETS:
        after = after_the_end_tag(ids, name)
        assert after and THINKING_TOKENS.index(END_THINK) not in after
        expected = [prompt + thought + after for prompt in ids[name][::2]]
        assert tiny.kl_ids(ids, name) == expected and len(expected) == 2
        # The first two prompts, of which the second's thinking ran out its budget.
        assert tiny.kl_ids(ids, name, limit=2) == expected[:1]
    key = tiny.thinking_key()
    record = json.loads(tiny.path("thinking", f"{key}.json").read_text())
    assert record["kept"] == dict.fromkeys(run.KL_SETS, 2)
    # The thinking is the model's own text: only its digest is outside the private folder.
    private = json.loads(tiny.path("private", "thinking", f"{key}.json").read_text())
    assert private == {
        name: [
            p if i % 2 == 0 else None
            for i, p in enumerate([prompt + thought + after_the_end_tag(ids, name) for prompt in ids[name]])
        ]
        for name in run.KL_SETS
    }


def test_the_unedited_thinking_is_kept_for_each_set_of_settings_and_never_sampled_over(tiny, monkeypatch):
    thinks_through(monkeypatch)
    first = thinking(tiny).thinking_traces()
    calls = []
    real = LlamaServer.thoughts
    monkeypatch.setattr(LlamaServer, "thoughts", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert tiny.thinking_traces() == first and calls == []
    tiny.settings.sampling.seed = 3
    tiny.thinking_traces()
    assert len(calls) == len(run.KL_SETS)
    # Going back finds the thinking sampled before, as it can't be sampled again exactly.
    tiny.settings.sampling.seed = 0
    assert tiny.thinking_traces() == first and len(calls) == len(run.KL_SETS)
    key = tiny.thinking_key()
    tiny.path("private", "thinking", f"{key}.json").unlink()
    with pytest.raises(
        RunError, match=r"missing or changed in private/thinking/.*restore it, or delete thinking/"
    ):
        tiny.thinking_traces()
    assert len(calls) == len(run.KL_SETS)


@pytest.mark.parametrize(
    "stage", ["adapters", "measurements", "scores", "exports", "directions_answering.pt"]
)
def test_the_unedited_thinking_doesnt_depend_on_the_stages_it_doesnt_use(tiny, monkeypatch, stage):
    key = thinking(tiny).thinking_key()
    monkeypatch.setitem(run.STAGES, stage, run.STAGES[stage] + 1)
    assert tiny.thinking_key() == key
    monkeypatch.setitem(run.STAGES, "thinking", run.STAGES["thinking"] + 1)
    assert tiny.thinking_key() != key


def test_a_limit_that_leaves_no_thinking_ended_by_itself_is_refused(tiny, monkeypatch):
    def last_only(self, prompts, sampling, seeds, thinking, done=None):
        return [None] * (len(prompts) - 1) + [[*self.tokenize("w2"), *self.tokenize(thinking.end)]]

    monkeypatch.setattr(LlamaServer, "thoughts", last_only)
    with pytest.raises(RunError, match="none of the first 2 held-out harmless prompts"):
        Measurement(thinking(tiny), None, "base", limit=2).measure()


def test_kl_cant_be_read_after_thinking_that_never_ends_by_itself(tiny, monkeypatch):
    monkeypatch.setattr(LlamaServer, "thoughts", lambda self, prompts, *a, **k: [None] * len(prompts))
    with pytest.raises(RunError, match="didn't end by itself on any of the 3 harmless test prompts"):
        thinking(tiny).thinking_traces()
    assert not tiny.path("thinking").exists() or not any(tiny.path("thinking").iterdir())


def test_a_thinking_model_is_judged_on_its_answers_alone_and_kl_is_read_where_they_start(tiny, monkeypatch):
    thinks_through(monkeypatch)
    thinking(tiny)
    generated, first_tokens = recording(monkeypatch, "generate"), recording(monkeypatch, "top_logprobs")
    results = Measurement(tiny, None, "base").measure()
    held_out = [answer for _, found in generated[: len(run.HELD_OUT)] for answer in found]
    assert FakeGuard.read == [a.text for a in held_out] and sum(a.thinking for a in held_out) > 0
    for name in measure.SETS:
        assert len(results[name]["thinking"]) == len(results[name]["forced"]) == 3
    ids = tiny.prompt_ids()
    # After the unedited model's own thinking, where it started, and after an empty thinking block.
    assert [read[0] for read in first_tokens] == [
        tiny.kl_ids(ids, "harmless_held_out"),
        ids["harmless_held_out"],
        ids["harmless_held_out" + run.ANSWERING],
    ]
    harmless = results["harmless_held_out"]
    assert [len(harmless[key]) for key in ("first_token", "first_token_thinking", "first_token_empty")] == [
        2,
        3,
        3,
    ]
    rows = dict(measure.summary(results))
    assert rows["Harmful, thinking cut"].endswith("/3") and rows["Maths, thinking tokens"].startswith(
        "median"
    )


def test_the_same_seed_samples_the_same_answers_and_another_is_still_compared(tiny, monkeypatch):
    thinks_through(monkeypatch)
    thinking(tiny)
    first = Measurement(tiny, None, "base").measure()
    again = Measurement(tiny, None, "again").measure()
    other = Measurement(tiny, None, "base-seed-5", seed=5).measure()
    assert first["maths"]["answers"] == again["maths"]["answers"] != other["maths"]["answers"]
    assert other["maths"]["seed"] == 5
    rows = {row[0]: row for row in measure.compare(first, other)}
    assert rows["Sampling seed"][1:3] == ("0", "5") and "Harmful, thinking cut" in rows
    assert "Sampling seed" not in {row[0] for row in measure.compare(first, again)}


def test_measurements_after_different_unedited_thinking_arent_compared(tiny, monkeypatch):
    thinks_through(monkeypatch)
    thinking(tiny)
    first = Measurement(tiny, None, "base").measure()

    def other_thinking(self, prompts, sampling, seeds, thinking, done=None):
        return [[*self.tokenize("w5"), *self.tokenize(thinking.end)]] * len(prompts)

    monkeypatch.setattr(LlamaServer, "thoughts", other_thinking)
    # A new seed thinks again, and here the unedited model's thinking comes out differently.
    tiny.settings.sampling.seed = 7
    other = Measurement(tiny, None, "other-thinking").measure()
    with pytest.raises(RunError, match="differ in more than the edit"):
        measure.compare(first, other)


def test_a_template_that_doesnt_open_the_thinking_is_refused(tiny):
    tiny.settings.deployment = config.Deployment(
        think_close="</think>", thinking_budget=6, thinking_budget_message=""
    )
    tiny.settings.scoring.answer_tokens = 16
    with pytest.raises(RunError, match="doesn't end a prompt by opening the thinking with <think>"):
        Measurement(tiny, None, "base").measure()


def test_a_greedy_model_that_doesnt_think_keeps_the_records_it_had_before_candor_sampled(tiny):
    tiny.gguf()
    measurement = Measurement(tiny, None, "base")
    inputs = measurement._answer_inputs("maths", [[1]], None, [])
    assert set(inputs) == {"paired", "edit", "slots"} and not {"sampling", "thinking"} & set(inputs["paired"])
    tiny.settings.sampling.temperature = 0.6
    inputs = measurement._answer_inputs("maths", [[1]], None, [])
    assert inputs["seed"] == 0 and inputs["paired"]["sampling"]["temperature"] == 0.6


def test_a_prompt_that_opens_thinking_needs_a_think_close(tiny):
    thinking(tiny).settings.deployment = config.Deployment()
    with pytest.raises(
        RunError, match=r"ends by opening the model's thinking, but deployment\.think_close isn't set"
    ):
        Measurement(tiny, None, "base").measure()


def test_comparing_thinking_measurements_shows_kl_at_each_place_it_is_read(tiny, monkeypatch):
    thinks_through(monkeypatch)
    thinking(tiny)
    base = Measurement(tiny, None, "base").measure()
    edited = Measurement(tiny, EDIT, "edited").measure()
    rows = {row[0]: row for row in measure.compare(base, edited)}
    for label in (
        "Harmless KL from A",
        "Harmless KL from A, empty thinking",
        "Harmless KL from A, thinking start",
    ):
        assert float(rows[label][2]) > 0
