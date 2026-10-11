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
from .tiny_run import FakeGuard, fake_prompts

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
