import copy
import io
import json
import math
import re
import shutil

import huggingface_hub
import numpy as np
import pytest
import torch
from rich.console import Console
from safetensors.torch import save_file
from typer.testing import CliRunner

from candor import cli, config, measure, menu, residuals, run, source
from candor.cli import app
from candor.engine import LlamaServer, ServerError
from candor.kernel import Kernel
from candor.measure import Edit, Measurement
from candor.run import Run, RunError
from candor.targets import Writer

from .helpers import EDIT, PARAMETERS, on_disk, recorded_syncs, write_journal
from .tiny_gguf import END_THINK, THINK, THINKING_TOKENS
from .tiny_run import (
    ALWAYS_THINKING_TEMPLATE,
    CHECK_JUDGE,
    REFUSED,
    FakeGuard,
    fake_prompts,
    thinking,
    thinks_through,
)


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
    # Labels counted the old way are never compared with ones counted the new way.
    with pytest.raises(RunError, match="aren't of its answers as candor counts them now"):
        measure.load(tiny, "base")
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
    assert measure.load(tiny, "base")


def test_a_measurement_stopped_part_way_never_shows_another_edits_answers_or_labels(tiny, monkeypatch):
    Measurement(tiny, None, "finalist").measure()

    def stopped(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(measure, "Guard", stopped)
    with pytest.raises(KeyboardInterrupt):
        Measurement(tiny, EDIT, "finalist").measure()
    with pytest.raises(
        RunError, match=r"missing harmful_held_out\.judged\.json, harmless_held_out\.judged\.json"
    ):
        measure.load(tiny, "finalist")
    real = LlamaServer.generate
    calls = []

    def stopping(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(self, *args, **kwargs)

    monkeypatch.setattr(LlamaServer, "generate", stopping)
    other = Edit("other", EDIT.direction_layer, {"attn": Kernel(0.9, 1.0, 0.8, 1.0)})
    with pytest.raises(KeyboardInterrupt):
        Measurement(tiny, other, "finalist").measure()
    with pytest.raises(RunError, match=r"missing harmless_held_out\.json, maths\.json"):
        measure.load(tiny, "finalist")


def test_labels_of_other_answers_are_refused(tiny):
    Measurement(tiny, None, "base").measure()
    Measurement(tiny, EDIT, "trial").measure()
    for name in ("harmful_held_out.json", "harmful_held_out.json.inputs.json"):
        shutil.copy(tiny.path("measurements", "trial", name), tiny.path("measurements", "base", name))
    with pytest.raises(RunError, match="labels for the held-out harmful prompts aren't of its answers"):
        measure.load(tiny, "base")


class StrictGuard(FakeGuard):
    """A second judge that calls every answer a refusal, so its labels can be told from the judge's."""

    def replies(self, prompts, answers, batch=16, done=None):
        super().replies(prompts, answers, batch, done)
        return [REFUSED] * len(answers)


def second_judge(tiny, monkeypatch):
    """The tiny run with a second judge, read from the tiny model's folder, so its size is known."""
    model = tiny.settings.model.source
    tiny.settings.measure.second_judge, tiny.settings.measure.second_judge_revision = model, "commit"
    monkeypatch.setattr(
        measure, "Guard", lambda m, revision: (StrictGuard if m == model else FakeGuard)(m, revision)
    )
    return tiny


def test_a_second_judge_judges_the_saved_answers_into_files_of_its_own_and_nothing_else_again(
    tiny, monkeypatch
):
    first = Measurement(tiny, None, "base").measure()
    folder = tiny.path("measurements", "base")
    labels = {path.name: path.read_bytes() for path in folder.glob("*.judged.json*")}
    second = second_judge(tiny, monkeypatch).settings.measure.second_judge
    # Not finished until the second judge has judged it.
    with pytest.raises(
        RunError, match=r"missing harmful_held_out\.second_judge\.json, harmless_held_out\.second_judge\.json"
    ):
        measure.load(tiny, "base")
    assert menu.measurements_of(tiny.folder, tiny.settings) == []
    answered = counting_answers(monkeypatch)
    FakeGuard.events.clear()
    again = Measurement(tiny, None, "base").measure()
    assert answered == [] and FakeGuard.events == [("load", second), ("close", second)]
    assert {path.name: path.read_bytes() for path in folder.glob("*.judged.json*")} == labels
    for name in run.HELD_OUT:
        assert again[name][measure.SECOND]["judge_refused"] == [True, True, True]
        assert {key: value for key, value in again[name].items() if key != measure.SECOND} == first[name]
        record = json.loads(run.record_of(folder / f"{name}.second_judge.json").read_text())
        assert record["judge"] == [second, "commit"] and record["answers"] == first[name]["answers"]
        private = tiny.path("private", "measurements", "base", f"{name}.second_judge.json")
        assert len(json.loads(private.read_text())["judge_replies"]) == 3
    assert measure.load(tiny, "base") == again
    assert menu.measurements_of(tiny.folder, tiny.settings) == ["base"]
    harmful = again["harmful_held_out"]
    judging = harmful["judge_seconds"] + harmful[measure.SECOND]["judge_seconds"]
    assert measure.seconds(harmful) == harmful["seconds"] + judging
    # Unset, it reads as it was measured before.
    tiny.settings.measure.second_judge = None
    assert measure.load(tiny, "base") == first


def test_two_judges_judge_one_at_a_time_each_freed_before_the_next_loads(tiny, monkeypatch):
    second = second_judge(tiny, monkeypatch).settings.measure.second_judge
    results = Measurement(tiny, None, "base").measure()
    judge = tiny.settings.measure.judge
    assert FakeGuard.events == [("load", judge), ("close", judge), ("load", second), ("close", second)]
    assert results["harmful_held_out"]["judge_refused"] == [True, False, False]


def weights_folder(folder, count):
    """A model folder holding count weights, for a judge whose size is read from its headers."""
    folder.mkdir(parents=True)
    save_file({"w": torch.zeros(count)}, folder / "model.safetensors")
    return str(folder)


def test_a_judge_the_gpu_cant_hold_stops_the_measurement_before_it_loads(tiny, monkeypatch, tmp_path):
    monkeypatch.setattr(Run, "check_judge", CHECK_JUDGE)
    second = second_judge(tiny, monkeypatch).settings.measure.second_judge
    judge = tiny.settings.measure.judge = weights_folder(tmp_path / "small", 10)
    needed = 2 * sum(math.prod(shape) for shape in source.tensor_shapes(second).values()) * 1.1
    monkeypatch.setattr(residuals, "memory_budget", lambda device: int(needed) - 1)
    with pytest.raises(
        RunError, match=rf"the judge {re.escape(second)} needs about 0\.0 GB of GPU memory and this machine"
    ):
        Measurement(tiny, None, "base").measure()
    # The run's judge fits, so it judged before the second was refused.
    assert FakeGuard.events == [("load", judge), ("close", judge)]
    monkeypatch.setattr(residuals, "memory_budget", lambda device: int(needed) + 1)
    answered = counting_answers(monkeypatch)
    Measurement(tiny, None, "base").measure()
    assert answered == [] and FakeGuard.events[-2:] == [("load", second), ("close", second)]
    # The run's judge is held to it as well.
    tiny.settings.measure.judge_revision = "another-commit"
    monkeypatch.setattr(residuals, "memory_budget", lambda device: 1)
    FakeGuard.events.clear()
    with pytest.raises(RunError, match=rf"the judge {re.escape(judge)} needs about"):
        Measurement(tiny, None, "base").measure()
    assert FakeGuard.events == []


def test_a_downloaded_judge_is_sized_from_the_hubs_cache_without_the_network(tiny, monkeypatch, tmp_path):
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / "models--org--judge"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("abc")
    weights_folder(repo / "snapshots" / "abc", 10)
    monkeypatch.setattr(source, "get_safetensors_metadata", lambda *_, **__: pytest.fail("asked the Hub"))
    monkeypatch.setattr(residuals, "memory_budget", lambda device: 1)
    with pytest.raises(RunError, match="the judge org/judge needs about"):
        CHECK_JUDGE(tiny, "org/judge", None)
    monkeypatch.setattr(residuals, "memory_budget", lambda device: 1000)
    CHECK_JUDGE(tiny, "org/judge", "abc")


def test_a_measurement_older_measuring_code_made_is_refused_saying_measuring_again_starts_afresh(
    tiny, monkeypatch
):
    Measurement(tiny, None, "base").measure()
    version = run.STAGES["measurements"]
    monkeypatch.setitem(run.STAGES, "measurements", version + 1)
    # Whole, it still loads, to be looked at.
    assert measure.load(tiny, "base")
    afresh = (
        "it was made by older measuring code, so candor measure would measure it afresh and keep this "
        f"one as base-measuring-{version}"
    )
    second_judge(tiny, monkeypatch)
    with pytest.raises(RunError, match=rf"is missing .*second_judge\.json; {afresh}$"):
        measure.load(tiny, "base")
    tiny.settings.measure.second_judge = None
    monkeypatch.setattr(measure, "COUNTING", measure.COUNTING + 1)
    with pytest.raises(RunError, match=rf"aren't of its answers as candor counts them now; {afresh}$"):
        measure.load(tiny, "base")


def test_answering_a_set_again_discards_the_second_judges_labels_too(tiny, monkeypatch):
    Measurement(second_judge(tiny, monkeypatch), None, "finalist").measure()
    stopping_after(monkeypatch, 1)
    with pytest.raises(KeyboardInterrupt):
        Measurement(tiny, EDIT, "finalist").measure()
    assert not list(tiny.path("measurements", "finalist").glob("*.second_judge.json*"))


def test_compare_shows_the_second_judges_refusals_beside_the_judges_and_never_mixes_judges(tiny, monkeypatch):
    second_judge(tiny, monkeypatch)
    a = Measurement(tiny, None, "base").measure()
    b = Measurement(tiny, None, "base-slots-2", slots=2).measure()
    b["harmful_held_out"][measure.SECOND]["judge_refused"] = [False, True, True]
    labels = [row[0] for row in measure.compare(a, b)]
    assert labels[:3] == [
        "Harmful refused, judge",
        "Harmful refused, second judge",
        "Harmful refused, keywords",
    ]
    assert labels[labels.index("Harmless refused, judge") + 1] == "Harmless refused, second judge"
    rows = {row[0]: row for row in measure.compare(a, b)}
    low, high = measure.change_interval(1, 0, 3)
    interval, p = f"{round(low):+d}, {round(high):+d}", f"{measure.mcnemar(1, 0):.2g}"
    assert rows["Harmful refused, second judge"][1:] == ("3/3", "2/3", "-1", interval, "1 / 0", p)
    assert rows["Harmful refused, judge"][1:4] == ("1/3", "1/3", "+0")
    assert dict(measure.summary(a))["Harmless refused, second judge"] == "3/3"
    # A measurement the second judge hasn't judged is compared on the judge's labels alone.
    alone = {name: {k: v for k, v in r.items() if k != measure.SECOND} for name, r in a.items()}
    assert not [label for label, *_ in measure.compare(alone, b) if "second judge" in label]
    other = copy.deepcopy(b)
    other["harmless_held_out"][measure.SECOND]["judge"] = "another"
    with pytest.raises(RunError, match="held-out harmless prompts were judged by different second judges"):
        measure.compare(a, other)
    elsewhere = copy.deepcopy(b)
    elsewhere["harmful_held_out"][measure.SECOND]["judge_machine"] = {"device": "cuda", "dtype": "bfloat16"}
    rows = {row[0]: row for row in measure.compare(a, elsewhere)}
    assert rows["Second judge on"][1:3] == ("cpu, float32", "cuda, bfloat16") and "Judged on" not in rows


def test_measure_and_compare_show_the_second_judge_from_the_command_line(tiny, monkeypatch):
    folder = on_disk(second_judge(tiny, monkeypatch))
    runner = CliRunner(env={"COLUMNS": "250"})
    measured = runner.invoke(app, ["measure", str(folder), "--limit", "2"])
    assert measured.exit_code == 0, measured.output
    for step in ("Harmful: second judge", "Harmless: second judge"):
        assert f"{step}: 2 of 2" in measured.output
    assert "Harmful refused, second judge" in measured.output
    compared = runner.invoke(app, ["compare", str(folder), "base-first-2", "base-first-2"])
    assert compared.exit_code == 0 and "Harmless refused, second judge" in compared.output
    narrow = CliRunner(env={"COLUMNS": "80"}).invoke(
        app, ["compare", str(folder), "base-first-2", "base-first-2"]
    )
    assert narrow.exit_code == 0 and "…" not in narrow.output


def test_compare_leaves_out_prompts_either_judge_didnt_label_and_names_other_slots(tiny):
    a = Measurement(tiny, None, "base").measure()
    b = Measurement(tiny, None, "base-slots-2", slots=2).measure()
    assert a["harmful_held_out"]["judge_refused"] == [True, False, False]
    b["harmful_held_out"]["judge_refused"][0] = None
    rows = {row[0]: row for row in measure.compare(a, b)}
    assert (
        rows["Harmful refused, judge"][1:4] == ("0/2", "0/2", "+0")
        and rows["Harmful refused, judge"][5] == "0 / 0"
    )
    assert rows["Slots"][1:3] == (str(a["maths"]["slots"]), "2")
    assert dict(measure.summary(b))["Harmful refused, judge"] == "0/2 (1 unlabelled)"
    assert "Slots" not in {row[0] for row in measure.compare(a, a)}


def test_the_median_thinking_of_an_even_count_is_halfway_between_the_middle_two(tiny):
    results = Measurement(tiny, None, "base").measure()
    results["maths"]["thinking"] = [10, 1, 3, 2]
    assert dict(measure.summary(results))["Maths, thinking tokens"] == "median 2.5, most 10"


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


def stopping_after(monkeypatch, saved: int) -> None:
    """The measurement stops, as at a signal, once this many answers are saved."""
    real = LlamaServer.generate
    count = []

    def generate(self, *args, each=None, **kwargs):
        def saving(place, answer):
            assert each is not None
            each(place, answer)
            count.append(1)
            if len(count) == saved:
                raise KeyboardInterrupt

        return real(self, *args, each=saving, **kwargs)

    monkeypatch.setattr(LlamaServer, "generate", generate)


def test_a_set_stopped_part_way_answers_only_the_prompts_left(tiny, monkeypatch):
    # Sampled, one answer at a time, so each answer depends on its seed alone.
    tiny.settings.sampling = config.Sampling(temperature=1.0, top_k=0, min_p=0.0)
    tiny.settings.slots.measure = 1
    whole = Measurement(tiny, None, "whole").measure()
    with monkeypatch.context() as stopped:
        stopping_after(stopped, 2)
        with pytest.raises(KeyboardInterrupt):
            Measurement(tiny, None, "base").measure()
    journal = tiny.path("private", "measurements", "base", "harmful_held_out.answering.jsonl")
    earlier = max(json.loads(line)["at"] for line in journal.read_text().splitlines()[1:])
    answered = counting_answers(monkeypatch)
    synced = recorded_syncs(monkeypatch)
    resumed = Measurement(tiny, None, "base").measure()
    # Each answer has its own seed, so the one asked again is the one an unbroken run gives.
    assert answered == [1, 3, 3] and not journal.exists()
    # The answers kept are written whole before any more are added.
    assert f"{journal.name}.building" in {name for name, _ in synced}
    private = tiny.path("private", "measurements")
    for name in measure.SETS:
        assert (private / "base" / f"{name}.json").read_text() == (
            private / "whole" / f"{name}.json"
        ).read_text()
        assert resumed[name]["finished"] == whole[name]["finished"]
    assert resumed["harmful_held_out"]["answer_seconds"] > earlier


def test_a_saved_answer_a_stop_cut_short_is_left_out(tmp_path):
    journal = tmp_path / "harmful_held_out.answering.jsonl"
    journal.write_text('{"record": {}}\n{"i": 0, "at": 1.0}\n{"i": 1, "answ')
    assert measure._journal(journal) == [{"record": {}}, {"i": 0, "at": 1.0}]


def test_answers_saved_from_other_inputs_are_answered_again(tiny, monkeypatch):
    with monkeypatch.context() as stopped:
        stopping_after(stopped, 2)
        with pytest.raises(KeyboardInterrupt):
            Measurement(tiny, None, "base").measure()
    answered = counting_answers(monkeypatch)
    Measurement(tiny, None, "base", slots=2).measure()
    assert answered == [3, 3, 3]


def test_a_failed_read_of_first_tokens_loses_no_answers(tiny, monkeypatch):
    def broken(*args, **kwargs):
        raise ServerError("injected")

    answered = counting_answers(monkeypatch)
    with monkeypatch.context() as failing:
        failing.setattr(LlamaServer, "top_logprobs", broken)
        with pytest.raises(ServerError, match="injected"):
            Measurement(tiny, None, "base").measure()
    # The harmless set's first tokens are read before it is answered, so nothing was answered for them.
    assert answered == [3]


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
    with pytest.raises(RunError, match="differ in more than the edit, seed, slots and machine"):
        measure.compare(q4, q8)


def test_measurements_with_other_settings_are_not_compared(tiny):
    first = Measurement(tiny, None, "base").measure()
    tiny.settings.measure.answer_tokens = 8
    other = Measurement(tiny, None, "shorter").measure()
    with pytest.raises(RunError, match="held-out harmful prompts differ in more than the edit, seed, slots"):
        measure.compare(first, other)
    tiny.settings.measure.answer_tokens = 16
    assert measure.compare(first, Measurement(tiny, None, "again", slots=2).measure())


def test_compare_names_the_machines_and_leaves_out_kl_read_on_another_backend(tiny):
    here = Measurement(tiny, None, "base").measure()
    assert here["maths"]["machine"] == tiny.machine()
    assert here["harmful_held_out"]["judge_machine"] == FakeGuard.machine
    assert measure.compare(here, here)[-1][0] == "Harmless KL from A"

    def on(machine, judged=None):
        sets = {name: dict(results, machine=machine) for name, results in here.items()}
        sets["harmful_held_out"]["judge_machine"] = judged or FakeGuard.machine
        return sets

    a40 = {"platform": "linux-cuda", "gpus": ["NVIDIA A40"]}
    rows = {row[0]: row for row in measure.compare(here, on(a40, {"device": "cuda", "dtype": "bfloat16"}))}
    assert rows["Harmless KL from A"][1:3] == ("", "other backend")
    assert rows["GPU"][1:3] == (", ".join(tiny.machine()["gpus"]), "NVIDIA A40")
    assert rows["Judged on"][1:3] == ("cpu, float32", "cuda, bfloat16")
    # Another card on the same backend is named, and its KL still shown.
    h100 = {"platform": "linux-cuda", "gpus": ["NVIDIA H100"]}
    rows = {row[0]: row for row in measure.compare(on(a40), on(h100))}
    assert (
        rows["GPU"][1:3] == ("NVIDIA A40", "NVIDIA H100") and rows["Harmless KL from A"][2] != "other backend"
    )
    # KL goes by the machine its own set was read on, which a set answered again apart may not share.
    mixed = on(a40)
    mixed["harmless_held_out"]["machine"] = here["harmless_held_out"]["machine"]
    rows = {row[0]: row for row in measure.compare(here, mixed)}
    assert rows["Harmless KL from A"][2] != "other backend"
    assert rows["GPU"][2] == f"NVIDIA A40; {', '.join(tiny.machine()['gpus'])}"
    # A new torch alone isn't another judge.
    newer = on(here["maths"]["machine"], FakeGuard.machine | {"torch": "newer"})
    assert "Judged on" not in {row[0] for row in measure.compare(here, newer)}


def test_measurements_by_other_measuring_code_are_measured_again_and_not_compared(tiny, monkeypatch):
    before = Measurement(tiny, None, "base").measure()
    monkeypatch.setitem(run.STAGES, "measurements", run.STAGES["measurements"] + 1)
    answered = counting_answers(monkeypatch)
    after = Measurement(tiny, None, "base").measure()
    assert answered == [3, 3, 3]
    with pytest.raises(RunError, match="a setting or candor's measuring changed between them"):
        measure.compare(before, after)


def test_what_older_measuring_code_measured_is_kept_under_a_name_of_its_own(tiny, monkeypatch):
    older = Measurement(tiny, None, "base").measure()
    version = run.STAGES["measurements"]
    monkeypatch.setitem(run.STAGES, "measurements", version + 1)
    Measurement(tiny, None, "base").measure()
    assert measure.load(tiny, f"base-measuring-{version}") == older
    assert tiny.path("private", "measurements", f"base-measuring-{version}").is_dir()
    assert measure.measured_by(tiny.path("measurements", "base")) == version + 1
    # Never measured over, even when that name is taken.
    monkeypatch.setitem(run.STAGES, "measurements", version + 2)
    tiny.path("measurements", f"base-measuring-{version + 1}").mkdir()
    with pytest.raises(RunError, match=f"base-measuring-{version + 1} is taken"):
        Measurement(tiny, None, "base").measure()
    assert measure.measured_by(tiny.path("measurements", "base")) == version + 1


def test_a_measurement_answers_slots_measure_at_once_unless_given_another_number(tiny, monkeypatch):
    started = []
    real = LlamaServer.start

    def start(self, *args, **kwargs):
        started.append((self.parallel, self.slots_setting))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(LlamaServer, "start", start)
    tiny.settings.slots.measure = 2
    for name, slots in (("two", None), ("three", 3)):
        Measurement(tiny, None, name, slots=slots, limit=1).measure()
        assert started[-1] == (slots or 2, "--slots, or slots.measure in candor.toml")
        record = run.record_of(tiny.path("measurements", name, "maths.json"))
        assert json.loads(record.read_text())["slots"] == (slots or 2)


def test_measure_names_a_measurement_for_slots_other_than_8_from_the_settings_or_the_command(tiny):
    tiny.settings.slots.measure = 2
    folder = on_disk(tiny)
    runner = CliRunner(env={"COLUMNS": "250"})
    for given, name in (([], "base-slots-2-first-1"), (["--slots", "8"], "base-first-1")):
        measured = runner.invoke(app, ["measure", str(folder), "--limit", "1", *given])
        assert measured.exit_code == 0, measured.output
        assert f"candor measure: {name}" in measured.output


def test_a_name_candor_gives_another_measurement_is_refused(tiny):
    folder = on_disk(tiny)
    runner = CliRunner(env={"COLUMNS": "250"})
    for taken in ("base", "trial-3", "heretic-2-exported"):
        measured = runner.invoke(app, ["measure", str(folder), "--limit", "1", "--name", taken])
        assert measured.exit_code == 1 and "it could replace one; choose another" in measured.output
    assert not (folder / "measurements").exists()
    for name in ("base-first-1", "regress-before"):
        measured = runner.invoke(app, ["measure", str(folder), "--limit", "1", "--name", name])
        assert measured.exit_code == 0, measured.output


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
    low, high = map(float, rows[-1][4].split(", "))
    assert low <= float(rows[-1][2]) <= high
    assert float(measure.compare(base, base)[-1][2]) == 0


def test_a_full_edit_is_measured_with_one_adapter_built_at_its_strengths(tiny):
    tiny.settings.edit.row_norm = "full"
    base = Measurement(tiny, None, "base").measure()
    edited = Measurement(tiny, EDIT, "trial").measure()
    assert sorted(p.name for p in tiny.path("measurements", "trial", "adapters").iterdir()) == [
        "edit.gguf",
        "edit.gguf.inputs.json",
    ]
    assert float(measure.compare(base, edited)[-1][2]) > 0


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
    with pytest.raises(RunError, match="held-out harmful prompts differ in more than the edit, seed, slots"):
        measure.compare(whole, part)


def test_a_direction_layer_outside_the_model_is_refused(tiny):
    edit = Edit("far", 2.5, EDIT.kernels)
    with pytest.raises(RunError, match=r"far reads its direction at layer 2\.5; this model has 2"):
        Measurement(tiny, edit, "far").measure()


def test_measure_and_compare_from_the_command_line_show_counts_only(tiny):
    folder = on_disk(tiny)
    runner = CliRunner(env={"COLUMNS": "250"})
    measured = runner.invoke(app, ["measure", str(folder), "--limit", "2"])
    assert measured.exit_code == 0, measured.output
    assert "candor measure: base-first-2" in measured.output
    assert "Harmful refused, judge" in measured.output
    assert "Measuring every prompt would take at most about" in measured.output
    assert "2 prompts can't keep all 8 slots busy" in measured.output
    for step in ("Harmful: answering", "Harmless: first tokens", "Harmless: judging"):
        assert f"{step}: 2 of 2" in measured.output
    compared = runner.invoke(app, ["compare", str(folder), "base-first-2", "base-first-2"])
    assert compared.exit_code == 0, compared.output
    assert "Harmless KL from A" in compared.output
    # In an 80-column terminal every label still shows whole, wrapped rather than cut.
    narrow = CliRunner(env={"COLUMNS": "80"}).invoke(
        app, ["compare", str(folder), "base-first-2", "base-first-2"]
    )
    assert "Harmless" in narrow.output and "\u2026" not in narrow.output
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
        ("Answer: \\boxed{18}\nCheck: 18 + 2 = 20", 18, True),
        ("She spent more than she had.\nAnswer: -$5", -5, True),
        ("A loss of -$5", -5, True),
    ],
)
def test_maths_answers_are_read_from_the_stated_answer_or_the_last_number(answer, gold, correct):
    assert measure.is_correct(answer, gold) is correct


def test_a_cut_off_answer_counts_only_if_it_stated_its_answer():
    assert not measure.is_correct("18 + 24 = 42. So the total is", 42, finished=False)
    assert measure.is_correct("So 42.\nAnswer: 42\nTo check, 6 times", 42, finished=False)
    assert measure.is_correct("So 42.\nAnswer: \\boxed{42}\nTo check, 6 times", 42, finished=False)


def test_gsm8k_answers_give_their_final_number():
    assert measure.gold_number("Work: 2 + 2 = 4\n#### 1,234") == 1234
    with pytest.raises(RunError, match="no '#### ' and number"):
        measure.gold_number("4")


def test_a_change_has_an_interval_even_when_nothing_differs():
    # K2 trial 205's maths against the unedited model's: 1,222 and 1,209 of 1,319 right.
    assert measure.change_interval(62, 49, 1319) == (
        pytest.approx(-33.8, abs=0.05),
        pytest.approx(7.8, abs=0.05),
    )
    assert measure.change_interval(0, 0, 300) == (
        pytest.approx(-2.75, abs=0.01),
        pytest.approx(2.75, abs=0.01),
    )
    assert measure.change_interval(0, 20, 300)[0] > 0
    assert measure.mean_interval([0.1] * 10) == (pytest.approx(0.1), pytest.approx(0.1))
    # Too few to bootstrap, and never outside the values themselves.
    assert measure.mean_interval([0.5]) is None and measure.mean_interval([0.0, 0.3]) is None
    assert interval([0.0] * 98 + [0.01, 0.2])[0] >= 0
    low, high = interval([0.0, 1.0] * 50)
    assert 0.35 < low < 0.45 and 0.55 < high < 0.65 and interval([0.0, 1.0] * 50) == (low, high)


def interval(values: list[float]) -> tuple[float, float]:
    found = measure.mean_interval(values)
    assert found is not None
    return found


def test_a_mean_carried_by_a_few_prompts_is_still_in_its_interval_most_of_the_time():
    # Skewed as KL is: a plain percentile bootstrap held this mean only 82% of the time.
    draws = np.random.default_rng(1)
    held = []
    for _ in range(400):
        low, high = interval(list(draws.lognormal(0, 1.5, 100)))
        held.append(low <= np.exp(1.5**2 / 2) <= high)
    assert sum(held) / len(held) > 0.88


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


def heretic_journal(path, settings):
    return write_journal(path, 12.5, settings, PARAMETERS)


def test_a_heretic_trial_is_read_as_an_edit(tmp_path):
    journal = heretic_journal(
        tmp_path / "study.jsonl", {"row_normalization": "pre", "orthogonalize_direction": True}
    )
    edit = measure.heretic_trial(journal, 1, "pre")
    assert edit.name == "heretic-1" and edit.direction_layer == 12.5
    assert edit.kernels["attn"] == Kernel(1.3, 16.5, 1.0, 13.5)
    with pytest.raises(RunError, match="has no finished trial 2"):
        measure.heretic_trial(journal, 2, "pre")


def test_a_heretic_trial_on_a_model_with_experts_is_refused(tmp_path):
    journal = heretic_journal(
        tmp_path / "study.jsonl", {"row_normalization": "pre", "orthogonalize_direction": True}
    )
    with pytest.raises(RunError, match=r"whether heretic edited the model's experts.*--outside"):
        measure.heretic_trial(journal, 1, "pre", experts=True)


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
    journal = heretic_journal(tmp_path / "study.jsonl", settings)
    with pytest.raises(RunError, match=f"can't be rebuilt like for like: .*{problem}"):
        measure.heretic_trial(journal, 1, "pre")


def test_a_full_heretic_trial_read_after_the_runs_think_close_is_rebuilt_where_the_answer_starts(tmp_path):
    settings = {
        "row_normalization": "full",
        "orthogonalize_direction": True,
        "response_prefix": "\n</think>\n\n",
    }
    journal = heretic_journal(tmp_path / "study.jsonl", settings)
    edit = measure.heretic_trial(journal, 1, "full", "\n</think>\n\n")
    assert edit.answering and edit.direction_layer == 12.5
    with pytest.raises(RunError, match=r"a response prefix, .* that isn't the run's deployment\.think_close"):
        measure.heretic_trial(journal, 1, "full", "</think>")
    with pytest.raises(RunError, match='normalised rows as "full"'):
        measure.heretic_trial(journal, 1, "pre", "\n</think>\n\n")
    plain = heretic_journal(tmp_path / "plain.jsonl", settings | {"response_prefix": ""})
    assert not measure.heretic_trial(plain, 1, "full").answering
    ranked = heretic_journal(tmp_path / "rank.jsonl", settings | {"full_normalization_lora_rank": 4})
    with pytest.raises(RunError, match='built "full" edits at rank 4, and candor at 3'):
        measure.heretic_trial(ranked, 1, "full", "\n</think>\n\n")


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


def test_unedited_thinking_stopped_part_way_resumes_from_the_set_it_stopped_in(tiny, monkeypatch):
    asked = []
    thinks_through(monkeypatch, asked=asked)
    whole = thinking(tiny).thinking_traces()
    key = tiny.thinking_key()
    private = tiny.path("private", "thinking")
    assert [p.name for p in private.iterdir()] == [f"{key}.json"]
    tiny.discard(tiny.path("thinking", f"{key}.json"))
    (private / f"{key}.json").unlink()

    real = LlamaServer.thoughts

    def stopped_in_the_second_set(self, prompts, *args, **kwargs):
        if len(asked) == len(run.KL_SETS) + 1:
            raise KeyboardInterrupt
        return real(self, prompts, *args, **kwargs)

    monkeypatch.setattr(LlamaServer, "thoughts", stopped_in_the_second_set)
    with pytest.raises(KeyboardInterrupt):
        tiny.thinking_traces()
    first, second = run.KL_SETS
    assert [p.name for p in private.iterdir()] == [f"{key}.{first}.json"]
    asked.clear()
    assert tiny.thinking_traces() == whole
    assert [prompts for prompts, *_ in asked] == [tiny.prompt_ids()[second]]
    assert [p.name for p in private.iterdir()] == [f"{key}.json"]
    # A set kept along the way that a pull from a pod brought back again goes once the record is there.
    (private / f"{key}.{first}.json").write_text("[]")
    assert tiny.thinking_traces() == whole
    assert [p.name for p in private.iterdir()] == [f"{key}.json"]


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


def test_a_thinking_runs_prompts_are_each_counted_once(tiny, monkeypatch):
    thinks_through(monkeypatch)
    done = thinking(tiny).prepare()
    texts, _ = tiny.prompt_texts()
    assert done["prompts"] == sum(map(len, texts.values()))


def test_another_seed_is_refused_for_a_run_that_answers_greedily(tiny):
    folder = on_disk(tiny)
    result = CliRunner().invoke(app, ["measure", str(folder), "--seed", "1"])
    assert result.exit_code == cli.FAILED_CHECK and "answers greedily" in result.output
    assert not (folder / "measurements").exists()


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


def test_a_model_whose_thinking_ends_at_its_end_tag_answers_and_reads_kl_right_after_it(tiny, monkeypatch):
    # As K2-Horizon's template writes an empty thinking block: the end tag, then the answer at once.
    thought = thinks_through(monkeypatch)
    thinking(tiny, template=ALWAYS_THINKING_TEMPLATE, close=END_THINK)
    ids = tiny.prompt_ids()
    end = THINKING_TOKENS.index(END_THINK)
    for name in run.CLOSED_SETS:
        assert ids[name + run.ANSWERING] == [[*prompt, end] for prompt in ids[name]]
    traces = tiny.thinking_traces()["harmless_test"]
    assert traces[0] == ids["harmless_test"][0] + thought and thought[-1] == end
    results = Measurement(tiny, None, "base").measure()
    assert sum(results["maths"]["thinking"]) > 0
    assert results["harmless_held_out"]["first_token"] and results["harmless_held_out"]["first_token_empty"]


def test_a_think_close_the_prompts_dont_open_is_caught_as_they_are_built(tiny):
    # As when K2-Horizon's reasoning effort is changed and think_close isn't: the prompts open another tag.
    thinking(tiny, template=ALWAYS_THINKING_TEMPLATE, close=END_THINK)
    tiny.settings.deployment.chat_template_kwargs = {"reasoning_effort": "low"}
    with pytest.raises(RunError, match="doesn't end a prompt by opening the thinking with <think>"):
        tiny.prompt_ids()
    assert not tiny.path("prompts.json").exists()


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


def test_progress_shows_as_lines_only_where_no_bar_can_be_seen():
    shown = io.StringIO()
    done = cli._lines(Console(file=shown, width=200), "Maths: answering", 20)
    for _ in range(20):
        done()
    assert shown.getvalue().splitlines() == [f"Maths: answering: {n} of 20" for n in range(2, 21, 2)]
    terminal = io.StringIO()
    done = cli._lines(Console(file=terminal, force_terminal=True), "Maths: answering", 20)
    done()
    assert terminal.getvalue() == ""
