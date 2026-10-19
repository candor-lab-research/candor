"""candor's menus, driven by a scripted set of answers, and their questions by keys in a terminal."""

import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import optuna
import pytest
import typer
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from typer.testing import CliRunner

from candor import cli, config, measure, menu, run, search
from candor.menu import BACK, Menus, Option, Questions, Status
from candor.run import Run
from candor.search import Search

from .helpers import settings, small

DOWN, ENTER, CTRL_C, CTRL_D = "\x1b[B", "\r", "\x03", "\x04"


class Scripted:
    """Answers the menus' questions in turn, each an answer or a function of the options shown, and
    keeps what was asked and which option each question started on. Choosing an option that is
    disabled or isn't there fails the test."""

    def __init__(self, *answers: Any):
        self.answers = list(answers)
        self.asked: list[tuple[str, list[Option]]] = []
        self.defaults: list[Any] = []

    def _next(self, message: str, options: list[Option]) -> Any:
        self.asked.append((message, options))
        assert self.answers, f"nothing left to answer {message!r}"
        answer = self.answers.pop(0)
        return answer(options) if callable(answer) else answer

    def choose(self, message: str, options: list[Option | None], default: Any = None) -> Any:
        shown = [o for o in options if o is not None]
        self.defaults.append(default)
        answer = self._next(message, shown)
        if answer is not None:
            [chosen] = [o for o in shown if o.value == answer]
            assert chosen.disabled is None, f"{chosen.title!r} is disabled: {chosen.disabled}"
        return answer

    def choose_two(self, message: str, options: list[Option]) -> list[Any] | None:
        return self._next(message, options)

    def text(self, message: str, check: Callable[[str], str | None] | None = None) -> str | None:
        answer = self._next(message, [])
        if answer is not None and check:
            assert check(answer) is None
        return answer

    def options(self, message: str) -> dict[Any, Option]:
        """The options the last question so worded showed, by value."""
        return {o.value: o for o in next(o for m, o in reversed(self.asked) if m == message)}


def titled(start: str):
    """The value of the option whose title starts so."""
    return lambda options: next(o.value for o in options if o.title.startswith(start))


def run_menus(root: Path, *answers) -> tuple[Scripted, str]:
    """The menus answered so, from the list of runs in root: what they asked, and what they showed."""
    out = io.StringIO()
    asked = Scripted(*answers)
    Menus(asked, Console(file=out, width=120), root).start()
    return asked, out.getvalue()


def a_run(root: Path, name: str = "run", source: str | None = None) -> Path:
    made = settings() if source is None else settings(source=source)
    config.write(root, name, made, "made by a test")
    return root / name


def measured(folder: Path, *names: str, code: int | None = None) -> None:
    """Finished measurements, judged too, as far as their files go, made by this version of the
    measuring code: candor's now unless given."""
    for name in names:
        (folder / "measurements" / name).mkdir(parents=True)
        for file in measure.finished(config.Measure()):
            (folder / "measurements" / name / file).write_text("{}")
        made = {"code": {"measurements": code or run.STAGES["measurements"]}}
        run.record_of(folder / "measurements" / name / "maths.json").write_text(json.dumps(made))


def test_a_runs_status_is_read_from_its_files(tmp_path):
    folder = a_run(tmp_path)
    found = menu.status(folder)
    assert found.built == dict.fromkeys(menu.STAGES, False) and not found.prepared
    assert found.line() == "not prepared"
    for stage, path in menu.STAGES.items():
        (folder / path).mkdir() if stage in ("model", "adapters") else (folder / path).touch()
    measured(folder, "base", "trial-12-exported")
    # Answered but not yet judged, so not finished.
    (folder / "measurements" / "trial-3").mkdir()
    for name in measure.SETS:
        (folder / "measurements" / "trial-3" / f"{name}.json").write_text("{}")
    # An export counts while its GGUF is here.
    for number, files in [(12, ["m.gguf"]), (13, ["m.gguf.inputs.json"])]:
        (folder / "exports" / f"trial-{number}").mkdir(parents=True)
        for file in files:
            (folder / "exports" / f"trial-{number}" / file).touch()
    found = menu.status(folder)
    assert found.prepared and found.measurements == ["base", "trial-12-exported"] and found.exports == [12]
    assert found.line() == "prepared · 2 measured · 1 exported"
    # A folder config.write left part way isn't a run.
    (tmp_path / ".run.partial-1").mkdir()
    (tmp_path / ".run.partial-1" / config.FILE).write_text("")
    assert [r.folder.name for r in menu.runs_in(tmp_path)] == ["run"]


def odd_search(folder: Path, kind: str) -> None:
    (folder / "search").mkdir()
    if kind == "garbage":
        (folder / "search" / "study.jsonl").write_text("not a journal\n")
        return
    # A study under candor's name whose trial lacks what candor records: anything, or all but its number.
    storage = search._storage(folder / "search")
    study = optuna.create_study(study_name=search.STUDIES["one layer"], storage=storage)
    trial = study.ask()
    if kind == "numbered":
        trial.set_user_attr("number", 1)
    study.tell(trial, 1.0)


@pytest.mark.parametrize("kind", ["garbage", "foreign", "numbered"])
def test_a_search_that_cant_be_read_is_noted_and_the_menus_carry_on(tmp_path, kind):
    folder = a_run(tmp_path)
    odd_search(folder, kind)
    found = menu.status(folder)
    assert found.found is None and found.problem and "the search can't be read" in found.problem
    asked, shown = run_menus(tmp_path, folder, BACK, "quit")
    assert "the search can't be read" in shown
    assert asked.options("run")["trials"].disabled == "nothing searched yet"


def test_settings_that_dont_load_are_said_so(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / config.FILE).write_text("not = [valid")
    [found] = menu.runs_in(tmp_path)
    assert found.settings is None and found.line() == "its settings don't load"
    _, shown = run_menus(tmp_path, titled("broken"), "quit")
    assert "candor.toml" in shown


def a_status(folder: Path, prepared: bool, trials: int, measurements: list[str]) -> Status:
    built = dict.fromkeys(menu.STAGES, prepared)
    return Status(folder, settings(), None, built, None, trials, 600, measurements, [])


@pytest.mark.parametrize(
    ("prepared", "trials", "measurements", "step"),
    [
        (False, 0, [], "run"),
        (True, 0, [], "search"),
        # Searched on a pod, which kept the model: carrying the search on doesn't need it here.
        (False, 313, [], "search"),
        (True, 600, ["base-first-32"], "base"),
        (True, 600, ["base"], "trials"),
    ],
)
def test_a_run_starts_on_the_step_it_is_ready_for(tmp_path, prepared, trials, measurements, step):
    menus = Menus(Scripted(), Console(file=io.StringIO()), tmp_path)
    assert menus._next(a_status(tmp_path, prepared, trials, measurements), "base") == step


def test_the_run_menu_starts_on_its_next_step_and_the_runs_on_the_one_last_opened(tmp_path):
    folder = a_run(tmp_path)
    asked, _ = run_menus(tmp_path, folder, BACK, "quit")
    assert asked.defaults == [None, "run", folder]
    assert asked.options("run")["run"].title.endswith("← next")
    # Search builds what it needs, so it is never shut off.
    assert asked.options("run")["search"].disabled is None


@pytest.mark.parametrize(
    ("where", "calls"),
    [("here", [{"trial": None, "limit": None, "exported": False}]), ("pod", [])],
)
def test_measuring_the_unedited_model_runs_the_command_or_shows_it_for_a_pod(
    tmp_path, monkeypatch, where, calls
):
    folder = a_run(tmp_path)
    called = []
    monkeypatch.setattr(cli, "measure_command", lambda folder, **given: called.append(given))
    asked, shown = run_menus(tmp_path, folder, "base", "every", where, BACK, "quit")
    assert called == calls
    assert asked.asked[3][0] == f"$ candor measure {folder}"
    # The run's panel shows again once a command has run here, as what it holds may have changed.
    assert shown.count("╭─ run ") == 1 + len(calls)
    if where == "pod":
        assert (
            "  # the run, if the pod doesn't have it yet:\n"
            f"  scripts/pod.sh push {folder}\n"
            f"  scripts/pod.sh start run-measure-base candor measure {folder}\n"
            f"  scripts/pod.sh wait run-measure-base {folder}\n"
        ) in shown


def test_a_quick_measurement_is_as_long_as_the_unedited_models_it_pairs_with(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(cli, "measure_command", lambda folder, **given: called.append(given["limit"]))
    folder = a_run(tmp_path)
    run_menus(tmp_path, folder, "base", menu.QUICK, "here", BACK, "quit")
    # Only the unedited model's own quick measurements, at the run's settings, set the length.
    measured(folder, "base-first-64", "outside-first-16", "base-seed-1-first-16")
    asked, _ = run_menus(tmp_path, folder, "base", 64, "here", BACK, "quit")
    assert called == [menu.QUICK, 64]
    assert list(asked.options("Measure the unedited model")) == ["every", 64, BACK]


@pytest.mark.parametrize(
    ("raised", "said"),
    [
        (KeyboardInterrupt(), "Running it again carries on where it stopped"),
        (typer.Exit(1), ""),
        (typer.BadParameter("no such thing"), "no such thing"),
    ],
)
def test_a_command_however_it_ends_comes_back_to_the_menu(tmp_path, monkeypatch, raised, said):
    folder = a_run(tmp_path)

    def ended(folder):
        raise raised

    monkeypatch.setattr(cli, "run", ended)
    asked, shown = run_menus(tmp_path, folder, "run", "here", None, None)
    assert said in shown and f"$ candor run {folder}" in shown
    assert [message for message, _ in asked.asked][-2:] == ["run", f"Runs in {tmp_path}/"]


def test_stopping_a_command_that_isnt_a_job_doesnt_promise_it_carries_on(tmp_path, monkeypatch):
    folder = a_run(tmp_path)
    measured(folder, "base", "trial-4")

    def stopped(folder, first, second):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "compare", stopped)
    _, shown = run_menus(tmp_path, folder, "compare", ["base", "trial-4"], BACK, "quit")
    assert "Stopped." in shown and "carries on" not in shown


def test_the_run_menu_remembers_the_last_choice_and_marks_what_is_done(tmp_path, monkeypatch):
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)
    folder = a_run(tmp_path)
    measured(folder, "base")
    asked, _ = run_menus(tmp_path, folder, "settings", "chat", BACK, BACK, "quit")
    assert asked.defaults[1:4] == ["run", "settings", None]
    assert asked.options("run")["base"].title.endswith("✓ measured")
    # The unedited model is served from the run's GGUF, which isn't here.
    assert asked.options("Chat with which model?")["base"].disabled == "its GGUF isn't here"
    assert a_status(folder, True, 313, ["base"]).line() == "313/600 trials · 1 measured"


def test_a_new_run_is_made_by_init_and_opened(tmp_path, monkeypatch):
    made = []

    def init(name, model, runs):
        made.append((name, model, runs))
        a_run(runs, name)

    monkeypatch.setattr(cli, "init", init)
    asked, _ = run_menus(tmp_path, "new", "fresh", "org/model", BACK, "quit")
    assert made == [("fresh", "org/model", tmp_path)]
    assert asked.asked[3][0] == "fresh"


def test_comparing_ticks_two_measurements_and_doesnt_show_the_run_again(tmp_path, monkeypatch):
    folder = a_run(tmp_path)
    measured(folder, "base", "trial-4")
    compared = []
    monkeypatch.setattr(cli, "compare", lambda folder, first, second: compared.append((first, second)))
    asked, shown = run_menus(tmp_path, folder, "compare", ["base", "trial-4"], BACK, "quit")
    assert compared == [("base", "trial-4")] and shown.count("╭─ run ") == 1
    assert [o.value for o in asked.asked[2][1]] == ["base", "trial-4"]


def test_an_editor_that_wont_start_is_said_so(tmp_path, monkeypatch):
    folder = a_run(tmp_path)
    monkeypatch.setenv("EDITOR", "no-such-editor --wait")
    _, shown = run_menus(tmp_path, folder, "settings", True, BACK, "quit")
    assert "no-such-editor --wait didn't open" in shown


@pytest.fixture
def searched(tiny):
    """The tiny run, searched for a few trials, with the slots of a thinking run."""
    small(tiny, trials=2)
    Search(tiny).search(4)
    tiny.settings.slots.measure = 28
    (tiny.folder / config.FILE).write_text(config.render(tiny.settings, "made by a test"))
    found = search.load(tiny)
    assert found is not None
    return tiny, found


def test_the_trials_listed_are_those_candor_search_names_and_those_worked_on(searched):
    run, found = searched
    space = run.settings.search
    named = {
        c.trial
        for scope in search.SCOPES
        for c in search.to_measure(found, scope, space)
        or [c for c in [search.fewest_refused(found, scope)] if c]
    }
    assert {p.candidate.trial for p in menu.picks(run, found, set())} == named
    listed = menu.picks(run, found, set(found.trials) - named)
    assert {p.candidate.trial for p in listed} == set(found.trials)
    # Trials scored in full first, fewest refused first; then those stopped early.
    order = [(p.stopped, p.candidate.refusals, p.candidate.kl) for p in listed]
    assert order == sorted(order)


def test_with_nothing_within_the_cap_each_kinds_fewest_refused_is_listed(searched):
    run, found = searched
    run.settings.search.max_refusals = -1
    fewest = {c.trial for scope in search.SCOPES if (c := search.fewest_refused(found, scope))}
    assert {p.candidate.trial for p in menu.picks(run, found, set())} == fewest


def test_the_trials_list_remembers_the_last_trial_and_a_trial_needs_no_export_here(searched):
    run, found = searched
    number = menu.picks(run, found, set())[0].candidate.trial
    asked, _ = run_menus(run.folder.parent, run.folder, "trials", number, BACK, BACK, BACK, "quit")
    trials = [
        d for (m, _), d in zip(asked.asked, asked.defaults, strict=True) if m.startswith("Trials worth")
    ]
    assert trials == [None, number]
    # Measuring its export, exported on a pod, needs none here; chatting with it does.
    trial = asked.options(f"Trial {number}")
    assert trial["measure export"].disabled is None and trial["chat"].disabled == "no export of it here"


def test_a_trials_row_fits_80_columns_at_its_widest():
    score = search.Score(100, 99, 0, 0.12345, 100)
    widest = menu.Pick(
        search.Candidate(600, "per layer, answer start", 100, 99, 0.1, {}), score, False, False
    )
    stopped = menu.Pick(
        search.Candidate(599, "layer 31", 7, 7, 0.1, {}), search.Score(7, 7, 0, 0.1, 8), True, False
    )
    lines = menu.rows([widest, stopped], {600: "export measured", 599: "export measured"})
    # The pointer takes 3 of a line's 80.
    assert all(len(line) <= 77 for line in lines)
    assert lines[0] == "600  per layer, answer  100 refused  KL 0.1235  over cap  ✓ export measured"
    assert "stopped early, 7 refused of 8" in lines[1]


def test_a_search_scored_by_an_older_candor_isnt_offered_to_carry_on(searched, monkeypatch):
    run, _ = searched
    run.settings.search.trials_per_scope = 5
    (run.folder / config.FILE).write_text(config.render(run.settings, "made by a test"))
    assert not menu.status(run.folder).older
    monkeypatch.setattr(search, "VERSION", search.VERSION + 1)
    older = menu.status(run.folder)
    assert older.older and older.trials < older.target
    menus = Menus(Scripted(), Console(file=io.StringIO()), run.folder.parent)
    assert menus._next(older, "base") == "base"
    asked, shown = run_menus(run.folder.parent, run.folder, BACK, "quit")
    assert asked.options(run.folder.name)["search"].disabled == "scored by an older candor"
    assert "scored by an older candor" in shown


def test_measurements_by_older_measuring_code_are_marked_and_the_unedited_model_measured_again(
    searched, monkeypatch
):
    run_, found = searched
    number = menu.picks(run_, found, set())[0].candidate.trial
    measured(run_.folder, "base-slots-28", "base-slots-28-first-16", code=1)
    measured(run_.folder, f"trial-{number}-exported-slots-28", "base-slots-28-first-8")
    status = menu.status(run_.folder)
    assert status.older_measured == {"base-slots-28", "base-slots-28-first-16"}
    menus = Menus(Scripted(), Console(file=io.StringIO()), run_.folder.parent)
    assert menus._next(status, "base-slots-28") == "base"
    answers = ["look", BACK, "trials", number, "measure", BACK, BACK, BACK, BACK]
    asked, shown = run_menus(run_.folder.parent, run_.folder, *answers, "quit")
    assert asked.options(run_.folder.name)["base"].title.endswith("✓ measured by an older candor   ← next")
    assert asked.options("Which measurement?")["base-slots-28"].title == "base-slots-28   by an older candor"
    assert asked.options(f"Trial {number}")["compare"].disabled == "measured by different versions of candor"
    assert "2 by an older candor" in shown
    # A quick measurement is offered only as long as one of the unedited model's it can pair with.
    assert list(asked.options(f"Measure trial {number}")) == ["every", 8, BACK]


def test_a_trial_is_measured_exported_and_chatted_with_under_the_runs_names(searched, monkeypatch):
    run, found = searched
    number = menu.picks(run, found, set())[0].candidate.trial
    calls = []
    monkeypatch.setattr(cli, "measure_command", lambda folder, **given: calls.append(("measure", given)))
    monkeypatch.setattr(cli, "export_command", lambda folder, trial: calls.append(("export", trial)))
    monkeypatch.setattr(cli, "serve_command", lambda folder, **given: calls.append(("serve", given)))
    monkeypatch.setenv("CANDOR_MCP", "/tools.json")
    # Measured with adapters, as candor measure names it at the run's slots, and exported.
    measured(run.folder, f"trial-{number}-slots-28")
    (run.folder / "exports" / f"trial-{number}").mkdir(parents=True)
    (run.folder / "exports" / f"trial-{number}" / "m.gguf").touch()
    answers = ["trials", number, "measure export", "every", "here", "export", "pod", "chat", BACK, BACK, BACK]
    asked, shown = run_menus(run.folder.parent, run.folder, *answers, "quit")
    assert calls == [
        ("measure", {"trial": number, "limit": None, "exported": True}),
        ("serve", {"trial": number, "mcp": Path("/tools.json")}),
    ]
    trial = asked.options(f"Trial {number}")
    assert trial["measure"].title.endswith("✓ measured") and trial["export"].title.endswith("✓ exported here")
    assert f"scripts/pod.sh start tiny-export-{number} candor export {run.folder} --trial {number}" in shown
    assert f"scripts/pod.sh pull {run.folder}/exports/trial-{number}" in shown
    # The model is a local folder, which the pod needs too.
    assert f"  # the model, a local folder:\n  scripts/pod.sh push {run.settings.model.source}\n" in shown


def test_a_runs_panel_names_a_local_model_from_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    folder = a_run(tmp_path, source=str(tmp_path / "models" / "mine"))
    found = menu.status(folder)
    assert found.settings is not None
    panel = Menus(Scripted(), Console(file=io.StringIO()), tmp_path)._panel(
        found, Run(folder, found.settings)
    )
    out = io.StringIO()
    Console(file=out, width=100).print(panel)
    assert "~/models/mine" in out.getvalue()


# The questions themselves, by keys


def asked_with(keys: str, question):
    with create_pipe_input() as typed:
        typed.send_text(keys)
        return question(Questions(input=typed, output=DummyOutput()))


def test_arrows_and_enter_choose_and_ctrl_c_or_ctrl_d_go_back():
    options = [Option("One", 1), None, Option("Off", 2, disabled="not now"), Option("Two", 3)]
    # The arrow passes over the line and the disabled option.
    assert asked_with(DOWN + ENTER, lambda q: q.choose("Which?", options)) == 3
    assert asked_with(ENTER, lambda q: q.choose("Which?", options, default=3)) == 3
    # A default that is disabled is passed over too.
    assert asked_with(ENTER, lambda q: q.choose("Which?", options, default=2)) == 1
    assert asked_with(CTRL_C, lambda q: q.choose("Which?", options)) is None
    assert asked_with(CTRL_D, lambda q: q.text("Trial")) is None


def test_two_must_be_ticked_and_text_is_checked():
    options = [Option("a", "a"), Option("b", "b"), Option("c", "c")]
    # Enter with one ticked is refused; the second tick then lets it through.
    keys = " " + ENTER + DOWN + DOWN + " " + ENTER
    assert asked_with(keys, lambda q: q.choose_two("Which two?", options)) == ["a", "c"]

    def check(value: str) -> str | None:
        return None if value.isdigit() else "a number"

    assert asked_with("x\x7f 12 " + ENTER, lambda q: q.text("Trial", check)) == "12"


def test_candor_alone_opens_the_menus_in_a_terminal_and_shows_help_elsewhere(monkeypatch):
    opened = []
    monkeypatch.setattr(menu, "start", lambda: opened.append(True))
    result = CliRunner().invoke(cli.app, [])
    assert result.exit_code == 2 and "Commands" in result.output and opened == []
    monkeypatch.setattr(cli, "_in_a_terminal", lambda: True)
    assert CliRunner().invoke(cli.app, []).exit_code == 0 and opened == [True]
