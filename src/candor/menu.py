"""The menus `candor` opens when it is run with no arguments in a terminal. They show what each run
holds and run the same commands the command line does, each shown before it runs, so a choice made
here is one you could type, or start on a pod."""

import os
import re
import shlex
import subprocess
from collections.abc import Callable, Set
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import questionary
import typer
from rich import box
from rich.console import Console, Group, RenderableType
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import cli, config, measure, search, source
from .run import Run, RunError

ACCENT = "#5fd7af"
STYLE = questionary.Style(
    [
        ("qmark", f"fg:{ACCENT} bold"),
        ("question", "bold"),
        ("pointer", f"fg:{ACCENT} bold"),
        ("highlighted", f"fg:{ACCENT} bold"),
        ("selected", f"fg:{ACCENT}"),
        ("answer", f"fg:{ACCENT}"),
        ("instruction", "fg:#767676 italic"),
        ("separator", "fg:#4e4e4e"),
        ("disabled", "fg:#6c6c6c italic"),
    ]
)
MARK, POINTER = "◆", "❯"  # noqa: RUF001
# How many prompts of each set a quick measurement answers, to try it and time it, when the unedited
# model has no quick measurement to pair it with.
QUICK = 32
# How many measurements a run's panel names; Look at a measurement lists them all.
MEASUREMENTS_SHOWN = 8
BACK = "back"
# A line between groups of options.
RULE = None


@dataclass(frozen=True)
class Option:
    """One line of a menu. Disabled says why it can't be chosen now."""

    title: str
    value: Any
    disabled: str | None = None


class Ask(Protocol):
    """The questions the menus ask. Each answer is None when the question was left with Ctrl-C."""

    def choose(self, message: str, options: list[Option | None], default: Any = None) -> Any: ...

    def choose_two(self, message: str, options: list[Option]) -> list[Any] | None: ...

    def text(self, message: str, check: Callable[[str], str | None] | None = None) -> str | None: ...


class Questions:
    """Ask in the terminal: arrows or j and k to move, enter to choose, Ctrl-C to go back. A question
    leaves the screen once answered, so only what the commands show stays."""

    def __init__(self, **terminal: Any):
        # Tests give prompt_toolkit's input and output here.
        self.terminal = {"erase_when_done": True, **terminal}

    def _answer(self, question: questionary.Question) -> Any:
        try:
            return question.unsafe_ask()
        except (KeyboardInterrupt, EOFError):
            return None

    def choose(self, message: str, options: list[Option | None], default: Any = None) -> Any:
        choices = [
            questionary.Separator(" ")
            if option is None
            else questionary.Choice(option.title, option.value, disabled=option.disabled)
            for option in options
        ]
        pointed = next(
            (
                c
                for c in choices
                if isinstance(c, questionary.Choice) and not c.disabled and c.value == default
            ),
            None,
        )
        question = questionary.select(
            message,
            choices,
            default=pointed,
            qmark=MARK,
            pointer=POINTER,
            style=STYLE,
            instruction=" ",
            **self.terminal,
        )
        return self._answer(question)

    def choose_two(self, message: str, options: list[Option]) -> list[Any] | None:
        question = questionary.checkbox(
            message,
            [questionary.Choice(option.title, option.value) for option in options],
            qmark=MARK,
            pointer=POINTER,
            style=STYLE,
            instruction="(space ticks, enter compares)",
            validate=lambda ticked: len(ticked) == 2 or "tick two",
            **self.terminal,
        )
        return self._answer(question)

    def text(self, message: str, check: Callable[[str], str | None] | None = None) -> str | None:
        question = questionary.text(
            message,
            qmark=MARK,
            style=STYLE,
            validate=(lambda value: check(value.strip()) or True) if check else None,
            **self.terminal,
        )
        answer = self._answer(question)
        return None if answer is None else answer.strip()


# What a run holds

STAGES = {"model": "gguf", "prompts": "prompts.json", "directions": "directions.pt", "adapters": "adapters"}


@dataclass(frozen=True)
class Status:
    """What a run folder holds, read from its files without building or checking anything."""

    folder: Path
    settings: config.RunConfig | None
    # Why its settings or its search don't load, when they don't.
    problem: str | None
    built: dict[str, bool]
    found: search.Found | None
    # The finished trials that count towards the search's target, of target.
    trials: int
    target: int
    measurements: list[str]
    exports: list[int]
    # Whether its search was scored by older code, so its trials can be measured and exported but no
    # more added.
    older: bool = False
    # Its measurements made by older measuring code.
    older_measured: frozenset[str] = frozenset()

    @property
    def model(self) -> str:
        return Path(self.settings.model.source).name if self.settings else "?"

    @property
    def prepared(self) -> bool:
        return all(self.built.values())

    def line(self) -> str:
        """The run in a few words, for the list of runs."""
        if self.settings is None:
            return "its settings don't load"
        if self.trials:
            parts = [f"{self.trials}/{self.target} trials"]
        else:
            parts = ["prepared" if self.prepared else "not prepared"]
        if self.measurements:
            parts.append(f"{len(self.measurements)} measured")
        if self.exports:
            parts.append(f"{len(self.exports)} exported")
        return " · ".join(parts)


def _number(name: str) -> int | None:
    """The trial a measurement or an export is of, by its name, as in trial-205-exported."""
    parts = name.split("-")
    return int(parts[1]) if len(parts) > 1 and parts[0] == "trial" and parts[1].isdigit() else None


def measurements_of(folder: Path, settings: config.RunConfig) -> list[str]:
    """The run's finished measurements, judged too: the unedited model's first, then by trial."""
    root = folder / "measurements"
    files = measure.finished(settings.measure)
    names = [
        path.name
        for path in (root.iterdir() if root.is_dir() else [])
        if path.is_dir() and all((path / file).exists() for file in files)
    ]
    return sorted(names, key=lambda name: (not name.startswith("base"), _number(name) or 0, name))


def older_of(folder: Path, names: list[str]) -> set[str]:
    """Those of the run's measurements made by older measuring code, which compare won't pair with
    one made now."""
    now = measure.STAGES["measurements"]
    return {name for name in names if measure.measured_by(folder / "measurements" / name) != now}


def exports_of(folder: Path) -> list[int]:
    """The trials whose GGUF candor export made is here."""
    root = folder / "exports"
    found = [
        number
        for path in (root.iterdir() if root.is_dir() else [])
        if (number := _number(path.name)) is not None and any(path.glob("*.gguf"))
    ]
    return sorted(found)


def status(folder: Path) -> Status:
    built = {stage: (folder / path).exists() for stage, path in STAGES.items()}
    try:
        settings = config.load(folder)
    except config.ConfigError as error:
        return Status(folder, None, str(error), built, None, 0, 0, [], [])
    if settings.deployment.think_close:
        built["thinking"] = any((folder / "thinking").glob("*.json"))
    found = problem = None
    older = False
    try:
        found = search.load(Run(folder, settings, report=lambda _: None))
        if found:
            # Read once here, so a search an older candor made fails here and not in the menu.
            for scope in search.SCOPES:
                search.best_so_far(found, scope, settings.search)
            older = search.older(folder)
    except RunError as error:
        found, problem = None, f"the search can't be read: {error}"
    except Exception as error:
        found, problem = None, f"the search can't be read ({type(error).__name__}): made by an older candor?"
    # Counted as candor search counts them: each kind's up to its share.
    share = settings.search.trials_per_scope
    kinds = [search.scope_of(trial.params) for trial in found.trials.values()] if found else []
    trials = sum(min(kinds.count(scope), share) for scope in search.SCOPES)
    target = len(search.SCOPES) * share
    measured, exported = measurements_of(folder, settings), exports_of(folder)
    older_measured = frozenset(older_of(folder, measured))
    return Status(
        folder, settings, problem, built, found, trials, target, measured, exported, older, older_measured
    )


def runs_in(root: Path) -> list[Status]:
    if not root.is_dir():
        return []
    # A folder whose name starts with a dot is one config.write hasn't finished.
    return [
        status(path)
        for path in sorted(root.iterdir())
        if not path.name.startswith(".") and (path / config.FILE).exists()
    ]


@dataclass(frozen=True)
class Pick:
    """A trial as the trials menu lists it."""

    candidate: search.Candidate
    score: search.Score
    stopped: bool
    # Within the search's cap on refusals, and on thinking that reached its budget when it has one.
    within: bool


def picks(run: Run, found: search.Found, worked_on: set[int]) -> list[Pick]:
    """The trials worth measuring, fewest refused first: those candor search names, each kind's least
    KL at each count within the cap, or its fewest refused when none is within it; and any in
    worked_on, those already measured or exported, even if they stopped early."""
    space = run.settings.search
    named = [
        candidate
        for scope in search.SCOPES
        for candidate in search.to_measure(found, scope, space)
        or [c for c in [search.fewest_refused(found, scope)] if c]
    ]
    others = [search.candidate(n, found.trials[n]) for n in sorted(worked_on) if n in found.trials]
    listed = []
    for candidate in {c.trial: c for c in [*named, *others]}.values():
        score = search.searched_score(found, candidate.trial)
        stopped = found.trials[candidate.trial].user_attrs["stopped"]
        listed.append(Pick(candidate, score, stopped, search.counts(score, space, stopped)))
    return sorted(listed, key=lambda p: (p.stopped, p.candidate.refusals, p.candidate.kl))


def scored(score: search.Score, stopped: bool, within: bool) -> str:
    """A trial's score in the search, as the menus show it."""
    if stopped:
        return f"stopped early, {score.refusals} refused of {score.judged}"
    return f"{score.refusals:>3} refused  KL {score.kl:.4f}" + ("" if within else "  over cap")


def rows(listed: list[Pick], marks: dict[int, str]) -> list[str]:
    """The trials menu's lines: each trial, the directions it reads, its score and how far it has
    got, in columns as narrow as their widest."""
    cells = [
        (
            str(p.candidate.trial),
            p.candidate.direction.replace(f", {search.ANSWER_START}", ", answer"),
            scored(p.score, p.stopped, p.within),
        )
        for p in listed
    ]
    widths = [max((len(cell[i]) for cell in cells), default=0) for i in range(3)]
    return [
        f"{number:>{widths[0]}}  {direction:<{widths[1]}}  {score:<{widths[2]}}"
        + (f"  ✓ {marks[p.candidate.trial]}" if marks.get(p.candidate.trial) else "")
        for p, (number, direction, score) in zip(listed, cells, strict=True)
    ]


# The menus


class Menus:
    def __init__(self, ask: Ask, console: Console, root: Path = Path("runs")):
        self.ask, self.console, self.root = ask, console, root
        # Whether the run's panel needs showing again: on opening it, and once a job has run.
        self.changed = True

    def start(self) -> None:
        self.console.print(Text.assemble(("candor", f"bold {ACCENT}"), "  edits a model's refusals away"))
        self.console.print(Text("arrows or j/k to move · enter to choose · ctrl-c to go back", style="dim"))
        last = None
        while True:
            runs = runs_in(self.root)
            name = max((len(r.folder.name) for r in runs), default=0)
            model = max((len(r.model) for r in runs), default=0)
            options: list[Option | None] = [
                Option(f"{r.folder.name:<{name}}  {r.model:<{model}}  {r.line()}", r.folder) for r in runs
            ]
            options += [
                RULE,
                Option("New run…", "new"),
                Option("Inspect a model…", "inspect"),
                Option("Quit", "quit"),
            ]
            where = f"{self.root}/"
            chosen = self.ask.choose(f"Runs in {where}" if runs else f"No runs in {where} yet", options, last)
            if chosen in (None, "quit"):
                return
            last = chosen
            if chosen == "new":
                if folder := self.new_run():
                    last = folder
                    self.open(folder)
            elif chosen == "inspect":
                if model_id := self.ask.text("Model, as a Hugging Face id or a local folder", _given):
                    self.call(["inspect", model_id], partial(cli.inspect, model_id))
            else:
                self.open(chosen)

    def new_run(self) -> Path | None:
        name = self.ask.text("Name for the run", _run_name)
        if not name:
            return None
        if (self.root / name).exists():
            self.console.print(f"[red]{escape(str(self.root / name))} already exists.[/]")
            return None
        model = self.ask.text("Model, as a Hugging Face id or a local folder", _given)
        if not model:
            return None
        self.call(["init", name, "--model", model], partial(cli.init, name, model, self.root))
        return self.root / name if (self.root / name / config.FILE).exists() else None

    # One run

    def open(self, folder: Path) -> None:
        self.changed = True
        last = None
        while True:
            found = status(folder)
            if found.settings is None:
                self.console.print(f"[red]{escape(str(found.problem))}[/]")
                return
            run = Run(folder, found.settings, report=lambda _: None)
            if self.changed:
                self.console.print(self._panel(found, run))
                self.changed = False
            base = _measured_as(run, None, False)
            next_step = self._next(found, base)
            searched = bool(found.found and found.found.trials)
            options: list[Option | None] = [
                Option("Prepare" + _done(found.prepared, "built"), "run"),
                Option(
                    "Search" + (f"   {found.trials}/{found.target} trials" if found.trials else ""),
                    "search",
                    "scored by an older candor" if found.older else None,
                ),
                Option("Trials…", "trials", None if searched else "nothing searched yet"),
                Option(
                    "Measure the unedited model"
                    + _done(base in found.measurements, _by(base, found.older_measured)),
                    "base",
                ),
                Option(
                    "Compare two measurements…",
                    "compare",
                    None if len(found.measurements) >= 2 else "measure two first",
                ),
                Option(
                    "Look at a measurement…", "look", None if found.measurements else "nothing measured yet"
                ),
                Option("Chat with…", "chat"),
                RULE,
                Option("Settings", "settings"),
                Option("Back", BACK),
            ]
            options = [
                Option(o.title + "   ← next", o.value, o.disabled) if o and o.value == next_step else o
                for o in options
            ]
            chosen = self.ask.choose(folder.name, options, default=last or next_step)
            if chosen in (None, BACK):
                return
            last = chosen
            if chosen == "run":
                self.job(run, "prepare", ["run", str(folder)], partial(cli.run, folder))
            elif chosen == "search":
                self.job(run, "search", ["search", str(folder)], partial(cli.search_command, folder))
            elif chosen == "trials" and found.found:
                self.trials(run, found.found)
            elif chosen == "base":
                self.measure(run, found.measurements, None, exported=False)
            elif chosen == "compare":
                self.compare(folder, found.measurements, found.older_measured)
            elif chosen == "look":
                self.look(run, found.measurements, found.older_measured)
            elif chosen == "chat":
                self.chat(folder, found)
            elif chosen == "settings":
                self.settings(folder)

    def _next(self, found: Status, base: str) -> str:
        """The step the run is ready for, which its menu starts on. A run searched on a pod has no
        model here, and needs none to carry its search on there."""
        if not found.prepared and not found.trials:
            return "run"
        if found.trials < found.target and not found.older:
            return "search"
        if base not in found.measurements or base in found.older_measured:
            return "base"
        return "trials"

    def _panel(self, found: Status, run: Run) -> Panel:
        settings = run.settings
        deployment = settings.deployment
        model = settings.model.source
        if source.is_local(model):
            path = Path(model).expanduser()
            model = f"~/{path.relative_to(Path.home())}" if path.is_relative_to(Path.home()) else str(path)
        revision = f" @ {settings.model.revision[:7]}" if settings.model.revision else ""
        thinks = f"thinks, budget {deployment.thinking_budget:,}" if deployment.think_close else "no thinking"
        lines: list[RenderableType] = [
            Text(f"{model}{revision}", style="bold"),
            Text(f"{deployment.quantization} · {thinks} · {settings.slots.measure} slots", style="dim"),
            Text("  ").join(
                Text(f"✓ {stage}", style=ACCENT) if done else Text(f"· {stage}", style="dim")
                for stage, done in found.built.items()
            ),
        ]
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        if found.found is not None:
            best = [
                f"best {scope}: trial {pick.trial}, {pick.refusals} refused, KL {pick.kl:.4f}"
                for scope in search.SCOPES
                if (pick := search.best_so_far(found.found, scope, settings.search))
            ]
            searched = f"{found.trials} of {found.target} trials"
            if found.older:
                searched += ", scored by an older candor"
            table.add_row("Search", escape("\n".join([searched, *best])))
        if found.measurements:
            shown = found.measurements[:MEASUREMENTS_SHOWN]
            more = len(found.measurements) - len(shown)
            listed = ", ".join(shown) + (f" and {more} more" if more else "")
            if older := len(found.older_measured):
                listed += f"\n{'all' if older == len(found.measurements) else older} by an older candor"
            table.add_row("Measured", escape(listed))
        if found.exports:
            table.add_row("Exports", escape(", ".join(f"trial {n}" for n in found.exports)))
        if found.problem:
            table.add_row("[yellow]Note[/]", escape(found.problem))
        if table.row_count:
            lines += [Text(), table]
        return Panel(
            Group(*lines), title=f"[bold]{escape(found.folder.name)}[/]", title_align="left", box=box.ROUNDED
        )

    # Trials

    def trials(self, run: Run, found: search.Found) -> None:
        last = None
        while True:
            measured, exported = measurements_of(run.folder, run.settings), exports_of(run.folder)
            worked_on = {n for n in map(_number, measured) if n is not None} | set(exported)
            listed = picks(run, found, worked_on)
            marks = {p.candidate.trial: _furthest(run, p.candidate.trial, measured, exported) for p in listed}
            options: list[Option | None] = [
                Option(title, p.candidate.trial) for p, title in zip(listed, rows(listed, marks), strict=True)
            ]
            options += [RULE, Option("Another trial, by its number…", "other"), Option("Back", BACK)]
            chosen = self.ask.choose(
                f"Trials worth measuring, of {len(found.trials)} searched", options, last
            )
            if chosen in (None, BACK):
                return
            if chosen == "other":
                given = self.ask.text("Trial number", partial(_trial_in, found))
                if not given:
                    continue
                chosen = int(given)
            last = chosen
            self.trial(run, found, chosen)

    def trial(self, run: Run, found: search.Found, number: int) -> None:
        folder = run.folder
        adapters, export = (_measured_as(run, number, exported) for exported in (False, True))
        base = _measured_as(run, None, False)
        trial = found.trials[number]
        stopped = trial.user_attrs["stopped"]
        score = search.searched_score(found, number)
        within = search.counts(score, run.settings.search, stopped)
        direction = search.candidate(number, trial).direction
        shown = " ".join(scored(score, stopped, within).split())
        self.console.print(Text(f"Trial {number}: {direction}, {shown}", style="dim"))
        while True:
            measured = measurements_of(folder, run.settings)
            older = older_of(folder, measured)
            exported = number in exports_of(folder)
            if not {base, export} <= set(measured):
                apart = "measure both first"
            elif measure.measured_by(folder / "measurements" / base) != measure.measured_by(
                folder / "measurements" / export
            ):
                apart = "measured by different versions of candor"
            else:
                apart = None
            options: list[Option | None] = [
                Option(
                    "Measure it with adapters" + _done(adapters in measured, _by(adapters, older)), "measure"
                ),
                Option("Export it, as the GGUF you run" + _done(exported, "exported here"), "export"),
                Option(
                    "Measure its export" + _done(export in measured, _by(export, older)), "measure export"
                ),
                Option("Compare its export with the unedited model", "compare", apart),
                Option("Chat with its export", "chat", None if exported else "no export of it here"),
                RULE,
                Option("Back", BACK),
            ]
            chosen = self.ask.choose(f"Trial {number}", options)
            if chosen in (None, BACK):
                return
            if chosen == "measure":
                self.measure(run, measured, number, exported=False)
            elif chosen == "export":
                self.job(
                    run,
                    f"export-{number}",
                    ["export", str(folder), "--trial", str(number)],
                    partial(cli.export_command, folder, trial=number),
                    pull=f"exports/trial-{number}",
                )
            elif chosen == "measure export":
                self.measure(run, measured, number, exported=True)
            elif chosen == "compare":
                self.call(["compare", str(folder), base, export], partial(cli.compare, folder, base, export))
            elif chosen == "chat":
                self.serve(folder, number)

    # Measuring and comparing

    def measure(self, run: Run, measured: list[str], trial: int | None, exported: bool) -> None:
        what = "the unedited model" if trial is None else f"trial {trial}" + ("'s export" if exported else "")
        # A quick measurement pairs only with one of the unedited model's that is as long.
        base = _measured_as(run, None, False)
        older = older_of(run.folder, measured)
        limits = sorted(
            {
                int(m[1])
                for name in measured
                if (m := re.fullmatch(rf"{base}-first-(\d+)", name)) and (trial is None or name not in older)
            }
        )
        pairs = "as" if trial is None else "to pair with"
        quick = [Option(f"The first {n} of each set, {pairs} {base}-first-{n}", n) for n in limits] or [
            Option(f"The first {QUICK} of each set, to try it and time it", QUICK)
        ]
        size = self.ask.choose(
            f"Measure {what}", [Option("Every prompt", "every"), *quick, RULE, Option("Back", BACK)]
        )
        if size in (None, BACK):
            return
        limit = None if size == "every" else size
        folder = run.folder
        args = ["measure", str(folder)]
        if trial is not None:
            args += ["--trial", str(trial)] + (["--exported"] if exported else [])
        if limit:
            args += ["--limit", str(limit)]
        name = "measure-" + ("base" if trial is None else str(trial)) + ("-exported" if exported else "")
        self.job(
            run, name, args, partial(cli.measure_command, folder, trial=trial, limit=limit, exported=exported)
        )

    def compare(self, folder: Path, names: list[str], older: frozenset[str]) -> None:
        options = [Option(_older_shown(n, older), n) for n in names]
        chosen = self.ask.choose_two("Compare which two? A is the higher one", options)
        if chosen:
            first, second = chosen
            self.call(["compare", str(folder), first, second], partial(cli.compare, folder, first, second))

    def look(self, run: Run, names: list[str], older: frozenset[str]) -> None:
        options: list[Option | None] = [Option(_older_shown(n, older), n) for n in names]
        name = self.ask.choose("Which measurement?", [*options, RULE, Option("Back", BACK)])
        if name in (None, BACK):
            return
        try:
            results = measure.load(run, name)
        except RunError as error:
            self.console.print(f"[red]{escape(str(error))}[/]")
            return
        table = Table(title=escape(name), title_justify="left", box=box.SIMPLE_HEAD)
        table.add_column("")
        table.add_column("Result", justify="right")
        for row in measure.summary(results):
            table.add_row(*map(escape, row))
        self.console.print(table)

    # Chatting

    def chat(self, folder: Path, found: Status) -> None:
        # The unedited model is served from the run's own GGUF, which a run pulled from a pod hasn't.
        options: list[Option | None] = [
            Option("The unedited model", "base", None if found.built["model"] else "its GGUF isn't here")
        ]
        options += [Option(f"Trial {n}'s export", n) for n in found.exports]
        chosen = self.ask.choose("Chat with which model?", [*options, RULE, Option("Back", BACK)])
        if chosen in (None, BACK):
            return
        self.serve(folder, None if chosen == "base" else chosen)

    def serve(self, folder: Path, trial: int | None) -> None:
        # As the command line takes it, so tools set for every chat are there too.
        mcp = Path(os.environ["CANDOR_MCP"]) if os.environ.get("CANDOR_MCP") else None
        args = ["serve", str(folder)] + ([] if trial is None else ["--trial", str(trial)])
        self.call(args, partial(cli.serve_command, folder, trial=trial, mcp=mcp))

    def settings(self, folder: Path) -> None:
        path = folder / config.FILE
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if not editor:
            self.console.print(
                f"The run's settings are in {escape(str(path))}; every command reads them afresh."
            )
            return
        if self.ask.choose(
            f"Open {path} in {editor}? Every command reads it afresh.",
            [Option("Open it", True), Option("Back", False)],
        ):
            try:
                subprocess.run([*shlex.split(editor), str(path)])
            except OSError as error:
                self.console.print(f"[red]{escape(editor)} didn't open: {escape(str(error))}[/]")
            self.changed = True

    # Running commands

    def job(self, run: Run, what: str, args: list[str], command: Callable[[], None], pull: str = "") -> None:
        """A command that can take hours: run it here, or show how to run it on a pod, as a job named
        after the run and what it does, pulling the run, and pull, back."""
        chosen = self.ask.choose(
            _command(args),
            [
                Option("Run it here", "here"),
                Option("Show how to run it on a pod", "pod"),
                RULE,
                Option("Back", BACK),
            ],
        )
        if chosen == "here":
            if self.call(args, command) is None:
                self.console.print("Running it again carries on where it stopped.")
            # What the run holds has changed, so the menu shows it again.
            self.changed = True
        elif chosen == "pod":
            name, folder = f"{run.folder.name}-{what}", args[1]
            lines = ["On a pod set up with scripts/pod.sh setup:"]
            model = run.settings.model.source
            if source.is_local(model):
                lines += ["  # the model, a local folder:", f"  scripts/pod.sh push {model}"]
            lines += [
                "  # the run, if the pod doesn't have it yet:",
                f"  scripts/pod.sh push {folder}",
                f"  scripts/pod.sh start {name} candor {shlex.join(args)}",
                f"  scripts/pod.sh wait {name} {folder}",
            ]
            if pull:
                lines.append(f"  scripts/pod.sh pull {folder}/{pull}")
            self.console.print(escape("\n".join(lines)), soft_wrap=True)

    def call(self, args: list[str], command: Callable[[], None]) -> bool | None:
        """Run a command as the command line would, and come back to the menu however it ends: whether
        it succeeded, or None when it was stopped with Ctrl-C."""
        self.console.print(Text(_command(args), style=f"bold {ACCENT}"), soft_wrap=True)
        try:
            command()
            return True
        except typer.Exit as stopped:
            return stopped.exit_code == 0
        except typer.BadParameter as error:
            self.console.print(f"[red]{escape(error.format_message())}[/]")
            return False
        except KeyboardInterrupt:
            self.console.print("[yellow]Stopped.[/]")
            return None


def _command(args: list[str]) -> str:
    return "$ candor " + shlex.join(args)


def _done(done: bool, what: str) -> str:
    return f"   ✓ {what}" if done else ""


def _by(name: str, older: Set[str]) -> str:
    """How a measurement is marked done: measured, or measured by older code."""
    return "measured by an older candor" if name in older else "measured"


def _older_shown(name: str, older: Set[str]) -> str:
    return f"{name}   by an older candor" if name in older else name


def _measured_as(run: Run, trial: int | None, exported: bool) -> str:
    """The name candor measure gives a measurement of the unedited model or a trial by default."""
    return cli._measurement_name(trial, False, run.settings.slots.measure, None, exported, None)


def _furthest(run: Run, trial: int, measured: list[str], exported: list[int]) -> str:
    """How far a trial has got: its export measured, exported, or measured with adapters."""
    if _measured_as(run, trial, True) in measured:
        return "export measured"
    if trial in exported:
        return "exported"
    return "measured" if _measured_as(run, trial, False) in measured else ""


def _given(value: str) -> str | None:
    return None if value else "type one, or ctrl-c to go back"


def _run_name(value: str) -> str | None:
    try:
        config.check_run_name(value)
    except config.ConfigError as error:
        return str(error)
    return None


def _trial_in(found: search.Found, value: str) -> str | None:
    if not value.isdigit() or int(value) not in found.trials:
        return f"the search has no finished trial {value}"
    return None


def start(root: Path = Path("runs")) -> None:
    Menus(Questions(), Console(), root).start()
