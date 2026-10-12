"""The candor command line."""

import json
import os
import shutil
import signal
import subprocess
import traceback
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any

import optuna
import typer
from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars
from huggingface_hub.utils import logging as hub_logging
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

from . import checks, config, measure, search, source, targets
from .convert import ConversionError
from .engine import LLAMA_CPP_RELEASE, ServerError
from .run import PROMPT_SETS, Run, RunError
from .toolchain import ToolchainError

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Remove a model's refusal direction, scored the way you actually run the model.",
)
MARKS = {
    "ok": "[green]ok[/]",
    "warn": "[yellow]warn[/]",
    "fail": "[red]fail[/]",
    "skipped": "[dim]skipped[/]",
}
# 2 is what typer uses for usage errors, so "couldn't inspect" gets its own code.
FAILED_CHECK, COULD_NOT_INSPECT = 1, 3
# A search's progress takes its speed from every trial of the session, since trials take minutes and
# rich otherwise looks back only 30 seconds.
DAY = 24 * 3600

Row = tuple[str, str, str]


class CannotInspect(Exception):
    pass


@app.callback()
def main() -> None:
    # The Hub's and transformers' progress bars and warnings would only clutter candor's own tables
    # and messages.
    disable_progress_bars()
    hub_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
    optuna.logging.set_verbosity(optuna.logging.WARNING)


@contextmanager
def _quiet_transformers() -> Iterator[None]:
    """transformers warns that a config it can't read without remote code "can yield errors", even
    when the tokenizer then loads fine from its own config."""
    level = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        yield
    finally:
        transformers_logging.set_verbosity(level)


def _examples(names: list[str], limit: int = 3) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


def _target_rows(model: str, hidden: int, layers: int, arch: str | None) -> list[Row]:
    try:
        shapes = source.tensor_shapes(model)
    except Exception as error:
        return [("Edit targets", "fail", f"couldn't read the tensor shapes: {error}")]
    found = targets.find_targets(shapes, hidden, layers)
    if not found.ok:
        problems = []
        if found.unclassified:
            problems.append(
                f"can't tell whether these write to the residual: {_examples(found.unclassified)}"
            )
        if found.missing:
            problems.append(_examples(found.missing))
        return [("Edit targets", "fail", "; ".join(problems))]
    parts = [
        f"{c}: " + ", ".join(f"{n} {k}" for k, n in kinds.items()) for c, kinds in found.summary().items()
    ]
    covered = f"{len(found.writers)} writers, every layer covered ({'; '.join(parts)})"
    rows: list[Row] = [("Edit targets", "ok", covered)]
    if found.extra_layers:
        extra = _examples([str(i) for i in found.extra_layers])
        detail = (
            f"layers {extra} sit beyond the {layers} the model runs, such as a token-prediction head, "
            "and are left alone"
        )
        rows.append(("Extra layers", "ok", detail))
    if found.experts:
        rows.append(("Experts", "fail", "candor doesn't edit mixture-of-experts models yet"))
    if found.quantized:
        detail = f"{_examples(found.quantized)} carry quantisation scales; use the unquantised original"
        rows.append(("Quantised", "fail", detail))
    if found.output_norms:
        detail = (
            "this model normalises each block's output before adding it to the residual stream, so the "
            "edit can't remove the direction exactly"
        )
        rows.append(("Output norms", "warn", detail))
    if arch and not found.experts:
        unmapped = [w.name for w in found.writers if targets.gguf_name(w, arch, layers) is None]
        if unmapped:
            rows.append(("GGUF names", "fail", f"no llama.cpp name for {_examples(unmapped)}"))
        else:
            first = found.writers[0]
            example = f"{first.name} -> {targets.gguf_name(first, arch, layers)}"
            rows.append(("GGUF names", "ok", f"all map, e.g. {example}"))
    return rows


def _tokenizer_rows(model: str) -> list[Row]:
    try:
        with _quiet_transformers():
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    except Exception as error:
        return [("Tokenizer", "fail", f"couldn't load it without running remote code: {error}")]
    template = checks.chat_template(tokenizer)
    if template is None:
        return [("Chat template", "fail", "the tokenizer has none, so this looks like a base model")]
    try:
        found = checks.begin_of_text(tokenizer)
    except Exception as error:
        return [("Chat template", "fail", f"transformers can't render a one-message chat with it: {error}")]
    counts = f"apps send {found.transformers}, llama.cpp {found.llama_cpp}, heretic {found.heretic}"
    rows: list[Row] = [
        ("Begin-of-text", "warn" if found.problems() else "ok", "; ".join([counts, *found.problems()]))
    ]
    if checks.pinnable_date_format(template):
        detail = "the chat template writes today's date into every prompt; a run can pin it"
        rows.append(("Date in prompt", "warn", detail))
    elif checks.writes_the_date(template):
        detail = (
            "the chat template writes today's date into every prompt and can't be given a fixed one, so "
            "finish each run on the day it starts"
        )
        rows.append(("Date in prompt", "warn", detail))
    else:
        rows.append(("Date in prompt", "ok", "none"))
    result = checks.app_template_parses(tokenizer, template)
    if result.status == "fails":
        detail = f"Bionic and LM Studio can't use it and fall back to another format: {result.detail}"
        rows.append(("Bionic template", "fail", detail))
    elif result.status == "skipped":
        rows.append(("Bionic template", "skipped", result.detail))
    else:
        rows.append(("Bionic template", "ok", "parses and renders in Bionic and LM Studio"))
    return rows


def _rows(model: str) -> list[Row]:
    try:
        config = source.read_config(model)
    except Exception as error:
        raise CannotInspect(f"candor couldn't read {model}: {error}") from None
    sizes = source.sizes(config)
    if sizes is None:
        raise CannotInspect(f"candor couldn't find the hidden size and layer count in {model}'s config")
    hidden, layers = sizes

    arch = targets.gguf_architecture(config)
    names = ", ".join(config.get("architectures") or [config.get("model_type", "unknown")])
    converts = f"llama.cpp calls it {arch}" if arch else "the pinned llama.cpp can't convert it"
    summary = f"{names}, {layers} layers, hidden size {hidden}; {converts}"
    rows: list[Row] = [("Architecture", "ok" if arch else "fail", summary)]
    code = checks.remote_code(config, source.read_json(model, "tokenizer_config.json"))
    if checks.needs_remote_code(config):
        detail = f"transformers can only load it by running {_examples(code)}; a run does, so read it first"
        rows.append(("Remote code", "warn", detail))
    elif code:
        detail = f"ships {_examples(code)}, but candor uses transformers' own code for it"
        rows.append(("Remote code", "ok", detail))
    else:
        rows.append(("Remote code", "ok", "none"))
    return rows + _target_rows(model, hidden, layers, arch) + _tokenizer_rows(model)


def _resolved(model: str) -> str:
    """Local folders are stored as absolute paths; anything else is a Hugging Face id."""
    folder = Path(model).expanduser()
    return str(folder.resolve()) if folder.is_dir() else model


def _checked_rows(console: Console, model: str) -> list[Row]:
    try:
        return _rows(model)
    except CannotInspect as error:
        console.print(f"[red]{escape(str(error))}[/]")
    except Exception as error:
        console.print(
            f"[red]candor hit an unexpected error inspecting {escape(model)}:[/] "
            f"{escape(f'{type(error).__name__}: {error}')}"
        )
    raise typer.Exit(COULD_NOT_INSPECT)


def _print_table(console: Console, title: str, rows: list[Row]) -> None:
    table = Table(title=escape(title), show_lines=True)
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Detail", overflow="fold")
    for check, status, detail in rows:
        table.add_row(check, MARKS[status], escape(detail))
    console.print(table)


@app.command()
def inspect(model: Annotated[str, typer.Argument(help="A Hugging Face model id or a local folder.")]) -> None:
    """Check a model before spending anything on it. Reads the config, tokenizer and file headers only.

    Exits with 1 when a check fails and 3 when the model can't be inspected at all.
    """
    console = Console()
    model = _resolved(model)
    rows = _checked_rows(console, model)
    _print_table(console, f"candor inspect {model}", rows)
    if any(status == "fail" for _, status, _ in rows):
        raise typer.Exit(FAILED_CHECK)


# Failures that matter for using the model in an app but don't stop candor from editing it.
NOT_BLOCKING = {"Bionic template"}


def _pinned_date(model: str, revision: str | None) -> str | None:
    with _quiet_transformers():
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    date_format = checks.pinnable_date_format(checks.chat_template(tokenizer) or "")
    return config.today_in(date_format) if date_format else None


def _revision(model: str) -> str | None:
    """The Hub commit the model is at now, so later steps use exactly what was checked."""
    return None if source.is_local(model) else HfApi().model_info(model).sha


def _pinned(prompts: config.Prompts, names: Iterable[str]) -> config.Prompts:
    """The named prompt sets at the Hub commits their datasets are at now, so a rebuild reads the same
    prompts."""
    sets = {name: getattr(prompts, name) for name in names}
    commits = {dataset: HfApi().dataset_info(dataset).sha for dataset in {s.dataset for s in sets.values()}}
    return prompts.model_copy(
        update={name: s.model_copy(update={"revision": commits[s.dataset]}) for name, s in sets.items()}
    )


@app.command()
def init(
    name: Annotated[
        str, typer.Argument(help="A name for the run. Its settings go in RUNS/NAME/candor.toml.")
    ],
    model: Annotated[str, typer.Option("--model", "-m", help="A Hugging Face model id or a local folder.")],
    runs: Annotated[Path, typer.Option(help="The folder that holds run folders.")] = Path("runs"),
) -> None:
    """Start a run: check the model, then write a settings file holding everything the run will use.

    Nothing is written if the model fails a check candor can't work around.
    """
    console = Console()
    try:
        config.check_run_name(name)
    except config.ConfigError as error:
        raise typer.BadParameter(str(error), param_hint="NAME") from None
    runs = runs.expanduser()
    if (runs / name).exists():
        existing = escape(str(runs / name))
        console.print(f"[red]{existing} already exists.[/] Pick another name, or delete that folder first.")
        raise typer.Exit(FAILED_CHECK)
    model = _resolved(model)
    rows = _checked_rows(console, model)
    _print_table(console, f"candor init {name}: checking {model}", rows)
    blocking = [check for check, status, _ in rows if status == "fail" and check not in NOT_BLOCKING]
    if blocking:
        failed = ", ".join(blocking)
        console.print(
            f"[red]candor can't edit this model until these pass: {failed}.[/] Nothing was written."
        )
        raise typer.Exit(FAILED_CHECK)

    try:
        revision = _revision(model)
        model_settings = config.Model(
            source=model,
            revision=revision,
            trust_remote_code=checks.needs_remote_code(source.read_config(model)),
            date=_pinned_date(model, revision),
        )
        header = (
            f"Settings for the candor run {name!r}, created {date.today()} by candor {version('candor')} "
            f"for llama.cpp {LLAMA_CPP_RELEASE}.\n"
            "Edit freely: every later command reads and checks this file."
        )
        record = [dict(zip(("check", "result", "detail"), row, strict=True)) for row in rows]
        path = config.write(
            runs,
            name,
            config.RunConfig(
                model=model_settings,
                prompts=_pinned(config.Prompts(), PROMPT_SETS),
                measure=config.Measure(judge_revision=_revision(config.Measure().judge)),
            ),
            header,
            {"inspect.json": json.dumps(record, indent=1)},
        )
    except FileExistsError as error:
        console.print(f"[red]{escape(str(error))}.[/] Pick another name, or delete that folder first.")
        raise typer.Exit(FAILED_CHECK) from None
    except Exception as error:
        console.print(f"[red]candor couldn't create the run:[/] {escape(f'{type(error).__name__}: {error}')}")
        raise typer.Exit(COULD_NOT_INSPECT) from None
    console.print(f"[green]Created {escape(str(path))}[/]")
    if any(check in NOT_BLOCKING and status == "fail" for check, status, _ in rows):
        console.print(
            "[yellow]Bionic can't read this model's chat template.[/] candor still works, but test the "
            "result in Bionic only after pasting in a fixed template."
        )


@app.command()
def upgrade(folder: Annotated[Path, typer.Argument(help="A run folder made by candor init.")]) -> None:
    """Add the settings this version of candor has and a run's candor.toml lacks, at their defaults,
    and remove those it no longer uses.

    Prompt sets it adds are pinned to their datasets' commits now. Every other value already in the
    file is kept, but the file is written afresh, so comments added by hand are lost.
    """
    console = Console()
    folder = folder.expanduser()
    try:
        settings, added, retired, header = config.load_for_upgrade(folder)
        if not added and not retired:
            console.print(f"{escape(str(folder / config.FILE))} already has every setting.")
            return
        new_sets = [name for name in PROMPT_SETS if {"prompts", f"prompts.{name}"} & set(added)]
        settings.prompts = _pinned(settings.prompts, new_sets)
        if {"measure", "measure.judge"} & set(added) and settings.measure.judge_revision is None:
            settings.measure.judge_revision = _revision(settings.measure.judge)
        changes = [f"added {', '.join(added)}"] if added else []
        changes += [f"removed {', '.join(retired)}"] if retired else []
        header += f"\nUpgraded {date.today()} by candor {version('candor')}: {'; '.join(changes)}."
        with Run(folder, settings).locked():
            config.rewrite(folder, settings, header)
    except config.ConfigError as error:
        console.print(f"[red]{escape(str(error))}[/]")
        raise typer.Exit(COULD_NOT_INSPECT) from None
    except RunError as error:
        console.print(f"[red]{escape(str(error))}.[/] Nothing was changed.")
        raise typer.Exit(FAILED_CHECK) from None
    except Exception as error:
        failure = escape(f"{type(error).__name__}: {error}")
        console.print(f"[red]candor couldn't upgrade the run:[/] {failure}")
        raise typer.Exit(COULD_NOT_INSPECT) from None
    if added:
        console.print(f"[green]Added to {escape(str(folder / config.FILE))}:[/]")
        for name in added:
            console.print(f"  {escape(config.setting(settings, name))}", soft_wrap=True)
    if retired:
        console.print(f"[green]Removed from {escape(str(folder / config.FILE))}:[/]")
        for name in retired:
            console.print(f"  {escape(name)}: {escape(config.RETIRED[name])}", soft_wrap=True)


@contextmanager
def _stopped_by_signals() -> Iterator[None]:
    """kill and a closed terminal end candor with SystemExit, so it stops llama-server and any other
    process it started on the way out, instead of leaving them running."""

    def stop(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    # A signal already ignored stays ignored, so nohup keeps a search going after its terminal closes.
    numbers = [n for n in (signal.SIGTERM, signal.SIGHUP) if signal.getsignal(n) is not signal.SIG_IGN]
    previous = {number: signal.signal(number, stop) for number in numbers}
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


@app.command()
def run(folder: Annotated[Path, typer.Argument(help="A run folder made by candor init.")]) -> None:
    """Carry a run through every stage built so far. Finished stages are reused, so it can resume."""
    console = Console()
    folder = folder.expanduser()

    def report(message: str) -> None:
        console.print(f"[dim]{escape(message)}[/]")

    try:
        with _stopped_by_signals():
            done = Run.open(folder, report).prepare()
    except config.ConfigError as error:
        console.print(f"[red]{escape(str(error))}[/]")
        raise typer.Exit(COULD_NOT_INSPECT) from None
    except (RunError, ToolchainError, ConversionError, ServerError) as error:
        console.print(f"[red]The run stopped:[/] {escape(str(error))}")
        raise typer.Exit(FAILED_CHECK) from None
    except Exception as error:
        console.print(
            f"[red]The run stopped on an unexpected error:[/] {escape(f'{type(error).__name__}: {error}')}"
        )
        raise typer.Exit(COULD_NOT_INSPECT) from None
    summary = ", ".join(f"{count} {what}" for what, count in done.items())
    console.print(
        f"[green]Prepared {escape(str(folder))}:[/] {summary}. Next, candor search {escape(str(folder))} "
        "looks for the best edit."
    )


@contextmanager
def _awake() -> Iterator[None]:
    """Keep a Mac awake while candor works. caffeinate stops by itself when candor exits."""
    caffeinate = shutil.which("caffeinate")
    process = subprocess.Popen([caffeinate, "-i", "-w", str(os.getpid())]) if caffeinate else None
    try:
        yield
    finally:
        if process:
            process.terminate()


def _where(error: Exception) -> str:
    """An unexpected error by its type and the line that raised it. Its message could quote a prompt
    or an answer, so it is never shown."""
    frame = traceback.extract_tb(error.__traceback__)[-1]
    return f"{type(error).__name__} at {Path(frame.filename).name}:{frame.lineno}"


def _measurement_name(trial: int | None, heretic: bool, slots: int, limit: int | None) -> str:
    name = "base" if trial is None else f"heretic-{trial}" if heretic else f"trial-{trial}"
    if slots != measure.DEFAULT_SLOTS:
        name += f"-slots-{slots}"
    return name + (f"-first-{limit}" if limit else "")


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60} h {minutes % 60} min" if minutes >= 60 else f"{max(minutes, 1)} min"


@app.command("measure")
def measure_command(
    folder: Annotated[Path, typer.Argument(help="A run folder made by candor init.")],
    heretic: Annotated[
        Path | None,
        typer.Option(help="A heretic study's journal (.jsonl), to measure one of its trials."),
    ] = None,
    trial: Annotated[
        int | None,
        typer.Option(help="A trial of candor search, or with --heretic of heretic's, by the number shown."),
    ] = None,
    slots: Annotated[
        int,
        typer.Option(
            min=1,
            help="How many answers llama-server works on at once. Results can shift slightly with it, so "
            "measuring the same edit with two different numbers shows how much is noise.",
        ),
    ] = measure.DEFAULT_SLOTS,
    limit: Annotated[
        int | None,
        typer.Option(min=1, help="Only the first N prompts of each set, to try the measurement and time it."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(
            help="The measurement's name. By default base or heretic-TRIAL, with -slots-N and -first-N "
            "added when those are set."
        ),
    ] = None,
) -> None:
    """Measure the unedited model, a trial of candor search, or a heretic trial rebuilt with this run's
    directions, on the prompts the search never sees and on maths. Answers go only to the run's private
    folder; this shows counts. A stopped measurement carries on where it stopped."""
    console = Console()
    folder = folder.expanduser()
    if heretic is not None and trial is None:
        raise typer.BadParameter("say which of the study's trials with --trial", param_hint="--heretic")
    name = name or _measurement_name(trial, heretic is not None, slots, limit)
    try:
        config.check_run_name(name)
    except config.ConfigError as error:
        raise typer.BadParameter(str(error), param_hint="--name") from None

    def report(message: str) -> None:
        console.print(f"[dim]{escape(message)}[/]")

    columns = (
        TextColumn("{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("left"),
        TimeRemainingColumn(),
    )
    try:
        with _stopped_by_signals(), _awake(), Progress(*columns, console=console) as bars:

            def progress(description: str, total: int) -> Callable[[], None]:
                task = bars.add_task(description, total=total)
                return lambda: bars.advance(task)

            run = Run.open(folder, report)
            edit = None
            if heretic is not None and trial is not None:
                edit = measure.heretic_trial(heretic.expanduser(), trial, run.settings.edit.row_norm)
            elif trial is not None:
                edit = search.trial_edit(run, trial)
            measurement = measure.Measurement(run, edit, name, slots, limit, progress)
            results = measurement.measure()
    except config.ConfigError as error:
        console.print(f"[red]{escape(str(error))}[/]")
        raise typer.Exit(COULD_NOT_INSPECT) from None
    except (RunError, ToolchainError, ConversionError, ServerError) as error:
        console.print(f"[red]The measurement stopped:[/] {escape(str(error))}")
        raise typer.Exit(FAILED_CHECK) from None
    except Exception as error:
        console.print(f"[red]The measurement stopped on an unexpected error:[/] {escape(_where(error))}")
        raise typer.Exit(COULD_NOT_INSPECT) from None
    table = Table(title=f"candor measure: {name}", title_justify="left", box=box.SIMPLE_HEAD)
    table.add_column("")
    table.add_column("Result", justify="right")
    for row in measure.summary(results):
        table.add_row(*map(escape, row))
    console.print(table)
    console.print(f"Numbers in {escape(str(measurement.folder))}; answers only in the run's private folder.")
    if limit:
        seconds = sum(
            measure.seconds(results[s]) * measurement.sizes[s] / len(results[s]["finished"])
            for s in measure.SETS
        )
        if limit < 4 * slots:
            # Too few prompts to keep every slot busy, so this speed undersells a full run.
            console.print(
                f"Measuring every prompt would take at most about {_duration(seconds)}: {limit} prompts "
                f"can't keep all {slots} slots busy, so a full run goes faster."
            )
        else:
            console.print(
                f"Measuring every prompt would take about {_duration(seconds)} at this speed. Edits "
                "that refuse less give longer answers, so allow more for them."
            )


@app.command()
def compare(
    folder: Annotated[Path, typer.Argument(help="A run folder made by candor init.")],
    first: Annotated[str, typer.Argument(help="A measurement's name, such as base.")],
    second: Annotated[str, typer.Argument(help="Another measurement's name, such as heretic-170.")],
) -> None:
    """Compare two measurements prompt by prompt: each count, how many prompts changed each way, and how
    likely so uneven a change would be by chance (McNemar)."""
    console = Console()
    try:
        run = Run.open(folder.expanduser())
        rows = measure.compare(measure.load(run, first), measure.load(run, second))
    except config.ConfigError as error:
        console.print(f"[red]{escape(str(error))}[/]")
        raise typer.Exit(COULD_NOT_INSPECT) from None
    except RunError as error:
        console.print(f"[red]{escape(str(error))}[/]")
        raise typer.Exit(FAILED_CHECK) from None
    except Exception as error:
        console.print(f"[red]candor couldn't compare them:[/] {escape(_where(error))}")
        raise typer.Exit(COULD_NOT_INSPECT) from None
    console.print(f"candor compare\n  A: {escape(first)}\n  B: {escape(second)}")
    table = Table(box=box.SIMPLE_HEAD)
    for heading in ["", "A", "B", "Change", "Only A/B", "p"]:
        table.add_column(heading, justify="left" if not heading else "right")
    for row in rows:
        table.add_row(*map(escape, row))
    console.print(table)
    console.print(
        "Harmful and Harmless are the held-out sets. Only A/B counts the prompts where just one of them "
        "says yes, and p is how likely a split that uneven would be if neither were really better. "
        "Two measurements of the same edit with different --slots show what noise alone does."
    )


@app.command("search")
def search_command(
    folder: Annotated[Path, typer.Argument(help="A run folder made by candor init.")],
    trials: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Stop once this many trials are finished, of both kinds together, instead of all of "
            "them. A few try the search and time it, and count towards the full search.",
        ),
    ] = None,
) -> None:
    """Search for the edit that changes the model least while it refuses almost nothing. The two kinds
    of edit, one layer's refusal direction for every layer or each layer its own, are searched apart
    and take turns. Each trial is scored in llama-server on the test prompts, its refusals counted by
    the judge; the held-out prompts are left for candor measure. A stopped search carries on where it
    stopped."""
    console = Console()
    folder = folder.expanduser()

    def report(message: str) -> None:
        console.print(f"[dim]{escape(message)}[/]")

    columns = (
        TextColumn("{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("left"),
        TimeRemainingColumn(),
    )
    try:
        with (
            _stopped_by_signals(),
            _awake(),
            Progress(*columns, console=console, speed_estimate_period=DAY) as bars,
        ):
            tasks = []

            def started(done: int, total: int) -> None:
                tasks.append(bars.add_task("Trials", total=total, completed=done))

            def finished(best: str | None) -> None:
                bars.advance(tasks[0])
                if best:
                    console.print(escape(best))

            run = Run.open(folder, report)
            space = run.settings.search
            total = len(search.SCOPES) * space.trials_per_scope
            found, seconds = search.Search(run).search(trials or total, started, finished)
    except config.ConfigError as error:
        console.print(f"[red]{escape(str(error))}[/]")
        raise typer.Exit(COULD_NOT_INSPECT) from None
    except (RunError, ToolchainError, ConversionError, ServerError) as error:
        console.print(f"[red]The search stopped:[/] {escape(str(error))}")
        raise typer.Exit(FAILED_CHECK) from None
    except Exception as error:
        console.print(f"[red]The search stopped on an unexpected error:[/] {escape(_where(error))}")
        raise typer.Exit(COULD_NOT_INSPECT) from None
    table = Table(
        title="Fully scored trials no other beats on both refusals and KL",
        title_justify="left",
        box=box.SIMPLE_HEAD,
    )
    for heading in ["Trial", "Direction", "Refused", "Keywords", "KL"]:
        table.add_column(heading, justify="left" if heading == "Direction" else "right")
    for candidate in search.candidates(found)[:10]:
        table.add_row(
            str(candidate.trial),
            candidate.direction,
            str(candidate.refusals),
            str(candidate.keyword_refusals),
            f"{candidate.kl:.4f}",
        )
    console.print(table)
    _, layers, _, _ = run.shape
    cap = space.max_refusals
    console.print(f"The least KL of each kind within the cap ({cap} refused):")
    picks = []
    for scope in search.SCOPES:
        kept = search.within_cap(found, scope, cap)
        top = kept[0] if kept else search.fewest_refused(found, scope)
        if top is None:
            console.print(f"  {scope}: no trial scored in full yet")
            continue
        described = f"trial {top.trial}, {top.direction}, {top.refusals} refused, KL {top.kl:.4f}"
        if kept:
            console.print(f"  {scope}: {described}")
        else:
            console.print(f"  {scope}: none refused {cap} or fewer; the fewest refused: {described}")
        # The best of the spread start is a point of a fixed grid, so its edges say nothing yet.
        aimed = (
            sum(search.scope_of(t.params) == scope for t in found.trials.values()) > space.startup_per_scope
        )
        if aimed and (edges := search.at_edges(top.params, space, layers)):
            console.print(
                f"    It presses on the edge of the search's ranges ({escape('; '.join(edges))}), so a "
                "better edit may lie outside them."
            )
        picks += [c.trial for c in kept[:2]] if kept else [top.trial]
    if len(found.trials) < total:
        console.print(
            f"Trials took about {seconds:.0f} s each, so all {total} would take about "
            f"{_duration(seconds * total)}. Trials that refuse almost nothing take longest, and later "
            "trials aim at them, so allow more. Run candor search again to carry on."
        )
    elif picks:
        console.print("Measure the best of each kind on the held-out prompts:")
        for number in picks:
            console.print(escape(f"  uv run candor measure {folder} --trial {number}"), soft_wrap=True)


def _direction(layer: float | None) -> str:
    return search.PER_LAYER if layer is None else f"layer {layer:g}"


def _trials(numbers: list[int]) -> str:
    return ("trial " if len(numbers) == 1 else "trials ") + ", ".join(map(str, numbers))


@app.command("score")
def score_command(
    folder: Annotated[Path, typer.Argument(help="A run folder made by candor init.")],
    trial: Annotated[
        int,
        typer.Option(help="A trial of candor search, or with --heretic of heretic's, by the number shown."),
    ],
    heretic: Annotated[
        Path | None,
        typer.Option(help="A heretic study's journal (.jsonl), to score one of its trials."),
    ] = None,
) -> None:
    """Score one edit exactly as the search scores its trials, without adding it to the search. A trial
    of the search shows whether its score repeats. A heretic trial, rebuilt with this run's directions,
    shows where it would sit among the search's trials and whether the search could have drawn it; the
    search's own best trial is scored again beside it, as a check. Shows counts only."""
    console = Console()
    folder = folder.expanduser()

    def report(message: str) -> None:
        console.print(f"[dim]{escape(message)}[/]")

    try:
        with _stopped_by_signals(), _awake():
            run = Run.open(folder, report)
            found = search.load(run)
            front = search.candidates(found) if found else []
            _, layers, _, writers = run.shape
            # Each edit, with its number when it is a trial of the search.
            edits: list[tuple[measure.Edit, int | None]] = []
            if heretic is None:
                edits.append((search.trial_edit(run, trial), trial))
            else:
                edit = measure.heretic_trial(heretic.expanduser(), trial, run.settings.edit.row_norm)
                nearest = search.nearest_layer(edit, run.settings.search, layers)
                edits += [(edit, None)] + ([(nearest, None)] if nearest else [])
                if front:
                    edits.append((search.trial_edit(run, front[0].trial), front[0].trial))
            scored = search.Search(run).score([edit for edit, _ in edits])
            searched = {
                number: search.searched_score(found, number) for _, number in edits if found and number
            }
    except config.ConfigError as error:
        console.print(f"[red]{escape(str(error))}[/]")
        raise typer.Exit(COULD_NOT_INSPECT) from None
    except (RunError, ToolchainError, ConversionError, ServerError) as error:
        console.print(f"[red]The scoring stopped:[/] {escape(str(error))}")
        raise typer.Exit(FAILED_CHECK) from None
    except Exception as error:
        console.print(f"[red]The scoring stopped on an unexpected error:[/] {escape(_where(error))}")
        raise typer.Exit(COULD_NOT_INSPECT) from None
    table = Table(box=box.SIMPLE_HEAD)
    for heading in ["Edit", "Direction", "Refused", "Keywords", "KL", "In the search"]:
        table.add_column(heading, justify="left" if heading in {"Edit", "Direction"} else "right")
    for (edit, number), score in zip(edits, scored.edits, strict=True):
        before = searched.get(number) if number else None
        table.add_row(
            escape(edit.name),
            _direction(edit.direction_layer),
            str(score.refusals),
            str(score.keyword_refusals),
            f"{score.kl:.4f}",
            f"{before.refusals}, {before.keyword_refusals}, {before.kl:.4f}" if before else "",
        )
    console.print(f"Scored as the search scores its trials, on its {scored.prompts} test prompts:")
    console.print(table)
    base = found.base_refusals if found else None
    console.print(
        f"The unedited model refuses {scored.base_refusals} of {scored.prompts}"
        + (f" ({base} in the search)." if base is not None else ".")
    )
    if not found or not front:
        console.print(
            f"The search has no {'finished trials' if found else 'trials'} yet to set these beside."
        )
        return
    reliable = True
    for (edit, number), score in zip(edits, scored.edits, strict=True):
        if number:
            if searched[number].judged < scored.prompts:
                console.print(
                    f"{escape(edit.name)} was stopped early in the search, after {searched[number].judged} "
                    "prompts, so there's no full score of it to repeat."
                )
            elif search.repeats(score, searched[number]):
                console.print(
                    f"{escape(edit.name)} scores as it did in the search, so the scores above can be set "
                    "beside the search's."
                )
            else:
                reliable = False
                console.print(
                    f"{escape(edit.name)} scores differently from the search (the last column), so the "
                    "scores above can't be set beside the search's, and the verdicts below don't hold."
                )
    components = sorted({writer.component for writer in writers})
    for (edit, number), score in zip(edits, scored.edits, strict=True):
        if number:
            continue
        name = escape(edit.name)
        beaten_by = [c.trial for c in front if search.beats(c, score)]
        tied = [c.trial for c in front if search.ties(c, score)]
        beaten = [c.trial for c in front if search.beats(score, c)]
        verdict = (
            f"{name}: the search's best {_trials(beaten_by)} {'beats' if len(beaten_by) == 1 else 'beat'} it "
            "on both refusals and KL."
            if beaten_by
            else f"{name}: no trial of the search beats it on both refusals and KL"
            + (f"; it ties with {_trials(tied)}" if tied else "")
            + (f"; it beats the search's best {_trials(beaten)} on both." if beaten else ".")
        )
        console.print(verdict if reliable else f"Not reliable: {verdict}")
        if outside := search.outside_space(edit, run.settings.search, layers, components):
            console.print(f"  The search couldn't have drawn it: {escape('; '.join(outside))}.")
        else:
            console.print("  The search could have drawn it.")
