"""The candor command line."""

import json
import os
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import traceback
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any

import click
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

from . import checks, config, export, measure, proxy, residuals, search, serve, source, targets
from .convert import ConversionError
from .engine import LLAMA_CPP_RELEASE, ServerError, terminate
from .run import PROMPT_SETS, Run, RunError, duration, every_tenth, record_of
from .toolchain import ToolchainError

app = typer.Typer(
    add_completion=False,
    help="Remove a model's refusal direction, scored the way you actually run the model. With no "
    "command, in a terminal, menus show each run in runs/ and run these commands for you.",
)
MARKS = {
    "ok": "[green]ok[/]",
    "warn": "[yellow]warn[/]",
    "fail": "[red]fail[/]",
}
# Every command exits with 1 when a check fails or candor stops it for a reason it gives, and 3 when it
# couldn't get that far: settings it can't read, a model it can't inspect or an unexpected error. 2 is
# typer's, for usage errors.
FAILED_CHECK, COULD_NOT_INSPECT = 1, 3
# A search's progress takes its speed from every trial of the session, since trials take minutes and
# rich otherwise looks back only 30 seconds.
DAY = 24 * 3600
GIB = 2**30
# How many edit targets inspect reads, spread over the layers, and about how many of each one's first
# values, to tell weights quantised before.
SAMPLED_WEIGHTS, SAMPLED_VALUES = 4, 4096
# How many of each kind of tensor export names that the export of a model edited elsewhere leaves out.
LEFT_OUT_SHOWN = 5
WITHOUT_TOOLS = " (drop --mcp, or put CANDOR_MCP= before the command, to serve without tools)"

Row = tuple[str, str, str]


class CannotInspect(Exception):
    pass


@app.callback(invoke_without_command=True)
def main(context: typer.Context) -> None:
    # The Hub's and transformers' progress bars and warnings would only clutter candor's own tables
    # and messages.
    disable_progress_bars()
    hub_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    residuals.protect_memory()
    if context.invoked_subcommand is not None:
        return
    if not _in_a_terminal():
        typer.echo(context.get_help())
        raise typer.Exit(2)
    from . import menu

    menu.start()


def _in_a_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


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


def _target_rows(model: str, hidden: int, layers: int, arch: str | None, revision: str | None) -> list[Row]:
    try:
        shapes = source.tensor_shapes(model, revision)
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
    if found.hidden_last:
        detail = (
            f"{_examples(found.hidden_last)} hold their experts as [experts, in, hidden], or square so "
            "candor can't tell which way, which it doesn't edit yet"
        )
        rows.append(("Experts", "fail", detail))
    elif found.experts:
        per_layer = Counter(w.layer for w in found.experts)
        counts = {
            shapes[w.name][0] if len(shapes[w.name]) == 3 else per_layer[w.layer] for w in found.experts
        }
        each = " or ".join(str(n) for n in sorted(counts))
        detail = (
            f"{each} experts in each of {len(per_layer)} layers, all edited along the layer's direction at "
            "its MLP strength"
        )
        rows.append(("Experts", "ok", detail))
    if found.quantized:
        detail = f"{_examples(found.quantized)} carry quantisation scales; use the unquantised original"
        rows.append(("Quantised", "fail", detail))
    rows += _quantised_before_rows(model, found.writers, revision)
    if found.output_norms:
        detail = (
            "this model normalises each block's output before adding it to the residual stream, so the "
            "edit can't remove the direction exactly"
        )
        rows.append(("Output norms", "warn", detail))
    if found.unedited:
        detail = (
            f"{_examples(found.unedited)}: the per-layer n-gram embedding it writes reaches the residual "
            "stream through a convolution too, so removing a direction from it wouldn't be exact"
        )
        rows.append(("Left unedited", "warn", detail))
    if arch and not found.hidden_last:
        unmapped = [w.name for w in found.writers if targets.gguf_name(w, arch, layers) is None]
        if unmapped:
            rows.append(("GGUF names", "fail", f"no llama.cpp name for {_examples(unmapped)}"))
        else:
            try:
                edit_targets = targets.edit_targets(found.writers, arch, layers)
            except ValueError as error:
                return [*rows, ("GGUF names", "fail", str(error))]
            first = found.writers[0]
            example = f"{first.name} -> {targets.gguf_name(first, arch, layers)}"
            if experts := [t for t in edit_targets.values() if t.writers[0].expert]:
                example += f"; {experts[0].name} -> {experts[0].gguf}, a layer's experts"
            rows.append(("GGUF names", "ok", f"all map, e.g. {example}"))
    return rows


def _quantised_before_rows(model: str, writers: list[targets.Writer], revision: str | None) -> list[Row]:
    """A warning when edit targets read across the layers look quantised to 4 bits before, as
    rude-llama's were: quantising them to 4 bits again rounds most of an edit away."""
    picked = [writers[i * (len(writers) - 1) // (SAMPLED_WEIGHTS - 1)].name for i in range(SAMPLED_WEIGHTS)]
    try:
        read = source.first_rows(model, list(dict.fromkeys(picked)), SAMPLED_VALUES, revision)
    except Exception as error:
        return [("Quantised before", "warn", f"couldn't read a few weights to tell: {error}")]
    if not (flagged := [name for name, rows in read.items() if source.quantised_before(rows)]):
        return []
    detail = (
        f"{_examples(flagged)} hold at most {source.FEWEST} values in most blocks of {source.BLOCK} (zeros "
        "aside), as "
        "weights quantised to 4 bits and saved unquantised do: quantising them to 4 bits again would "
        "round most of an edit away, so candor export keeps such weights as Q8_0 or F32 and the file is "
        "larger"
    )
    return [("Quantised before", "warn", detail)]


def _tokenizer_rows(model: str, revision: str | None) -> list[Row]:
    try:
        with _quiet_transformers():
            tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
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
    if efforts := checks.reasoning_efforts(tokenizer):
        rows.append(("Reasoning effort", "ok", _efforts(efforts)))
    if close := checks.think_close(tokenizer):
        detail = f"thinks before answering, and its thinking ends with {close!r}; it is scored thinking"
        if not checks.turns_thinking_off(tokenizer):
            detail += ", as its template can't turn thinking off"
        rows.append(("Thinking", "ok", detail))
    elif checks.turns_thinking_off(tokenizer):
        detail = (
            "its template turns thinking off, but its prompts don't open the thinking, so the model opens "
            "it itself and candor would judge its thinking as its answer"
        )
        rows.append(("Thinking", "fail", detail))
    elif checks.may_write_its_own_thinking(tokenizer, template):
        detail = (
            "its template names a think tag, but its prompts neither open the thinking nor close it: if the "
            "model writes its own thinking, candor would judge it as its answer"
        )
        rows.append(("Thinking", "warn", detail))
    else:
        rows.append(("Thinking", "ok", "answers without thinking, as far as its template shows"))
    return rows


def _kv_cache_rows(model_config: dict[str, Any], arch: str | None) -> list[Row]:
    """The memory a token of context and a slot take, at the KV cache type init gives a run, where candor
    knows them; nothing otherwise."""
    cache = config.Deployment().kv_cache
    per_token = targets.kv_bytes_per_token(model_config, arch, config.KV_CACHE_TYPES[cache])
    per_slot = targets.recurrent_bytes_per_slot(model_config, arch)
    if per_token is None or per_slot is None:
        return []
    detail = (
        f"{per_token / 1024:g} KiB a token at {cache}, so {int(GIB // per_token):,} tokens of context per GiB"
    )
    if per_slot:
        state = "linear-attention layers' state"
        if arch == "qwen4exp" and source.text_config(model_config).get("ple_layer_ids"):
            state += " and its n-gram embedding's convolution history"
        detail += f", plus {per_slot / 2**20:.2f} MiB a slot for its {state}"
    return [("KV cache", "ok", detail)]


def _efforts(efforts: checks.Efforts) -> str:
    """The reasoning_effort values a chat template takes, as a sentence."""
    *values, last = [f'"{value}"' for value in efforts.values]
    listed = f"{', '.join(values)} or {last}" if values else last
    default = f'; "{efforts.default}" when given none' if efforts.default else ""
    return f"the chat template takes {listed}{default}"


def _rows(model: str, revision: str | None = None) -> list[Row]:
    """The checks of a model at a Hub commit, or as it is now."""
    try:
        config = source.read_config(model, revision)
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
    code = checks.remote_code(config, source.read_json(model, "tokenizer_config.json", revision))
    if checks.needs_remote_code(config):
        detail = f"transformers can only load it by running {_examples(code)}; a run does, so read it first"
        rows.append(("Remote code", "warn", detail))
    elif code:
        detail = f"ships {_examples(code)}, but candor uses transformers' own code for it"
        rows.append(("Remote code", "ok", detail))
    else:
        rows.append(("Remote code", "ok", "none"))
    return (
        rows
        + _target_rows(model, hidden, layers, arch, revision)
        + _tokenizer_rows(model, revision)
        + _kv_cache_rows(config, arch)
    )


def _resolved(model: str) -> str:
    """Local folders are stored as absolute paths; anything else is a Hugging Face id."""
    folder = Path(model).expanduser()
    return str(folder.resolve()) if folder.is_dir() else model


def _checked_rows(console: Console, model: str, revision: str | None = None) -> list[Row]:
    try:
        return _rows(model, revision)
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


@app.command(short_help="Check a model before spending anything on it.")
def inspect(model: Annotated[str, typer.Argument(help="A Hugging Face model id or a local folder.")]) -> None:
    """Check a model before spending anything on it. Reads only the config, the tokenizer, the file
    headers and the first rows of a few weights.

    Exits with 1 when a check fails and 3 when the model can't be inspected at all.
    """
    console = Console()
    model = _resolved(model)
    rows = _checked_rows(console, model)
    _print_table(console, f"candor inspect {model}", rows)
    if any(status == "fail" for _, status, _ in rows):
        raise typer.Exit(FAILED_CHECK)


def _from_template(
    model: str, revision: str | None
) -> tuple[str | None, dict[str, Any] | None, str | None, checks.Efforts | None]:
    """The date to pin, if the chat template writes one it can be given; the settings to render it
    with, a reasoning_effort at the template's own default if it takes one; what ends the model's
    thinking with those settings, if it thinks; and the reasoning_effort values it takes."""
    with _quiet_transformers():
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    date_format = checks.pinnable_date_format(checks.chat_template(tokenizer) or "")
    efforts = checks.reasoning_efforts(tokenizer)
    settings = {checks.EFFORT: efforts.default} if efforts and efforts.default else None
    close = checks.think_close(tokenizer, **(settings or {}))
    return config.today_in(date_format) if date_format else None, settings, close, efforts


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


@app.command(short_help="Start a run: check the model and write its settings.")
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
    # The commit the run pins is the one checked.
    try:
        revision = _revision(model)
    except Exception as error:
        console.print(f"[red]candor couldn't find the commit {escape(model)} is at:[/] {escape(str(error))}")
        raise typer.Exit(COULD_NOT_INSPECT) from None
    rows = _checked_rows(console, model, revision)
    _print_table(console, f"candor init {name}: checking {model}", rows)
    blocking = [check for check, status, _ in rows if status == "fail"]
    if blocking:
        failed = ", ".join(blocking)
        console.print(
            f"[red]candor can't edit this model until these pass: {failed}.[/] Nothing was written."
        )
        raise typer.Exit(FAILED_CHECK)

    try:
        pinned_date, template_settings, close, efforts = _from_template(model, revision)
        model_settings = config.Model(
            source=model,
            revision=revision,
            trust_remote_code=checks.needs_remote_code(source.read_config(model, revision)),
            date=pinned_date,
        )
        deployment = config.Deployment(
            chat_template_kwargs=template_settings,
            think_close=close,
            thinking_budget=config.THINKING_BUDGET if close else None,
        )
        header = (
            f"Settings for the candor run {name!r}, created {date.today()} by candor {version('candor')} "
            f"for llama.cpp {LLAMA_CPP_RELEASE}.\n"
            "Edit freely: every later command reads and checks this file."
        )
        if efforts:
            header += f"\nReasoning effort: {_efforts(efforts)}."
        record = [dict(zip(("check", "result", "detail"), row, strict=True)) for row in rows]
        path = config.write(
            runs,
            name,
            config.RunConfig(
                model=model_settings,
                deployment=deployment,
                slots=config.Slots(thinking=config.THINKING_MODEL_SLOTS, measure=config.THINKING_MODEL_SLOTS)
                if close
                else config.Slots(),
                prompts=_pinned(config.Prompts(), PROMPT_SETS),
                scoring=config.Scoring(kl_tokens=config.KL_TOKENS),
                measure=config.Measure(
                    judge_revision=_revision(config.Measure().judge),
                    second_judge=config.SECOND_JUDGE,
                    second_judge_revision=_revision(config.SECOND_JUDGE),
                ),
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
    if efforts:
        chosen = (
            f'The run uses "{efforts.default}"'
            if template_settings
            else "The run leaves it unset, so the template uses its own default"
        )
        console.print(
            escape(
                f"Reasoning effort: {_efforts(efforts)}. {chosen} (deployment.chat_template_kwargs): set it "
                "to the one you run at, and think_close to match."
            )
        )
    if close:
        console.print(
            f"[yellow]This model thinks before it answers.[/] Its thinking is capped at "
            f"{config.THINKING_BUDGET} tokens (deployment.thinking_budget). Set {escape('[sampling]')} to "
            "the thinking settings its model card recommends: the defaults answer greedily."
        )


@app.command(short_help="Bring a run's candor.toml up to this version's settings.")
def upgrade(folder: Annotated[Path, typer.Argument(help="A run folder made by candor init.")]) -> None:
    """Add the settings this version of candor has and a run's candor.toml lacks, at their defaults.

    Prompt sets it adds are pinned to their datasets' commits now. Every other value already in the
    file is kept, but the file is written afresh, so comments added by hand are lost.
    """
    console = Console()
    folder = folder.expanduser()
    try:
        settings, added, header = config.load_for_upgrade(folder)
        if not added:
            console.print(f"{escape(str(folder / config.FILE))} already has every setting.")
            return
        new_sets = [name for name in PROMPT_SETS if {"prompts", f"prompts.{name}"} & set(added)]
        settings.prompts = _pinned(settings.prompts, new_sets)
        if {"measure", "measure.judge"} & set(added) and settings.measure.judge_revision is None:
            settings.measure.judge_revision = _revision(settings.measure.judge)
        header += f"\nUpgraded {date.today()} by candor {version('candor')}: added {', '.join(added)}."
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
    console.print(f"[green]Added to {escape(str(folder / config.FILE))}:[/]")
    for name in added:
        console.print(f"  {escape(config.setting(settings, name))}", soft_wrap=True)


def _lines(console: Console, what: str, total: int, done: int = 0) -> Callable[[], None]:
    """Progress as a line at every tenth, where output goes to a file such as a pod's log, which shows no
    progress bar."""
    if console.is_terminal:
        return lambda: None
    return every_tenth(lambda line: console.print(escape(line)), what, total, done)


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


@app.command(short_help="Build every stage the search needs, reusing what is built.")
def run(folder: Annotated[Path, typer.Argument(help="A run folder made by candor init.")]) -> None:
    """Carry a run through every stage built so far. Finished stages are reused, so it can resume."""
    console = Console()
    folder = folder.expanduser()

    report = _reporter(console)
    with _stopping(console, "The run"), _stopped_by_signals(), _awake():
        done = Run.open(folder, report).prepare()
    summary = ", ".join(f"{count} {what}" for what, count in done.items())
    console.print(
        f"[green]Prepared {escape(str(folder))}:[/] {summary}. Next, candor search {escape(str(folder))} "
        "looks for the best edit."
    )


@contextmanager
def _awake() -> Iterator[None]:
    """Keep a Mac awake while candor works. caffeinate stops by itself when candor exits."""
    caffeinate = shutil.which("caffeinate")
    process = subprocess.Popen([caffeinate, "-is", "-w", str(os.getpid())]) if caffeinate else None
    try:
        yield
    finally:
        if process:
            process.terminate()


def _where(error: Exception) -> str:
    """An unexpected error by its type, the line that raised it and the lines of candor's own that led
    there, innermost first. Its message could quote a prompt or an answer, so it is never shown."""
    frames = traceback.extract_tb(error.__traceback__)
    # _stopping's own frame, through which every command's error passes, says nothing.
    ours = [
        f for f in frames[:-1] if Path(f.filename).parent == Path(__file__).parent and f.name != "_stopping"
    ]
    shown = [f"{Path(f.filename).name}:{f.lineno}" for f in reversed([*ours, frames[-1]])]
    return f"{type(error).__name__} at {shown[0]}" + (f", from {', '.join(shown[1:])}" if ours else "")


@contextmanager
def _stopping(console: Console, what: str, *more: type[Exception]) -> Iterator[None]:
    """End a command with its exit code and a line saying why when what it runs fails: a check that
    failed, or an unexpected error, by where it happened only."""
    try:
        yield
    except (typer.Exit, click.ClickException):
        raise
    except config.ConfigError as error:
        console.print(f"[red]{escape(str(error))}[/]")
        raise typer.Exit(COULD_NOT_INSPECT) from None
    except (RunError, ToolchainError, ConversionError, ServerError, *more) as error:
        console.print(f"[red]{what} stopped:[/] {escape(str(error))}")
        raise typer.Exit(FAILED_CHECK) from None
    except Exception as error:
        console.print(f"[red]{what} stopped on an unexpected error:[/] {escape(_where(error))}")
        raise typer.Exit(COULD_NOT_INSPECT) from None


def _multiple(low: float, high: float) -> str:
    """How many times its trial's strength an edit's strengths are, as the proxy shows it."""
    shown = f"{low:.3g}", f"{high:.3g}"
    return f"x{shown[0]}" + (f"-{shown[1]}" if shown[1] != shown[0] else "")


def _reporter(console: Console) -> Callable[[str], None]:
    """How a command shows what candor is doing: dimmed, under its own lines."""
    return lambda message: console.print(f"[dim]{escape(message)}[/]")


def _bars(console: Console, **settings: Any) -> Progress:
    return Progress(
        TextColumn("{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("left"),
        TimeRemainingColumn(),
        console=console,
        **settings,
    )


def _heretic_trial(run: Run, journal: Path, trial: int, rebuilt: bool = True) -> measure.Edit:
    """A heretic study's trial, as this run rebuilds it. Serving only names it to find its export, so it
    needn't read the model, whose weights a machine that only serves may not hold."""
    settings = run.settings
    experts = rebuilt and any(w.expert for w in run.shape[3])
    return measure.heretic_trial(
        journal.expanduser(), trial, settings.edit.row_norm, settings.deployment.think_close, experts
    )


def _edit_name(trial: int, heretic: bool) -> str:
    return f"heretic-{trial}" if heretic else f"trial-{trial}"


def _measurement_name(
    trial: int | None,
    heretic: bool,
    slots: int,
    limit: int | None,
    exported: bool,
    seed: int | None,
    outside: str | None = None,
) -> str:
    name = "base" if trial is None else _edit_name(trial, heretic) + ("-exported" if exported else "")
    name = outside or name
    if slots != config.MEASURE_SLOTS:
        name += f"-slots-{slots}"
    if seed is not None:
        name += f"-seed-{seed}"
    return name + (f"-first-{limit}" if limit else "")


@app.command("measure", short_help="Measure the unedited model or a trial, on prompts and maths.")
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
        int | None,
        typer.Option(
            min=1,
            help="How many answers llama-server works on at once, instead of slots.measure. Results can "
            "shift slightly with it, so measuring the same edit with two different numbers shows how much "
            "is noise.",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(min=1, help="Only the first N prompts of each set, to try the measurement and time it."),
    ] = None,
    exported: Annotated[
        bool,
        typer.Option(
            help="Measure the trial's GGUF made by candor export, the file you run, instead of the "
            "run's model with the trial's adapters."
        ),
    ] = False,
    outside: Annotated[
        str | None,
        typer.Option(help="Measure a model edited elsewhere, by the name candor export --outside gave it."),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            min=0,
            max=2**32 - 1,
            help="Sample with this seed instead of the run's (sampling.seed). Measuring the same edit with "
            "two seeds shows how much sampling alone moves the counts.",
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(
            help="The measurement's name. By default base, trial-N or heretic-N, with -exported, "
            f"-slots-N (at other than {config.MEASURE_SLOTS}), -seed-N and -first-N added when those are "
            "set."
        ),
    ] = None,
    sets: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            help=f"Measure only this set, one of {', '.join(measure.SETS)}; repeat it for more. It is kept "
            "as a full measurement keeps it, so the others can be measured later under the same name.",
        ),
    ] = None,
) -> None:
    """Measure the unedited model, a trial of candor search, or a heretic trial rebuilt with this run's
    directions, on the prompts the search never sees and on maths. Answers go only to the run's private
    folder; this shows counts. A stopped measurement carries on where it stopped."""
    console = Console()
    folder = folder.expanduser()
    if unknown := [given for given in dict.fromkeys(sets or ()) if given not in measure.SETS]:
        raise typer.BadParameter(
            f"not a set candor measure answers: {', '.join(unknown)}; choose from {', '.join(measure.SETS)}",
            param_hint="--set",
        )
    if heretic is not None and trial is None:
        raise typer.BadParameter("say which of the study's trials with --trial", param_hint="--heretic")
    if exported and trial is None:
        raise typer.BadParameter("say which trial's export with --trial", param_hint="--exported")
    if outside is not None and (trial is not None or exported):
        raise typer.BadParameter("measures a model edited elsewhere on its own", param_hint="--outside")
    for given, hint in ((name, "--name"), (outside, "--outside")):
        if given is not None:
            try:
                config.check_run_name(given)
            except config.ConfigError as error:
                raise typer.BadParameter(str(error), param_hint=hint) from None

    report = _reporter(console)
    with (
        _stopping(console, "The measurement"),
        _stopped_by_signals(),
        _awake(),
        _bars(console) as bars,
    ):

        def progress(description: str, total: int) -> Callable[[], None]:
            task = bars.add_task(description, total=total)
            line = _lines(console, description, total)

            def done() -> None:
                bars.advance(task)
                line()

            return done

        run = Run.open(folder, report)
        if seed is not None and run.settings.sampling.temperature == 0:
            raise RunError(
                "--seed samples the answers with other draws, but this run answers greedily, which "
                "every seed answers alike"
            )
        slots = run.settings.slots.measure if slots is None else slots
        default = _measurement_name(trial, heretic is not None, slots, limit, exported, seed, outside)
        if name not in (None, default) and export.RESERVED.fullmatch(name):
            raise RunError(
                f"--name {name} is how candor names a measurement of the unedited model or of a trial, "
                "so it could replace one; choose another"
            )
        name = name or default
        edit = None
        if heretic is not None and trial is not None:
            edit = _heretic_trial(run, heretic, trial)
        elif trial is not None:
            # The export holds the edit, so the trial needn't be rebuildable as scored; the
            # export itself is checked against the run as it is now.
            edit = search.recorded_edit(run, trial) if exported else search.trial_edit(run, trial)
        measurement = measure.Measurement(
            run, edit, name, slots, limit, progress, exported, seed, outside, sets or measure.SETS
        )
        results = measurement.measure()
    table = Table(title=f"candor measure: {name}", title_justify="left", box=box.SIMPLE_HEAD)
    table.add_column("")
    table.add_column("Result", justify="right")
    for row in measure.summary(results):
        table.add_row(*map(escape, row))
    console.print(table)
    console.print(f"Numbers in {escape(str(measurement.folder))}; answers only in the run's private folder.")
    if outside is not None:
        _warn_left_out(console, run, outside)
    if limit:
        seconds = sum(
            measure.seconds(results[s]) * measurement.sizes[s] / len(results[s]["finished"]) for s in results
        )
        every = "every prompt of these sets" if sets else "every prompt"
        if limit < 4 * slots:
            # Too few prompts to keep every slot busy, so this speed undersells a full run.
            console.print(
                f"Measuring {every} would take at most about {duration(seconds)}: {limit} prompts "
                f"can't keep all {slots} slots busy, so a full run goes faster."
            )
        else:
            console.print(
                f"Measuring {every} would take about {duration(seconds)} at this speed. Edits "
                "that refuse less give longer answers, so allow more for them."
            )


@app.command(short_help="Compare two measurements prompt by prompt.")
def compare(
    folder: Annotated[Path, typer.Argument(help="A run folder made by candor init.")],
    first: Annotated[str, typer.Argument(help="A measurement's name, such as base.")],
    second: Annotated[str, typer.Argument(help="Another measurement's name, such as heretic-170.")],
) -> None:
    """Compare two measurements prompt by prompt: each count, how many prompts changed each way, how
    likely so uneven a change would be by chance (McNemar), and how far the change could be from what
    was seen."""
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
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    for heading in ["", "A", "B", "Change", "95% interval", "Only A/B", "p"]:
        table.add_column(heading, justify="left" if not heading else "right", no_wrap=bool(heading))
    for row in rows:
        table.add_row(*map(escape, row))
    console.print(table)
    console.print(
        "Harmful and Harmless are the held-out sets. Only A/B counts the prompts where just one of them "
        "says yes, and p is how likely a split that uneven would be if neither were really better, either "
        "way (two-sided). The 95% interval is the change's, in prompts, or the KL's: it allows for which "
        "prompts were asked, not for what batching moves, and a KL that a few prompts carry may still lie "
        "outside it. Two measurements of the same edit with different --slots or --seed show what noise "
        "alone does."
    )


@app.command("export", short_help="Write a trial's edit into the GGUF you run.")
def export_command(
    folder: Annotated[Path, typer.Argument(help="A run folder made by candor init.")],
    trial: Annotated[
        int | None,
        typer.Option(help="A trial of candor search, or with --heretic of heretic's, by the number shown."),
    ] = None,
    heretic: Annotated[
        Path | None,
        typer.Option(help="A heretic study's journal (.jsonl), to export one of its trials."),
    ] = None,
    outside: Annotated[
        Path | None,
        typer.Option(
            help="A copy of the run's model edited elsewhere, such as heretic's own save of a trial, as a "
            "folder or one unquantised GGUF file, to export as candor exports its own edits. Needs --name."
        ),
    ] = None,
    name: Annotated[str | None, typer.Option(help="What to call a model given with --outside.")] = None,
) -> None:
    """Write a trial's edit into a copy of the run's GGUF, quantised as the run deploys: the GGUF you
    run. Measure that exact file before using it, with candor measure --exported. With
    --outside, a model edited elsewhere is exported the same way, to measure it like for like."""
    console = Console()
    folder = folder.expanduser()
    if (trial is None) == (outside is None):
        raise typer.BadParameter(
            "say which trial with --trial, or give a model with --outside", param_hint="--trial"
        )
    if (outside is None) != (name is None):
        raise typer.BadParameter(
            "--name names a model given with --outside, and only that", param_hint="--name"
        )
    if outside is not None and heretic is not None:
        raise typer.BadParameter("exports a model edited elsewhere on its own", param_hint="--outside")
    if name is not None:
        try:
            config.check_run_name(name)
        except config.ConfigError as error:
            raise typer.BadParameter(str(error), param_hint="--name") from None

    report = _reporter(console)
    with _stopping(console, "The export"), _stopped_by_signals(), _awake():
        run = Run.open(folder, report)
        if outside is not None:
            assert name is not None
            shipped = export.export_outside(run, name, outside)
        elif heretic is not None:
            assert trial is not None
            edit = _heretic_trial(run, heretic, trial)
            shipped = export.export(run, edit)
        else:
            assert trial is not None
            shipped = export.export(run, search.trial_edit(run, trial))
    shown = name or _edit_name(trial or 0, heretic is not None)
    console.print(f"{escape(shown)}: {escape(str(shipped))}, {shipped.stat().st_size / 1e9:.1f} GB")
    if name is not None:
        _warn_left_out(console, run, name, record_of(shipped))
    # Named as candor measure names them at the run's slots.
    slots = run.settings.slots.measure
    base = _measurement_name(None, False, slots, None, False, None)
    if outside is not None:
        console.print("Measure this file, then compare it with the unedited model:")
        measured = _measurement_name(None, False, slots, None, False, None, name)
        commands = [
            ["measure", str(folder), "--outside", str(name)],
            ["compare", str(folder), base, measured],
        ]
    else:
        assert trial is not None
        console.print(
            "Measure this file, then compare it with the unedited model, and with the trial as measured "
            "with adapters to see what writing it into the weights and quantising changed:"
        )
        picked = ["--trial", str(trial), *(["--heretic", str(heretic)] if heretic is not None else [])]
        adapters, exported = (
            _measurement_name(trial, heretic is not None, slots, None, made, None) for made in (False, True)
        )
        commands = [
            ["measure", str(folder), *picked, "--exported"],
            ["compare", str(folder), base, exported],
            ["compare", str(folder), adapters, exported],
        ]
    for command in commands:
        console.print(escape("  " + shlex.join(["uv", "run", "candor", *command])), soft_wrap=True)


def _warn_left_out(console: Console, run: Run, name: str, listed: Path | None = None) -> None:
    """Warn of the tensors of the model edited elsewhere exported as name that its export leaves out, by
    kind: counted, and given the record that lists them all, the first few named."""
    given, left = export.left_out(run, name)
    shown = escape(str(given))
    if left is None:
        console.print(
            f"[yellow]{escape(name)}'s export was made before candor listed the tensors of {shown} it leaves "
            "out[/]: remove its folder in exports/ and export it again to list them."
        )
        return
    kinds = {
        "differ": (
            "differs from the run's model in {} outside candor's edit targets",
            "; the export takes the run's model's own, so any edit there is left out",
        ),
        "missing": ("holds {} missing from the run's model", ", which the export leaves out"),
    }
    for kind, (found, rest) in kinds.items():
        if not (names := left[kind]):
            continue
        counted = f"{len(names)} tensor" + ("" if len(names) == 1 else "s")
        if listed is not None:
            more = len(names) - LEFT_OUT_SHOWN
            rest += ": " + ", ".join(names[:LEFT_OUT_SHOWN])
            rest += f" and {more} more, all listed in {listed}" if more > 0 else ""
        console.print(f"[yellow]{shown} {found.format(counted)}[/]{escape(rest)}")


@app.command("serve", short_help="Chat with the unedited model or a trial's export, as measured.")
def serve_command(
    folder: Annotated[Path, typer.Argument(help="A run folder made by candor init.")],
    trial: Annotated[
        int | None,
        typer.Option(
            help="Serve this trial's export, made by candor export, by the number shown; without it, the "
            "unedited model."
        ),
    ] = None,
    heretic: Annotated[
        Path | None,
        typer.Option(help="A heretic study's journal (.jsonl), to serve the export of one of its trials."),
    ] = None,
    port: Annotated[
        int, typer.Option(min=1, max=65535, help="The port for the chat page and the API.")
    ] = 8080,
    context: Annotated[
        int | None,
        typer.Option(
            min=256,
            help="Tokens of context, at most the model's own limit. \\[default: 16384, or 32768 with tools]",
        ),
    ] = None,
    mcp: Annotated[
        Path | None,
        typer.Option(
            envvar="CANDOR_MCP",
            help="A file of MCP servers in Cursor's format (llama-server's --mcp-servers-config) whose "
            "tools, such as a sandbox or web search, the model can call from the chat page; CANDOR_MCP "
            "set to one gives every serve its tools, and set empty, none. Their descriptions then reach "
            "the model too, so it no longer runs exactly as measured.",
        ),
    ] = None,
) -> None:
    """Run the unedited model or a trial's export in llama-server as candor measures it: the run's KV
    cache type, chat template settings, sampling and thinking budget, with today's date. It first shows
    the command, the settings and how the chat template renders a first message, so the mode (such as
    thinking) is plain before you chat; then it serves llama-server's chat page until Ctrl-C."""
    console = Console()
    folder = folder.expanduser()
    if heretic is not None and trial is None:
        raise typer.BadParameter("say which of the study's trials with --trial", param_hint="--heretic")
    if mcp is not None:
        mcp = mcp.expanduser().resolve()
        if not mcp.is_file():
            raise typer.BadParameter(f"{mcp} is not a file{WITHOUT_TOOLS}", param_hint="--mcp or CANDOR_MCP")

    report = _reporter(console)
    with _stopping(console, "candor serve", serve.ServeError), _stopped_by_signals():
        run = Run.open(folder, report)
        if trial is None:
            name = "the unedited model"
            with run.locked():
                _, model = run.gguf()
        else:
            if heretic is not None:
                edit = _heretic_trial(run, heretic, trial, rebuilt=False)
            else:
                edit = search.recorded_edit(run, trial)
            name = edit.name
            with run.locked():
                model = export.built(run, edit)
        # Tool calls and the pages they return fill a chat's context fast.
        context = context or (32768 if mcp else 16384)
        limit = serve.context_limit(model)
        context = min(context, limit) if limit else context
        args = serve.arguments(run, model, port, context, mcp)
        settings = serve.chat_settings(run)
        system = run.settings.deployment.system_prompt
        console.print(f"Serving {escape(name)}: {escape(str(model))}")
        temperature = run.settings.sampling.temperature
        sampled = "greedily" if temperature == 0 else f"at temperature {temperature:g}"
        budget = run.settings.deployment.thinking_budget
        console.print(
            f"KV cache {run.settings.deployment.kv_cache}, {context} tokens of context, chat template "
            f"settings: {escape(json.dumps(settings)) if settings else 'none'}. Answers are sampled as "
            f"measured, {sampled}" + (f", thinking for at most {budget} tokens." if budget else ".")
        )
        console.print(shlex.join(args), markup=False, highlight=False, soft_wrap=True, style="dim")
        log = run.path("private", "logs", "serve.log")
        process = serve.start(args, log)
        try:
            console.print("A first message of Hello reaches the model as:")
            console.print(serve.sample(run, port), markup=False, highlight=False)
            tools = serve.mcp_tools(port) if mcp else []
            if mcp and not tools:
                raise serve.ServeError(f"no MCP server in {mcp} listed a tool; see {log}{WITHOUT_TOOLS}")
            console.print(
                "In the chat page's Settings, then in a new chat (a chat keeps the settings it began with):"
            )
            if system:
                console.print("  set the system message to the one measured, between the lines:")
                console.print("---", system, "---", sep="\n", markup=False, highlight=False)
            else:
                console.print("  leave the system message empty, as measured")
            if tools:
                listed = escape(", ".join(tools))
                console.print(
                    f"  under Tools, keep Server on ({listed}, from {escape(str(mcp))}; a server "
                    f"missing there failed to start, see {escape(str(log))}) and turn Browser off: the "
                    "page's own tools, such as get_info. Each call asks first: the menu beside Allow "
                    "once can always allow every tool from Server"
                )
            else:
                console.print(
                    "  turn off every tool under Tools: the page offers its own (such as the time and "
                    "get_info), and chat templates such as Llama 3's turn any tool into a request for a "
                    "function call"
                )
            console.print(
                "  keep the reasoning menu on Default: its other choices replace the template settings "
                "and the thinking budget (its Low, Medium and High are budgets of 512, 2048 and 8192 "
                "tokens, not the model's own effort levels)"
            )
            console.print(
                "  leave the sampling settings empty, so the page uses the server's: a value set there "
                "before stays in the browser and overrides them"
            )
            console.print(f"Chat at http://127.0.0.1:{port}, llama-server's own page. Ctrl-C stops it.")
            code = process.wait()
            stopped = f"llama-server stopped by itself with code {code}; its log is {log}"
            if code:
                raise serve.ServeError(stopped)
            console.print(escape(stopped))
        except KeyboardInterrupt:
            console.print("Stopped.")
        finally:
            terminate(process)


@app.command("search", short_help="Find the smallest edit that leaves almost no refusals.")
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

    report = _reporter(console)
    with (
        _stopping(console, "The search"),
        _stopped_by_signals(),
        _awake(),
        _bars(console, speed_estimate_period=DAY) as bars,
    ):
        tasks, lines = [], []

        def started(done: int, total: int) -> None:
            tasks.append(bars.add_task("Trials", total=total, completed=done))
            lines.append(_lines(console, "Trials", total, done))

        def finished(best: str | None) -> None:
            bars.advance(tasks[0])
            lines[0]()
            if best:
                console.print(escape(best))

        run = Run.open(folder, report)
        space = run.settings.search
        total = len(search.SCOPES) * space.trials_per_scope
        found, seconds = search.Search(run).search(trials or total, started, finished)
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
    cap, hits = space.max_refusals, space.max_budget_hits
    caps = f"{cap} refused" + ("" if hits is None else f", {hits} at the thinking budget")
    console.print(
        f"The least KL of each kind at each count within the cap ({caps}), unless a trial "
        "refusing fewer has less:"
    )
    picks, gaps_shown = [], False
    for scope in search.SCOPES:
        best = search.to_measure(found, scope, space)
        fewest = search.fewest_refused(found, scope)
        if fewest is None:
            console.print(f"  {scope}: no trial scored in full yet")
            continue
        aimed = search.aimed(found, scope, space)
        drawn_from = search.kind_space(space, found, scope)
        if scope in found.strength:
            low, high = drawn_from.strength
            console.print(f"  {scope}: the search widened strength to {low:g}-{high:g}")
        for candidate in best or [fewest]:
            described = (
                f"trial {candidate.trial}, {candidate.direction}, {candidate.refusals} refused, "
                f"KL {candidate.kl:.4f}"
            )
            if best:
                against = _against_least(found, candidate, best[-1])
                gaps_shown = gaps_shown or bool(against)
                console.print(f"  {scope}: {described}{against}")
            else:
                none = f"none refused {cap} or fewer" if hits is None else "none within the cap"
                console.print(f"  {scope}: {none}; the fewest refused: {described}")
            if aimed and (edges := search.at_edges(candidate.params, drawn_from, layers)):
                console.print(
                    f"    It presses on the edge of the search's ranges ({escape('; '.join(edges))}), "
                    "so a better edit may lie outside them."
                )
            picks.append(candidate.trial)
    if gaps_shown:
        console.print(
            f"A gap within {search.TIE_SE} standard errors is a tie: the test prompts can't tell the two "
            "apart by KL."
        )
    if run.settings.deployment.think_close:
        _thinking_summary(console, run, found)
    if (left := total - len(found.trials)) > 0:
        console.print(
            f"Trials took about {seconds:.0f} s each, so the {left} left of {total} would take about "
            f"{duration(seconds * left)}. Trials that refuse almost nothing take longest, and later trials "
            "aim at them, so allow more. Run candor search again to carry on."
        )
    elif picks:
        console.print("Measure each on the held-out prompts:")
        for number in picks:
            console.print(escape(f"  uv run candor measure {folder} --trial {number}"), soft_wrap=True)


def _against_least(found: search.Found, candidate: search.Candidate, least: search.Candidate) -> str:
    """How much more KL a trial has than its kind's least within the cap, paired by prompt, and whether
    that is a tie; nothing for the least itself, or where either kept no KL at each prompt."""
    if candidate.trial == least.trial:
        return ""
    paired = search.kl_gap(
        search.searched_score(found, candidate.trial), search.searched_score(found, least.trial)
    )
    if paired is None:
        return ""
    gap, error, tie = paired
    verdict = "a tie" if tie else "not a tie"
    return f"; {gap:.4f} more KL than trial {least.trial}, paired by prompt (SE {error:.4f}): {verdict}"


def _searched(score: search.Score, prompts: int) -> str:
    """A trial's score in the search: its refusals by the judge and by keywords, and its KL, or that it
    stopped early, as a trial that couldn't win does."""
    if score.judged < prompts:
        return f"{score.refusals} of {score.judged}, stopped early"
    return f"{score.refusals}, {score.keyword_refusals}, {score.kl:.4f}"


def _thinking_summary(console: Console, run: Run, found: search.Found) -> None:
    """How long the trials' answers thought, to choose the budget a search of a thinking model scores
    at."""
    scores = [search.searched_score(found, number) for number in found.trials]
    # A search scored with thinking off has no thinking to show.
    if not scores or not search.Search(run).thinks:
        return
    judged = sum(score.judged for score in scores)
    means = sorted(score.thinking for score in scores)
    budget = run.settings.scoring.thinking_budget or run.settings.deployment.thinking_budget
    console.print(
        f"Thinking: {means[0]:.0f} to {means[-1]:.0f} tokens per answer on average across trials "
        f"(median {statistics.median(means):.0f}); {sum(s.forced for s in scores)} of {judged} answers "
        f"reached the search's budget of {budget} and were made to stop."
    )


def _trials(numbers: list[int]) -> str:
    return ("trial " if len(numbers) == 1 else "trials ") + ", ".join(map(str, numbers))


@app.command("score", short_help="Score one edit as the search scores its trials.")
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

    report = _reporter(console)
    with _stopping(console, "The scoring"), _stopped_by_signals(), _awake():
        run = Run.open(folder, report)
        found = search.load(run)
        front = search.candidates(found) if found else []
        _, layers, _, writers = run.shape
        # Each edit, with its number when it is a trial of the search.
        edits: list[tuple[measure.Edit, int | None]] = []
        if heretic is None:
            edits.append((search.recorded_edit(run, trial), trial))
        else:
            edit = _heretic_trial(run, heretic, trial)
            nearest = search.nearest_layer(edit, run.settings.search, layers)
            edits += [(edit, None)] + ([(nearest, None)] if nearest else [])
            if front:
                edits.append((search.recorded_edit(run, front[0].trial), front[0].trial))
        # A trial the search scored in full, scored again, shows whether scoring repeats, so the
        # unedited model is scored again only where there is none.
        repeated = found is not None and any(
            number and not found.trials[number].user_attrs["stopped"] for _, number in edits
        )
        scored = search.Search(run).score([edit for edit, _ in edits], base=not repeated)
        searched = {number: search.searched_score(found, number) for _, number in edits if found and number}
    table = Table(box=box.SIMPLE_HEAD)
    for heading in ["Edit", "Direction", "Refused", "Keywords", "KL", "In the search"]:
        table.add_column(heading, justify="left" if heading in {"Edit", "Direction"} else "right")
    for (edit, number), score in zip(edits, scored.edits, strict=True):
        before = searched.get(number) if number else None
        table.add_row(
            escape(edit.name),
            search.direction(edit.direction_layer, edit.answering),
            str(score.refusals),
            str(score.keyword_refusals),
            f"{score.kl:.4f}",
            _searched(before, scored.prompts) if before else "",
        )
    console.print(f"Scored as the search scores its trials, on its {scored.prompts} test prompts:")
    console.print(table)
    base = found.base_refusals if found else None
    if scored.base_refusals is None:
        console.print(f"The unedited model refuses {base} of {scored.prompts} in the search.")
    else:
        console.print(
            f"The unedited model refuses {scored.base_refusals} of {scored.prompts}"
            + (f" ({base} in the search)." if base is not None else ".")
        )
    for (edit, number), score in zip(edits, scored.edits, strict=True):
        if number:
            if searched[number].judged < scored.prompts:
                console.print(
                    f"{escape(edit.name)} was stopped early in the search, after {searched[number].judged} "
                    "prompts, so there's no full score of it to repeat."
                )
            elif differs := search.differences(score, searched[number]):
                console.print(
                    f"{escape(edit.name)} scores differently from the search, by more than noise: "
                    f"{'; '.join(differs)}. The scores above can't be set beside the search's, so the "
                    "verdicts below don't hold."
                )
            else:
                paired = search.kl_gap(score, searched[number])
                gap = f" (KL {paired[0]:+.4f} paired by prompt, SE {paired[1]:.4f})" if paired else ""
                console.print(
                    f"{escape(edit.name)} scores as it did in the search, within noise{gap}, so the scores "
                    "above can be set beside the search's."
                )
    if not front:
        what = "trial scored in full" if found and found.trials else "trials"
        console.print(f"The search has no {what} yet to set these beside.")
        return
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
        console.print(verdict)
        scope = search.PER_LAYER if edit.direction_layer is None else search.ONE_LAYER
        drawn_from = search.kind_space(run.settings.search, found, scope)
        if outside := search.outside_space(edit, drawn_from, layers, components):
            console.print(f"  The search couldn't have drawn it: {escape('; '.join(outside))}.")
        else:
            console.print("  The search could have drawn it.")


@app.command("proxy", short_help="Check if a thinking model may be searched with thinking off.")
def proxy_command(
    folder: Annotated[Path, typer.Argument(help="A run folder made by candor init.")],
    trial: Annotated[
        list[int] | None,
        typer.Option(
            help="A trial of candor search to check, by the number shown; repeat it for more. Without it, "
            "the best of each kind and place are checked.",
        ),
    ] = None,
) -> None:
    """Check whether a thinking model's search may score its trials with thinking off, many times
    faster. The best edits a short search scored with thinking off found, of each kind and place, or
    the trials named, are scored on the test prompts with thinking off and thinking as the model is
    used, also at more strength, and the two orders compared (for trials named, the counts only,
    without the verdict); the edit refusing fewest thinking is scored again with another seed, to show
    how far sampling alone moves a score. A stopped check carries on. Shows counts only."""
    console = Console()
    folder = folder.expanduser()

    report = _reporter(console)
    with _stopping(console, "The check"), _stopped_by_signals(), _awake():
        run = Run.open(folder, report)
        checked = proxy.check(run, trial or ())
    prompts = checked.off[0].judged
    table = Table(box=box.SIMPLE_HEAD)
    headings = [
        "Edit",
        "Trial",
        "Direction",
        "Strength",
        "Refused off",
        "Refused on",
        "Thinking",
        "Forced",
        "KL",
    ]
    for heading in headings:
        table.add_column(heading, justify="left" if heading in {"Edit", "Direction"} else "right")
    rows = zip(checked.edits, checked.trials, checked.factors, checked.off, checked.on, strict=True)
    for edit, number, factor, off, on in rows:
        table.add_row(
            edit.name,
            str(number),
            search.direction(edit.direction_layer, edit.answering),
            _multiple(*factor),
            str(off.refusals),
            str(on.refusals),
            f"{on.thinking:.0f}",
            str(on.forced),
            f"{on.kl:.4f}",
        )
    console.print(f"Refusals of the {prompts} test prompts, thinking off and on; thinking in mean tokens:")
    console.print(table)
    base = checked.base
    console.print(f"The unedited model refuses {base['off']} with thinking off and {base['on']} with it on.")
    more = sum(on.refusals > off.refusals for off, on in zip(checked.off, checked.on, strict=True))
    console.print(f"Thinking refuses more on {more} of {len(checked.edits)} edits.")
    first, again = checked.on[checked.repeated], checked.repeat
    console.print(
        f"{checked.edits[checked.repeated].name} refused {first.refusals} thinking, and {again.refusals} "
        "with another seed."
    )
    space = run.settings.search
    cap = str(space.max_refusals)
    if space.max_budget_hits is not None:
        cap += f" refused, {space.max_budget_hits} at the thinking budget,"
    within = {
        mode: [e.name for e, s in zip(checked.edits, scores, strict=True) if search.counts(s, space)]
        for mode, scores in (("off", checked.off), ("on", checked.on))
    }
    for mode, names in within.items():
        console.print(f"Within the cap of {cap} with thinking {mode}: {', '.join(names) or 'none'}.")
    if not trial:
        agreement = checked.agreement
        shown = f"{agreement:.2f}" if agreement == agreement else "none: one way scores every edit alike"
        console.print(f"Rank agreement (Spearman) between the two: {shown}.")
    near = f"{proxy.NEARLY_WORKS:.0%} of the prompts"
    if checked.offset is None:
        console.print(f"No edit refused at most {near} with thinking off, so none came near working.")
    else:
        console.print(
            f"Where thinking off refuses at most {near}, thinking refuses a median of "
            f"{checked.offset:g} more."
        )
    if trial:
        console.print(
            "The trials were named, so whether the search may score thinking off isn't judged here: "
            "candor proxy without --trial judges it on the best of each kind and place."
        )
        return
    verdict = checked.verdict()
    if verdict == "holds":
        console.print(
            "[green]The cheap score ranks edits as thinking does[/]: "
            'scoring.thinking = "off" may be set for the search.'
        )
    elif verdict == "offset":
        console.print(
            "[yellow]The cheap score ranks edits as thinking does, but thinking needs stronger ones[/]: "
            'with scoring.thinking = "off", the edits the search suggests must also be scored thinking '
            "at more strength."
        )
    else:
        console.print(
            "[yellow]The cheap score can't be trusted to rank edits as thinking does[/]: "
            'keep scoring.thinking = "on".'
        )
