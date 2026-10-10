"""The candor command line."""

import json
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer
from huggingface_hub import HfApi
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

from . import checks, config, source, targets
from .engine import LLAMA_CPP_RELEASE

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

Row = tuple[str, str, str]


class CannotInspect(Exception):
    pass


@app.callback()
def main() -> None:
    pass


def _examples(names: list[str], limit: int = 3) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


def _first(config: dict, *keys: str) -> int | None:
    return next((config[key] for key in keys if isinstance(config.get(key), int)), None)


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
        rows.append(("Extra layers", "ok", f"layers {extra} sit beyond the {layers} the model runs, such as "
                     "a token-prediction head, and are left alone"))  # fmt: skip
    if found.experts:
        rows.append(("Experts", "fail", "candor doesn't edit mixture-of-experts models yet"))
    if found.quantized:
        rows.append(("Quantised", "fail", f"{_examples(found.quantized)} carry quantisation scales; "
                     "use the unquantised original"))  # fmt: skip
    if found.output_norms:
        rows.append(("Output norms", "warn", "this model normalises each block's output before adding it "
                     "to the residual stream, so the edit can't remove the direction exactly"))  # fmt: skip
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
        rows.append(("Date in prompt", "warn", "the chat template writes today's date into every prompt; "
                     "a run can pin it"))  # fmt: skip
    elif checks.writes_the_date(template):
        rows.append(("Date in prompt", "warn", "the chat template writes today's date into every prompt and "
                     "can't be given a fixed one, so finish each run on the day it starts"))  # fmt: skip
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
    text = source.text_config(config)
    hidden = _first(text, "hidden_size", "n_embd", "d_model") or _first(
        config, "hidden_size", "n_embd", "d_model"
    )
    layers = _first(text, "num_hidden_layers", "n_layer") or _first(config, "num_hidden_layers", "n_layer")
    if hidden is None or layers is None:
        raise CannotInspect(f"candor couldn't find the hidden size and layer count in {model}'s config")

    arch = targets.gguf_architecture(config)
    names = ", ".join(config.get("architectures") or [config.get("model_type", "unknown")])
    converts = f"llama.cpp calls it {arch}" if arch else "the pinned llama.cpp can't convert it"
    summary = f"{names}, {layers} layers, hidden size {hidden}; {converts}"
    rows: list[Row] = [("Architecture", "ok" if arch else "fail", summary)]
    code = checks.remote_code(config, source.read_json(model, "tokenizer_config.json"))
    if code:
        rows.append(("Remote code", "warn", f"loading runs {_examples(code)}, so read it first"))
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
        console.print(f"[red]{error}[/]")
    except Exception as error:
        console.print(
            f"[red]candor hit an unexpected error inspecting {model}:[/] {type(error).__name__}: {error}"
        )
    raise typer.Exit(COULD_NOT_INSPECT)


def _print_table(console: Console, title: str, rows: list[Row]) -> None:
    table = Table(title=title, show_lines=True)
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Detail", overflow="fold")
    for check, status, detail in rows:
        table.add_row(check, MARKS[status], detail)
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
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    date_format = checks.pinnable_date_format(checks.chat_template(tokenizer) or "")
    return config.today_in(date_format) if date_format else None


def _revision(model: str) -> str | None:
    """The Hub commit the model is at now, so later steps use exactly what was checked."""
    return None if Path(model).is_dir() else HfApi().model_info(model).sha


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
        console.print(
            f"[red]{runs / name} already exists.[/] Pick another name, or delete that folder first."
        )
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
        model_settings = config.Model(source=model, revision=revision, date=_pinned_date(model, revision))
        header = (
            f"Settings for the candor run {name!r}, created {date.today()} by candor {version('candor')} "
            f"for llama.cpp {LLAMA_CPP_RELEASE}.\n"
            "Edit freely: every later command reads and checks this file."
        )
        record = [dict(zip(("check", "result", "detail"), row, strict=True)) for row in rows]
        path = config.write(
            runs,
            name,
            config.RunConfig(model=model_settings),
            header,
            {"inspect.json": json.dumps(record, indent=1)},
        )
    except FileExistsError as error:
        console.print(f"[red]{error}.[/] Pick another name, or delete that folder first.")
        raise typer.Exit(FAILED_CHECK) from None
    except Exception as error:
        console.print(f"[red]candor couldn't create the run:[/] {type(error).__name__}: {error}")
        raise typer.Exit(COULD_NOT_INSPECT) from None
    console.print(f"[green]Created {path}[/]")
    if any(check in NOT_BLOCKING and status == "fail" for check, status, _ in rows):
        console.print(
            "[yellow]Bionic can't read this model's chat template.[/] candor still works, but test the "
            "result in Bionic only after pasting in a fixed template."
        )
