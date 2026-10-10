"""The candor command line."""

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

from . import checks, source, targets

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


@app.callback()
def main() -> None:
    pass


def _examples(names: list[str], limit: int = 3) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


def _targets_row(found: targets.Targets, arch: str | None) -> list[tuple[str, str, str]]:
    if not found.ok:
        problems = []
        if found.unclassified:
            problems.append(
                f"can't tell whether these write to the residual: {_examples(found.unclassified)}"
            )
        if found.missing:
            problems.append(_examples(found.missing))
        return [("Edit targets", "fail", "; ".join(problems))]
    parts = []
    for component, kinds in found.summary().items():
        parts.append(f"{component}: " + ", ".join(f"{count} {kind}" for kind, count in kinds.items()))
    rows = [("Edit targets", "ok", f"{len(found.writers)} writers, every layer covered ({'; '.join(parts)})")]
    if arch:
        unmapped = [w.name for w in found.writers if targets.gguf_name(w, arch) is None]
        if unmapped:
            rows.append(("GGUF names", "fail", f"no llama.cpp name for {_examples(unmapped)}"))
        else:
            first = found.writers[0]
            rows.append(
                ("GGUF names", "ok", f"all map, e.g. {first.name} -> {targets.gguf_name(first, arch)}")
            )
    return rows


@app.command()
def inspect(model: Annotated[str, typer.Argument(help="A Hugging Face model id or a local folder.")]) -> None:
    """Check a model before spending anything on it. Reads the config, tokenizer and file headers only."""
    config = source.read_config(model)
    text = source.text_config(config)
    hidden, layers = text["hidden_size"], text["num_hidden_layers"]
    model_type = text.get("model_type", config["model_type"])
    arch = targets.gguf_architecture(model_type) or targets.gguf_architecture(config["model_type"])
    rows = [
        (
            "Architecture",
            "ok" if arch else "fail",
            f"{model_type}, {layers} layers, hidden size {hidden}; "
            + (f"llama.cpp calls it {arch}" if arch else "the pinned llama.cpp doesn't support it"),
        )
    ]

    code = checks.remote_code(config)
    rows.append(
        ("Remote code", "warn", f"loading runs {_examples(code)}, so read it first")
        if code
        else ("Remote code", "ok", "none")
    )

    rows += _targets_row(targets.find_targets(source.tensor_shapes(model), hidden, layers), arch)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    except Exception as error:
        rows.append(("Tokenizer", "fail", f"couldn't load without running remote code: {error}"))
    else:
        found = checks.begin_of_text(tokenizer)
        counts = f"apps send {found.transformers}, llama.cpp {found.llama_cpp}, heretic {found.heretic}"
        rows.append(
            ("Begin-of-text", "warn" if found.problems() else "ok", "; ".join([counts, *found.problems()]))
        )
        if checks.writes_the_date(tokenizer):
            rows.append(("Date in prompt", "warn", "the chat template writes today's date, so runs pin it"))
        else:
            rows.append(("Date in prompt", "ok", "none"))
        template = tokenizer.chat_template
        if isinstance(template, str):
            result = checks.app_template_parses(template)
            status = {"ok": "ok", "fails": "fail", "skipped": "skipped"}[result.status]
            detail = result.detail or "parses in Bionic and LM Studio"
            if result.status == "fails":
                detail = f"Bionic and LM Studio can't parse it and will use another format: {detail}"
            rows.append(("Bionic template", status, detail))
        else:
            rows.append(("Bionic template", "fail", "the tokenizer has no single chat template"))

    table = Table(title=f"candor inspect {model}", show_lines=True)
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Detail", overflow="fold")
    for check, status, detail in rows:
        table.add_row(check, MARKS[status], detail)
    Console().print(table)
    if any(status == "fail" for _, status, _ in rows):
        raise typer.Exit(1)
