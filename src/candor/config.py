"""A run's settings: one TOML file per run, written by `candor init` and read by every later command.

Every setting a run uses lives here, search ranges included, so nothing is hidden in the code. The
file must spell out every setting: a missing one is an error, not a silent default, so a later
candor release can't change an existing run. Each field's description is also the comment
written above it in the file.
"""

import os
import re
import shutil
import tomllib
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

FILE = "candor.toml"
QUANTIZATIONS = {"Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "F16", "BF16"}
KV_CACHE_TYPES = {"f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"}
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Range = tuple[Finite, Finite]
Text = Annotated[str, Field(strict=True, min_length=1)]


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Model(Section):
    source: Text = Field(description="The model to edit: a Hugging Face id or a local folder.")
    revision: Text | None = Field(
        default=None,
        description="The Hugging Face commit the model was checked at, so later steps use exactly those "
        "weights and that chat template. Not set for a local folder.",
    )
    date: Text | None = Field(
        default=None,
        description="The date written into every prompt, pinned so a run's prompts don't change from day "
        "to day. Only set when the chat template writes the date and can be given a fixed one.",
    )


class Deployment(Section):
    quantization: str = Field(default="Q4_K_M", description="The GGUF type you will load in Bionic.")
    kv_cache: str = Field(
        default="f16", description="The KV cache type you run with (llama.cpp's --cache-type-k and -v)."
    )
    system_prompt: Annotated[str, Field(strict=True)] = Field(
        default="",
        description="The system prompt you use. Bionic sends none by default; heretic used "
        '"You are a helpful assistant."',
    )

    @field_validator("quantization")
    @classmethod
    def _known_quantization(cls, value: str) -> str:
        if value not in QUANTIZATIONS:
            raise ValueError(f"must be one of {', '.join(sorted(QUANTIZATIONS))}")
        return value

    @field_validator("kv_cache")
    @classmethod
    def _known_cache(cls, value: str) -> str:
        if value not in KV_CACHE_TYPES:
            raise ValueError(f"must be one of {', '.join(sorted(KV_CACHE_TYPES))}")
        return value


class PromptSet(Section):
    dataset: Text
    split: Text
    column: Text = "text"


class Prompts(Section):
    harmful: PromptSet = Field(
        default=PromptSet(dataset="mlabonne/harmful_behaviors", split="train[:400]"),
        description="Harmful prompts the refusal direction is read from.",
    )
    harmless: PromptSet = Field(
        default=PromptSet(dataset="mlabonne/harmless_alpaca", split="train[:400]"),
        description="Harmless prompts the direction is measured against.",
    )
    harmful_test: PromptSet = Field(
        default=PromptSet(dataset="mlabonne/harmful_behaviors", split="test[:100]"),
        description="Harmful prompts each trial's refusals are counted on.",
    )
    harmless_test: PromptSet = Field(
        default=PromptSet(dataset="mlabonne/harmless_alpaca", split="test[:100]"),
        description="Harmless prompts each trial's KL divergence is measured on.",
    )


class Edit(Section):
    row_norm: str = Field(
        default="pre",
        description='How the edit treats each weight row\'s length: "none" removes the direction as is; '
        '"pre" computes the edit on rows of length 1 and scales them back, as heretic does on the Mac.',
    )

    @field_validator("row_norm")
    @classmethod
    def _known_row_norm(cls, value: str) -> str:
        if value not in {"none", "pre"}:
            raise ValueError('must be "none" or "pre"')
        return value


class Scoring(Section):
    answer_tokens: StrictInt = Field(
        default=100, ge=1, le=8192, description="How long an answer to each harmful test prompt may be."
    )
    kl_tokens: StrictInt = Field(
        default=20,
        ge=1,
        le=1000,
        description="How many of the most likely first tokens KL divergence is measured over (20 is "
        "within 0.5% of the full vocabulary).",
    )


class Search(Section):
    trials: StrictInt = Field(default=200, ge=1, description="How many edits to try.")
    startup_trials: StrictInt = Field(
        default=60, ge=1, description="Random trials before the search starts aiming."
    )
    seed: StrictInt = Field(
        default=0, ge=0, lt=2**32, description="Fixes the search's randomness, so a run can be repeated."
    )
    strength: Range = Field(
        default=(0.8, 3.0),
        description="Edit strength at the peak layer. 1 removes the direction; above 1 pushes past it, "
        "which stubborn and thinking models need.",
    )
    peak_layer: Range = Field(
        default=(0.6, 1.0), description="Where the edit is strongest, as a fraction of the model's depth."
    )
    edge_strength: Range = Field(
        default=(0.0, 1.0), description="Strength at the edges of the edited band, as a fraction of the peak."
    )
    reach: Range = Field(
        default=(0.01, 0.6),
        description="How far either side of the peak the edit reaches, as a fraction of depth.",
    )
    direction_layer: Range = Field(
        default=(0.4, 0.9),
        description="Which layer's refusal direction a single-direction trial uses, as a fraction of depth.",
    )

    @field_validator("strength", "peak_layer", "edge_strength", "reach", "direction_layer")
    @classmethod
    def _ordered(cls, value: tuple[float, float]) -> tuple[float, float]:
        if value[0] > value[1]:
            raise ValueError("the first number must not be larger than the second")
        return value

    @field_validator("peak_layer", "edge_strength", "reach", "direction_layer")
    @classmethod
    def _fraction(cls, value: tuple[float, float]) -> tuple[float, float]:
        if not 0 <= value[0] <= value[1] <= 1:
            raise ValueError("must be fractions between 0 and 1")
        return value

    @field_validator("strength")
    @classmethod
    def _positive(cls, value: tuple[float, float]) -> tuple[float, float]:
        if value[0] <= 0:
            raise ValueError("must be above 0")
        return value

    @model_validator(mode="after")
    def _startup_fits(self) -> "Search":
        if self.startup_trials > self.trials:
            raise ValueError("startup_trials can't be more than trials")
        return self


class RunConfig(Section):
    model: Model = Field(description="The model this run edits.")
    deployment: Deployment = Field(
        default_factory=Deployment,
        description="How you actually run the model, so trials are scored that way.",
    )
    prompts: Prompts = Field(default_factory=Prompts, description="Where the prompts come from.")
    edit: Edit = Field(default_factory=Edit, description="The weight edit itself.")
    scoring: Scoring = Field(default_factory=Scoring, description="How each candidate edit is measured.")
    search: Search = Field(default_factory=Search, description="The search over edits.")


class ConfigError(Exception):
    pass


def _optional(field: Any) -> bool:
    return field.default is None


def _missing(settings: BaseModel, prefix: str = "") -> list[str]:
    """Settings the file didn't spell out. Only those whose absence means "not set" may be left out."""
    found = []
    for name, field in type(settings).model_fields.items():
        if name not in settings.model_fields_set and not _optional(field):
            found.append(f"{prefix}{name}")
        elif isinstance(value := getattr(settings, name), Section):
            found += _missing(value, f"{prefix}{name}.")
    return found


def parse(text: str, where: str) -> RunConfig:
    try:
        settings = RunConfig.model_validate(tomllib.loads(text))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{where} isn't valid TOML: {error}") from None
    except ValidationError as error:
        problems = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()]
        raise ConfigError(f"{where} has settings candor can't use:\n  " + "\n  ".join(problems)) from None
    if missing := _missing(settings):
        raise ConfigError(f"{where} is missing settings, so candor won't guess them: {', '.join(missing)}")
    return settings


def load(folder: Path) -> RunConfig:
    path = folder / FILE
    try:
        text = path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        raise ConfigError(f"{folder} has no {FILE}; create one with candor init") from None
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(f"candor couldn't read {path}: {error}") from None
    return parse(text, str(path))


def _string(value: str) -> str:
    escapes = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}
    out = []
    for char in value:
        if char in escapes:
            out.append(escapes[char])
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            out.append(f"\\u{ord(char):04X}")
        else:
            out.append(char)
    return '"' + "".join(out) + '"'


def _value(value: Any) -> str:
    if isinstance(value, BaseModel):
        return "{ " + ", ".join(f"{k} = {_value(v)}" for k, v in value.model_dump().items()) + " }"
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple | list):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    return repr(value)


def _comment(text: str | None) -> list[str]:
    return [f"# {line}" for line in (text or "").splitlines()]


def render(config: RunConfig, header: str) -> str:
    """The settings as TOML, with each field's description as a comment above it."""
    lines = _comment(header)
    for name, field in RunConfig.model_fields.items():
        section = getattr(config, name)
        lines += ["", *_comment(field.description), f"[{name}]"]
        for key, inner in type(section).model_fields.items():
            value = getattr(section, key)
            setting = f"# {key} =" if value is None else f"{key} = {_value(value)}"
            lines += [*_comment(inner.description), setting]
    return "\n".join(lines) + "\n"


def check_run_name(name: str) -> None:
    if not RUN_NAME.fullmatch(name) or name in {".", ".."}:
        raise ConfigError(
            f"{name!r} can't be a run name: use one plain folder name of letters, digits, '.', '_' or '-'"
        )


def write(
    runs: Path, name: str, config: RunConfig, header: str, extras: dict[str, str] | None = None
) -> Path:
    """Create runs/name holding the settings file and any extra text files. Nothing is left behind
    if any step fails."""
    check_run_name(name)
    text = render(config, header)
    if parse(text, "the new settings") != config:
        raise ConfigError("the new settings don't read back as written")
    folder = runs / name
    if folder.exists():
        raise FileExistsError(f"{folder} already exists")
    runs.mkdir(parents=True, exist_ok=True)
    partial = runs / f".{name}.partial-{os.getpid()}"
    shutil.rmtree(partial, ignore_errors=True)
    try:
        partial.mkdir()
        (partial / FILE).write_text(text, encoding="utf-8")
        for file, content in (extras or {}).items():
            (partial / file).write_text(content, encoding="utf-8")
        partial.rename(folder)
    finally:
        shutil.rmtree(partial, ignore_errors=True)
    return folder / FILE


def today_in(format: str) -> str:
    return date.today().strftime(format)
