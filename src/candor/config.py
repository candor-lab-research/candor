"""A run's settings: one TOML file per run, written by `candor init` and read by every later command.

Every setting a run uses lives here, search ranges included, so nothing is hidden in the code. The
file must spell out every setting: a missing one is an error, not a silent default, so a later
candor release can't change an existing run. Each field's description is also the comment
written above it in the file.
"""

import json
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
    StrictBool,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

FILE = "candor.toml"
# Each GGUF type a run can deploy, with its bits per weight.
QUANTIZATIONS = {
    "Q4_K_S": 4.6,
    "Q4_K_M": 4.9,
    "Q5_K_S": 5.5,
    "Q5_K_M": 5.7,
    "Q6_K": 6.6,
    "Q8_0": 8.5,
    "F16": 16,
    "BF16": 16,
}
KV_CACHE_TYPES = {"f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"}
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")

Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Range = tuple[Finite, Finite]
Text = Annotated[str, Field(strict=True, min_length=1)]


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Model(Section):
    source: Text = Field(description="The model to edit: a Hugging Face id, or a local folder's full path.")
    revision: Text | None = Field(
        default=None,
        description="The Hugging Face commit the model was checked at, so later steps use exactly those "
        "weights and that chat template. Not set for a local folder.",
    )
    trust_remote_code: StrictBool = Field(
        default=False,
        description="Whether transformers may load the model with the Python files it ships, which runs "
        "the model author's code on this machine, at the pinned revision. init turns it on only when "
        "transformers has no code of its own for the model; read those files first.",
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
    # The dataset's configuration, for datasets that have several, such as GSM8K's "main".
    subset: Text | None = None
    split: Text
    column: Text = "text"
    # The dataset's Hugging Face commit, pinned by init so a rebuild reads the same prompts.
    revision: Text | None = None
    # Rows to leave out, by the values of other columns, such as { source = ["AdvBench"] }.
    exclude: dict[Text, list[Text]] | None = None


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
    harmful_held_out: PromptSet = Field(
        default=PromptSet(
            dataset="walledai/StrongREJECT", split="train", column="prompt", exclude={"source": ["AdvBench"]}
        ),
        description="Harmful prompts the search never sees, for comparing finished edits. Prompts also "
        "in another set are dropped, and so are StrongREJECT's prompts taken from AdvBench, which the "
        "other harmful sets come from.",
    )
    harmless_held_out: PromptSet = Field(
        default=PromptSet(dataset="mlabonne/harmless_alpaca", split="test[100:400]"),
        description="Harmless prompts the search never sees, for comparing finished edits' KL "
        "divergence and checking the judge on answers that aren't refusals. Prompts also in another "
        "set are dropped.",
    )
    maths: PromptSet = Field(
        default=PromptSet(dataset="openai/gsm8k", subset="main", split="test", column="question"),
        description="Maths questions, for checking that a finished edit keeps the model's reasoning. "
        "Each needs an answer column ending in '#### ' and the number, as GSM8K's does.",
    )


class Edit(Section):
    row_norm: str = Field(
        default="pre",
        description='How the edit treats each weight row\'s length: "none" edits the weights as they are; '
        '"pre" computes the edit on rows scaled to length 1, then scales them back, as heretic does on '
        "the Mac.",
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


class Measure(Section):
    answer_tokens: StrictInt = Field(
        default=256,
        ge=1,
        le=8192,
        description="How long an answer to each held-out prompt may be, for the judge to read.",
    )
    maths_answer_tokens: StrictInt = Field(
        default=512,
        ge=1,
        le=8192,
        description="How long an answer to each maths question may be. One cut off before it states "
        "its answer counts as wrong.",
    )
    maths_instruction: Text = Field(
        default="Solve this step by step. End with the final answer on its own line, as: Answer: <number>",
        description="Added below each maths question, so every answer ends with a number candor can check.",
    )
    judge: Text = Field(
        default="Qwen/Qwen3Guard-Gen-0.6B",
        description="The Qwen3Guard-Gen model on Hugging Face that decides whether each answer is a "
        "refusal. Keyword counts are kept beside its verdicts.",
    )
    judge_revision: Text | None = Field(
        default=None,
        description="The judge's Hugging Face commit, pinned by init so its verdicts can't change.",
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
        default=(0.8, 1.5),
        description='Edit strength at the peak layer. With row_norm "none", 1 removes the direction from '
        'what each edited weight writes, and above 1 pushes past it. With "pre", 1 removes it from what '
        "the weights write with every row scaled to length 1; scaling the rows back leaves a part no "
        "strength removes, larger the more the rows' lengths differ.",
    )
    peak_layer: Range = Field(
        default=(0.6, 1.0), description="Where the edit is strongest, as a fraction of the model's depth."
    )
    edge_strength: Range = Field(
        default=(0.0, 1.0), description="Strength at the edges of the edited band, as a fraction of the peak."
    )
    reach: Range = Field(
        default=(0.01, 0.6),
        description="How far either side of the peak the edit reaches, as a fraction of depth. The search "
        "never goes below one layer, as in heretic, so every edit changes some layer.",
    )
    direction_layer: Range = Field(
        default=(0.4, 0.9),
        description="Which layer's refusal direction a single-direction trial uses, as a fraction of depth.",
    )

    kl_target: Finite = Field(
        default=0.01,
        ge=0,
        description="Below this KL divergence an edit is scored by its refusals instead, so the search "
        "doesn't dwell on edits that change almost nothing (heretic's default).",
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

    def direction_layers(self, layers: int) -> tuple[int, int]:
        """The first and last whole layer a single-direction trial reads its direction from, in a model
        with this many layers."""
        last = layers - 1
        low, high = self.direction_layer
        return round(low * last), round(high * last)

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
    measure: Measure = Field(
        default_factory=Measure,
        description="How finished edits are measured, on prompts the search never sees.",
    )
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


def _validated(text: str, where: str) -> RunConfig:
    """The settings, with any the text leaves out at their defaults."""
    try:
        return RunConfig.model_validate(tomllib.loads(text))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{where} isn't valid TOML: {error}") from None
    except ValidationError as error:
        problems = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()]
        raise ConfigError(f"{where} has settings candor can't use:\n  " + "\n  ".join(problems)) from None


def parse(text: str, where: str) -> RunConfig:
    settings = _validated(text, where)
    if missing := _missing(settings):
        raise ConfigError(
            f"{where} is missing settings, so candor won't guess them: {', '.join(missing)}. If this "
            "version of candor added them, candor upgrade adds them at their defaults."
        )
    return settings


def _read(folder: Path) -> tuple[Path, str]:
    path = folder / FILE
    try:
        return path, path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        raise ConfigError(f"{folder} has no {FILE}; create one with candor init") from None
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(f"candor couldn't read {path}: {error}") from None


def load(folder: Path) -> RunConfig:
    path, text = _read(folder)
    return parse(text, str(path))


def load_for_upgrade(folder: Path) -> tuple[RunConfig, list[str], str]:
    """The settings with those the file leaves out at their defaults, the names of those, and the
    file's header."""
    path, text = _read(folder)
    settings = _validated(text, str(path))
    header = []
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        header.append(line.removeprefix("#").removeprefix(" "))
    return settings, _missing(settings), "\n".join(header)


def rewrite(folder: Path, settings: RunConfig, header: str) -> None:
    """Replace the run's settings file in one step, so it is never left half written."""
    text = render(settings, header)
    if parse(text, "the new settings") != settings:
        raise ConfigError("the new settings don't read back as written")
    partial = folder / f".{FILE}.partial-{os.getpid()}"
    try:
        partial.write_text(text, encoding="utf-8")
        partial.replace(folder / FILE)
    finally:
        partial.unlink(missing_ok=True)


def _key(key: str) -> str:
    return key if BARE_KEY.fullmatch(key) else _value(key)


def _value(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, dict):
        pairs = [f"{_key(k)} = {_value(v)}" for k, v in value.items() if v is not None]
        return "{ " + ", ".join(pairs) + " }"
    if isinstance(value, str):
        # JSON's string escapes are all valid in TOML, which also forbids a raw DEL character.
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple | list):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    return repr(value)


def setting(settings: RunConfig, name: str) -> str:
    """One setting, named by its dotted path, as a line of TOML."""
    value: Any = settings
    for part in name.split("."):
        value = getattr(value, part)
    return f"{name} = {_value(value)}"


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
