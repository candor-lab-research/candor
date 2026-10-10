"""A run's settings: one TOML file per run, written by `candor init` and read by every later command.

Every setting a run uses lives here, search ranges included, so nothing is hidden in the code.
Each field's description is also the comment written above it in the file.
"""

import json
import tomllib
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

FILE = "candor.toml"
QUANTIZATIONS = {"Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "F16", "BF16"}
KV_CACHE_TYPES = {"f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"}


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Model(Section):
    source: str = Field(description="The model to edit: a Hugging Face id or a local folder.")
    date: str | None = Field(
        default=None,
        description="The date written into every prompt, pinned so a run's prompts don't change "
        "from day to day. Only set when the chat template writes the date.",
    )


class Deployment(Section):
    quantization: str = Field(default="Q4_K_M", description="The GGUF type you will load in Bionic.")
    kv_cache: str = Field(
        default="f16", description="The KV cache type you run with (llama.cpp's --cache-type-k and -v)."
    )
    system_prompt: str = Field(
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
    dataset: str
    split: str
    column: str = "text"


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


Range = tuple[float, float]


class Search(Section):
    trials: int = Field(default=200, ge=1, description="How many edits to try.")
    startup_trials: int = Field(
        default=60, ge=1, description="Random trials before the search starts aiming."
    )
    seed: int = Field(default=0, description="Fixes the search's randomness, so a run can be repeated.")
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
    def _ordered(cls, value: Range) -> Range:
        low, high = value
        if low > high:
            raise ValueError("the first number must not be larger than the second")
        return value

    @field_validator("peak_layer", "edge_strength", "reach", "direction_layer")
    @classmethod
    def _fraction(cls, value: Range) -> Range:
        if not 0 <= value[0] <= value[1] <= 1:
            raise ValueError("must be fractions between 0 and 1")
        return value

    @field_validator("strength")
    @classmethod
    def _positive(cls, value: Range) -> Range:
        if value[0] <= 0:
            raise ValueError("must be above 0")
        return value

    @model_validator(mode="after")
    def _startup_fits(self) -> "Search":
        if self.startup_trials > self.trials:
            raise ValueError("startup_trials can't be more than trials")
        return self


class RunConfig(Section):
    model: Model
    deployment: Deployment = Field(
        default_factory=Deployment,
        description="How you actually run the model, so trials are scored that way.",
    )
    prompts: Prompts = Field(default_factory=Prompts, description="Where the prompts come from.")
    search: Search = Field(default_factory=Search, description="The search over edits.")


class ConfigError(Exception):
    pass


def load(folder: Path) -> RunConfig:
    path = folder / FILE
    try:
        return RunConfig.model_validate(tomllib.loads(path.read_text()))
    except FileNotFoundError:
        raise ConfigError(f"{folder} has no {FILE}; create one with candor init") from None
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path} isn't valid TOML: {error}") from None
    except ValidationError as error:
        problems = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()]
        raise ConfigError(f"{path} has settings candor can't use:\n  " + "\n  ".join(problems)) from None


def _value(value: Any) -> str:
    if isinstance(value, BaseModel):
        return "{ " + ", ".join(f"{k} = {_value(v)}" for k, v in value.model_dump().items()) + " }"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple | list):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    return repr(value)


def _comment(text: str | None) -> list[str]:
    return [f"# {line}" for line in (text or "").splitlines()] if text else []


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


def write(folder: Path, config: RunConfig, header: str) -> Path:
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / FILE
    path.write_text(render(config, header))
    if load(folder) != config:
        raise ConfigError(f"{path} doesn't read back as written")
    return path


def today_in(format: str) -> str:
    return date.today().strftime(format)
