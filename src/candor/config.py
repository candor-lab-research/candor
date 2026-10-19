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
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

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

from .files import write_atomic

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
# Each KV cache type llama-server takes, with its bits per value.
KV_CACHE_TYPES = {
    "f32": 32,
    "f16": 16,
    "bf16": 16,
    "q8_0": 8.5,
    "q5_1": 6,
    "q5_0": 5.5,
    "q4_1": 5,
    "q4_0": 4.5,
    "iq4_nl": 4.5,
}
# The thinking budget init gives a model that thinks: the longest llama-server's chat page offers.
THINKING_BUDGET = 8192
# How many first tokens init has KL read over, and the second judge it gives, for a new run.
KL_TOKENS = 200
SECOND_JUDGE = "Qwen/Qwen3Guard-Gen-8B"
# How many answers llama-server works on at once unless [slots] says otherwise: what runs used before
# slots were settings, so a candor.toml without them reads as it did.
THINKING_SLOTS = 8
SEARCH_SLOTS = 8
MEASURE_SLOTS = 8
# A thinking model's answers are long, and the server makes more tokens a second the more it answers
# together, so init gives a thinking model this many for every stage, and the search takes it unset.
THINKING_MODEL_SLOTS = 32
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
END_TAG = re.compile(r"\s*</[^<>\s]+>\s*")

Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Range = tuple[Finite, Finite]
Text = Annotated[str, Field(strict=True, min_length=1)]
# A value a chat template is rendered with: one that TOML, JSON and llama-server all hold as it is.
TemplateValue = Annotated[str, Field(strict=True)] | StrictBool | StrictInt | Finite


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
    quantization: str = Field(default="Q4_K_M", description="The GGUF type you run.")
    kv_cache: str = Field(
        default="f16", description="The KV cache type you run with (llama.cpp's --cache-type-k and -v)."
    )
    system_prompt: Annotated[str, Field(strict=True)] = Field(
        default="",
        description="The system prompt you use. Chat apps such as llama-server's page send none by "
        'default; heretic used "You are a helpful assistant."',
    )
    chat_template_kwargs: dict[Text, TemplateValue] | None = Field(
        default=None,
        description="Settings the chat template is rendered with, as llama-server's --chat-template-kwargs, "
        'such as { reasoning_effort = "high" }. Every prompt is rendered with them and candor serve '
        "passes them to the server. init sets reasoning_effort for a template that takes one, at the "
        "template's own default, and names the values it takes at the top of this file. A "
        "reasoning_effort can change the tag a model's thinking opens with, as K2-Horizon's does, and "
        "then think_close must change with it. The date is model.date.",
    )
    think_close: Text | None = Field(
        default=None,
        description="For a model that thinks before it answers: the text that ends its thinking, as its "
        'chat template writes it with thinking off, such as Qwen\'s "\\n</think>\\n\\n", or for a '
        "template that can't turn thinking off, the end tag of the tag its prompts open the thinking "
        'with, such as K2-Horizon\'s "</ifm|think>". init sets it for a model that thinks. Answers are '
        "judged on what follows it, and KL is read where the answer starts. Unset for a model that "
        "doesn't think.",
    )
    thinking_budget: StrictInt | None = Field(
        default=None,
        ge=1,
        description="How many tokens a thinking model may think before it is made to stop and answer, "
        "as you run it. Set for every model that thinks.",
    )
    thinking_budget_message: Annotated[str, Field(strict=True)] = Field(
        default="\n\nConsidering the limited time by the user, I have to give the solution based on the "
        "thinking directly now.",
        description="What a thinking model is made to write when its budget runs out, before its "
        "thinking is closed. The default is Qwen's own, as the end of think_close gives the newline "
        "before the end tag.",
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

    @field_validator("chat_template_kwargs")
    @classmethod
    def _not_the_date(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value and "date_string" in value:
            raise ValueError("the date the template writes is set by model.date")
        return value

    @model_validator(mode="after")
    def _thinking(self) -> "Deployment":
        if self.think_close is None:
            if self.thinking_budget is not None:
                raise ValueError("thinking_budget is set, but think_close isn't, so the model doesn't think")
        elif self.thinking_budget is None:
            raise ValueError("a model that thinks needs a thinking_budget")
        elif not END_TAG.fullmatch(self.think_close):
            raise ValueError(
                "think_close must hold one end tag, such as </think>, and nothing else but space"
            )
        return self

    def tags(self) -> tuple[str, str]:
        """The tags a thinking model's thinking starts and ends with."""
        assert self.think_close is not None
        end = self.think_close.strip()
        return "<" + end[2:], end


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


class RowNorm(StrEnum):
    NONE = "none"
    PRE = "pre"
    FULL = "full"


class Edit(Section):
    row_norm: str = Field(
        default="pre",
        description='How the edit treats each weight row\'s length: "none" edits the weights as they are; '
        '"pre" computes the edit on rows scaled to length 1, then scales them back, as heretic does on '
        'the Mac; "full" also scales every edited row back to its own length, as heretic\'s "full" '
        "does, at its rank of 3. A \"full\" edit isn't linear in strength, so each trial's adapter is "
        'built for it and llama-server restarted with it: slower, but it can work where "pre" can\'t, '
        "as on Qwen3.5-9B.",
    )

    @field_validator("row_norm")
    @classmethod
    def _known_row_norm(cls, value: str) -> str:
        if value not in set(RowNorm):
            *most, last = (f'"{norm}"' for norm in RowNorm)
            raise ValueError(f"must be {', '.join(most)} or {last}")
        return value


class Sampling(Section):
    temperature: Finite = Field(
        default=0.0,
        ge=0,
        description="0 answers greedily, always taking the likeliest token. A thinking model should be "
        "sampled at the settings its model card recommends, as greedy answers can repeat themselves "
        "without end.",
    )
    top_k: StrictInt = Field(
        default=40, ge=0, description="Only the k likeliest tokens are kept; 0 keeps all."
    )
    top_p: Finite = Field(
        default=0.95, gt=0, le=1, description="Only the likeliest tokens making up this share are kept."
    )
    min_p: Finite = Field(
        default=0.05,
        ge=0,
        lt=1,
        description="Tokens less likely than this share of the likeliest token's chance are dropped.",
    )
    presence_penalty: Finite = Field(
        default=0.0,
        description="Lowers the chance of every token among the last 64 the model has seen, its prompt "
        "included (llama-server's repeat_last_n), against repetition: not the whole answer, as vLLM's does.",
    )
    repeat_penalty: Finite = Field(
        default=1.0, gt=0, description="Divides the chance of recently used tokens; 1 changes nothing."
    )
    seed: StrictInt = Field(
        default=0,
        ge=0,
        lt=2**32,
        description="Each prompt is answered with a seed of its own made from this one, the same for every "
        "edit, so two edits meet the same random draws and are compared prompt by prompt.",
    )


class Scoring(Section):
    answer_tokens: StrictInt = Field(
        default=100,
        ge=1,
        le=8192,
        description="How long an answer to each harmful test prompt may be, after any thinking.",
    )
    thinking_budget: StrictInt | None = Field(
        default=None,
        ge=1,
        description="A shorter thinking budget for the search than deployment.thinking_budget, so a "
        "search of a thinking model finishes in time; the edits it suggests are measured at the "
        "deployment's. Unset: the deployment's.",
    )
    kl_tokens: StrictInt = Field(
        default=20,
        ge=1,
        le=1000,
        description="How many of the most likely first tokens KL divergence is measured over (20 came "
        "within 0.5% of the full vocabulary on Llama-3.2-3B's first answer token without thinking; the cut "
        "alone costs an estimated 0.4-0.7% where K2-Horizon-7B's and Qwen3.5-9B's answers start after "
        "thinking, and the floor given to the base's tokens an edited read doesn't list can understate "
        "KL more for an edit that pushes them out, an estimated 2-7% for heretic's Trial 161). init sets "
        "200 for a new run, which costs little and shrinks both.",
    )
    thinking: Literal["on", "off"] = Field(
        default="on",
        description='For a model that thinks: "on" scores the search\'s trials as the model is used, '
        'thinking to the budget; "off" closes the thinking with an empty block and answers greedily, '
        'many times faster. Set "off" only where candor proxy shows it ranks edits as thinking does: '
        "the edits the search suggests are measured as used either way. Only for a model whose chat "
        "template can turn thinking off, as Qwen's can and K2-Horizon's can't.",
    )
    closed_answer_tokens: StrictInt = Field(
        default=150,
        ge=1,
        le=8192,
        description='How long an answer to each harmful test prompt may be with thinking "off". '
        "Shorter answers miss refusals that come late (heretic #487).",
    )


class Measure(Section):
    answer_tokens: StrictInt = Field(
        default=256,
        ge=1,
        le=8192,
        description="How long an answer to each held-out prompt may be after any thinking, for the "
        "judge to read.",
    )
    maths_answer_tokens: StrictInt = Field(
        default=512,
        ge=1,
        le=8192,
        description="How long an answer to each maths question may be after any thinking. One cut off "
        "before it states its answer counts as wrong.",
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
    second_judge: Text | None = Field(
        default=None,
        description="Another Qwen3Guard-Gen model that judges the held-out answers again, after the judge, "
        "so a bar on refusals can be held under both; candor compare shows its counts beside the "
        "judge's. The search keeps to the judge. init sets Qwen3Guard-Gen-8B, which takes about 16 GB "
        "at 16 bits, more than a Mac with 18 GB allows, so it judges on a pod. Unset: one judge.",
    )
    second_judge_revision: Text | None = Field(
        default=None,
        description="The second judge's Hugging Face commit, pinned by init so its verdicts can't change.",
    )

    @model_validator(mode="after")
    def _second_judge_named(self) -> "Measure":
        if self.second_judge_revision is not None and self.second_judge is None:
            raise ValueError("second_judge_revision is set, but second_judge isn't")
        return self


class Search(Section):
    trials_per_scope: StrictInt = Field(
        default=300,
        ge=1,
        description="How many edits to try of each kind: every layer reading the refusal direction found "
        "at one layer, and each layer reading its own. The two kinds are searched apart and take turns. "
        "Fewer than 200 each makes missing the best edit far likelier.",
    )
    startup_per_scope: StrictInt = Field(
        default=20,
        ge=1,
        description="Evenly spread trials of each kind before the search starts aiming.",
    )
    max_refusals: StrictInt = Field(
        default=2,
        ge=0,
        description="The most test prompts an edit may refuse and still count. The search looks for the "
        "least KL among edits that refuse no more than this.",
    )
    max_budget_hits: StrictInt | None = Field(
        default=None,
        ge=0,
        description="The most test prompts whose thinking may reach the budget, so the model is made to "
        "stop and answer, for an edit still to count beside max_refusals; only for a search whose trials "
        "think. Unset, thinking to the budget never stops an edit counting. Changing it carries a search "
        "on, its finished trials judged again by the new number.",
    )
    stop_early: StrictBool = Field(
        default=True,
        description="Stop scoring a trial once it can't win, to save time. On refusals, it stops once it "
        "has refused more test prompts than max_refusals and than 25 fully scored trials of its kind "
        "(with max_budget_hits set, once it is further past both caps together than 25 of them), so "
        "every trial the search draws after it is as it would have been. With max_budget_hits set, it "
        "also stops once more of its answers than that reach the thinking budget, which can change the "
        "trials drawn after it.",
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
    widest_strength: Range = Field(
        default=(0.5, 3.0),
        description="How far the search may widen strength for a kind of edit. From a kind's 60th trial, "
        "and every 30th after, if its best edit within the cap (max_refusals and any max_budget_hits) "
        "presses an end of strength, or none is within it yet (then the ceiling), that end moves out to "
        "this one, once. Its trials so far carry on and a few evenly spread trials start the wider range. "
        "Set it to strength to never widen.",
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
    report_every: StrictInt = Field(
        default=25,
        ge=0,
        description="After every this many finished trials, of both kinds together, the search reports how "
        "it is going and writes the report to search/progress.json: its pace and time left, and each "
        "kind's best trial so far, strength range and trials since that best improved. 0 turns it off. "
        "Only a report: changing it never stops a search carrying on.",
    )

    @field_validator("strength", "widest_strength", "peak_layer", "edge_strength", "reach", "direction_layer")
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

    @field_validator("strength", "widest_strength")
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
        if self.startup_per_scope > self.trials_per_scope:
            raise ValueError("startup_per_scope can't be more than trials_per_scope")
        return self

    @model_validator(mode="after")
    def _widest_holds_strength(self) -> "Search":
        if not self.widest_strength[0] <= self.strength[0] <= self.strength[1] <= self.widest_strength[1]:
            raise ValueError("widest_strength must hold all of strength")
        return self

    def widens(self) -> bool:
        """Whether the search may widen strength at all."""
        return self.widest_strength != self.strength


class Slots(Section):
    thinking: StrictInt = Field(
        default=THINKING_SLOTS,
        ge=1,
        description="How many answers the unedited model thinks at once for thinking/, the thinking KL is "
        "read after. Sampled thinking can shift with it, so changing it samples that thinking afresh.",
    )
    search: StrictInt | None = Field(
        default=None,
        ge=1,
        description="How many answers the search, candor score and candor proxy work on at once. A search "
        f"is carried on only at the number it was made with. Unset: {THINKING_MODEL_SLOTS} for a "
        "model that thinks, whose answers are long and come faster the more are made together, and "
        f"{SEARCH_SLOTS} otherwise.",
    )
    measure: StrictInt = Field(
        default=MEASURE_SLOTS,
        ge=1,
        description="How many answers candor measure works on at once, unless --slots gives another "
        f"number. A measurement at other than {MEASURE_SLOTS} has -slots-N in its name.",
    )


class RunConfig(Section):
    model: Model = Field(description="The model this run edits.")
    deployment: Deployment = Field(
        default_factory=Deployment,
        description="How you actually run the model, so trials are scored that way.",
    )
    sampling: Sampling = Field(
        default_factory=Sampling,
        description="How the model picks each token as you chat with it. The search, measurements and "
        "candor serve all answer this way.",
    )
    prompts: Prompts = Field(default_factory=Prompts, description="Where the prompts come from.")
    edit: Edit = Field(default_factory=Edit, description="The weight edit itself.")
    scoring: Scoring = Field(default_factory=Scoring, description="How each candidate edit is measured.")
    measure: Measure = Field(
        default_factory=Measure,
        description="How finished edits are measured, on prompts the search never sees.",
    )
    search: Search = Field(default_factory=Search, description="The search over edits.")
    slots: Slots = Field(
        default_factory=Slots,
        description="How many answers llama-server works on at once. Each slot's context is held in the KV "
        "cache, all of it from the server's start, so more slots need more GPU memory (candor inspect "
        "shows how much a token takes); answers can shift slightly with the number.",
    )

    @model_validator(mode="after")
    def _search_thinks(self) -> "RunConfig":
        if self.scoring.thinking_budget is not None:
            if self.deployment.think_close is None:
                raise ValueError("scoring.thinking_budget is set for a model that doesn't think")
            if self.scoring.thinking_budget > (self.deployment.thinking_budget or 0):
                raise ValueError(
                    "scoring.thinking_budget is above deployment.thinking_budget, so the search would "
                    "score edits with more thinking than they are used with"
                )
        if self.scoring.thinking == "off" and self.deployment.think_close is None:
            raise ValueError('scoring.thinking is "off" for a model that doesn\'t think')
        if self.search.max_budget_hits is not None:
            if self.deployment.think_close is None:
                raise ValueError("search.max_budget_hits is set for a model that doesn't think")
            if self.scoring.thinking == "off":
                raise ValueError(
                    'search.max_budget_hits is set, but with scoring.thinking "off" the search\'s trials '
                    "don't think"
                )
        return self


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
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{where} isn't valid TOML: {error}") from None
    try:
        return RunConfig.model_validate(data)
    except ValidationError as error:
        problems = []
        for found in error.errors():
            # pydantic puts "Value error, " before what candor's own checks say, and a check of several
            # settings at once names none.
            message = found["msg"].removeprefix("Value error, ")
            setting = ".".join(map(str, found["loc"]))
            problems.append(f"{setting}: {message}" if setting else message)
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
    write_atomic(folder / FILE, text)


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
    if not RUN_NAME.fullmatch(name):
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
