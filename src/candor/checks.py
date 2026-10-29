"""Checks on a model's tokenizer and chat template that decide whether a measurement means anything."""

import re
from dataclasses import dataclass
from typing import Any, cast

from jinja2 import TemplateError
from transformers import CONFIG_MAPPING, PreTrainedTokenizerBase

from .config import END_TAG

# A prompt that ends by opening a thinking tag, such as Qwen3.5's <think> or K2-Horizon's <ifm|think>.
OPENS_THINKING = re.compile(r"<([^<>/]*think[^<>]*)>\s*$", re.IGNORECASE)
# A prompt that ends by closing an empty thinking block, as Qwen3.5's small models do unless asked to
# think.
CLOSES_THINKING = re.compile(r"</[^<>\s]*think[^<>\s]*>\s*$", re.IGNORECASE)
# Any think tag.
THINK_TAG = re.compile(r"</?[^<>\s]*think[^<>\s]*>", re.IGNORECASE)
EFFORT = "reasoning_effort"
# The reasoning_effort values a chat template may take, by the names OpenAI's API and chat templates use,
# and one no template knows.
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
UNKNOWN_EFFORT = "x-unknown"


def remote_code(*configs: dict[str, Any] | None) -> list[str]:
    """The Python files a model ships and runs when loaded, from the auto_map of each config given
    (config.json and tokenizer_config.json)."""
    modules = set()
    for config in configs:
        for value in (config or {}).get("auto_map", {}).values():
            for reference in value if isinstance(value, list) else [value]:
                if reference:
                    modules.add(reference.split("--")[-1].rsplit(".", 1)[0] + ".py")
    return sorted(modules)


def needs_remote_code(model_config: dict[str, Any]) -> bool:
    """Whether transformers can load the model only with the Python files it ships: its config names
    its own classes, and transformers has none for its type."""
    return bool(remote_code(model_config)) and model_config.get("model_type") not in CONFIG_MAPPING


def chat_template(tokenizer: PreTrainedTokenizerBase) -> str | None:
    """The template apps use: the only one, or the one named default when there are several."""
    template = tokenizer.chat_template
    if isinstance(template, dict):
        template = template.get("default")
    return template if isinstance(template, str) else None


USER = {"role": "user", "content": "hi"}


def _leading(ids: list[int], token: int) -> int:
    return next((i for i, t in enumerate(ids) if t != token), len(ids))


@dataclass(frozen=True)
class BeginOfText:
    """How many begin-of-text tokens each way of running the model puts in front of a chat."""

    template: int
    tokenizer_adds: int

    @property
    def transformers(self) -> int:
        """transformers' apply_chat_template and mlx-lm send the template's own."""
        return self.template

    @property
    def llama_cpp(self) -> int:
        """llama.cpp drops the template's copy and adds its own, so it sends one if either would."""
        return max(self.template, self.tokenizer_adds)

    @property
    def heretic(self) -> int:
        """heretic tokenizes the rendered chat with the tokenizer's own special tokens added too."""
        return self.template + self.tokenizer_adds

    def problems(self) -> list[str]:
        found = []
        if self.heretic != self.transformers:
            found.append(
                f"heretic scores this model with {self.heretic} begin-of-text tokens, but apps send "
                f"{self.transformers}, so heretic's refusal and KL numbers describe input you never use"
            )
        if self.llama_cpp != self.transformers:
            found.append(
                f"llama.cpp sends {self.llama_cpp} begin-of-text tokens and transformers-based apps send "
                f"{self.transformers}, so score the model in the engine you will run it with"
            )
        return found


def begin_of_text(tokenizer: PreTrainedTokenizerBase) -> BeginOfText:
    bos = tokenizer.bos_token_id
    if not isinstance(bos, int):
        return BeginOfText(template=0, tokenizer_adds=0)
    chat = [{"role": "user", "content": "Hi"}]
    text = cast(str, tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False))
    template = _leading(tokenizer(text, add_special_tokens=False)["input_ids"], bos)
    with_special = _leading(tokenizer(text)["input_ids"], bos)
    return BeginOfText(template=template, tokenizer_adds=with_special - template)


STRFTIME = re.compile(r"strftime_now\s*\(")
DATE_STRING = re.compile(r"set\s+date_string\s*=\s*strftime_now\s*\(\s*[\"']([^\"']+)[\"']\s*\)")
DATE_GUARD = re.compile(r"date_string\s+is\s+(not\s+)?defined")


def _ready(tokenizer: PreTrainedTokenizerBase, **settings: Any) -> str:
    """A one-message chat as the template renders it with these settings, ready for the answer."""
    rendered = tokenizer.apply_chat_template([USER], tokenize=False, add_generation_prompt=True, **settings)
    return cast(str, rendered)


def turns_thinking_off(tokenizer: PreTrainedTokenizerBase, **settings: Any) -> bool:
    """Whether enable_thinking=False changes the prompt the template renders with these settings."""
    return _ready(tokenizer, **{**settings, "enable_thinking": False}) != _ready(tokenizer, **settings)


def think_close(tokenizer: PreTrainedTokenizerBase, **settings: Any) -> str | None:
    """The text that ends a thinking model's thinking, its chat template rendered with these settings.

    For a template that turns thinking off, as Qwen's does: what enable_thinking=False writes after the
    tag that opens the thinking, one end tag such as </think> (Qwen3.8's also leaves out the system
    message it adds at every reasoning effort but medium). For one without that switch whose prompts end by
    opening the thinking with a tag, as K2-Horizon's do with <ifm|think>: that tag's end tag alone, which
    is what K2-Horizon's template writes after an empty thinking block and how llama-server closes the
    thinking when a budget runs out. For one that turns thinking off by writing that end tag in place of
    the tag that opens it, as Spark-X's does: the end tag, which closes the thinking the prompt opens
    where its answer starts in use, though its template writes an empty thinking as the end tag alone.
    None for a model that doesn't think, or a
    template that writes thinking off some other way, such as one whose prompts don't open the thinking
    themselves.
    """
    thinking = _ready(tokenizer, **settings)
    answering = _ready(tokenizer, **{**settings, "enable_thinking": False})
    opened = OPENS_THINKING.search(thinking)
    if answering == thinking:
        close = f"</{opened[1]}>" if opened else ""
    elif answering.startswith(thinking):
        close = answering[len(thinking) :]
    elif opened and opened[0] in answering:
        close = answering[answering.rindex(opened[0]) + len(opened[0]) :]
    elif opened and answering[: opened.start()] == thinking[: opened.start()]:
        close = answering[opened.start() :]
        close = close if close.strip() == f"</{opened[1]}>" else ""
    else:
        close = ""
    return close if END_TAG.fullmatch(close) else None


def thinking_off_closes_an_empty_block(tokenizer: PreTrainedTokenizerBase, **settings: Any) -> bool:
    """Whether turning thinking off only writes, after the prompt, what closes the thinking it opens, as
    Qwen's template does, so answers after an empty thinking block are the model with its thinking off."""
    thinking = _ready(tokenizer, **settings)
    return _ready(tokenizer, **{**settings, "enable_thinking": False}).startswith(thinking)


def may_write_its_own_thinking(tokenizer: PreTrainedTokenizerBase, template: str) -> bool:
    """Whether a model whose prompts neither open its thinking nor turn it off may still think: its
    template names a think tag, as one for a model that writes its own thinking may, to keep it out of
    past answers, and its prompts don't end by closing the thinking."""
    return THINK_TAG.search(template) is not None and not CLOSES_THINKING.search(_ready(tokenizer))


@dataclass(frozen=True)
class Efforts:
    """The reasoning_effort values a chat template takes, and the one that renders as the template does
    when given none, if any does."""

    values: tuple[str, ...]
    default: str | None


def reasoning_efforts(tokenizer: PreTrainedTokenizerBase) -> Efforts | None:
    """The reasoning_effort values the chat template takes: those of EFFORTS it names and renders a
    prompt with, unlike a value it doesn't know, such as K2-Horizon's low, medium and high. None for a
    template that takes none."""
    template = chat_template(tokenizer) or ""
    if not re.search(rf"\b{EFFORT}\b", template):
        return None
    given_none = _ready(tokenizer)
    unknown = _effort(tokenizer, UNKNOWN_EFFORT)
    values, default = [], None
    for value in EFFORTS:
        if f"'{value}'" not in template and f'"{value}"' not in template:
            continue
        rendered = _effort(tokenizer, value)
        if rendered is None or rendered == unknown:
            continue
        values.append(value)
        if default is None and rendered == given_none:
            default = value
    return Efforts(tuple(values), default) if values else None


def _effort(tokenizer: PreTrainedTokenizerBase, value: str) -> str | None:
    """A one-message chat rendered with this reasoning_effort, or None if the template refuses it."""
    try:
        return _ready(tokenizer, **{EFFORT: value})
    except TemplateError:
        return None


def pinnable_date_format(template: str) -> str | None:
    """The date format to pin, when the template's only date is a date_string it accepts from outside.

    Llama 3 templates set date_string from strftime_now only when none is passed in, so passing
    one fixes the date. Any other call to strftime_now would still write today's date.
    """
    match = DATE_STRING.search(template)
    if match and len(STRFTIME.findall(template)) == 1 and DATE_GUARD.search(template):
        return match[1]
    return None


def writes_the_date(template: str) -> bool:
    """True when the chat template puts today's date in every prompt, so prompts change daily."""
    return STRFTIME.search(template) is not None
