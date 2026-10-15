"""Checks on a model's tokenizer and chat template that decide whether a measurement means anything."""

import re
from dataclasses import dataclass
from typing import Any, cast

from transformers import CONFIG_MAPPING, PreTrainedTokenizerBase

from .config import END_TAG


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


def think_close(tokenizer: PreTrainedTokenizerBase) -> str | None:
    """The text that ends a thinking model's thinking, as its chat template writes it with thinking off:
    what enable_thinking=False adds to a prompt ready for the answer, one end tag such as </think>. None
    for a template without that switch, or one that writes thinking off some other way, such as one
    whose prompts don't open the thinking themselves."""

    def ready(**settings: Any) -> str:
        rendered = tokenizer.apply_chat_template(
            [USER], tokenize=False, add_generation_prompt=True, **settings
        )
        return cast(str, rendered)

    thinking, answering = ready(), ready(enable_thinking=False)
    close = answering[len(thinking) :] if answering.startswith(thinking) else ""
    return close if END_TAG.fullmatch(close) else None


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
