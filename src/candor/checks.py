"""Checks on a model's tokenizer and chat template that decide whether a measurement means anything."""

from dataclasses import dataclass
from typing import cast

from transformers import PreTrainedTokenizerBase


def _leading(ids: list[int], token: int) -> int:
    return next((i for i, t in enumerate(ids) if t != token), len(ids))


@dataclass(frozen=True)
class BeginOfText:
    """How many begin-of-text tokens each way of running the model puts in front of a chat."""

    template: int
    tokenizer_adds: int

    @property
    def transformers(self) -> int:
        """transformers' apply_chat_template, mlx-lm and Bionic's MLX engine send the template's own."""
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


def writes_the_date(tokenizer: PreTrainedTokenizerBase) -> bool:
    """True when the chat template puts today's date in every prompt, so prompts change daily.

    Pin it for a run by passing the template a fixed date_string.
    """
    template = tokenizer.chat_template
    return isinstance(template, str) and "strftime_now" in template
