"""Prompt sets, and how they become token ids."""

from dataclasses import dataclass
from typing import cast

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True)
class PromptSet:
    dataset: str
    split: str
    column: str = "text"

    def load(self) -> list[str]:
        return list(load_dataset(self.dataset, split=self.split)[self.column])


HARMFUL_TRAIN = PromptSet("mlabonne/harmful_behaviors", "train[:400]")
HARMLESS_TRAIN = PromptSet("mlabonne/harmless_alpaca", "train[:400]")
HARMFUL_TEST = PromptSet("mlabonne/harmful_behaviors", "test[:100]")
HARMLESS_TEST = PromptSet("mlabonne/harmless_alpaca", "test[:100]")


def encode(
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    system: str = SYSTEM_PROMPT,
    prefix: str = "",
    add_special_tokens: bool = False,
) -> list[list[int]]:
    """Token ids for each prompt, up to the point where the model starts answering.

    The chat template already writes the special tokens the model expects. add_special_tokens=True
    lets the tokenizer add its own as well, which is what heretic does: for Llama 3 that means two
    begin-of-text tokens. It exists only to reproduce heretic's numbers.
    """
    encoded = []
    for prompt in prompts:
        chat = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        text = cast(str, tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False))
        encoded.append(tokenizer(text + prefix, add_special_tokens=add_special_tokens)["input_ids"])
    return encoded
