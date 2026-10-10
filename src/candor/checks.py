"""Checks on a model's tokenizer and chat template that decide whether a measurement means anything."""

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from transformers import PreTrainedTokenizerBase

JS = Path(__file__).parent / "js"


def remote_code(config: dict[str, Any]) -> list[str]:
    """The Python files a model ships and runs when loaded, listed in its config's auto_map."""
    modules = set()
    for value in config.get("auto_map", {}).values():
        for reference in value if isinstance(value, list) else [value]:
            if reference:
                modules.add(reference.split("--")[-1].rsplit(".", 1)[0] + ".py")
    return sorted(modules)


@dataclass(frozen=True)
class TemplateCheck:
    status: str
    detail: str = ""


def app_template_parses(template: str, variables: dict[str, Any] | None = None) -> TemplateCheck:
    """Whether Bionic and LM Studio's template parser (@huggingface/jinja) can read the template.

    If it can't, those apps fall back to a prompt format the model wasn't trained on.
    """
    node = shutil.which("node")
    if node is None:
        return TemplateCheck("skipped", "Node.js isn't installed")
    if not (JS / "node_modules" / "@huggingface" / "jinja").exists():
        return TemplateCheck("skipped", f"run once: npm install --prefix {JS}")
    with tempfile.NamedTemporaryFile("w", suffix=".jinja") as file:
        file.write(template)
        file.flush()
        args = [node, str(JS / "check_template.mjs"), file.name, json.dumps(variables or {})]
        result = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
    if result["ok"]:
        return TemplateCheck("ok")
    lines = ", ".join(map(str, result["lines"])) or "unknown"
    return TemplateCheck("fails", f"{result['error']} (statements on lines {lines})")


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
