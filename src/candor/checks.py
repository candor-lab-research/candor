"""Checks on a model's tokenizer and chat template that decide whether a measurement means anything."""

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from transformers import PreTrainedTokenizerBase

JS = Path(__file__).parent / "js"


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


def chat_template(tokenizer: PreTrainedTokenizerBase) -> str | None:
    """The template apps use: the only one, or the one named default when there are several."""
    template = tokenizer.chat_template
    if isinstance(template, dict):
        template = template.get("default")
    return template if isinstance(template, str) else None


@dataclass(frozen=True)
class TemplateCheck:
    status: str
    detail: str = ""


SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
USER = {"role": "user", "content": "hi"}
ASSISTANT = {"role": "assistant", "content": "Hello."}
CONVERSATIONS = [[USER], [SYSTEM, USER], [SYSTEM, USER, ASSISTANT, USER], [USER, ASSISTANT, USER]]


def app_template_parses(tokenizer: PreTrainedTokenizerBase, template: str) -> TemplateCheck:
    """Whether Bionic and LM Studio's template parser (@huggingface/jinja) can use the template.

    If it can't, those apps fall back to a prompt format the model wasn't trained on. Only the
    conversations transformers renders are tried, so a template that deliberately refuses one
    (some reject system messages) isn't reported as unreadable.
    """
    renderable = []
    for conversation in CONVERSATIONS:
        try:
            tokenizer.apply_chat_template(
                conversation, chat_template=template, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            continue
        renderable.append(conversation)
    if not renderable:
        return TemplateCheck("skipped", "transformers can't render any test conversation with it either")
    node = shutil.which("node")
    if node is None:
        return TemplateCheck("skipped", "Node.js isn't installed")
    if not (JS / "node_modules" / "@huggingface" / "jinja").exists():
        return TemplateCheck("skipped", f"run once: npm install --prefix {JS}")
    with tempfile.NamedTemporaryFile("w", suffix=".jinja") as file:
        file.write(template)
        file.flush()
        args = [node, str(JS / "check_template.mjs"), file.name, json.dumps(renderable)]
        try:
            run = subprocess.run(args, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return TemplateCheck("skipped", "the checker timed out")
    try:
        result = json.loads(run.stdout)
    except json.JSONDecodeError:
        last_line = run.stderr.strip().splitlines()[-1:] or ["no output"]
        return TemplateCheck("skipped", f"the checker itself failed: {last_line[0]}")
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


STRFTIME = re.compile(r"strftime_now\s*\(")
DATE_STRING = re.compile(r"set\s+date_string\s*=\s*strftime_now\s*\(\s*[\"']([^\"']+)[\"']\s*\)")
DATE_GUARD = re.compile(r"date_string\s+is\s+(not\s+)?defined")


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
