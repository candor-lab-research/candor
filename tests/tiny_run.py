"""A run on a tiny random model with made-up prompts and a fake judge, for measuring and searching."""

import json
from typing import ClassVar

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from candor import config, measure, run, search
from candor.engine import LlamaServer
from candor.run import Run

from .helpers import (
    HF_NAMES,
    LLAMA_SERVER,
    TEMPLATE,
    THINKING_TEMPLATE,
    fake_everything_but_the_stages,
    pinned,
)
from .tiny_gguf import HIDDEN, LAYERS, random_weights, write_model
from .tiny_moe import random_moe_weights, write_moe_folder, write_moe_model

# Qwen3.8's: thinking adds a system message, at every reasoning effort but medium, that turning it
# off leaves out.
MESSAGE_WHEN_THINKING_TEMPLATE = (
    "{% if enable_thinking is not defined or enable_thinking %}Think hard.\n{% endif %}" + THINKING_TEMPLATE
)
# One that always opens the thinking, as K2-Horizon's does, with a tag that depends on the reasoning
# effort: "high" when none is given, and refused unless it is one the template knows.
ALWAYS_THINKING_TEMPLATE = TEMPLATE.replace(
    "assistant:{% endif %}",
    "assistant:{% set effort = reasoning_effort | default('high') %}"
    "{% if effort == 'high' %}<think>\n{% elif effort == 'low' %}<think_less>\n"
    "{% else %}{{ raise_exception('no such effort') }}{% endif %}{% endif %}",
)

REFUSED = "Safety: Safe\nCategories: None\nRefusal: Yes"
ANSWERED = "Safety: Unsafe\nCategories: Violent\nRefusal: No"


def fake_prompts(spec: config.PromptSet) -> list[str]:
    """Three made-up prompts per set, none repeated across sets, and GSM8K-style answers."""
    if spec.column == "answer":
        return ["Four. #### 4", "#### 1,000", "#### -2.5"]
    return [f"prompt {i} from {spec.dataset} {spec.split}" for i in range(3)]


class FakeGuard:
    """Labels answers by position, without a model: the first of each batch refused, the rest not."""

    judged: ClassVar[list[int]] = []
    read: ClassVar[list[str]] = []
    # Each judge loaded and closed, by its model, in order.
    events: ClassVar[list[tuple[str, str]]] = []
    machine: ClassVar[dict[str, str]] = {"device": "cpu", "dtype": "float32", "torch": "fake"}

    def __init__(self, model: str, revision: str | None):
        FakeGuard.judged = []
        FakeGuard.read = []
        FakeGuard.events.append(("load", model))
        self.model = model

    def replies(self, prompts, answers, batch=16, done=None):
        assert len(prompts) == len(answers)
        FakeGuard.judged.append(len(answers))
        FakeGuard.read += answers
        for _ in answers:
            if done:
                done()
        return [REFUSED if i == 0 else ANSWERED for i in range(len(answers))]

    def close(self):
        FakeGuard.events.append(("close", self.model))


def write_matching_llama(folder):
    """A Hugging Face folder holding the tiny GGUF's own layer weights, so adapters made from it fit
    the model llama-server runs."""
    folder.mkdir()
    tensors = {}
    for name, weight in random_weights().items():
        if name.startswith("blk."):
            _, layer, kind, _ = name.split(".")
            tensors[f"model.layers.{layer}.{HF_NAMES[kind]}.weight"] = torch.from_numpy(weight.copy())
    save_file(tensors, folder / "model.safetensors")
    model_config = {"architectures": ["LlamaForCausalLM"], "hidden_size": HIDDEN, "num_hidden_layers": LAYERS}
    (folder / "config.json").write_text(json.dumps(model_config | {"max_position_embeddings": 256}))
    return folder


# The run's own, which the tiny run replaces, for a test that needs them.
READ_DIRECTIONS, PLANNED_DIRECTIONS, CHECK_JUDGE = Run.directions, Run.planned_directions, Run.check_judge


def fixed_plan(self):
    """The made-up directions named in the run's inputs, as Run.planned_directions names real ones."""
    self.prompt_ids()
    self.inputs["directions"] = "fixed"
    if self.settings.deployment.think_close:
        self.inputs["answering directions"] = "fixed, answering"
    return {}


def fixed_directions(self, answering=False):
    """Made-up directions, different where the answer starts, instead of reading residuals."""
    fixed_plan(self)
    seed = torch.Generator().manual_seed(1 if answering else 0)
    # A row more is drawn and the first dropped, so the records pinned from these draws hold.
    return F.normalize(torch.randn(LAYERS + 1, HIDDEN, generator=seed), dim=-1)[1:]


def tiny_run(tmp_path, monkeypatch, moe: bool = False, fused: bool = False) -> Run:
    """A run whose model is a tiny random GGUF and a Hugging Face folder of the same weights, with
    made-up prompts and a fake judge, ready to measure or search: a Llama, or a Qwen3-MoE whose
    folder stores each expert apart, or each layer's in one tensor when fused."""
    if not LLAMA_SERVER.exists():
        pytest.skip("no llama-server")
    if moe:
        weights = random_moe_weights()
        write_moe_model(tmp_path / "tiny.gguf", weights, chat_template=TEMPLATE)
        model = write_moe_folder(tmp_path / "model", weights, fused)
    else:
        write_model(tmp_path / "tiny.gguf", random_weights(), chat_template=TEMPLATE)
        model = write_matching_llama(tmp_path / "model")
    fake_everything_but_the_stages(monkeypatch, tmp_path / "tiny.gguf")
    monkeypatch.setattr(Run, "directions", fixed_directions)
    monkeypatch.setattr(Run, "planned_directions", fixed_plan)
    monkeypatch.setattr(run, "load_prompts", fake_prompts)
    monkeypatch.setattr(measure, "load_prompts", fake_prompts)
    monkeypatch.setattr(measure, "Guard", FakeGuard)
    monkeypatch.setattr(search, "Guard", FakeGuard)
    monkeypatch.setattr(FakeGuard, "events", [])
    # The judge is fake, so a real one's size is never read from the Hub.
    monkeypatch.setattr(Run, "check_judge", lambda self, model, revision: None)
    folder = tmp_path / "runs" / "tiny"
    folder.mkdir(parents=True)
    return Run(folder, pinned(model), report=lambda _: None)


def thinks_through(monkeypatch, words: int = 2, asked: list | None = None) -> list[int]:
    """Make the unedited model's thinking end by itself on every other prompt, as the tiny random model's
    seldom does within its budget, so KL can be read after it. Returns the made-up thinking, this many
    words and the end tag; each call's prompts, sampling, seeds and thinking go to asked."""
    thought: list[int] = []

    def thoughts(self, prompts, sampling, seeds, thinking, done=None):
        if asked is not None:
            asked.append((prompts, sampling, seeds, thinking))
        thought[:] = [*self.tokenize(" ".join(["w2"] * words)), *self.tokenize(thinking.end)]
        for _ in prompts:
            if done:
                done()
        return [list(thought) if i % 2 == 0 else None for i in range(len(prompts))]

    monkeypatch.setattr(LlamaServer, "thoughts", thoughts)
    return thought


def thinking(
    tiny: Run, budget: int = 6, template: str = THINKING_TEMPLATE, close: str = "\n</think>\n\n"
) -> Run:
    """The tiny run with a model that thinks before it answers, sampled, so its answers vary."""
    write_model(tiny.folder.parents[1] / "tiny.gguf", random_weights(), template, thinking=True)
    tiny.settings.deployment = config.Deployment(
        think_close=close, thinking_budget=budget, thinking_budget_message="w1"
    )
    tiny.settings.sampling = config.Sampling(temperature=1.0, top_k=0, min_p=0.0)
    # Room for thinking and an answer in the tiny model's 256 tokens.
    tiny.settings.scoring.answer_tokens = 16
    return tiny
