"""A run on a tiny random model with made-up prompts and a fake judge, for measuring and searching."""

import json
from typing import ClassVar

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from candor import config, measure, run, search
from candor.run import Run

from .test_engine import LLAMA_SERVER
from .test_run import TEMPLATE, fake_everything_but_the_stages, pinned
from .tiny_gguf import HIDDEN, LAYERS, random_weights, write_model

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

    def __init__(self, model: str, revision: str | None):
        FakeGuard.judged = []

    def replies(self, prompts, answers, batch=16, done=None):
        assert len(prompts) == len(answers)
        FakeGuard.judged.append(len(answers))
        for _ in answers:
            if done:
                done()
        return [REFUSED if i == 0 else ANSWERED for i in range(len(answers))]

    def close(self):
        pass


HF_NAMES = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
}


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


def fixed_directions(self):
    self.prompt_ids()
    self.inputs["directions"] = "fixed"
    return F.normalize(torch.randn(LAYERS + 1, HIDDEN, generator=torch.Generator().manual_seed(0)), dim=-1)


def tiny_run(tmp_path, monkeypatch) -> Run:
    """A run whose model is a tiny random GGUF and a Hugging Face folder of the same weights, with
    made-up prompts and a fake judge, ready to measure or search."""
    if not LLAMA_SERVER.exists():
        pytest.skip("no llama-server")
    write_model(tmp_path / "tiny.gguf", random_weights(), chat_template=TEMPLATE)
    model = write_matching_llama(tmp_path / "model")
    fake_everything_but_the_stages(monkeypatch, tmp_path / "tiny.gguf")
    monkeypatch.setattr(Run, "directions", fixed_directions)
    monkeypatch.setattr(run, "load_prompts", fake_prompts)
    monkeypatch.setattr(measure, "load_prompts", fake_prompts)
    monkeypatch.setattr(measure, "Guard", FakeGuard)
    monkeypatch.setattr(search, "Guard", FakeGuard)
    folder = tmp_path / "runs" / "tiny"
    folder.mkdir(parents=True)
    return Run(folder, pinned(model), report=lambda _: None)
