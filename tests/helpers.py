"""Helpers the tests share: models, prompts and runs made for them, and checks of what they print."""

import json
import os
import shutil
import zlib
from pathlib import Path
from types import SimpleNamespace

import gguf
import torch
from safetensors.torch import load_file, save_file

from candor import config, files, kl, run, toolchain
from candor.config import Model, RunConfig
from candor.kernel import Kernel
from candor.measure import Edit

from .tiny_gguf import (
    random_weights,
    write_model,
)
from .tiny_tokenizer import WITHOUT_BOS

# The width and depth of the models these helpers make, which aren't tiny_gguf's.
HIDDEN, LAYERS = 64, 2


def llama(layers: int = 2, prefix: str = "model.layers") -> dict[str, list[int]]:
    shapes = {}
    for i in range(layers):
        shapes |= {
            f"{prefix}.{i}.self_attn.q_proj.weight": [HIDDEN, HIDDEN],
            f"{prefix}.{i}.self_attn.k_proj.weight": [16, HIDDEN],
            f"{prefix}.{i}.self_attn.o_proj.weight": [HIDDEN, HIDDEN],
            f"{prefix}.{i}.mlp.up_proj.weight": [128, HIDDEN],
            f"{prefix}.{i}.mlp.gate_proj.weight": [128, HIDDEN],
            f"{prefix}.{i}.mlp.down_proj.weight": [HIDDEN, 128],
            f"{prefix}.{i}.input_layernorm.weight": [HIDDEN],
        }
    return shapes


# Qwen3.5-9B's sizes: a full-attention layer after every three linear-attention ones.
QWEN35 = {
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "hidden_size": 4096,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_value_head_dim": 128,
    "linear_num_value_heads": 32,
}


# rude-llama's, a Llama 3.2 3B.
LLAMA_3B = {"num_hidden_layers": 28, "num_attention_heads": 24, "num_key_value_heads": 8, "hidden_size": 3072}


THINKING = WITHOUT_BOS + (
    "{% if add_generation_prompt %}<think>\n{% if enable_thinking is defined and not enable_thinking %}"
    "\n</think>\n\n{% endif %}{% endif %}"
)


# Qwen3's own: the model opens its thinking itself, so turning it off writes a whole empty block.
OPENS_ITS_OWN = WITHOUT_BOS + (
    "{% if add_generation_prompt %}A:{% if enable_thinking is defined and not enable_thinking %}"
    "<think>\n\n</think>\n\n{% endif %}{% endif %}"
)


# Qwen3.8's: thinking adds a system message, at every reasoning effort but medium, that turning it
# off leaves out.
MESSAGE_WHEN_THINKING = (
    "{% if enable_thinking is not defined or enable_thinking %}Think hard.\n{% endif %}" + THINKING
)


def stub(folder: Path, script: str) -> Path:
    """A shell script standing in for llama-server."""
    path = folder / "llama-server"
    path.write_text(f"#!/bin/sh\n{script}\n")
    path.chmod(0o755)
    return path


# What llama.cpp logs when a KV cache doesn't fit on the GPU, as K2-Horizon's search did on an A40.
NO_ROOM = (
    "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 61488.00 MiB on device 0: cudaMalloc failed: "
    "out of memory"
)


KV_FAILED = "llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache"


EDIT = Edit("trial", 0.4, {"attn": Kernel(1.2, 1.0, 0.8, 1.0), "mlp": Kernel(1.0, 0.0, 0.5, 1.0)})


def write_llama(folder: Path) -> Path:
    folder.mkdir()
    torch.manual_seed(7)
    save_file({name: torch.randn(shape) for name, shape in llama().items()}, folder / "model.safetensors")
    model_config = {"architectures": ["LlamaForCausalLM"], "hidden_size": HIDDEN, "num_hidden_layers": LAYERS}
    (folder / "config.json").write_text(json.dumps(model_config | {"max_position_embeddings": 256}))
    return folder


def short() -> config.Measure:
    """Measure settings that fit the tiny test models' 256 tokens of context."""
    return config.Measure(answer_tokens=16, maths_answer_tokens=16, maths_instruction="Answer.")


class Loaded(torch.nn.Module):
    """Stands in for a model loaded by transformers when the residuals are faked."""


def residuals_of(prompts: list[list[int]]) -> torch.Tensor:
    """The residuals fake_residuals gives for these prompts."""
    generator = torch.Generator().manual_seed(zlib.crc32(json.dumps(prompts).encode()))
    return torch.randn(len(prompts), LAYERS + 1, HIDDEN, generator=generator)


def fake_residuals(monkeypatch) -> SimpleNamespace:
    """Residuals fixed by the prompts they are read for. Returns how each model and tokenizer was loaded."""
    loads = SimpleNamespace(models=[], tokenizers=[])

    def residuals(model, prompts, pad_id):
        return residuals_of(prompts)

    def load_model(folder, **kwargs):
        loads.models.append(kwargs)
        return Loaded()

    def pad_token_id_for(folder, trust_remote_code):
        loads.tokenizers.append(trust_remote_code)
        return 0

    monkeypatch.setattr(run.residuals, "last_token_residuals", residuals)
    monkeypatch.setattr(run.residuals, "pad_token_id_for", pad_token_id_for)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", load_model)
    return loads


def on_disk(run, name="cli") -> Path:
    """A run folder beside run's own holding its settings as they are now, for candor's commands to
    read."""
    return config.write(run.folder.parent, name, run.settings, "made by a test").parent


def adapter_factors(path, name=""):
    """An adapter's (b, a), for the weight named when it holds several."""
    tensors = {t.name: torch.from_numpy(t.data.copy()) for t in gguf.GGUFReader(path).tensors}
    a = next(v for k, v in tensors.items() if k.startswith(name) and k.endswith("lora_a"))
    b = next(v for k, v in tensors.items() if k.startswith(name) and k.endswith("lora_b"))
    return b, a


def small(run, trials=2):
    run.settings.search.trials_per_scope = trials
    run.settings.search.startup_per_scope = 1


# A trial's settings as heretic records them.
HERETIC_KERNELS = {
    "attn.o_proj": {
        "max_weight": 1.2,
        "max_weight_position": 0.8,
        "min_weight": 0.6,
        "min_weight_distance": 0.6,
    },
    "mlp.down_proj": {
        "max_weight": 1.0,
        "max_weight_position": 1.0,
        "min_weight": 0.5,
        "min_weight_distance": 0.6,
    },
}


def write_journal(path, direction_index, settings=None, parameters=HERETIC_KERNELS):
    """A heretic study's journal holding one finished trial, its number 1, with these settings."""
    settings = {"row_normalization": "pre", "orthogonalize_direction": True} if settings is None else settings
    ops = [
        {"op_code": 2, "study_id": 0, "user_attr": {"settings": json.dumps(settings)}},
        {"op_code": 8, "trial_id": 0, "user_attr": {"index": 1, "direction_index": direction_index}},
        {"op_code": 8, "trial_id": 0, "user_attr": {"parameters": parameters, "refusals": 4}},
        {"op_code": 8, "trial_id": 0, "user_attr": {"base_refusals": 96, "kl_divergence": 0.03}},
    ]
    path.write_text("".join(json.dumps(op) + "\n" for op in ops))
    return path


def mean_from_top(base: list[dict[int, float]], edited: list[dict[int, float]]) -> float:
    """Mean KL(base || edited) over prompts, from the top tokens a server lists for each (see
    kl.from_top)."""
    return sum(map(kl.from_top, base, edited)) / len(base)


def flat(output: str) -> str:
    """Command output with Rich's line breaks undone."""
    return " ".join(output.split())


def recorded_syncs(monkeypatch) -> list[tuple[str, str]]:
    """Each file flushed, by name, with what it held then."""
    synced = []
    real = files.sync

    def sync(path: Path) -> None:
        synced.append((path.name, path.read_text()))
        real(path)

    monkeypatch.setattr(files, "sync", sync)
    return synced


PARAMETERS = {
    "attn.o_proj": {
        "max_weight": 1.3,
        "max_weight_position": 16.5,
        "min_weight": 1.0,
        "min_weight_distance": 13.5,
    },
    "mlp.down_proj": {
        "max_weight": 1.2,
        "max_weight_position": 16.0,
        "min_weight": 0.5,
        "min_weight_distance": 12.0,
    },
}


def settings(**model) -> RunConfig:
    return RunConfig(model=Model.model_validate({"source": "org/model"} | model))


# A tiny Qwen3.5: layer 0 is linear attention with 2 key heads and 4 value heads of 8 dims, layer 1 is
# full attention. llama.cpp b11524's converter writes the value heads in the order 0, 2, 1, 3, tiled
# across the key heads (checked against its own _reorder_v_heads).
TINY_QWEN35 = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "text_config": {
        "hidden_size": HIDDEN,
        "num_hidden_layers": 2,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 8,
    },
}


CONVERTER_ORDER = [*range(0, 8), *range(16, 24), *range(8, 16), *range(24, 32)]


OUT_PROJ = "model.language_model.layers.0.linear_attn.out_proj.weight"


def qwen35() -> dict[str, list[int]]:
    prefix = "model.language_model.layers"
    return {
        f"{prefix}.0.linear_attn.out_proj.weight": [HIDDEN, 32],
        f"{prefix}.1.self_attn.o_proj.weight": [HIDDEN, HIDDEN],
        **{f"{prefix}.{i}.mlp.down_proj.weight": [HIDDEN, 128] for i in range(2)},
    }


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
GGUF_NAMES = {hf: kind for kind, hf in HF_NAMES.items()}


def gguf_from_folder(folder, destination, log, rounded_to=torch.float32, thinks=False, types=None):
    """The tiny model's GGUF with the layer weights of a Hugging Face folder, as the converter would
    write it, rounded to a type, its matrices stored as types gives, and of the thinking model when it
    thinks: the tiny run's own converter only copies the unedited GGUF."""
    weights = random_weights()
    for name, tensor in load_file(folder / "model.safetensors").items():
        _, _, layer, *kind, _ = name.split(".")
        weights[f"blk.{layer}.{GGUF_NAMES['.'.join(kind)]}.weight"] = tensor.to(rounded_to).float().numpy()
    kinds = {name: types for name, array in weights.items() if array.ndim == 2} if types else None
    write_model(destination, weights, THINKING_TEMPLATE if thinks else TEMPLATE, thinking=thinks, types=kinds)


def installed_server() -> Path:
    if folder := os.environ.get("CANDOR_LLAMA_BIN"):
        return Path(folder).expanduser() / "llama-server"
    return toolchain.cache() / toolchain.this_platform() / "bin" / "llama-server"


LLAMA_SERVER = installed_server()


TEMPLATE = (
    "{%- if not date_string is defined %}{%- set date_string = strftime_now('%d %b %Y') %}{%- endif %}"
    "<s>Today: {{ date_string }}\n{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant:{% endif %}"
)


def fake_everything_but_the_stages(monkeypatch, tiny: Path) -> None:
    """The converter copies a tiny random GGUF, and residuals are random but fixed by their prompts."""
    monkeypatch.setenv("CANDOR_LLAMA_BIN", str(LLAMA_SERVER.parent))
    monkeypatch.setattr(
        run.convert, "to_gguf", lambda folder, destination, log: shutil.copy(tiny, destination)
    )
    monkeypatch.setattr(run.convert, "quantize", lambda base, destination, *_: shutil.copy(base, destination))
    monkeypatch.setattr(run.convert, "check_targets", lambda *_: [])
    monkeypatch.setattr(run, "load_prompts", lambda spec: [f"first prompt {spec.split}", "second"])
    fake_residuals(monkeypatch)


def pinned(model: Path) -> config.RunConfig:
    return config.RunConfig(model=config.Model(source=str(model), date="01 Jan 2020"), measure=short())


# A thinking model's template that turns thinking off with an empty thinking block, as Qwen3.5's does.
# It writes past thinking too, by which llama-server knows it thinks and leaves its thinking on.
THINKING_TEMPLATE = TEMPLATE.replace(
    "{{ m['content'] }}",
    "{% if m.reasoning_content %}<think>\n{{ m.reasoning_content }}\n</think>\n\n{% endif %}"
    "{{ m['content'] }}",
).replace(
    "assistant:{% endif %}",
    "assistant:<think>\n{% if enable_thinking is defined and not enable_thinking %}"
    "{{ '\\n</think>\\n\\n' }}{% endif %}{% endif %}",
)
