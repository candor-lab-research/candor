"""A tiny random model that transformers can load only with the Python files it ships, like K2-Horizon.

Its code subclasses Llama under a model type transformers doesn't know, and its config names the
llama.cpp architecture, so candor's checks see an ordinary Llama.
"""

import json
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from .tiny_tokenizer import WITH_BOS, tokenizer

CONFIGURATION = """from transformers import LlamaConfig


class TinyConfig(LlamaConfig):
    model_type = "candor_tiny"
"""
MODELING = """from transformers import LlamaForCausalLM

from .configuration_tiny import TinyConfig


class TinyForCausalLM(LlamaForCausalLM):
    config_class = TinyConfig
"""


def write_custom_code_model(folder: Path, hidden: int = 64, layers: int = 2) -> Path:
    torch.manual_seed(3)
    shape = LlamaConfig(
        vocab_size=96,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    LlamaForCausalLM(shape).save_pretrained(folder)
    model_config = json.loads((folder / "config.json").read_text())
    model_config |= {
        "model_type": "candor_tiny",
        "auto_map": {
            "AutoConfig": "configuration_tiny.TinyConfig",
            "AutoModelForCausalLM": "modeling_tiny.TinyForCausalLM",
        },
    }
    (folder / "config.json").write_text(json.dumps(model_config))
    (folder / "configuration_tiny.py").write_text(CONFIGURATION)
    (folder / "modeling_tiny.py").write_text(MODELING)
    tiny = tokenizer(WITH_BOS, adds_bos=True)
    tiny.pad_token = "[UNK]"
    tiny.save_pretrained(folder)
    return folder
