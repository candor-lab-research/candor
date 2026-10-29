"""A tiny random model that transformers can load only with the Python files it ships, like K2-Horizon.

Its code subclasses Llama under a model type transformers doesn't know, and its config names the
llama.cpp architecture, so candor's checks see an ordinary Llama.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
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
# Code written for transformers 4, as Spark-X's is: it lists the weights tied to its input embedding,
# which it names as Spark-X does, and makes its mask with the arguments transformers 4 took.
TRANSFORMERS_4 = """import torch
from transformers import LlamaForCausalLM, LlamaModel
from transformers.masking_utils import create_causal_mask

from .configuration_tiny import TinyConfig


class TinyModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.embedding = self.embed_tokens
        del self.embed_tokens

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        embeds = self.embedding(input_ids)
        mask = create_causal_mask(
            config=self.config, input_embeds=embeds, attention_mask=attention_mask,
            cache_position=torch.arange(embeds.shape[1]), past_key_values=None, position_ids=position_ids,
        )
        return super().forward(inputs_embeds=embeds, attention_mask=mask, position_ids=position_ids, **kwargs)


class TinyForCausalLM(LlamaForCausalLM):
    config_class = TinyConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = TinyModel(config)
        self.post_init()
"""


def write_custom_code_model(
    folder: Path, hidden: int = 64, layers: int = 2, transformers_4: bool = False
) -> Path:
    torch.manual_seed(3)
    shape = LlamaConfig(
        vocab_size=96,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=transformers_4,
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
    (folder / "modeling_tiny.py").write_text(TRANSFORMERS_4 if transformers_4 else MODELING)
    if transformers_4:
        weights = load_file(folder / "model.safetensors")
        weights["model.embedding.weight"] = weights.pop("model.embed_tokens.weight")
        save_file(weights, folder / "model.safetensors", metadata={"format": "pt"})
    tiny = tokenizer(WITH_BOS, adds_bos=True)
    tiny.pad_token = "[UNK]"
    tiny.save_pretrained(folder)
    return folder
