"""A tiny random Qwen3-MoE written straight to GGUF, its experts stacked as llama.cpp's converter
stacks them, and the same weights as a Hugging Face folder that stores each expert apart."""

import json
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .tiny_gguf import HEADS, HIDDEN, KV_HEADS, TOKENS, TYPES

LAYERS, EXPERTS, USED, EXPERT_FFN = 2, 12, 4, 16
HEAD = HIDDEN // HEADS
# Each layer tensor's name in the Hugging Face folder, by its GGUF kind; the experts' are per expert.
HF_NAMES = {
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "ffn_norm": "post_attention_layernorm",
    "ffn_gate_inp": "mlp.gate",
}
EXPERT_NAMES = {"ffn_gate_exps": "gate_proj", "ffn_up_exps": "up_proj", "ffn_down_exps": "down_proj"}


def random_moe_weights(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def matrix(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32)

    ones = np.ones(HIDDEN, np.float32)
    weights = {
        "token_embd.weight": matrix(len(TOKENS), HIDDEN),
        "output_norm.weight": ones,
        "output.weight": matrix(len(TOKENS), HIDDEN),
    }
    for i in range(LAYERS):
        weights |= {
            f"blk.{i}.attn_norm.weight": ones,
            f"blk.{i}.attn_q.weight": matrix(HEADS * HEAD, HIDDEN),
            f"blk.{i}.attn_k.weight": matrix(KV_HEADS * HEAD, HIDDEN),
            f"blk.{i}.attn_v.weight": matrix(KV_HEADS * HEAD, HIDDEN),
            f"blk.{i}.attn_output.weight": matrix(HIDDEN, HEADS * HEAD),
            f"blk.{i}.attn_q_norm.weight": np.ones(HEAD, np.float32),
            f"blk.{i}.attn_k_norm.weight": np.ones(HEAD, np.float32),
            f"blk.{i}.ffn_norm.weight": ones,
            f"blk.{i}.ffn_gate_inp.weight": matrix(EXPERTS, HIDDEN),
            f"blk.{i}.ffn_gate_exps.weight": matrix(EXPERTS, EXPERT_FFN, HIDDEN),
            f"blk.{i}.ffn_up_exps.weight": matrix(EXPERTS, EXPERT_FFN, HIDDEN),
            f"blk.{i}.ffn_down_exps.weight": matrix(EXPERTS, HIDDEN, EXPERT_FFN),
        }
    return weights


def write_moe_model(
    path: Path,
    weights: dict[str, np.ndarray],
    chat_template: str | None = None,
    types: dict[str, gguf.GGMLQuantizationType] | None = None,
) -> None:
    """A Qwen3-MoE of the weights' size, each tensor stored as types gives it or as its array is."""
    writer = gguf.GGUFWriter(path, "qwen3moe")
    writer.add_context_length(256)
    writer.add_embedding_length(HIDDEN)
    writer.add_block_count(LAYERS)
    writer.add_feed_forward_length(EXPERT_FFN)
    writer.add_expert_feed_forward_length(EXPERT_FFN)
    writer.add_expert_count(EXPERTS)
    writer.add_expert_used_count(USED)
    writer.add_head_count(HEADS)
    writer.add_head_count_kv(KV_HEADS)
    writer.add_key_length(HEAD)
    writer.add_value_length(HEAD)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_dimension_count(HEAD)
    writer.add_vocab_size(len(TOKENS))
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    writer.add_tokenizer_model("llama")
    writer.add_token_list(TOKENS)
    writer.add_token_scores([0.0] * len(TOKENS))
    writer.add_token_types(TYPES)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    if chat_template:
        writer.add_chat_template(chat_template)
    for name, tensor in weights.items():
        if types and name in types:
            writer.add_tensor(name, gguf.quants.quantize(tensor, types[name]), raw_dtype=types[name])
        else:
            writer.add_tensor(name, tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def write_moe_folder(folder: Path, weights: dict[str, np.ndarray], fused: bool = False) -> Path:
    """The GGUF's layer weights as a Hugging Face Qwen3-MoE folder, each expert stored apart, or each
    layer's experts held in one tensor of each kind, [experts, out, in], when fused."""
    folder.mkdir()
    tensors = {}
    for name, weight in weights.items():
        if not name.startswith("blk."):
            continue
        _, layer, kind, _ = name.split(".")
        prefix = f"model.layers.{layer}"
        if kind in EXPERT_NAMES and fused:
            tensors[f"{prefix}.mlp.experts.{EXPERT_NAMES[kind]}"] = torch.from_numpy(weight.copy())
        elif kind in EXPERT_NAMES:
            for expert in range(EXPERTS):
                tensors[f"{prefix}.mlp.experts.{expert}.{EXPERT_NAMES[kind]}.weight"] = torch.from_numpy(
                    weight[expert].copy()
                )
        else:
            tensors[f"{prefix}.{HF_NAMES[kind]}.weight"] = torch.from_numpy(weight.copy())
    save_file(tensors, folder / "model.safetensors")
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD,
        "num_experts": EXPERTS,
        "num_experts_per_tok": USED,
        "moe_intermediate_size": EXPERT_FFN,
        "max_position_embeddings": 256,
    }
    (folder / "config.json").write_text(json.dumps(config))
    return folder


def moe_gguf_from_folder(
    folder: Path,
    destination: Path,
    chat_template: str | None = None,
    kind: gguf.GGMLQuantizationType | None = None,
) -> None:
    """The tiny MoE's GGUF with the layer weights of a Hugging Face folder, each layer's experts stacked
    in expert order, as the converter writes them, its matrices stored as kind when given."""
    weights = random_moe_weights()
    kinds = {hf: kind for kind, hf in HF_NAMES.items()}
    experts = {hf: kind for kind, hf in EXPERT_NAMES.items()}
    stacked: dict[str, dict[int, np.ndarray]] = {}
    for name, tensor in load_file(folder / "model.safetensors").items():
        if name.endswith(tuple(f"experts.{hf}" for hf in experts)):
            _, _, layer, _, _, projection = name.split(".")
            weights[f"blk.{layer}.{experts[projection]}.weight"] = tensor.float().numpy()
            continue
        _, _, layer, *rest, _ = name.split(".")
        if rest[:2] == ["mlp", "experts"]:
            stacked.setdefault(f"blk.{layer}.{experts[rest[3]]}.weight", {})[int(rest[2])] = (
                tensor.float().numpy()
            )
        else:
            weights[f"blk.{layer}.{kinds['.'.join(rest)]}.weight"] = tensor.float().numpy()
    for name, by_expert in stacked.items():
        weights[name] = np.stack([by_expert[i] for i in range(EXPERTS)])
    types = {name: kind for name, array in weights.items() if array.ndim >= 2} if kind else None
    write_moe_model(destination, weights, chat_template, types)
