"""A tiny random Llama written straight to GGUF, so engine tests need no downloads or converter."""

from pathlib import Path

import gguf
import numpy as np

HIDDEN, HEADS, KV_HEADS, FFN, LAYERS = 32, 4, 2, 64, 2
SPECIAL = ["<unk>", "<s>", "</s>"]
BYTES = [f"<0x{b:02X}>" for b in range(256)]
WORDS = [f"w{i}" for i in range(29)]
TOKENS = SPECIAL + BYTES + WORDS
FIRST_WORD = len(SPECIAL) + len(BYTES)
TYPES = [gguf.TokenType.UNKNOWN, gguf.TokenType.CONTROL, gguf.TokenType.CONTROL]
TYPES += [gguf.TokenType.BYTE] * len(BYTES) + [gguf.TokenType.NORMAL] * len(WORDS)
# A thinking model's vocabulary: the last two words become think tags, added tokens as in Qwen3.5, so
# the vocabulary's size and the random weights stay the same.
THINK, END_THINK = "<think>", "</think>"
THINKING_TOKENS = [*TOKENS[:-2], THINK, END_THINK]
THINKING_TYPES = [*TYPES[:-2], gguf.TokenType.USER_DEFINED, gguf.TokenType.USER_DEFINED]


def random_weights(
    seed: int = 0, hidden: int = HIDDEN, ffn: int = FFN, layers: int = LAYERS
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def matrix(rows: int, columns: int) -> np.ndarray:
        return (rng.standard_normal((rows, columns)) / np.sqrt(columns)).astype(np.float32)

    ones = np.ones(hidden, np.float32)
    kv = hidden // HEADS * KV_HEADS
    weights = {
        "token_embd.weight": matrix(len(TOKENS), hidden),
        "output_norm.weight": ones,
        "output.weight": matrix(len(TOKENS), hidden),
    }
    for i in range(layers):
        weights |= {
            f"blk.{i}.attn_norm.weight": ones,
            f"blk.{i}.attn_q.weight": matrix(hidden, hidden),
            f"blk.{i}.attn_k.weight": matrix(kv, hidden),
            f"blk.{i}.attn_v.weight": matrix(kv, hidden),
            f"blk.{i}.attn_output.weight": matrix(hidden, hidden),
            f"blk.{i}.ffn_norm.weight": ones,
            f"blk.{i}.ffn_gate.weight": matrix(ffn, hidden),
            f"blk.{i}.ffn_up.weight": matrix(ffn, hidden),
            f"blk.{i}.ffn_down.weight": matrix(hidden, ffn),
        }
    return weights


def write_model(
    path: Path,
    weights: dict[str, np.ndarray],
    chat_template: str | None = None,
    thinking: bool = False,
    types: dict[str, gguf.GGMLQuantizationType] | None = None,
) -> None:
    """A llama of the weights' size, each tensor stored as types gives it or as its array is."""
    tokens, token_types = (THINKING_TOKENS, THINKING_TYPES) if thinking else (TOKENS, TYPES)
    hidden = weights["token_embd.weight"].shape[1]
    writer = gguf.GGUFWriter(path, "llama")
    writer.add_context_length(256)
    writer.add_embedding_length(hidden)
    writer.add_block_count(sum(name.endswith(".attn_norm.weight") for name in weights))
    writer.add_feed_forward_length(weights["blk.0.ffn_up.weight"].shape[0])
    writer.add_head_count(HEADS)
    writer.add_head_count_kv(KV_HEADS)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_dimension_count(hidden // HEADS)
    writer.add_vocab_size(len(TOKENS))
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * len(tokens))
    writer.add_token_types(token_types)
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
