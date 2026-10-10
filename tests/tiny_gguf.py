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


def random_weights(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def matrix(rows: int, columns: int) -> np.ndarray:
        return (rng.standard_normal((rows, columns)) / np.sqrt(columns)).astype(np.float32)

    ones = np.ones(HIDDEN, np.float32)
    kv = HIDDEN // HEADS * KV_HEADS
    weights = {
        "token_embd.weight": matrix(len(TOKENS), HIDDEN),
        "output_norm.weight": ones,
        "output.weight": matrix(len(TOKENS), HIDDEN),
    }
    for i in range(LAYERS):
        weights |= {
            f"blk.{i}.attn_norm.weight": ones,
            f"blk.{i}.attn_q.weight": matrix(HIDDEN, HIDDEN),
            f"blk.{i}.attn_k.weight": matrix(kv, HIDDEN),
            f"blk.{i}.attn_v.weight": matrix(kv, HIDDEN),
            f"blk.{i}.attn_output.weight": matrix(HIDDEN, HIDDEN),
            f"blk.{i}.ffn_norm.weight": ones,
            f"blk.{i}.ffn_gate.weight": matrix(FFN, HIDDEN),
            f"blk.{i}.ffn_up.weight": matrix(FFN, HIDDEN),
            f"blk.{i}.ffn_down.weight": matrix(HIDDEN, FFN),
        }
    return weights


def write_model(path: Path, weights: dict[str, np.ndarray], chat_template: str | None = None) -> None:
    writer = gguf.GGUFWriter(path, "llama")
    writer.add_context_length(256)
    writer.add_embedding_length(HIDDEN)
    writer.add_block_count(LAYERS)
    writer.add_feed_forward_length(FFN)
    writer.add_head_count(HEADS)
    writer.add_head_count_kv(KV_HEADS)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_dimension_count(HIDDEN // HEADS)
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
        writer.add_tensor(name, tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
