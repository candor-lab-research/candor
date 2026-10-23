"""LoRA adapters in GGUF, one per edit target, so llama.cpp can set each one's strength per request."""

from pathlib import Path

import gguf
from torch import Tensor


def write_adapter(path: Path, arch: str, factors: dict[str, tuple[Tensor, Tensor]]) -> None:
    """factors maps a GGUF weight name to its (b, a): (out, rank) and (rank, in), or for a layer's
    stacked experts (experts, out, rank) and (experts, rank, in), which llama.cpp applies to each
    expert it routes to.

    alpha is set to the rank, so the scale llama.cpp applies is exactly the strength in the request.
    """
    ranks = {a.shape[-2] for _, a in factors.values()}
    if len(ranks) != 1:
        raise ValueError(f"one adapter can't mix ranks {sorted(ranks)}")
    writer = gguf.GGUFWriter(path, arch)
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, float(ranks.pop()))
    for name, (b, a) in factors.items():
        writer.add_tensor(f"{name}.lora_a", a.float().cpu().numpy())
        writer.add_tensor(f"{name}.lora_b", b.float().cpu().numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
