"""LoRA adapters in GGUF, one per layer and component, so llama.cpp can set each layer's strength
per request."""

from collections.abc import Sequence
from pathlib import Path

import gguf
from torch import Tensor

from .edit import RowNorm, rank_one
from .weights import ModelFiles, Target


def architecture(model_gguf: Path) -> str:
    """The architecture name a GGUF model declares, which its adapters must repeat."""
    return gguf.GGUFReader(model_gguf).fields["general.architecture"].contents()


def write_adapter(path: Path, arch: str, factors: dict[str, tuple[Tensor, Tensor]]) -> None:
    """factors maps a GGUF weight name to its (b, a).

    alpha is set to the rank, so the scale llama.cpp applies is exactly the strength in the request.
    """
    ranks = {a.shape[0] for _, a in factors.values()}
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


def layer_adapter_paths(folder: Path, targets: Sequence[Target], layers: int) -> dict[tuple[str, int], Path]:
    return {
        (target.component, layer): folder / f"{target.component}.{layer}.gguf"
        for layer in range(layers)
        for target in targets
    }


def write_layer_adapters(
    folder: Path,
    arch: str,
    files: ModelFiles,
    targets: Sequence[Target],
    directions: Tensor,
    row_norm: RowNorm,
) -> dict[tuple[str, int], Path]:
    """One adapter per target and layer. directions holds the direction to remove at each layer."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = layer_adapter_paths(folder, targets, len(directions))
    for layer, direction in enumerate(directions):
        for target in targets:
            hf_name, gguf_name = target.names(layer)
            factors = rank_one(files.tensor(hf_name), direction, row_norm)
            write_adapter(paths[target.component, layer], arch, {gguf_name: factors})
    return paths
