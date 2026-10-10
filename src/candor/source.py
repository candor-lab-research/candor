"""Read a model's config and tensor shapes without loading its weights, from a folder or the Hub."""

import json
import struct
from pathlib import Path
from typing import Any

from huggingface_hub import get_safetensors_metadata, hf_hub_download, list_repo_files


def is_local(model: str) -> bool:
    return Path(model).expanduser().is_dir()


def read_config(model: str) -> dict[str, Any]:
    path = (
        Path(model).expanduser() / "config.json"
        if is_local(model)
        else Path(hf_hub_download(model, "config.json"))
    )
    return json.loads(path.read_text())


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    """Multimodal models keep the language model's settings under text_config."""
    return config.get("text_config", config)


def files(model: str) -> list[str]:
    if is_local(model):
        return sorted(p.name for p in Path(model).expanduser().iterdir())
    return list_repo_files(model)


def tensor_shapes(model: str) -> dict[str, list[int]]:
    """Every tensor's shape, read from the safetensors headers only."""
    if not is_local(model):
        metadata = get_safetensors_metadata(model)
        return {
            name: list(info.shape)
            for file in metadata.files_metadata.values()
            for name, info in file.tensors.items()
        }
    shapes = {}
    for path in sorted(Path(model).expanduser().glob("*.safetensors")):
        with path.open("rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        header.pop("__metadata__", None)
        shapes |= {name: info["shape"] for name, info in header.items()}
    return shapes
