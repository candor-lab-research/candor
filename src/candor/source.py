"""Read a model's config and tensor shapes without loading its weights, from a folder or the Hub."""

import json
import struct
from pathlib import Path
from typing import Any

from huggingface_hub import get_safetensors_metadata, hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError

INDEX = "model.safetensors.index.json"


def is_local(model: str) -> bool:
    return Path(model).expanduser().is_dir()


def read_json(model: str, name: str) -> dict[str, Any] | None:
    """A JSON file from the model, or None if it has none by that name."""
    if is_local(model):
        path = Path(model).expanduser() / name
        return json.loads(path.read_text()) if path.exists() else None
    try:
        return json.loads(Path(hf_hub_download(model, name)).read_text())
    except RemoteEntryNotFoundError:
        return None


def read_config(model: str) -> dict[str, Any]:
    config = read_json(model, "config.json")
    if config is None:
        raise FileNotFoundError(f"{model} has no config.json")
    return config


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    """Multimodal models keep the language model's settings under text_config."""
    return config.get("text_config", config)


def weight_files(folder: Path) -> list[str]:
    """The safetensors files a loader reads: those in the index, or the single model.safetensors.

    Other .safetensors files, such as Mistral's consolidated.safetensors, are a different format.
    """
    index = folder / INDEX
    if index.exists():
        return sorted(set(json.loads(index.read_text())["weight_map"].values()))
    if (folder / "model.safetensors").exists():
        return ["model.safetensors"]
    return []


def local_headers(folder: Path) -> dict[str, dict[str, Any]]:
    """Every tensor's dtype and shape in the weight files a loader reads, from the headers only."""
    headers = {}
    for name in weight_files(folder):
        with (folder / name).open("rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        header.pop("__metadata__", None)
        headers |= header
    return headers


def tensor_shapes(model: str) -> dict[str, list[int]]:
    """Every tensor's shape, read from the safetensors headers only."""
    if not is_local(model):
        metadata = get_safetensors_metadata(model)
        return {
            name: list(info.shape)
            for file in metadata.files_metadata.values()
            for name, info in file.tensors.items()
        }
    return {name: info["shape"] for name, info in local_headers(Path(model).expanduser()).items()}
