"""Read a model's config and tensor shapes without loading its weights, from a folder or the Hub."""

import json
import struct
from pathlib import Path
from typing import Any

from huggingface_hub import get_safetensors_metadata, hf_hub_download, try_to_load_from_cache
from huggingface_hub.errors import RemoteEntryNotFoundError

INDEX = "model.safetensors.index.json"


def is_local(model: str) -> bool:
    """Local folders are given by their full path, which is how init stores them, so a Hugging Face id
    is never mistaken for a folder that happens to share its name."""
    return Path(model).expanduser().is_absolute()


def read_json(model: str, name: str, revision: str | None = None) -> dict[str, Any] | None:
    """A JSON file from the model, or None if it has none by that name."""
    if is_local(model):
        path = Path(model).expanduser() / name
        return json.loads(path.read_text()) if path.exists() else None
    try:
        return json.loads(Path(hf_hub_download(model, name, revision=revision)).read_text())
    except RemoteEntryNotFoundError:
        return None


def read_config(model: str, revision: str | None = None) -> dict[str, Any]:
    config = read_json(model, "config.json", revision)
    if config is None:
        raise FileNotFoundError(f"{model} has no config.json")
    return config


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    """Multimodal models keep the language model's settings under text_config."""
    return config.get("text_config", config)


def sizes(config: dict[str, Any]) -> tuple[int, int] | None:
    """The hidden size and layer count, or None when the config doesn't give both."""
    text = text_config(config)
    hidden, layers = text.get("hidden_size"), text.get("num_hidden_layers")
    return (hidden, layers) if isinstance(hidden, int) and isinstance(layers, int) else None


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


def downloaded(model: str, revision: str | None = None) -> Path | None:
    """The folder of the Hub's cache that holds every weight file of a Hugging Face model at this
    commit, found without the network; None for a local folder, or while any is missing."""
    if is_local(model):
        return None
    for name in (INDEX, "model.safetensors"):
        found = try_to_load_from_cache(model, name, revision=revision)
        if isinstance(found, str):
            folder = Path(found).parent
            return folder if all((folder / file).exists() for file in weight_files(folder)) else None
    return None


def local_headers(folder: Path) -> dict[str, dict[str, Any]]:
    """Every tensor's dtype and shape in the weight files a loader reads, from the headers only."""
    headers = {}
    for name in weight_files(folder):
        with (folder / name).open("rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        header.pop("__metadata__", None)
        headers |= header
    return headers


def tensor_headers(model: str, revision: str | None = None) -> dict[str, dict[str, Any]]:
    """Every tensor's dtype and shape, read from the safetensors headers only."""
    if not is_local(model):
        metadata = get_safetensors_metadata(model, revision=revision)
        return {
            name: {"dtype": info.dtype, "shape": list(info.shape)}
            for file in metadata.files_metadata.values()
            for name, info in file.tensors.items()
        }
    return local_headers(Path(model).expanduser())


def tensor_shapes(model: str, revision: str | None = None) -> dict[str, list[int]]:
    """Every tensor's shape, read from the safetensors headers only."""
    return {name: info["shape"] for name, info in tensor_headers(model, revision).items()}


def float_type(model: str, revision: str | None = None) -> str:
    """The GGUF type a model converts to without rounding: F16 when every weight matrix is stored
    as F16, otherwise BF16, which also holds the range of F32 weights."""
    headers = tensor_headers(model, revision).values()
    stored = {info["dtype"] for info in headers if len(info["shape"]) > 1}
    if not stored:
        raise ValueError(f"{model} has no weights to read their type from")
    return "F16" if stored == {"F16"} else "BF16"
