"""Read a model's config, tensor shapes and the first rows of a few weights without loading the rest,
from a folder or the Hub."""

import json
import struct
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import (
    constants,
    get_safetensors_metadata,
    get_session,
    hf_hub_download,
    hf_hub_url,
    try_to_load_from_cache,
)
from huggingface_hub.errors import RemoteEntryNotFoundError
from huggingface_hub.utils import build_hf_headers, hf_raise_for_status

INDEX = "model.safetensors.index.json"
# The bytes first read of a safetensors file: the whole header of most, so one read finds it.
HEADER_BYTES = 100_000
# Unquantised weights' values as their raw bits, by safetensors dtype.
BITS = {"BF16": np.uint16, "F16": np.uint16, "F32": np.uint32}
# Weights quantised to 4 bits share a scale over blocks of 32 or more values along a row, so once saved
# unquantised each such block holds at most 16 values; weights never quantised hold nearly all 32.
BLOCK, FEWEST = 32, 16


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


def _read(model: str, file: str, start: int, length: int, revision: str | None) -> bytes:
    """length bytes of one of the model's files from start, without reading the rest of it."""
    if is_local(model):
        with (Path(model).expanduser() / file).open("rb") as f:
            f.seek(start)
            return f.read(length)
    url = hf_hub_url(model, file, revision=revision)
    headers = build_hf_headers() | {"range": f"bytes={start}-{start + length - 1}"}
    response = get_session().get(url, headers=headers, timeout=constants.HF_HUB_DOWNLOAD_TIMEOUT)
    hf_raise_for_status(response)
    return response.content


def _header(model: str, file: str, revision: str | None) -> tuple[int, dict[str, Any]]:
    """Where a safetensors file's data starts, and its header."""
    start = _read(model, file, 0, HEADER_BYTES, revision)
    size = struct.unpack("<Q", start[:8])[0]
    header = start[8 : 8 + size] if 8 + size <= len(start) else _read(model, file, 8, size, revision)
    return 8 + size, json.loads(header)


def _first_rows(model: str, file: str, name: str, values: int, revision: str | None) -> np.ndarray | None:
    """One weight's first rows, as first_rows reads them; None for a type other than BITS."""
    start, header = _header(model, file, revision)
    info = header[name]
    if info["dtype"] not in BITS:
        return None
    bits = np.dtype(BITS[info["dtype"]])
    row = info["shape"][-1]
    begin, end = info["data_offsets"]
    length = min(max(1, values // row) * row * bits.itemsize, end - begin)
    return np.frombuffer(_read(model, file, start + begin, length, revision), bits).reshape(-1, row)


def first_rows(
    model: str, names: list[str], values: int, revision: str | None = None
) -> dict[str, np.ndarray]:
    """The first rows of each weight named, as many as hold this many values and at least one, as their
    raw bits, each read alone from its file, all at once. A weight stored as a type other than BITS is
    left out."""
    index = read_json(model, INDEX, revision)
    files = [index["weight_map"][name] if index else "model.safetensors" for name in names]
    with ThreadPoolExecutor() as pool:
        read = list(pool.map(partial(_first_rows, model, values=values, revision=revision), files, names))
    return {name: rows for name, rows in zip(names, read, strict=True) if rows is not None}


def quantised_before(rows: np.ndarray) -> bool:
    """Whether a weight's rows look quantised before and saved unquantised: most of their blocks of
    BLOCK values hold at most FEWEST values. Blocks of one value, as zeroed weights hold, tell nothing."""
    blocks = np.sort(rows[:, : rows.shape[1] // BLOCK * BLOCK].reshape(-1, BLOCK), axis=1)
    distinct = 1 + np.count_nonzero(np.diff(blocks, axis=1), axis=1)
    told = distinct[distinct > 1]
    return told.size > 0 and bool(np.median(told) <= FEWEST)


def float_type(model: str, revision: str | None = None) -> str:
    """The GGUF type a model converts to without rounding: F16 when every weight matrix is stored
    as F16, otherwise BF16, which also holds the range of F32 weights."""
    headers = tensor_headers(model, revision).values()
    stored = {info["dtype"] for info in headers if len(info["shape"]) > 1}
    if not stored:
        raise ValueError(f"{model} has no weights to read their type from")
    return "F16" if stored == {"F16"} else "BF16"
