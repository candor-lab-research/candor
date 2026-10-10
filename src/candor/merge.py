"""Write an edited copy of a model folder without going through transformers."""

import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from .source import INDEX, local_headers, weight_files
from .targets import QUANTIZATION

EDITABLE = {"BF16", "F16", "F32"}
OTHER_WEIGHT_FORMATS = {".bin", ".pt", ".pth", ".ckpt", ".gguf", ".h5", ".msgpack"}


def _check(source: Path, names: set[str]) -> None:
    """Refuse before writing anything if a target is missing or isn't plain floating point."""
    headers = local_headers(source)
    missing = names - headers.keys()
    if missing:
        raise KeyError(f"{len(missing)} weights not found in {source}, e.g. {min(missing)}")
    for name in sorted(names):
        if headers[name]["dtype"] not in EDITABLE:
            raise ValueError(
                f"{name} is {headers[name]['dtype']}, which merge can't edit; use an unquantised copy"
            )
        module = name.removesuffix(".weight")
        if any(f"{module}.{suffix}" in headers for suffix in QUANTIZATION):
            raise ValueError(
                f"{name} has quantisation scales, which merge can't edit; use an unquantised copy"
            )


def merge(source: Path, destination: Path, deltas: dict[str, tuple[Tensor, Tensor]]) -> list[str]:
    """Copy a model folder, adding b @ a to each named weight, and return the files left out.

    Every other file, config and tokenizer included, is copied byte for byte, so nothing gets
    rewritten the way transformers' save_pretrained does it. Weight files the model's index doesn't
    list, and weights in other formats, are left out, so the copy never ships an unedited duplicate.
    """
    shards = weight_files(source)
    if not shards:
        raise FileNotFoundError(f"{source} has no model.safetensors and no {INDEX}")
    _check(source, set(deltas))
    destination.mkdir(parents=True)
    pending = dict(deltas)
    left_out = []
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        if path.suffix in OTHER_WEIGHT_FORMATS:
            left_out.append(path.name)
            continue
        if path.suffix != ".safetensors":
            shutil.copy2(path, destination / path.name)
            continue
        if path.name not in shards:
            left_out.append(path.name)
            continue
        with safe_open(path, "pt") as f:
            metadata = f.metadata()
        tensors = load_file(path)
        for name in tensors.keys() & pending.keys():
            b, a = pending.pop(name)
            weight = tensors[name]
            tensors[name] = (weight.float() + b.float() @ a.float()).to(weight.dtype)
        save_file(tensors, destination / path.name, metadata=metadata)
    return left_out
