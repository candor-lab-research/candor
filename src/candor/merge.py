"""Write an edited copy of a model folder without going through transformers."""

import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from .source import INDEX, weight_files

FLOAT8 = {torch.float8_e4m3fn, torch.float8_e5m2}


def merge(source: Path, destination: Path, deltas: dict[str, tuple[Tensor, Tensor]]) -> list[str]:
    """Copy a model folder, adding b @ a to each named weight, and return the files left out.

    Every other file, config and tokenizer included, is copied byte for byte, so nothing gets
    rewritten the way transformers' save_pretrained does it. Safetensors files that the model's
    index doesn't list are left out, so the copy never ships an unedited duplicate of the weights.
    """
    shards = weight_files(source)
    if not shards:
        raise FileNotFoundError(f"{source} has no model.safetensors and no {INDEX}")
    destination.mkdir(parents=True)
    pending = dict(deltas)
    left_out = []
    for path in sorted(source.iterdir()):
        if not path.is_file():
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
            if weight.dtype in FLOAT8:
                raise ValueError(f"{name} is FP8 with separate scales, which merge can't edit correctly")
            tensors[name] = (weight.float() + b.float() @ a.float()).to(weight.dtype)
        save_file(tensors, destination / path.name, metadata=metadata)
    if pending:
        raise KeyError(f"{len(pending)} weights not found in {source}, e.g. {min(pending)}")
    return left_out
