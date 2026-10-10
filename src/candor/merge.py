"""Write an edited copy of a model folder without going through transformers."""

import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor


def merge(source: Path, destination: Path, deltas: dict[str, tuple[Tensor, Tensor]]) -> None:
    """Copy a model folder, adding b @ a to each named weight.

    Every other file, config and tokenizer included, is copied byte for byte, so nothing gets
    rewritten the way transformers' save_pretrained does it.
    """
    destination.mkdir(parents=True)
    pending = dict(deltas)
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        if path.suffix != ".safetensors":
            shutil.copy2(path, destination / path.name)
            continue
        with safe_open(path, "pt") as f:
            metadata = f.metadata()
        tensors = load_file(path)
        for name in tensors.keys() & pending.keys():
            b, a = pending.pop(name)
            weight = tensors[name]
            tensors[name] = (weight.float() + b.float() @ a.float()).to(weight.dtype)
        save_file(tensors, destination / path.name, metadata=metadata)
    if pending:
        raise KeyError(f"{len(pending)} weights not found in {source}, e.g. {min(pending)}")
