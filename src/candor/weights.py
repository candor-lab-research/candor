"""Read weights straight from a Hugging Face model folder."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor

from .source import INDEX, weight_files


class ModelFiles:
    """Single weights by name, from the files a loader reads. Each file is opened once, as a layer's
    experts can be hundreds of weights."""

    def __init__(self, folder: Path):
        self.folder = folder
        shards = weight_files(folder)
        if not shards:
            raise FileNotFoundError(f"{folder} has no model.safetensors and no {INDEX}")
        self.shard_of: dict[str, str] = {}
        for shard in shards:
            with safe_open(folder / shard, "pt") as f:
                self.shard_of |= dict.fromkeys(f.keys(), shard)
        self._opened: dict[str, Any] = {}

    def tensor(self, name: str) -> Tensor:
        shard = self.shard_of[name]
        if shard not in self._opened:
            self._opened[shard] = safe_open(self.folder / shard, "pt")
        return self._opened[shard].get_tensor(name)

    def read(self, names: Sequence[str], stacked: bool = False) -> Tensor:
        """One weight, or when stacked these weights along a new first axis in this order, as llama.cpp's
        converter stacks a layer's experts."""
        if not stacked:
            [name] = names
            return self.tensor(name)
        return torch.stack([self.tensor(name) for name in names])
