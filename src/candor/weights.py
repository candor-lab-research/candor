"""Read weights straight from a Hugging Face model folder."""

import json
from pathlib import Path

from safetensors import safe_open
from torch import Tensor


class ModelFiles:
    def __init__(self, folder: Path):
        self.folder = folder
        index = folder / "model.safetensors.index.json"
        if index.exists():
            self.shard_of: dict[str, str] = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(folder / "model.safetensors", "pt") as f:
                self.shard_of = dict.fromkeys(f.keys(), "model.safetensors")

    def tensor(self, name: str) -> Tensor:
        with safe_open(self.folder / self.shard_of[name], "pt") as f:
            return f.get_tensor(name)
