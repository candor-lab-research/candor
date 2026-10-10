"""Read weights straight from a Hugging Face model folder."""

from pathlib import Path

from safetensors import safe_open
from torch import Tensor

from .source import INDEX, weight_files


class ModelFiles:
    """Single weights by name, from the files a loader reads."""

    def __init__(self, folder: Path):
        self.folder = folder
        shards = weight_files(folder)
        if not shards:
            raise FileNotFoundError(f"{folder} has no model.safetensors and no {INDEX}")
        self.shard_of: dict[str, str] = {}
        for shard in shards:
            with safe_open(folder / shard, "pt") as f:
                self.shard_of |= dict.fromkeys(f.keys(), shard)

    def tensor(self, name: str) -> Tensor:
        with safe_open(self.folder / self.shard_of[name], "pt") as f:
            return f.get_tensor(name)
