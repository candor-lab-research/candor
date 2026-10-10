"""Read weights straight from a Hugging Face model folder, and name them the way GGUF does."""

import json
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open
from torch import Tensor


@dataclass(frozen=True)
class Target:
    """A weight matrix that writes into the residual stream, named in both formats."""

    component: str
    hf: str
    gguf: str

    def names(self, layer: int) -> tuple[str, str]:
        return self.hf.format(layer), self.gguf.format(layer)


DENSE = (
    Target("attn", "model.layers.{}.self_attn.o_proj.weight", "blk.{}.attn_output.weight"),
    Target("mlp", "model.layers.{}.mlp.down_proj.weight", "blk.{}.ffn_down.weight"),
)


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
