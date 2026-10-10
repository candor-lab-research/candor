import json

import pytest
import torch
from safetensors.torch import save_file

from candor.weights import ModelFiles


def test_reads_only_the_shards_the_index_lists(tmp_path):
    torch.manual_seed(5)
    shards = {
        "model-00001-of-00002.safetensors": {"a.weight": torch.randn(4, 6)},
        "model-00002-of-00002.safetensors": {"b.weight": torch.randn(5, 4)},
    }
    for name, tensors in shards.items():
        save_file(tensors, tmp_path / name)
    weight_map = {tensor: shard for shard, tensors in shards.items() for tensor in tensors}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    save_file({"layers.0.attention.wo.weight": torch.zeros(2, 2)}, tmp_path / "consolidated.safetensors")

    files = ModelFiles(tmp_path)
    for tensors in shards.values():
        for name, tensor in tensors.items():
            assert torch.equal(files.tensor(name), tensor)
    with pytest.raises(KeyError):
        files.tensor("layers.0.attention.wo.weight")


def test_a_folder_without_weights_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelFiles(tmp_path)
