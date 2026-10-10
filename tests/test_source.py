import json

import torch
from safetensors.torch import save_file

from candor.source import read_json, tensor_shapes


def test_shapes_follow_the_index_and_skip_other_weight_files(tmp_path):
    save_file({"a.weight": torch.zeros(2, 3)}, tmp_path / "model-00001-of-00001.safetensors")
    save_file({"layers.0.wo.weight": torch.zeros(5, 5)}, tmp_path / "consolidated.safetensors")
    index = {"weight_map": {"a.weight": "model-00001-of-00001.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    assert tensor_shapes(str(tmp_path)) == {"a.weight": [2, 3]}


def test_read_json_returns_none_for_a_missing_file(tmp_path):
    assert read_json(str(tmp_path), "tokenizer_config.json") is None
