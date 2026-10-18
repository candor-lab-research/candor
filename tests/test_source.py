import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from candor import source
from candor.source import is_local, read_json, sizes, tensor_shapes


def test_shapes_follow_the_index_and_skip_other_weight_files(tmp_path):
    save_file({"a.weight": torch.zeros(2, 3)}, tmp_path / "model-00001-of-00001.safetensors")
    save_file({"layers.0.wo.weight": torch.zeros(5, 5)}, tmp_path / "consolidated.safetensors")
    index = {"weight_map": {"a.weight": "model-00001-of-00001.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    assert tensor_shapes(str(tmp_path)) == {"a.weight": [2, 3]}


def test_hub_shapes_and_float_type_are_read_at_the_pinned_commit(monkeypatch):
    asked = []
    stored = {
        "a.weight": SimpleNamespace(dtype="F16", shape=[4, 2]),
        "norm": SimpleNamespace(dtype="F32", shape=[2]),
    }

    def metadata(model, revision):
        asked.append((model, revision))
        return SimpleNamespace(files_metadata={"model.safetensors": SimpleNamespace(tensors=stored)})

    monkeypatch.setattr(source, "get_safetensors_metadata", metadata)
    assert tensor_shapes("org/model", "abc123") == {"a.weight": [4, 2], "norm": [2]}
    assert asked == [("org/model", "abc123")]
    # Only the weight matrices decide it: F16 when every one is F16, as the conversion keeps them.
    assert source.float_type("org/model", "abc123") == "F16"
    stored["b.weight"] = SimpleNamespace(dtype="BF16", shape=[4, 2])
    assert source.float_type("org/model", "abc123") == "BF16"


def test_read_json_returns_none_for_a_missing_file(tmp_path):
    assert read_json(str(tmp_path), "tokenizer_config.json") is None


def test_only_a_full_path_is_a_local_folder(tmp_path, monkeypatch):
    (tmp_path / "org" / "model").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert not is_local("org/model")
    assert is_local(str(tmp_path / "org" / "model"))
    assert is_local("~/org/model")
    assert is_local(str(tmp_path / "gone"))


@pytest.mark.parametrize(
    ("config", "found"),
    [
        ({"hidden_size": 64, "num_hidden_layers": 2}, (64, 2)),
        ({"text_config": {"hidden_size": 64, "num_hidden_layers": 2}}, (64, 2)),
        ({"text_config": {}, "hidden_size": 64, "num_hidden_layers": 2}, None),
        ({"n_embd": 64, "n_layer": 2}, None),
        ({"hidden_size": "64", "num_hidden_layers": 2}, None),
    ],
)
def test_sizes_come_from_the_language_models_own_settings(config, found):
    assert sizes(config) == found
