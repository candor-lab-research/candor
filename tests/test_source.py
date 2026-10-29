import json
from types import SimpleNamespace

import huggingface_hub
import numpy as np
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


def test_a_hub_model_is_found_downloaded_in_the_cache_only_once_every_weight_file_is_there(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / "models--org--judge"
    snapshot = repo / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc")
    index = {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}
    (snapshot / source.INDEX).write_text(json.dumps(index))
    save_file({"a": torch.zeros(2, 3)}, snapshot / "model-00001-of-00002.safetensors")
    assert source.downloaded("org/judge") is None
    save_file({"b": torch.zeros(4)}, snapshot / "model-00002-of-00002.safetensors")
    assert source.downloaded("org/judge") == source.downloaded("org/judge", "abc") == snapshot
    assert tensor_shapes(str(snapshot)) == {"a": [2, 3], "b": [4]}
    assert source.downloaded("org/judge", "other") is None
    assert source.downloaded("org/other") is None and source.downloaded(str(snapshot)) is None


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


def test_the_first_rows_of_hub_weights_are_read_alone_at_the_pinned_commit(tmp_path, monkeypatch):
    torch.manual_seed(0)
    weights = {
        "a.weight": torch.randn(3, 40).bfloat16(),
        "b.weight": torch.randn(70, 8),
        "packed.weight": torch.zeros(2, 4, dtype=torch.int32),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(weights, tmp_path / shard)
    (tmp_path / source.INDEX).write_text(json.dumps({"weight_map": dict.fromkeys(weights, shard)}))
    asked = []

    class Hub:
        """Each file of the local folder, served a range at a time."""

        def get(self, url, headers, timeout):
            asked.append(url)
            start, end = map(int, headers["range"].removeprefix("bytes=").split("-"))
            return SimpleNamespace(content=(tmp_path / url.rsplit("/", 1)[1]).read_bytes()[start : end + 1])

    monkeypatch.setattr(source, "get_session", Hub)
    monkeypatch.setattr(source, "hf_raise_for_status", lambda response: None)
    monkeypatch.setattr(source, "hf_hub_download", lambda model, name, revision: str(tmp_path / name))
    read = source.first_rows("org/model", list(weights), 100, "abc123")
    # Whole rows holding 100 values, at least one, as their bits.
    assert np.array_equal(read["a.weight"], weights["a.weight"][:2].view(torch.int16).numpy().view(np.uint16))
    assert np.array_equal(
        read["b.weight"], weights["b.weight"][:12].view(torch.int32).numpy().view(np.uint32)
    )
    assert set(read) == {"a.weight", "b.weight"} and all("/resolve/abc123/" in url for url in asked)
    local = source.first_rows(str(tmp_path), ["a.weight"], 10_000)
    assert np.array_equal(local["a.weight"], weights["a.weight"].view(torch.int16).numpy().view(np.uint16))


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
