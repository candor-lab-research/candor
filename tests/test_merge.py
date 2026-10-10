import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from candor.merge import merge


@pytest.fixture
def model_folder(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    torch.manual_seed(3)
    shards = {
        "model-00001-of-00002.safetensors": {"a.weight": torch.randn(4, 6).bfloat16()},
        "model-00002-of-00002.safetensors": {"b.weight": torch.randn(5, 4).bfloat16()},
    }
    for name, tensors in shards.items():
        save_file(tensors, folder / name, metadata={"format": "pt"})
    weight_map = {tensor: shard for shard, tensors in shards.items() for tensor in tensors}
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (folder / "config.json").write_text('{"rope_theta": 500000.0}\n')
    (folder / "tokenizer_config.json").write_text('{"tokenizer_class": "PreTrainedTokenizerFast"}\n')
    return folder


def test_adds_the_delta_and_leaves_everything_else_alone(model_folder, tmp_path):
    b, a = torch.randn(5, 1), torch.randn(1, 4)
    merge(model_folder, tmp_path / "merged", {"b.weight": (b, a)})
    original = load_file(model_folder / "model-00002-of-00002.safetensors")["b.weight"]
    edited = load_file(tmp_path / "merged" / "model-00002-of-00002.safetensors")["b.weight"]
    assert edited.dtype == torch.bfloat16
    torch.testing.assert_close(edited, (original.float() + b @ a).bfloat16())
    untouched = "model-00001-of-00002.safetensors"
    assert torch.equal(
        load_file(model_folder / untouched)["a.weight"],
        load_file(tmp_path / "merged" / untouched)["a.weight"],
    )


def test_copies_non_weight_files_byte_for_byte(model_folder, tmp_path):
    merge(model_folder, tmp_path / "merged", {})
    for name in ("config.json", "tokenizer_config.json", "model.safetensors.index.json"):
        assert (tmp_path / "merged" / name).read_bytes() == (model_folder / name).read_bytes()


def test_keeps_safetensors_metadata(model_folder, tmp_path):
    merge(model_folder, tmp_path / "merged", {})
    with safe_open(tmp_path / "merged" / "model-00001-of-00002.safetensors", "pt") as f:
        assert f.metadata() == {"format": "pt"}


def test_unknown_weight_is_an_error_before_anything_is_written(model_folder, tmp_path):
    with pytest.raises(KeyError):
        merge(model_folder, tmp_path / "merged", {"missing.weight": (torch.zeros(1, 1), torch.zeros(1, 1))})
    assert not (tmp_path / "merged").exists()


def test_weight_files_outside_the_index_are_left_out(model_folder, tmp_path):
    save_file({"layers.0.attention.wo.weight": torch.zeros(2, 2)}, model_folder / "consolidated.safetensors")
    left_out = merge(model_folder, tmp_path / "merged", {})
    assert left_out == ["consolidated.safetensors"]
    assert not (tmp_path / "merged" / "consolidated.safetensors").exists()


def test_fp8_weights_are_refused_before_anything_is_written(tmp_path):
    folder = tmp_path / "fp8"
    folder.mkdir()
    save_file({"a.weight": torch.zeros(4, 6).to(torch.float8_e4m3fn)}, folder / "model.safetensors")
    with pytest.raises(ValueError, match="F8_E4M3"):
        merge(folder, tmp_path / "merged", {"a.weight": (torch.zeros(4, 1), torch.zeros(1, 6))})
    assert not (tmp_path / "merged").exists()


def test_scaled_int8_weights_are_refused_before_anything_is_written(tmp_path):
    folder = tmp_path / "int8"
    folder.mkdir()
    tensors = {"a.weight": torch.full((4, 6), 10, dtype=torch.int8), "a.weight_scale": torch.ones(4, 1)}
    save_file(tensors, folder / "model.safetensors")
    with pytest.raises(ValueError):
        merge(folder, tmp_path / "merged", {"a.weight": (torch.zeros(4, 1), torch.zeros(1, 6))})
    assert not (tmp_path / "merged").exists()


def test_scales_on_a_float_weight_are_refused(tmp_path):
    folder = tmp_path / "scaled"
    folder.mkdir()
    save_file(
        {"a.weight": torch.zeros(4, 6), "a.weight_scale": torch.ones(4, 1)}, folder / "model.safetensors"
    )
    with pytest.raises(ValueError, match="quantisation scales"):
        merge(folder, tmp_path / "merged", {"a.weight": (torch.zeros(4, 1), torch.zeros(1, 6))})


def test_a_failed_merge_leaves_no_model_and_the_next_one_succeeds(model_folder, tmp_path, monkeypatch):
    written = []

    def fail_after_one_shard(tensors, path, metadata=None):
        if written:
            raise OSError("disk full")
        save_file(tensors, path, metadata=metadata)
        written.append(path)

    monkeypatch.setattr("candor.merge.save_file", fail_after_one_shard)
    with pytest.raises(OSError, match="disk full"):
        merge(model_folder, tmp_path / "merged", {})
    assert written
    assert not (tmp_path / "merged").exists()

    monkeypatch.undo()
    merge(model_folder, tmp_path / "merged", {})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["merged", "model"]
    assert sorted(p.name for p in (tmp_path / "merged").iterdir()) == sorted(
        p.name for p in model_folder.iterdir()
    )


def test_an_existing_destination_is_refused_and_left_alone(model_folder, tmp_path):
    (tmp_path / "merged").mkdir()
    with pytest.raises(FileExistsError):
        merge(model_folder, tmp_path / "merged", {})
    assert list((tmp_path / "merged").iterdir()) == []


def test_other_weight_formats_are_left_out(model_folder, tmp_path):
    (model_folder / "pytorch_model.bin").write_bytes(b"old weights")
    assert merge(model_folder, tmp_path / "merged", {}) == ["pytorch_model.bin"]
    assert not (tmp_path / "merged" / "pytorch_model.bin").exists()
