import gguf
import numpy as np
import pytest
import torch

from candor.adapters import write_adapter


def read_back(path):
    reader = gguf.GGUFReader(path)
    fields = {name: field.contents() for name, field in reader.fields.items()}
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    return fields, tensors


def test_writes_what_llama_cpp_expects(tmp_path):
    b, a = torch.randn(6, 1), torch.randn(1, 10)
    path = tmp_path / "attn.3.gguf"
    write_adapter(path, "llama", {"blk.3.attn_output.weight": (b, a)})
    fields, tensors = read_back(path)
    assert fields["general.architecture"] == "llama"
    assert fields["general.type"] == "adapter"
    assert fields["adapter.type"] == "lora"
    assert fields["adapter.lora.alpha"] == 1.0
    lora_a, lora_b = tensors["blk.3.attn_output.weight.lora_a"], tensors["blk.3.attn_output.weight.lora_b"]
    # llama.cpp wants a's first dimension to match the weight's inputs and b's second its outputs.
    assert list(lora_a.shape) == [10, 1]
    assert list(lora_b.shape) == [1, 6]
    np.testing.assert_array_equal(lora_a.data, a.numpy())
    np.testing.assert_array_equal(lora_b.data, b.numpy())


def test_alpha_matches_rank(tmp_path):
    path = tmp_path / "rank2.gguf"
    write_adapter(path, "llama", {"blk.0.ffn_down.weight": (torch.randn(6, 2), torch.randn(2, 10))})
    fields, _ = read_back(path)
    assert fields["adapter.lora.alpha"] == 2.0


def test_refuses_mixed_ranks(tmp_path):
    factors = {
        "blk.0.ffn_down.weight": (torch.randn(6, 1), torch.randn(1, 10)),
        "blk.1.ffn_down.weight": (torch.randn(6, 2), torch.randn(2, 10)),
    }
    with pytest.raises(ValueError):
        write_adapter(tmp_path / "mixed.gguf", "llama", factors)
