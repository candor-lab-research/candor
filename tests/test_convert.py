import gguf
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from candor.convert import check_targets
from candor.targets import find_targets

from .test_targets import llama

HIDDEN = 64


def hf_folder(tmp_path, dtype=torch.bfloat16):
    folder = tmp_path / "hf"
    folder.mkdir()
    torch.manual_seed(6)
    tensors = {name: torch.randn(shape).to(dtype) for name, shape in llama().items()}
    save_file(tensors, folder / "model.safetensors")
    return folder, tensors


def gguf_file(tmp_path, tensors, change=lambda name, array: array, drop=()):
    path = tmp_path / "model-BF16.gguf"
    writer = gguf.GGUFWriter(path, "llama")
    names = {"self_attn.o_proj": "attn_output", "mlp.down_proj": "ffn_down"}
    for name, tensor in tensors.items():
        for hf, short in names.items():
            if hf in name and name not in drop:
                layer = name.split(".")[2]
                bits = tensor.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
                gguf_name = f"blk.{layer}.{short}.weight"
                writer.add_tensor(
                    gguf_name, change(gguf_name, bits), raw_dtype=gguf.GGMLQuantizationType.BF16
                )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def check(tmp_path, folder, path):
    writers = find_targets(llama(), HIDDEN, 2).writers
    return check_targets(path, folder, writers, "llama", 2)


def test_identical_targets_pass(tmp_path):
    folder, tensors = hf_folder(tmp_path)
    assert check(tmp_path, folder, gguf_file(tmp_path, tensors)) == []


def test_reordered_rows_are_caught(tmp_path):
    folder, tensors = hf_folder(tmp_path)

    def permute(name, array):
        return array[::-1].copy() if name == "blk.1.attn_output.weight" else array

    problems = check(tmp_path, folder, gguf_file(tmp_path, tensors, permute))
    assert problems == [
        "blk.1.attn_output.weight doesn't hold the same values as "
        "model.layers.1.self_attn.o_proj.weight; the converter changed it"
    ]


def test_a_missing_target_is_caught(tmp_path):
    folder, tensors = hf_folder(tmp_path)
    problems = check(
        tmp_path, folder, gguf_file(tmp_path, tensors, drop={"model.layers.0.mlp.down_proj.weight"})
    )
    assert problems == ["model.layers.0.mlp.down_proj.weight has no tensor blk.0.ffn_down.weight in the GGUF"]


def test_a_reshaped_target_is_caught(tmp_path):
    folder, tensors = hf_folder(tmp_path)

    def transpose(name, array):
        return array.T.copy() if name == "blk.0.ffn_down.weight" else array

    problems = check(tmp_path, folder, gguf_file(tmp_path, tensors, transpose))
    assert len(problems) == 1
    assert "blk.0.ffn_down.weight is" in problems[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_other_precisions_pass_within_bf16_rounding(tmp_path, dtype):
    folder, tensors = hf_folder(tmp_path, dtype)
    assert check(tmp_path, folder, gguf_file(tmp_path, tensors)) == []


def test_other_precisions_still_catch_a_real_change(tmp_path):
    folder, tensors = hf_folder(tmp_path, torch.float32)

    def nudge(name, array):
        if name != "blk.0.ffn_down.weight":
            return array
        changed = torch.from_numpy(array.view(np.int16).copy()).view(torch.bfloat16).float() * 1.05
        return changed.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)

    problems = check(tmp_path, folder, gguf_file(tmp_path, tensors, nudge))
    assert len(problems) == 1 and problems[0].startswith("blk.0.ffn_down.weight doesn't hold")
