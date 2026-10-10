import json

import gguf
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from candor import convert
from candor.convert import check_targets, float_type
from candor.targets import find_targets, gguf_name

from .test_targets import llama
from .tiny_tokenizer import tokenizer

HIDDEN = 64
GGUF_TYPE = {torch.bfloat16: gguf.GGMLQuantizationType.BF16, torch.float16: gguf.GGMLQuantizationType.F16}


def hf_folder(tmp_path, dtype=torch.bfloat16):
    folder = tmp_path / "hf"
    folder.mkdir()
    torch.manual_seed(6)
    tensors = {name: torch.randn(shape).to(dtype) for name, shape in llama().items()}
    save_file(tensors, folder / "model.safetensors")
    return folder, tensors


def gguf_file(tmp_path, tensors, change=lambda name, bits: bits, drop=(), dtype=torch.bfloat16):
    """The targets written as a converter that changes nothing would, then passed through change."""
    path = tmp_path / "model.gguf"
    writer = gguf.GGUFWriter(path, "llama")
    for target in find_targets({n: list(t.shape) for n, t in tensors.items()}, HIDDEN, 2).writers:
        if target.name in drop:
            continue
        name = str(gguf_name(target, "llama", 2))
        bits = tensors[target.name].to(dtype).view(torch.int16).numpy().view(np.uint16)
        writer.add_tensor(name, change(name, bits), raw_dtype=GGUF_TYPE[dtype])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def check(folder, path):
    writers = find_targets(llama(), HIDDEN, 2).writers
    return check_targets(path, folder, writers, "llama", 2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_identical_targets_pass(tmp_path, dtype):
    folder, tensors = hf_folder(tmp_path, dtype)
    assert check(folder, gguf_file(tmp_path, tensors, dtype=dtype)) == []


def test_reordered_rows_are_caught(tmp_path):
    folder, tensors = hf_folder(tmp_path)

    def permute(name, bits):
        return bits[::-1].copy() if name == "blk.1.attn_output.weight" else bits

    problems = check(folder, gguf_file(tmp_path, tensors, permute))
    assert problems == [
        "blk.1.attn_output.weight doesn't hold the same values as "
        "model.layers.1.self_attn.o_proj.weight; the converter changed it"
    ]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_a_one_ulp_change_is_caught(tmp_path, dtype):
    folder, tensors = hf_folder(tmp_path, dtype)

    def nudge(name, bits):
        if name != "blk.0.ffn_down.weight":
            return bits
        changed = bits.copy()
        changed[0, 0] += 1
        return changed

    problems = check(folder, gguf_file(tmp_path, tensors, nudge, dtype=dtype))
    assert problems == [
        "blk.0.ffn_down.weight doesn't hold the same values as "
        "model.layers.0.mlp.down_proj.weight; the converter changed it"
    ]


def test_a_missing_target_is_caught(tmp_path):
    folder, tensors = hf_folder(tmp_path)
    problems = check(folder, gguf_file(tmp_path, tensors, drop={"model.layers.0.mlp.down_proj.weight"}))
    assert problems == ["model.layers.0.mlp.down_proj.weight has no tensor blk.0.ffn_down.weight in the GGUF"]


def test_a_reshaped_target_is_caught(tmp_path):
    folder, tensors = hf_folder(tmp_path)

    def transpose(name, bits):
        return bits.T.copy() if name == "blk.0.ffn_down.weight" else bits

    problems = check(folder, gguf_file(tmp_path, tensors, transpose))
    assert len(problems) == 1
    assert "blk.0.ffn_down.weight is" in problems[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_other_precisions_pass_within_bf16_rounding(tmp_path, dtype):
    folder, tensors = hf_folder(tmp_path, dtype)
    assert check(folder, gguf_file(tmp_path, tensors)) == []


def test_other_precisions_still_catch_a_real_change(tmp_path):
    folder, tensors = hf_folder(tmp_path, torch.float32)

    def nudge(name, bits):
        if name != "blk.0.ffn_down.weight":
            return bits
        changed = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16).float() * 1.05
        return changed.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)

    problems = check(folder, gguf_file(tmp_path, tensors, nudge))
    assert len(problems) == 1 and problems[0].startswith("blk.0.ffn_down.weight doesn't hold")


@pytest.mark.parametrize(
    ("dtypes", "expected"),
    [
        ({"F16"}, "F16"),
        ({"BF16"}, "BF16"),
        ({"F32"}, "BF16"),
        ({"F16", "BF16"}, "BF16"),
    ],
)
def test_models_convert_at_their_own_float_type(tmp_path, dtypes, expected):
    torch_types = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}
    tensors = {f"m{i}": torch.zeros(2, 2, dtype=torch_types[name]) for i, name in enumerate(sorted(dtypes))}
    save_file(tensors | {"norm": torch.zeros(2)}, tmp_path / "model.safetensors")
    assert float_type(tmp_path) == expected


@pytest.fixture
def converter_calls(monkeypatch):
    """The converter's arguments, and what the model folder it was given held while it ran."""
    calls = []

    def run(program, args, log):
        folder = args[2]
        files = {p.name: p.readlink() if p.is_symlink() else None for p in folder.iterdir()}
        settings = json.loads((folder / "tokenizer_config.json").read_text())
        calls.append({"args": args, "folder": folder, "files": files, "settings": settings})

    monkeypatch.setattr(convert.toolchain, "converter", lambda: ("python", "convert_hf_to_gguf.py"))
    monkeypatch.setattr(convert, "_run", run)
    return calls


def model_with_tokenizer(tmp_path, saved_tokenizer, dtype=torch.bfloat16):
    folder, _ = hf_folder(tmp_path, dtype)
    saved_tokenizer.save_pretrained(folder)
    return folder


@pytest.mark.parametrize(("dtype", "outtype"), [(torch.float16, "f16"), (torch.bfloat16, "bf16")])
def test_the_converter_writes_the_models_float_type_to_the_destination(
    tmp_path, converter_calls, dtype, outtype
):
    folder = model_with_tokenizer(tmp_path, tokenizer(None, adds_bos=True), dtype)
    destination = tmp_path / "out" / "model.gguf"
    destination.parent.mkdir()
    convert.to_bf16(folder, destination, tmp_path / "convert.log")
    assert converter_calls[0]["args"][3:] == ["--outtype", outtype, "--outfile", destination]


def test_a_tokenizer_saved_by_transformers_5_is_given_to_the_converter_renamed(tmp_path, converter_calls):
    saved = tokenizer(None, adds_bos=True)
    saved.add_special_tokens({"additional_special_tokens": ["<|tool|>"]})
    folder = model_with_tokenizer(tmp_path, saved)
    settings = json.loads((folder / "tokenizer_config.json").read_text())
    assert settings["tokenizer_class"] == "TokenizersBackend"
    assert settings["extra_special_tokens"] == ["<|tool|>"]
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    destination = tmp_path / "out" / "model.gguf"
    destination.parent.mkdir()

    convert.to_bf16(folder, destination, tmp_path / "convert.log")

    [call] = converter_calls
    assert call["folder"] != folder and call["folder"].name == folder.name
    links = {name: (folder / name).resolve() for name in before}
    assert call["files"] == links | {"tokenizer_config.json": None}
    expected = {k: v for k, v in settings.items() if k != "extra_special_tokens"}
    expected |= {"tokenizer_class": "PreTrainedTokenizerFast", "additional_special_tokens": ["<|tool|>"]}
    assert call["settings"] == expected
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
    assert list(destination.parent.iterdir()) == []


def test_a_tokenizer_the_converter_can_read_is_given_as_it_is(tmp_path, converter_calls):
    folder = model_with_tokenizer(tmp_path, tokenizer(None, adds_bos=True))
    settings = json.loads((folder / "tokenizer_config.json").read_text())
    settings["tokenizer_class"] = "PreTrainedTokenizerFast"
    (folder / "tokenizer_config.json").write_text(json.dumps(settings))
    convert.to_bf16(folder, tmp_path / "model.gguf", tmp_path / "convert.log")
    assert converter_calls[0]["folder"] == folder
