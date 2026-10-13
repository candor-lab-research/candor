"""A run's GGUF stage for models stored as F16, and its adapters for a model the converter reorders."""

import json
import math
import shutil

import gguf
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from candor import config, run
from candor.edit import RowNorm, rank_one
from candor.run import Run, RunError
from candor.weights import ModelFiles

from .test_convert import CONVERTER_ORDER, OUT_PROJ, QWEN35, qwen35
from .test_targets import llama

HIDDEN, LAYERS = 64, 2


def make_run(tmp_path, shapes, model_config, dtype):
    folder = tmp_path / "model"
    folder.mkdir()
    torch.manual_seed(4)
    tensors = {name: torch.randn(shape).to(dtype) for name, shape in shapes.items()}
    save_file(tensors, folder / "model.safetensors")
    (folder / "config.json").write_text(json.dumps(model_config))
    runs = tmp_path / "runs" / "test"
    runs.mkdir(parents=True)
    return Run(runs, config.RunConfig(model=config.Model(source=str(folder))), report=lambda _: None)


def llama_run(tmp_path, dtype):
    model_config = {"architectures": ["LlamaForCausalLM"], "hidden_size": HIDDEN, "num_hidden_layers": LAYERS}
    return make_run(tmp_path, llama(), model_config, dtype)


@pytest.fixture
def conversions(monkeypatch):
    calls = []

    def to_gguf(folder, destination, log):
        calls.append("convert")
        destination.write_bytes(b"converted")

    def quantize(base, destination, quantization, log):
        calls.append(f"{base.name} -> {quantization}")
        destination.write_bytes(quantization.encode())

    monkeypatch.setattr(run.convert, "to_gguf", to_gguf)
    monkeypatch.setattr(run.convert, "quantize", quantize)
    monkeypatch.setattr(run.convert, "check_targets", lambda *_: [])
    return calls


def test_an_f16_model_keeps_its_f16_gguf_as_the_f16_deployment(tmp_path, conversions):
    this = llama_run(tmp_path, torch.float16)
    this.settings.deployment.quantization = "F16"
    base, quantized = this.gguf()
    assert base.name == quantized.name == "model-F16.gguf"
    assert conversions == ["convert"]


def test_a_bf16_deployment_of_an_f16_model_is_quantised_from_its_f16_gguf(tmp_path, conversions):
    this = llama_run(tmp_path, torch.float16)
    this.settings.deployment.quantization = "BF16"
    base, quantized = this.gguf()
    assert (base.name, quantized.name) == ("model-F16.gguf", "model-BF16.gguf")
    assert conversions == ["convert", "model-F16.gguf -> BF16"]


@pytest.mark.parametrize(("quantization", "fits"), [("BF16", True), ("Q8_0", False)])
def test_the_disk_check_counts_only_the_files_it_will_write(
    tmp_path, conversions, monkeypatch, quantization, fits
):
    this = llama_run(tmp_path, torch.bfloat16)
    this.settings.deployment.quantization = quantization
    weights = sum(math.prod(shape) for shape in llama().values())
    free = int(1.1 * weights * 2) + 1
    monkeypatch.setattr(run.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(0, 0, free))
    if fits:
        this.gguf()
    else:
        with pytest.raises(RunError, match="GB free"):
            this.gguf()


def adapter_factors(path):
    tensors = {t.name: torch.from_numpy(t.data.copy()) for t in gguf.GGUFReader(path).tensors}
    a = next(v for k, v in tensors.items() if k.endswith("lora_a"))
    b = next(v for k, v in tensors.items() if k.endswith("lora_b"))
    return b, a


@pytest.mark.parametrize("row_norm", list(RowNorm))
def test_qwen35_adapters_are_built_in_the_gguf_layout(tmp_path, monkeypatch, row_norm):
    this = make_run(tmp_path, qwen35(), QWEN35, torch.bfloat16)
    this.settings.edit.row_norm = row_norm.value
    found = F.normalize(torch.randn(LAYERS + 1, HIDDEN), dim=-1)

    def fake_directions(self, answering=False):
        self.inputs["directions"] = "fixed"
        return found

    monkeypatch.setattr(Run, "directions", fake_directions)
    made = this.adapters()["per-layer"]
    files = ModelFiles(this.model_folder)

    weight = files.tensor(OUT_PROJ).float()
    b, a = adapter_factors(made["blk.0.ssm_out.weight"])
    hf_b, hf_a = rank_one(weight, found[1], row_norm)
    merged_then_converted = (weight + hf_b @ hf_a)[:, CONVERTER_ORDER]
    torch.testing.assert_close(weight[:, CONVERTER_ORDER] + b @ a, merged_then_converted)

    unchanged = files.tensor("model.language_model.layers.1.self_attn.o_proj.weight")
    b, a = adapter_factors(made["blk.1.attn_output.weight"])
    torch.testing.assert_close((b, a), rank_one(unchanged, found[2], row_norm))
