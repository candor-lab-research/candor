"""A run's GGUF stage for models stored as F16, and its adapters for a model the converter reorders."""

import json
import math
import shutil

import gguf
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from candor import config, convert, export, run
from candor.edit import RowNorm, edit_factors, rank_one
from candor.kernel import Kernel
from candor.measure import Edit
from candor.run import Run, RunError
from candor.weights import ModelFiles

from .helpers import CONVERTER_ORDER, HIDDEN, LAYERS, OUT_PROJ, TINY_QWEN35, adapter_factors, llama, qwen35


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


@pytest.mark.parametrize("row_norm", [RowNorm.NONE, RowNorm.PRE])
def test_qwen35_adapters_are_built_in_the_gguf_layout(tmp_path, monkeypatch, row_norm):
    this = make_run(tmp_path, qwen35(), TINY_QWEN35, torch.bfloat16)
    this.settings.edit.row_norm = row_norm.value
    found = F.normalize(torch.randn(LAYERS, HIDDEN), dim=-1)

    def fake_directions(self, answering=False):
        self.inputs["directions"] = "fixed"
        return found

    monkeypatch.setattr(Run, "directions", fake_directions)
    made = this.adapters()["per-layer"]
    files = ModelFiles(this.model_folder)

    weight = files.tensor(OUT_PROJ).float()
    b, a = adapter_factors(made["blk.0.ssm_out.weight"])
    hf_b, hf_a = rank_one(weight, found[0], row_norm)
    merged_then_converted = (weight + hf_b @ hf_a)[:, CONVERTER_ORDER]
    torch.testing.assert_close(weight[:, CONVERTER_ORDER] + b @ a, merged_then_converted)

    unchanged = files.tensor("model.language_model.layers.1.self_attn.o_proj.weight")
    b, a = adapter_factors(made["blk.1.attn_output.weight"])
    torch.testing.assert_close((b, a), rank_one(unchanged, found[1], row_norm))


def test_a_full_qwen35_edits_adapter_is_built_in_the_gguf_layout(tmp_path, monkeypatch):
    this = make_run(tmp_path, qwen35(), TINY_QWEN35, torch.bfloat16)
    this.settings.edit.row_norm = "full"
    found = F.normalize(torch.randn(LAYERS, HIDDEN), dim=-1)

    def fake_directions(self, answering=False):
        self.inputs["directions"] = "fixed"
        return found

    monkeypatch.setattr(Run, "directions", fake_directions)
    edit = Edit("e", None, {"attn": Kernel(1.2, 0.0, 1.2, 1.0)})
    path, _ = this.edit_adapter(this.path("scores", "edits", "e"), edit)
    b, a = adapter_factors(path, "blk.0.ssm_out.weight")
    weight = ModelFiles(this.model_folder).tensor(OUT_PROJ).float()
    hf_b, hf_a = edit_factors(weight, found[0], 1.2, RowNorm.FULL)
    torch.testing.assert_close(weight[:, CONVERTER_ORDER] + b @ a, (weight + hf_b @ hf_a)[:, CONVERTER_ORDER])


def test_a_qwen35_export_writes_and_reads_its_edit_in_the_converters_order(tmp_path, monkeypatch):
    this = make_run(tmp_path, qwen35(), TINY_QWEN35, torch.bfloat16)
    this.settings.edit.row_norm = "full"
    found = F.normalize(torch.randn(LAYERS, HIDDEN), dim=-1)

    def fake_directions(self, answering=False):
        self.inputs["directions"] = "fixed"
        return found

    monkeypatch.setattr(Run, "directions", fake_directions)
    edit = Edit("e", None, {"attn": Kernel(1.2, 0.0, 1.2, 1.0)})
    changes = export.deltas(this, edit)
    weights = ModelFiles(this.model_folder)
    weight = weights.tensor(OUT_PROJ).float()
    # The export writes the weight with the very change its adapter holds, in the converter's order.
    written = torch.from_numpy(export._written(this, OUT_PROJ, changes[OUT_PROJ]))
    b, a = adapter_factors(this.edit_adapter(this.path("scores", "e"), edit)[0], "blk.0.ssm_out.weight")
    torch.testing.assert_close(written, weight[:, CONVERTER_ORDER] + b @ a)
    # A GGUF holding that reads back as the whole edit, and as the model's own weight changed.
    path = tmp_path / "edited.gguf"
    writer = gguf.GGUFWriter(path, "qwen35")
    for name, gguf_name in this.target_names.items():
        values = (
            written
            if name == OUT_PROJ
            else convert.as_converted(weights.tensor(name).float(), name, "qwen35", TINY_QWEN35)
        )
        writer.add_tensor(gguf_name, values.numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    assert export.kept(this, path, {OUT_PROJ: changes[OUT_PROJ]})["blk.0.ssm_out.weight"] == pytest.approx(
        1, abs=1e-5
    )
    taken = export.taken_from_gguf(this, path)
    assert set(taken) == {OUT_PROJ}
    torch.testing.assert_close(
        export.edited(weight, taken[OUT_PROJ]), export.edited(weight, changes[OUT_PROJ])
    )
