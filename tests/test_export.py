import re

import pytest
import torch
from safetensors.torch import load_file
from typer.testing import CliRunner

from candor import config, convert, export, measure, run
from candor.cli import app
from candor.convert import check_targets
from candor.edit import RowNorm, rank_one
from candor.kernel import Kernel
from candor.measure import Edit, Measurement
from candor.run import Run, RunError

from .test_measure import EDIT
from .test_run import TEMPLATE
from .test_search import write_journal
from .tiny_gguf import random_weights, write_model
from .tiny_run import HF_NAMES

GGUF_NAMES = {hf: kind for kind, hf in HF_NAMES.items()}


def gguf_from_folder(folder, destination, log):
    """The tiny model's GGUF with the layer weights of a Hugging Face folder, as the converter would
    write it: the tiny run's own converter only copies the unedited GGUF."""
    weights = random_weights()
    for name, tensor in load_file(folder / "model.safetensors").items():
        _, _, layer, *kind, _ = name.split(".")
        weights[f"blk.{layer}.{GGUF_NAMES['.'.join(kind)]}.weight"] = tensor.numpy()
    write_model(destination, weights, chat_template=TEMPLATE)


@pytest.fixture
def converting(tiny, monkeypatch):
    calls = []

    def to_gguf(folder, destination, log):
        if "exports" in destination.parts:  # not the run's own conversion of the unedited model
            calls.append(folder)
        gguf_from_folder(folder, destination, log)

    monkeypatch.setattr(convert, "to_gguf", to_gguf)
    monkeypatch.setattr(convert, "check_targets", check_targets)
    return calls


def kl_between(a, b):
    return float(measure.compare(a, b)[-1][2])


@pytest.mark.parametrize("row_norm", ["pre", "none"])
@pytest.mark.parametrize("edit", [EDIT, Edit("trial", None, EDIT.kernels)], ids=["one layer", "per layer"])
def test_the_export_answers_as_the_model_with_the_edits_adapters(tiny, converting, edit, row_norm):
    tiny.settings.edit.row_norm = row_norm
    shipped = export.export(tiny, edit)
    assert shipped == tiny.path("exports", "trial", "tiny-trial-Q4_K_M.gguf")
    assert sorted(p.name for p in shipped.parent.iterdir()) == [shipped.name, f"{shipped.name}.inputs.json"]
    base = Measurement(tiny, None, "base").measure()
    adapters = Measurement(tiny, edit, "adapters").measure()
    merged = Measurement(tiny, edit, "merged", exported=True).measure()
    assert kl_between(adapters, merged) < 1e-5
    assert kl_between(base, merged) > 1e-2
    for name in measure.SETS:
        assert adapters[name]["tokens"] == merged[name]["tokens"]


@pytest.mark.parametrize("row_norm", ["pre", "none"])
def test_each_edited_weight_gains_its_strength_times_its_factors(tiny, converting, row_norm):
    tiny.settings.edit.row_norm = row_norm
    deltas = export.deltas(tiny, EDIT)
    _, _, _, writers = tiny.shape
    directions = tiny.edit_directions(EDIT.name, EDIT.direction_layer)
    original = load_file(tiny.model_folder / "model.safetensors")
    edited = {w.name for w in writers if EDIT.strength(w)}
    assert set(deltas) == edited and edited
    for writer in writers:
        if writer.name in edited:
            b, a = rank_one(original[writer.name], directions[writer.layer], RowNorm(row_norm))
            torch.testing.assert_close(
                deltas[writer.name][0] @ deltas[writer.name][1], EDIT.strength(writer) * b @ a
            )


def test_an_export_is_reused_until_the_edit_or_the_quantisation_changes(tiny, converting):
    first = export.export(tiny, EDIT)
    assert export.export(tiny, EDIT) == first and len(converting) == 1
    stronger = Edit("trial", 0.4, EDIT.kernels | {"attn": Kernel(1.5, 1.0, 0.8, 1.0)})
    export.export(tiny, stronger)
    assert len(converting) == 2
    with pytest.raises(RunError, match="trial has no export matching this run"):
        Measurement(tiny, EDIT, "stale", exported=True).measure()
    tiny.settings.deployment.quantization = "Q8_0"
    again = export.export(tiny, stronger)
    assert again.name == "tiny-trial-Q8_0.gguf" and len(converting) == 3
    assert sorted(p.name for p in again.parent.iterdir()) == [again.name, f"{again.name}.inputs.json"]


@pytest.mark.parametrize("row_norm", list(RowNorm))
def test_merging_before_the_qwen35_reorder_gives_the_adapter_built_after_it(row_norm):
    config = {"linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_value_head_dim": 3}
    name = "model.language_model.layers.0.linear_attn.out_proj.weight"
    torch.manual_seed(0)
    weight, direction = torch.randn(5, 12), torch.nn.functional.normalize(torch.randn(5), dim=0)
    b, a = rank_one(weight, direction, row_norm)
    reordered = convert.as_converted(weight, name, "qwen35", config)
    assert not torch.equal(reordered, weight)
    b_converted, a_converted = rank_one(reordered, direction, row_norm)
    merged = convert.as_converted(weight + 1.3 * b @ a, name, "qwen35", config)
    torch.testing.assert_close(merged, reordered + 1.3 * b_converted @ a_converted)


def test_the_deployed_float_type_is_exported_without_quantising(tiny, converting, monkeypatch):
    def refuse(*_):
        raise AssertionError("quantised")

    tiny.settings.deployment.quantization = "BF16"  # the tiny model's float type
    monkeypatch.setattr(convert, "quantize", refuse)
    shipped = export.export(tiny, EDIT)
    assert shipped.name == "tiny-trial-BF16.gguf" and shipped.stat().st_size > 0


def test_an_exported_measurement_is_redone_when_the_export_code_changes(tiny, converting, monkeypatch):
    export.export(tiny, EDIT)
    before = Measurement(tiny, EDIT, "merged", exported=True).measure()
    original = export.deltas

    def doubled(run, edit):
        return {name: (2 * b, a) for name, (b, a) in original(run, edit).items()}

    monkeypatch.setattr(export, "deltas", doubled)
    monkeypatch.setitem(run.STAGES, "exports", run.STAGES["exports"] + 1)
    with pytest.raises(RunError, match="no export matching"):
        Measurement(tiny, EDIT, "merged", exported=True).measure()
    export.export(tiny, EDIT)
    after = Measurement(tiny, EDIT, "merged", exported=True).measure()
    assert kl_between(before, after) > 1e-3


def test_what_killed_exports_left_behind_is_removed_before_the_disk_check(tiny, converting, monkeypatch):
    stranded = [tiny.path("exports", name, "tmpkilled") for name in ("trial", "heretic-1")]
    for folder in stranded:
        (folder / "model").mkdir(parents=True)
    other = tiny.path("exports", "heretic-1", "tiny-heretic-1-Q4_K_M.gguf")
    other.write_bytes(b"another edit's export")
    checked = []
    real = Run.check_disk

    def check_disk(self, *args):
        checked.append([folder.exists() for folder in stranded])
        real(self, *args)

    monkeypatch.setattr(Run, "check_disk", check_disk)
    export.export(tiny, EDIT)
    assert checked[-1] == [False, False] and other.exists()


def test_a_failed_export_leaves_nothing_behind(tiny, monkeypatch):
    def broken(folder, destination, log):
        if "exports" not in destination.parts:
            return gguf_from_folder(folder, destination, log)
        destination.write_bytes(b"half")
        raise convert.ConversionError("llama.cpp's converter failed")

    monkeypatch.setattr(convert, "to_gguf", broken)
    with pytest.raises(convert.ConversionError):
        export.export(tiny, EDIT)
    assert list(tiny.path("exports", "trial").iterdir()) == []


def test_an_export_whose_converter_changed_an_edited_weight_is_refused(tiny, monkeypatch):
    def unedited(folder, destination, log):
        gguf_from_folder(tiny.model_folder, destination, log)

    monkeypatch.setattr(convert, "to_gguf", unedited)
    monkeypatch.setattr(convert, "check_targets", check_targets)
    with pytest.raises(RunError, match="doesn't hold the edited weights unchanged"):
        export.export(tiny, EDIT)
    assert not tiny.path("exports", "trial", "tiny-trial-Q4_K_M.gguf").exists()


def test_only_an_edit_has_an_export_to_measure(tiny):
    with pytest.raises(ValueError, match="only an edit"):
        Measurement(tiny, None, "base", exported=True)


def test_export_and_measure_it_from_the_command_line(tiny, converting):
    config.write(tiny.folder.parent, "cli", tiny.settings, "made by a test")
    folder = tiny.folder.parent / "cli"
    journal = write_journal(tiny.folder.parent / "heretic.jsonl", 0.25)
    runner = CliRunner(env={"COLUMNS": "250"})
    early = runner.invoke(
        app, ["measure", str(folder), "--heretic", str(journal), "--trial", "1", "--exported"]
    )
    assert early.exit_code == 1 and "heretic-1 has no export matching this run" in early.output
    exported = runner.invoke(app, ["export", str(folder), "--heretic", str(journal), "--trial", "1"])
    assert exported.exit_code == 0, exported.output
    shipped = folder / "exports" / "heretic-1" / "cli-heretic-1-Q4_K_M.gguf"
    assert f"heretic-1: {shipped}" in exported.output
    assert f"candor measure {folder} --trial 1 --heretic {journal} --exported" in exported.output
    assert f"candor compare {folder} heretic-1 heretic-1-exported" in exported.output
    picked = ["--heretic", str(journal), "--trial", "1"]
    assert runner.invoke(app, ["measure", str(folder), *picked]).exit_code == 0
    measured = runner.invoke(app, ["measure", str(folder), *picked, "--exported"])
    assert measured.exit_code == 0, measured.output
    assert "candor measure: heretic-1-exported" in measured.output
    compared = runner.invoke(app, ["compare", str(folder), "heretic-1", "heretic-1-exported"])
    assert compared.exit_code == 0, compared.output
    assert re.search(r"Harmless KL from A +0\.0000", compared.output)


def test_exported_needs_a_trial(tiny):
    result = CliRunner().invoke(app, ["measure", str(tiny.folder), "--exported"])
    assert result.exit_code == 2 and "--trial" in result.output
