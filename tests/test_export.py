import re
import shutil
from dataclasses import replace

import gguf
import pytest
import torch
from safetensors.torch import load_file, save_file
from typer.testing import CliRunner

from candor import config, convert, export, measure, merge, run
from candor.cli import app
from candor.convert import check_targets
from candor.convert import quantize as real_quantize
from candor.edit import RowNorm, edit_factors, rank_one
from candor.kernel import Kernel
from candor.measure import Edit, Measurement
from candor.run import Run, RunError

from .test_engine import LLAMA_SERVER
from .test_measure import EDIT
from .test_run import TEMPLATE
from .test_search import write_journal
from .tiny_gguf import HIDDEN, random_weights, write_model
from .tiny_run import HF_NAMES, thinking

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
    return float(next(row for row in measure.compare(a, b) if row[0] == "Harmless KL from A")[2])


@pytest.mark.parametrize("row_norm", ["pre", "none", "full"])
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


@pytest.mark.parametrize("answering", [False, True])
def test_a_full_edit_ships_the_adapter_it_was_scored_with(tiny, converting, answering):
    thinking(tiny)
    tiny.settings.edit.row_norm = "full"
    edit = replace(EDIT, name="trial-full", answering=answering)
    deltas = export.deltas(tiny, edit)
    path, _ = tiny.edit_adapter(tiny.path("scores", "edits", edit.name), edit)
    stored = {t.name: torch.from_numpy(t.data.copy()) for t in gguf.GGUFReader(path).tensors}
    for writer in tiny.shape[3]:
        name = tiny.target_names[writer.name]
        if EDIT.strength(writer):
            b, a = deltas[writer.name]
            assert a.shape[0] == 3
            torch.testing.assert_close(stored[f"{name}.lora_b"] @ stored[f"{name}.lora_a"], b @ a)
        else:
            assert f"{name}.lora_a" not in stored


def test_an_edit_read_where_the_answer_starts_ships_that_direction(tiny, converting):
    thinking(tiny)
    answering = replace(EDIT, name="trial-answering", answering=True)
    deltas = export.deltas(tiny, answering)
    directions = tiny.edit_directions(EDIT.name, EDIT.direction_layer, answering=True)
    assert not torch.equal(directions, tiny.edit_directions(EDIT.name, EDIT.direction_layer))
    original = load_file(tiny.model_folder / "model.safetensors")
    for writer in tiny.shape[3]:
        if EDIT.strength(writer):
            b, a = rank_one(
                original[writer.name], directions[writer.layer], RowNorm(tiny.settings.edit.row_norm)
            )
            torch.testing.assert_close(
                deltas[writer.name][0] @ deltas[writer.name][1], EDIT.strength(writer) * b @ a
            )
    export.export(tiny, EDIT)
    export.export(tiny, answering)
    assert len(converting) == 2


def test_an_edit_read_where_the_answer_starts_is_exported_again_when_those_directions_change(
    tiny, converting, monkeypatch
):
    thinking(tiny)
    answering = replace(EDIT, answering=True)
    export.export(tiny, answering)
    real = Run.directions

    def changed(self, answering=False):
        found = real(self, answering)
        self.inputs["answering directions"] = "changed"
        return found

    monkeypatch.setattr(Run, "directions", changed)
    export.export(tiny, answering)
    assert len(converting) == 2


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
    # Enough rows for a "full" edit's rank-3 SVD; its randomised SVD of reordered columns agrees to
    # within a few parts in a million.
    weight, direction = torch.randn(16, 12), torch.nn.functional.normalize(torch.randn(16), dim=0)
    b, a = edit_factors(weight, direction, 1.3, row_norm)
    reordered = convert.as_converted(weight, name, "qwen35", config)
    assert not torch.equal(reordered, weight)
    b_converted, a_converted = edit_factors(reordered, direction, 1.3, row_norm)
    merged = convert.as_converted(weight + b @ a, name, "qwen35", config)
    torch.testing.assert_close(merged, reordered + b_converted @ a_converted)


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


def test_an_export_that_holds_the_whole_edit_keeps_all_of_it(tiny, converting):
    shipped = export.export(tiny, EDIT)
    held = export.kept(tiny, shipped, export.deltas(tiny, EDIT))
    assert held and all(share == pytest.approx(1, abs=1e-4) for share in held.values())


def losing(tiny, share, honoured=True):
    """A quantiser that keeps share of each edit, and all of it in tensors given a type of their own
    when honoured; the run's own quantising copies, as the tiny run's does."""
    tiny.gguf()
    original = load_file(tiny.model_folder / "model.safetensors")
    made = export.deltas(tiny, EDIT)
    asked = []

    def quantize(base, destination, quantization, log, types=None):
        if "exports" not in destination.parts:
            return shutil.copy(base, destination)
        asked.append(types)
        weights = random_weights()
        for name, (b, a) in made.items():
            gguf_name = tiny.target_names[name]
            kept = 1.0 if honoured and types and gguf_name in types else share
            weights[gguf_name] = (original[name] + kept * b @ a).numpy()
        write_model(destination, weights, chat_template=TEMPLATE)

    return quantize, asked


def test_weights_the_quantisation_rounds_the_edit_away_from_are_quantised_to_keep_it(
    tiny, converting, monkeypatch
):
    quantize, asked = losing(tiny, 0.5)
    monkeypatch.setattr(convert, "quantize", quantize)
    said = []
    tiny.report = said.append
    shipped = export.export(tiny, EDIT)
    edited = {tiny.target_names[name] for name in export.deltas(tiny, EDIT)}
    assert asked == [None, dict.fromkeys(edited, "Q8_0")]
    moved = f"in {len(edited)} of {len(edited)} edited weights; quantising those as Q8_0"
    assert f"Q4_K_M keeps 50.0% to 50.0% of the edit {moved}" in said
    assert "Each edited weight keeps 100.0% to 100.0% of its edit" in said
    assert all(
        share == pytest.approx(1, abs=1e-4)
        for share in export.kept(tiny, shipped, export.deltas(tiny, EDIT)).values()
    )


@pytest.mark.parametrize("share", [0.0, 0.5])
def test_an_edit_still_lost_after_quantising_to_keep_it_is_refused(tiny, converting, monkeypatch, share):
    quantize, asked = losing(tiny, share, honoured=False)
    monkeypatch.setattr(convert, "quantize", quantize)
    with pytest.raises(
        RunError, match=f"keep less than 98% of the edit even quantised to keep it, e.g. .* {share:.1%}"
    ):
        export.export(tiny, EDIT)
    assert len(asked) == 2
    assert not tiny.path("exports", "trial", "tiny-trial-Q4_K_M.gguf").exists()


def test_llama_quantize_gives_the_named_tensors_their_own_type(tmp_path, monkeypatch):
    if not LLAMA_SERVER.exists():
        pytest.skip("no llama.cpp")
    monkeypatch.setenv("CANDOR_LLAMA_BIN", str(LLAMA_SERVER.parent))
    write_model(tmp_path / "f32.gguf", random_weights())
    out = tmp_path / "q.gguf"
    real_quantize(
        tmp_path / "f32.gguf", out, "Q4_K_M", tmp_path / "log", {"blk.0.attn_output.weight": "Q8_0"}
    )
    types = {t.name: t.tensor_type.name for t in gguf.GGUFReader(out).tensors}
    assert types["blk.0.attn_output.weight"] == "Q8_0"
    assert types["blk.1.attn_output.weight"] != "Q8_0"  # the pattern is anchored: blk.0 only


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


def saved_elsewhere(tiny, rename=lambda name: name, called="saved"):
    """A copy of the tiny model with EDIT merged in, as another tool would save it."""
    folder = tiny.folder.parent / called
    merge.merge(tiny.model_folder, folder, export.deltas(tiny, EDIT))
    weights = load_file(folder / "model.safetensors")
    save_file({rename(name): tensor for name, tensor in weights.items()}, folder / "model.safetensors")
    return folder


def test_a_model_edited_elsewhere_ships_and_measures_as_the_same_edit_of_candors(tiny, converting):
    tiny.gguf()
    folder = saved_elsewhere(tiny)
    said = []
    tiny.report = said.append
    shipped = export.export_outside(tiny, "saved", folder)
    assert shipped == tiny.path("exports", "saved", "tiny-saved-Q4_K_M.gguf")
    assert f"Taking the {len(export.deltas(tiny, EDIT))} edited weights from {folder}" in said
    export.export(tiny, EDIT)
    theirs = Measurement(tiny, None, "saved", outside="saved").measure()
    ours = Measurement(tiny, EDIT, "merged", exported=True).measure()
    assert kl_between(ours, theirs) < 1e-5
    assert export.export_outside(tiny, "saved", folder) == shipped and len(converting) == 2


def test_a_model_edited_elsewhere_is_exported_again_once_it_changes(tiny, converting):
    tiny.gguf()
    folder = saved_elsewhere(tiny)
    export.export_outside(tiny, "saved", folder)
    weights = folder / "model.safetensors"
    save_file(load_file(weights), weights)
    with pytest.raises(RunError, match="saved's export no longer matches this run or"):
        Measurement(tiny, None, "saved", outside="saved").measure()
    export.export_outside(tiny, "saved", folder)
    assert len(converting) == 2
    Measurement(tiny, None, "saved", outside="saved").measure()


def test_a_model_edited_elsewhere_names_its_layers_without_language_model(tiny, converting, monkeypatch):
    tiny.gguf()
    folder = saved_elsewhere(tiny)
    _, layers, arch, writers = tiny.shape
    wrapped = [replace(w, name=w.name.replace("model.", "model.language_model.", 1)) for w in writers]
    monkeypatch.setitem(tiny.__dict__, "shape", (HIDDEN, layers, arch, wrapped))
    monkeypatch.setattr(export, "ModelFiles", wrapping(tiny.model_folder))
    changes = export.taken_from(tiny, folder)
    assert set(changes) == {w.name for w in wrapped if EDIT.strength(w)}


def wrapping(model_folder):
    """ModelFiles that read model_folder as if its layers were named model.language_model.*, as a
    multimodal model's are."""
    real = export.ModelFiles

    class Wrapped(real):
        def tensor(self, name):
            return super().tensor(name.replace("model.language_model.", "model.", 1))

    return lambda folder: Wrapped(folder) if folder == model_folder else real(folder)


def test_models_edited_elsewhere_that_cant_be_taken_are_refused(tiny, converting):
    tiny.gguf()
    with pytest.raises(RunError, match="holds the run's model unchanged in every edit target"):
        export.export_outside(tiny, "same", tiny.model_folder)
    for reserved in ("trial-3", "base", "heretic-161-exported"):
        with pytest.raises(RunError, match=f"{reserved} is how candor names its own edits and measurements"):
            export.export_outside(tiny, reserved, tiny.model_folder)
    saved_elsewhere(tiny)
    folder = saved_elsewhere(tiny, lambda name: name.replace("down_proj", "w2"), "renamed")
    with pytest.raises(
        RunError, match=re.escape("has no weight for the edit target model.layers.0.mlp.down_proj.weight")
    ):
        export.export_outside(tiny, "renamed", folder)
    with pytest.raises(RunError, match="has no export in Q4_K_M; run candor export --outside first"):
        Measurement(tiny, None, "missing", outside="missing").measure()


def test_the_command_line_exports_and_measures_a_model_edited_elsewhere(tiny, converting):
    tiny.gguf()
    folder = saved_elsewhere(tiny)
    config.write(tiny.folder.parent, "cli", tiny.settings, "made by a test")
    run_folder = tiny.folder.parent / "cli"
    runner = CliRunner(env={"COLUMNS": "200"})
    unnamed = runner.invoke(app, ["export", str(run_folder), "--outside", str(folder)])
    assert unnamed.exit_code == 2 and "--name" in unnamed.output
    both = runner.invoke(
        app, ["export", str(run_folder), "--trial", "1", "--outside", str(folder), "--name", "x"]
    )
    assert both.exit_code == 2
    exported = runner.invoke(app, ["export", str(run_folder), "--outside", str(folder), "--name", "saved"])
    assert exported.exit_code == 0, exported.output
    assert f"uv run candor measure {run_folder} --outside saved" in exported.output
    measured = runner.invoke(app, ["measure", str(run_folder), "--outside", "saved"])
    assert measured.exit_code == 0, measured.output
    assert (run_folder / "measurements" / "saved" / "maths.json").exists()
