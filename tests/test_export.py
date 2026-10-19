import json
import re
import shutil
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import gguf
import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from typer.testing import CliRunner

from candor import convert, export, measure, residuals, run, search
from candor.cli import app
from candor.convert import quantize as real_quantize
from candor.edit import RowNorm, full_device, rank_one
from candor.kernel import Kernel
from candor.measure import Edit, Measurement
from candor.run import Run, RunError

from .helpers import EDIT, LLAMA_SERVER, TEMPLATE, gguf_from_folder, on_disk, small, write_journal
from .tiny_gguf import HIDDEN, random_weights, write_model
from .tiny_run import thinking

TYPE = gguf.GGMLQuantizationType


def kl_between(a, b):
    return float(next(row for row in measure.compare(a, b) if row[0] == "Harmless KL from A")[2])


@pytest.mark.parametrize(
    ("row_norm", "edit"),
    [("pre", EDIT), ("none", Edit("trial", None, EDIT.kernels)), ("full", Edit("trial", None, EDIT.kernels))],
    ids=["pre, one layer", "none, per layer", "full, per layer"],
)
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


def test_a_full_change_gives_the_gpus_memory_back_before_a_server_needs_it(tiny, converting, monkeypatch):
    thinking(tiny)
    tiny.settings.edit.row_norm = "full"
    freed = []
    monkeypatch.setattr(residuals, "free_memory", freed.append)
    tiny.changes(replace(EDIT, name="trial-full"))
    assert freed == [full_device()]


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


def test_an_export_and_a_scored_edit_are_kept_when_only_the_measuring_code_changes(
    tiny, converting, monkeypatch
):
    shipped = export.export(tiny, EDIT)
    edit = Edit("other", 0.6, EDIT.kernels)
    scored = tiny.path("scores", "layer-0.6")
    measure.edit_adapters(tiny, edit, scored)
    monkeypatch.setitem(run.STAGES, "measurements", run.STAGES["measurements"] + 1)
    assert export.export(tiny, EDIT) == shipped and len(converting) == 1
    Measurement(tiny, EDIT, "merged", exported=True, limit=1).measure()
    # Neither is built from a measurement, so neither holds its code's version.
    for record in [run.record_of(shipped), *scored.glob("*.inputs.json")]:
        assert "measurements" not in json.loads(record.read_text())["code"]


def test_an_older_measurement_stays_where_it_is_until_something_can_be_measured_in_its_place(
    tiny, converting, monkeypatch
):
    export.export(tiny, EDIT)
    Measurement(tiny, EDIT, "merged", exported=True, limit=1).measure()
    version = run.STAGES["measurements"]
    monkeypatch.setitem(run.STAGES, "measurements", version + 1)
    monkeypatch.setitem(run.STAGES, "exports", run.STAGES["exports"] + 1)
    with pytest.raises(RunError, match="no export matching"):
        Measurement(tiny, EDIT, "merged", exported=True, limit=1).measure()
    assert measure.measured_by(tiny.path("measurements", "merged")) == version
    assert not tiny.path("measurements", f"merged-measuring-{version}").exists()


def test_a_trials_export_is_measured_after_its_edit_can_no_longer_be_rebuilt_as_scored(
    tiny, converting, monkeypatch
):
    small(tiny)
    folder = str(on_disk(tiny))
    runner = CliRunner(env={"COLUMNS": "200"})
    assert runner.invoke(app, ["search", folder, "--trials", "2"]).exit_code == 0
    assert runner.invoke(app, ["export", folder, "--trial", "2"]).exit_code == 0
    monkeypatch.setattr(search, "EDITS", search.EDITS + 1)
    rebuilt = runner.invoke(app, ["measure", folder, "--trial", "2", "--limit", "1"])
    assert rebuilt.exit_code == 1 and "can't be rebuilt as they were scored" in rebuilt.output
    shipped = runner.invoke(app, ["measure", folder, "--trial", "2", "--exported", "--limit", "1"])
    assert shipped.exit_code == 0, shipped.output


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


def losing(tiny, monkeypatch, share=0.5, at_q8_0: float | Callable[[str], float] = 1.0, at_f32=1.0):
    """A quantiser that stores each tensor at the type it is given, by default a matrix in F16, keeping
    of each edited weight's change in the GGUF it quantises: share in F16, at_q8_0 in Q8_0 (a share,
    or a function of the GGUF name) and at_f32 in F32. The run's own GGUF is quantised by it too.
    Returns the types each export asked for."""
    original = load_file(tiny.model_folder / "model.safetensors")
    _, _, _, writers = tiny.shape
    edited = {tiny.target_names[w.name]: original[w.name].numpy() for w in writers if EDIT.strength(w)}
    asked = []

    def quantize(base, destination, quantization, log, types=None):
        if "exports" in destination.parts:
            asked.append(types)
        weights, kinds = {}, {}
        for tensor in gguf.GGUFReader(base).tensors:
            values = convert.stored(tensor).numpy()
            kind = (types or {}).get(tensor.name) or ("F16" if values.ndim == 2 else "F32")
            kept = {"F16": share, "Q8_0": at_q8_0, "F32": at_f32}[kind]
            kept = kept(tensor.name) if callable(kept) else kept
            if tensor.name in edited and kept != 1:
                values = edited[tensor.name] + kept * (values - edited[tensor.name])
            weights[tensor.name], kinds[tensor.name] = values, TYPE[kind]
        write_model(destination, weights, TEMPLATE, types=kinds)

    monkeypatch.setattr(convert, "quantize", quantize)
    tiny.gguf()
    return asked


def edited_in_gguf(tiny):
    return {tiny.target_names[name] for name in export.deltas(tiny, EDIT)}


def test_weights_the_quantisation_rounds_the_edit_away_from_are_quantised_to_keep_it(
    tiny, converting, monkeypatch
):
    asked = losing(tiny, monkeypatch)
    said = []
    tiny.report = said.append
    shipped = export.export(tiny, EDIT)
    edited = edited_in_gguf(tiny)
    plain = convert.tensor_types(tiny.gguf()[1])
    assert asked == [None, plain | dict.fromkeys(edited, "Q8_0")]
    assert convert.tensor_types(shipped) == plain | dict.fromkeys(edited, "Q8_0")
    assert (
        f"Q4_K_M keeps 50.0% to 50.0% of the edit in {len(edited)} of {len(edited)} edited weights; "
        "storing those as Q8_0"
    ) in said
    assert all(
        share == pytest.approx(1, abs=0.01)
        for share in export.kept(tiny, shipped, export.deltas(tiny, EDIT)).values()
    )


def test_weights_q8_0_still_rounds_the_edit_away_from_stay_f32_and_the_rest_q8_0(
    tiny, converting, monkeypatch
):
    asked = losing(tiny, monkeypatch, at_q8_0=lambda name: 0.5 if "ffn_down" in name else 1.0)
    said = []
    tiny.report = said.append
    shipped = export.export(tiny, EDIT)
    edited = edited_in_gguf(tiny)
    down = {name for name in edited if "ffn_down" in name}
    assert down and down != edited
    plain = convert.tensor_types(tiny.gguf()[1])
    held = dict.fromkeys(edited - down, "Q8_0") | dict.fromkeys(down, "F32")
    # F32 is written straight into the file, not asked of llama-quantize.
    assert asked == [None, plain | dict.fromkeys(edited, "Q8_0")]
    assert convert.tensor_types(shipped) == plain | held
    assert any(
        line.startswith("Q8_0 keeps ")
        and line.endswith(f"in {len(down)} of {len(edited)} edited weights; storing those as F32")
        for line in said
    )


def test_a_run_deploying_q8_0_stores_what_it_rounds_away_as_f32(tiny, converting, monkeypatch):
    tiny.settings.deployment.quantization = "Q8_0"
    asked = losing(tiny, monkeypatch)
    shipped = export.export(tiny, EDIT)
    assert shipped.name == "tiny-trial-Q8_0.gguf"
    plain = convert.tensor_types(tiny.gguf()[1])
    assert asked == [None]
    assert convert.tensor_types(shipped) == plain | dict.fromkeys(edited_in_gguf(tiny), "F32")


def test_a_weight_kept_in_f32_holds_its_change_unrounded_in_one_gguf(tiny, converting, monkeypatch):
    # The tiny model's float type is BF16, which the converter rounds the weights to.
    monkeypatch.setattr(
        convert,
        "to_gguf",
        lambda folder, destination, log: gguf_from_folder(
            folder, destination, log, torch.bfloat16, types=TYPE.BF16
        ),
    )
    asked = losing(tiny, monkeypatch, at_q8_0=0.5)
    said = []
    tiny.report = said.append
    shipped = export.export(tiny, EDIT)
    assert sorted(p.name for p in shipped.parent.iterdir()) == [shipped.name, f"{shipped.name}.inputs.json"]
    original = load_file(tiny.model_folder / "model.safetensors")
    stored = {tensor.name: tensor for tensor in gguf.GGUFReader(shipped).tensors}
    for name, (b, a) in export.deltas(tiny, EDIT).items():
        tensor = stored[tiny.target_names[name]]
        assert tensor.tensor_type == TYPE.F32
        assert torch.equal(convert.stored(tensor), original[name].float() + b.float() @ a.float())
    assert any(line.startswith("Written in BF16, those kept ") for line in said)
    edited = edited_in_gguf(tiny)
    record = json.loads(run.record_of(shipped).read_text())
    assert record["apart"] == dict.fromkeys(edited, "F32")
    assert export.export(tiny, EDIT) == shipped and len(asked) == 2
    Measurement(tiny, EDIT, "merged", exported=True).measure()


def test_an_export_recorded_before_it_listed_what_it_holds_apart_is_reused_unless_others_moved(
    tiny, converting, monkeypatch
):
    asked = losing(tiny, monkeypatch, at_q8_0=0.5)
    shipped = export.export(tiny, EDIT)
    record = json.loads(run.record_of(shipped).read_text())
    del record["apart"]  # as candor recorded exports before it listed the tensors they hold apart
    run.record_of(shipped).write_text(json.dumps(record))
    assert export.export(tiny, EDIT) == shipped and len(asked) == 2
    Measurement(tiny, EDIT, "reused", exported=True).measure()
    # An unedited tensor's type moved, as llama-quantize moved some past a tensor given its own type.
    moved = tiny.folder / "moved.gguf"
    convert.quantize(
        shipped,
        moved,
        "Q4_K_M",
        tiny.folder / "log",
        convert.tensor_types(shipped) | {"blk.0.attn_q.weight": "Q8_0"},
    )
    moved.replace(shipped)
    with pytest.raises(RunError, match="trial has no export matching this run"):
        Measurement(tiny, EDIT, "stale", exported=True).measure()


def to_a_float_type(base, destination, quantization, log, types=None):
    """llama-quantize given a float type: every matrix in that type, whatever types asks."""
    weights = {tensor.name: convert.stored(tensor).numpy() for tensor in gguf.GGUFReader(base).tensors}
    kinds = {name: TYPE[quantization] for name, values in weights.items() if values.ndim == 2}
    write_model(destination, weights, TEMPLATE, types=kinds)


# An edit too small for 16 bits to hold: under half a step of the weights it changes.
TINY = Edit("trial", 0.4, {"attn": Kernel(1e-5, 1.0, 1e-5, 1.0), "mlp": Kernel(1e-5, 0.0, 1e-5, 1.0)})


@pytest.mark.parametrize(
    "deployed",
    # BF16 weights are exact in F16, so only the edit's rounding moves what the export reads.
    ["BF16", "F16"],
    ids=["its float type", "another float type"],
)
def test_a_float_deployment_stores_what_16_bits_round_away_in_f32(tiny, converting, monkeypatch, deployed):
    # A checkpoint stored in BF16, as real ones are, so its GGUF holds every weight exactly.
    weights = tiny.model_folder / "model.safetensors"
    save_file({name: tensor.to(torch.bfloat16) for name, tensor in load_file(weights).items()}, weights)
    monkeypatch.setattr(
        convert,
        "to_gguf",
        lambda folder, destination, log: gguf_from_folder(
            folder, destination, log, torch.bfloat16, types=TYPE.BF16
        ),
    )
    monkeypatch.setattr(convert, "quantize", to_a_float_type)
    tiny.settings.deployment.quantization = deployed
    said = []
    tiny.report = said.append
    shipped = export.export(tiny, TINY)
    edited = {tiny.target_names[name] for name in export.deltas(tiny, TINY)}
    assert edited and {convert.tensor_types(shipped)[name] for name in edited} == {"F32"}
    assert json.loads(run.record_of(shipped).read_text())["apart"] == dict.fromkeys(edited, "F32")
    assert any(line.endswith("storing those as F32") for line in said)
    held = export.kept(tiny, shipped, export.deltas(tiny, TINY))
    assert all(share == pytest.approx(1, abs=1e-6) for share in held.values())


def test_a_bf16_gguf_rewritten_with_some_tensors_in_f32_keeps_everything_else(tmp_path):
    weights = random_weights()
    write_model(
        tmp_path / "bf16.gguf",
        weights,
        TEMPLATE,
        types={name: TYPE.BF16 for name, array in weights.items() if array.ndim == 2},
    )
    # Values BF16 can't hold, as an unrounded change's.
    replaced = {
        name: weights[name] * np.float32(1.001)
        for name in ("blk.0.ffn_down.weight", "blk.1.attn_output.weight")
    }
    functions = {name: partial(values.copy) for name, values in replaced.items()}
    export._rewritten(tmp_path / "bf16.gguf", tmp_path / "f32.gguf", functions, export.F32)
    # Without a type, each keeps its own, rounded as the converter rounds.
    export._rewritten(tmp_path / "bf16.gguf", tmp_path / "rounded.gguf", functions)
    for tensor in gguf.GGUFReader(tmp_path / "rounded.gguf").tensors:
        if tensor.name in replaced:
            assert tensor.tensor_type == TYPE.BF16
            expected = torch.from_numpy(replaced[tensor.name]).to(torch.bfloat16).float()
            assert torch.equal(convert.stored(tensor), expected)
    before, after = gguf.GGUFReader(tmp_path / "bf16.gguf"), gguf.GGUFReader(tmp_path / "f32.gguf")

    def fields(reader):
        return [
            (f.name, f.types, f.contents()) for f in reader.fields.values() if not f.name.startswith("GGUF.")
        ]

    assert fields(after) == fields(before)
    assert [t.name for t in after.tensors] == [t.name for t in before.tensors]
    unchanged = {t.name: t for t in before.tensors}
    for tensor in after.tensors:
        if tensor.name in replaced:
            assert tensor.tensor_type == TYPE.F32
            assert np.array_equal(convert.stored(tensor).numpy(), replaced[tensor.name])
        else:
            assert tensor.tensor_type == unchanged[tensor.name].tensor_type
            assert np.array_equal(np.asarray(tensor.data), np.asarray(unchanged[tensor.name].data))


def test_a_file_that_still_falls_short_in_f32_is_refused(tiny, converting, monkeypatch):
    asked = losing(tiny, monkeypatch, 0.5, 0.5)
    by_gguf_name = {gguf_name: name for name, gguf_name in tiny.target_names.items()}
    # As if what was written in F32 didn't hold the change.
    monkeypatch.setattr(
        export, "_values", lambda tensor: tiny_weight(tiny, by_gguf_name[tensor.name]).numpy()
    )
    with pytest.raises(RunError, match=r"keep less than 98% of the edit even stored as F32, e\.g\. .* 0\.0%"):
        export.export(tiny, EDIT)
    assert len(asked) == 2
    assert not tiny.path("exports", "trial", "tiny-trial-Q4_K_M.gguf").exists()


def tiny_weight(tiny, name):
    return export.ModelFiles(tiny.model_folder).tensor(name).float()


def test_an_edit_lost_even_written_unrounded_is_refused(tiny, converting, monkeypatch):
    asked = losing(tiny, monkeypatch)
    monkeypatch.setattr(export, "_written", lambda run, name, change: tiny_weight(run, name).numpy())
    with pytest.raises(
        RunError, match="keep less than 98% of the edit even written unrounded, in F32, before quantising"
    ):
        export.export(tiny, EDIT)
    assert asked == [None]


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


@pytest.mark.parametrize("kind", ["Q8_0", "F32"])
def test_llama_quantize_given_every_tensors_type_changes_only_the_one_given_another(
    tmp_path, monkeypatch, kind
):
    if not LLAMA_SERVER.exists():
        pytest.skip("no llama.cpp")
    monkeypatch.setenv("CANDOR_LLAMA_BIN", str(LLAMA_SERVER.parent))
    # Wide enough for k-quants, and deep enough for Q4_K_M to give ffn_down a type by its layer.
    write_model(tmp_path / "f32.gguf", random_weights(hidden=256, ffn=512, layers=8))
    real_quantize(tmp_path / "f32.gguf", tmp_path / "plain.gguf", "Q4_K_M", tmp_path / "log")
    plain = convert.tensor_types(tmp_path / "plain.gguf")
    assert {plain[f"blk.{i}.ffn_down.weight"] for i in range(8)} == {"Q4_K", "Q6_K"}
    apart = tmp_path / "apart.gguf"
    real_quantize(
        tmp_path / "f32.gguf", apart, "Q4_K_M", tmp_path / "log", plain | {"blk.1.ffn_down.weight": kind}
    )
    assert convert.tensor_types(apart) == plain | {"blk.1.ffn_down.weight": kind}


def test_a_failed_export_leaves_nothing_behind(tiny, monkeypatch):
    tiny.gguf()

    def broken(base, destination, *_):
        destination.write_bytes(b"half")
        raise convert.ConversionError("llama-quantize failed")

    monkeypatch.setattr(convert, "quantize", broken)
    with pytest.raises(convert.ConversionError):
        export.export(tiny, EDIT)
    assert list(tiny.path("exports", "trial").iterdir()) == []


def test_only_an_edit_has_an_export_to_measure(tiny):
    with pytest.raises(ValueError, match="only an edit"):
        Measurement(tiny, None, "base", exported=True)


def test_export_and_measure_it_from_the_command_line(tiny, converting):
    folder = on_disk(tiny)
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


def test_the_commands_export_suggests_name_the_measurements_measure_makes_at_the_runs_slots(tiny, converting):
    tiny.settings.slots.measure = 3
    folder = on_disk(tiny)
    journal = write_journal(tiny.folder.parent / "heretic.jsonl", 0.25)
    runner = CliRunner(env={"COLUMNS": "250"})
    exported = runner.invoke(app, ["export", str(folder), "--heretic", str(journal), "--trial", "1"])
    assert exported.exit_code == 0, exported.output
    suggested = [line.split()[3:] for line in exported.output.splitlines() if "uv run candor" in line]
    assert suggested[1:] == [
        ["compare", str(folder), "base-slots-3", "heretic-1-exported-slots-3"],
        ["compare", str(folder), "heretic-1-slots-3", "heretic-1-exported-slots-3"],
    ]
    measured = runner.invoke(app, suggested[0])
    assert "candor measure: heretic-1-exported-slots-3" in measured.output


def test_exported_needs_a_trial(tiny):
    result = CliRunner().invoke(app, ["measure", str(tiny.folder), "--exported"])
    assert result.exit_code == 2 and "--trial" in result.output


def saved_elsewhere(tiny, rename=lambda name: name, called="saved"):
    """A copy of the tiny model with EDIT in its weights, as another tool would save it."""
    folder = tiny.folder.parent / called
    shutil.copytree(tiny.model_folder, folder)
    weights = load_file(folder / "model.safetensors")
    for name, change in export.deltas(tiny, EDIT).items():
        weights[name] = export.edited(weights[name], change).to(weights[name].dtype)
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


def gguf_holding(tiny, path, folder, kind=TYPE.F32, arch: str | None = "llama", only=None):
    """A GGUF of only the architecture, if any, and the edit targets of a Hugging Face folder in one
    type, as taken_from_gguf reads one."""
    weights = load_file(folder / "model.safetensors")
    writer = gguf.GGUFWriter(path, arch or "llama")
    if arch is None:
        del writer.kv_data[0][gguf.Keys.General.ARCHITECTURE]
    for name, gguf_name in list(tiny.target_names.items())[:only]:
        array = weights[name].numpy()
        if kind in (TYPE.F32, TYPE.F16):
            writer.add_tensor(gguf_name, array.astype(np.float32 if kind == TYPE.F32 else np.float16))
        else:
            writer.add_tensor(gguf_name, gguf.quants.quantize(array, kind), raw_dtype=kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def test_a_gguf_edited_elsewhere_ships_and_measures_as_the_same_edit_of_candors(tiny, converting):
    tiny.gguf()
    path = gguf_holding(tiny, tiny.folder.parent / "saved.gguf", saved_elsewhere(tiny))
    said = []
    tiny.report = said.append
    shipped = export.export_outside(tiny, "saved", path)
    assert f"Taking the {len(export.deltas(tiny, EDIT))} edited weights from {path}" in said
    export.export(tiny, EDIT)
    theirs = Measurement(tiny, None, "saved", outside="saved").measure()
    ours = Measurement(tiny, EDIT, "merged", exported=True).measure()
    assert kl_between(ours, theirs) < 1e-5
    assert export.export_outside(tiny, "saved", path) == shipped and len(converting) == 2
    path.touch()
    with pytest.raises(RunError, match="saved's export no longer matches this run or"):
        Measurement(tiny, None, "saved", outside="saved").measure()


@pytest.mark.parametrize("kind", [TYPE.F16, TYPE.BF16], ids=["F16", "BF16"])
def test_a_gguf_in_a_narrower_type_counts_only_weights_the_edit_moved(tiny, converting, kind):
    tiny.gguf()
    unedited = gguf_holding(tiny, tiny.folder.parent / "unedited.gguf", tiny.model_folder, kind)
    with pytest.raises(RunError, match="holds the run's model unchanged in every edit target"):
        export.export_outside(tiny, "unedited", unedited)
    edited = gguf_holding(tiny, tiny.folder.parent / "edited.gguf", saved_elsewhere(tiny), kind)
    _, _, _, writers = tiny.shape
    assert set(export.taken_from_gguf(tiny, edited)) == {w.name for w in writers if EDIT.strength(w)}


def test_ggufs_edited_elsewhere_that_cant_be_taken_are_refused(tiny, converting):
    tiny.gguf()
    folder = saved_elsewhere(tiny)
    made = tiny.folder.parent
    refused = {
        "quantised already": gguf_holding(tiny, made / "q8.gguf", folder, TYPE.Q8_0),
        "holds a qwen2 model": gguf_holding(tiny, made / "qwen.gguf", folder, arch="qwen2"),
        "doesn't say which architecture": gguf_holding(tiny, made / "unnamed.gguf", folder, arch=None),
        "has no tensor blk.": gguf_holding(tiny, made / "part.gguf", folder, only=1),
        "one part of a split GGUF": gguf_holding(tiny, made / "model-00001-of-00002.gguf", folder),
    }
    for message, path in refused.items():
        with pytest.raises(RunError, match=message):
            export.export_outside(tiny, path.stem, path)
    text = made / "notes.txt"
    text.write_text("not a model")
    with pytest.raises(RunError, match="isn't a folder or a GGUF file"):
        export.export_outside(tiny, "notes", text)
    assert converting == []


def test_the_command_line_exports_and_measures_a_model_edited_elsewhere(tiny, converting):
    tiny.gguf()
    folder = saved_elsewhere(tiny)
    run_folder = on_disk(tiny)
    runner = CliRunner(env={"COLUMNS": "200"})
    unnamed = runner.invoke(app, ["export", str(run_folder), "--outside", str(folder)])
    assert unnamed.exit_code == 2 and "--name" in unnamed.output
    both = runner.invoke(
        app, ["export", str(run_folder), "--trial", "1", "--outside", str(folder), "--name", "x"]
    )
    assert both.exit_code == 2
    heretic = ["--heretic", str(folder / "study.jsonl")]
    ignored = runner.invoke(
        app, ["export", str(run_folder), "--outside", str(folder), "--name", "x", *heretic]
    )
    assert ignored.exit_code == 2 and "edited elsewhere on its own" in " ".join(ignored.output.split())
    exported = runner.invoke(app, ["export", str(run_folder), "--outside", str(folder), "--name", "saved"])
    assert exported.exit_code == 0, exported.output
    assert f"uv run candor measure {run_folder} --outside saved" in exported.output
    measured = runner.invoke(app, ["measure", str(run_folder), "--outside", "saved"])
    assert measured.exit_code == 0, measured.output
    assert (run_folder / "measurements" / "saved" / "maths.json").exists()
