"""Mixture-of-experts models: a layer's experts are one edit target, stacked as llama.cpp stores them,
each expert edited along its layer's direction; served as 3D adapters, exported merged."""

import shutil
from functools import partial
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from typer.testing import CliRunner

from candor import convert, export, run
from candor.adapters import write_adapter
from candor.cli import _target_rows, app
from candor.convert import check_targets
from candor.edit import RowNorm, edit_factors, full, rank_one
from candor.engine import LlamaServer
from candor.measure import Edit, Measurement
from candor.run import RunError
from candor.weights import ModelFiles

from .helpers import EDIT, LLAMA_SERVER, TEMPLATE, mean_from_top, on_disk, write_journal
from .test_export import TINY, TYPE, kl_between
from .tiny_gguf import FIRST_WORD, HIDDEN
from .tiny_moe import EXPERT_FFN, EXPERTS, LAYERS, moe_gguf_from_folder, random_moe_weights, write_moe_model
from .tiny_run import tiny_run

PROMPTS = [[1, FIRST_WORD + i, FIRST_WORD + 2 * i + 1, FIRST_WORD + 3] for i in range(6)]
EXPERTS_1 = "blk.1.ffn_down_exps.weight"


@pytest.fixture(params=[False, True], ids=["experts apart", "experts fused"])
def tiny_moe(tmp_path, monkeypatch, request):
    made = tiny_run(tmp_path, monkeypatch, moe=True, fused=request.param)
    monkeypatch.setattr(
        run.convert,
        "to_gguf",
        lambda folder, destination, log: moe_gguf_from_folder(folder, destination, TEMPLATE),
    )
    monkeypatch.setattr(run.convert, "check_targets", check_targets)
    return made


def direction(seed: int = 5) -> torch.Tensor:
    return F.normalize(torch.randn(HIDDEN, generator=torch.Generator().manual_seed(seed)), dim=0)


def test_stacked_experts_get_the_factors_each_would_get_alone():
    experts = torch.from_numpy(random_moe_weights()[EXPERTS_1])
    r = direction()
    for row_norm in (RowNorm.NONE, RowNorm.PRE):
        b, a = rank_one(experts, r, row_norm)
        assert b.shape == (EXPERTS, HIDDEN, 1) and a.shape == (EXPERTS, 1, EXPERT_FFN)
        for e in range(EXPERTS):
            alone = rank_one(experts[e], r, row_norm)
            torch.testing.assert_close((b[e], a[e]), alone)
    b, a = full(experts, r, 1.3)
    for e in range(EXPERTS):
        torch.testing.assert_close(b[e] @ a[e], torch.matmul(*full(experts[e], r, 1.3)), atol=1e-6, rtol=1e-5)


def test_removing_a_direction_from_every_expert_removes_it_from_the_blocks_output():
    """Without row normalisation the edit is a projection of the block's output, whichever experts the
    router picks and however it weighs them, as the router's weights scale each expert's output."""
    torch.manual_seed(0)
    experts, shared = torch.randn(EXPERTS, HIDDEN, EXPERT_FFN), torch.randn(HIDDEN, EXPERT_FFN)
    r = direction()
    (b, a), (b_shared, a_shared) = rank_one(experts, r, RowNorm.NONE), rank_one(shared, r, RowNorm.NONE)
    edited, edited_shared = experts + b @ a, shared + b_shared @ a_shared
    x, weights = torch.randn(EXPERTS, EXPERT_FFN), torch.rand(EXPERTS)
    out = torch.einsum("e,eoi,ei->o", weights, experts, x) + shared @ x[0]
    out_edited = torch.einsum("e,eoi,ei->o", weights, edited, x) + edited_shared @ x[0]
    torch.testing.assert_close(out_edited, out - r * (r @ out), atol=1e-5, rtol=1e-5)


def test_expert_adapters_at_their_strengths_match_the_merged_model(tmp_path):
    if not LLAMA_SERVER.exists():
        pytest.skip("no llama-server")
    weights = random_moe_weights()
    write_moe_model(tmp_path / "base.gguf", weights)
    merged, scales = dict(weights), {}
    edits = {EXPERTS_1: (RowNorm.PRE, 1.5), "blk.0.ffn_down_exps.weight": (RowNorm.NONE, 0.7)}
    for name, (row_norm, strength) in edits.items():
        b, a = rank_one(torch.from_numpy(weights[name]), direction(), row_norm)
        write_adapter(tmp_path / f"{name}.gguf", "qwen3moe", {name: (b, a)})
        scales[tmp_path / f"{name}.gguf"] = strength
        merged[name] = weights[name] + strength * (b @ a).numpy()
    write_moe_model(tmp_path / "merged.gguf", merged)

    def server(model, adapters=()):
        return LlamaServer(
            LLAMA_SERVER,
            tmp_path / model,
            tmp_path / "server.log",
            list(adapters),
            parallel=2,
            context_per_slot=64,
        )

    with server("base.gguf") as base, server("merged.gguf") as merged_model:
        unedited, expected = base.top_logprobs(PROMPTS, 50), merged_model.top_logprobs(PROMPTS, 50)
    with server("base.gguf", scales) as s:
        served = s.top_logprobs(PROMPTS, 50, scales)
    assert mean_from_top(served, expected) < 1e-6
    assert mean_from_top(unedited, expected) > 1e-4


def test_a_layers_experts_are_one_edit_target_in_expert_order(tiny_moe):
    found = tiny_moe.edit_targets
    fused = "model.layers.0.mlp.experts.down_proj" in found
    experts = "model.layers.{}.mlp.experts.down_proj" + ("" if fused else ".weight")
    assert list(found) == [
        "model.layers.0.self_attn.o_proj.weight",
        experts.format(0),
        "model.layers.1.self_attn.o_proj.weight",
        experts.format(1),
    ]
    target = found[experts.format(1)]
    assert target.gguf == EXPERTS_1 and target.writers[0].component == "mlp"
    apart = [f"model.layers.1.mlp.experts.{e}.down_proj.weight" for e in range(EXPERTS)]
    assert target.names == ([experts.format(1)] if fused else apart)
    assert len(tiny_moe.target_names) == LAYERS * (2 if fused else EXPERTS + 1)


def test_the_converted_gguf_holds_each_layers_experts_stacked(tiny_moe, tmp_path):
    base, _ = tiny_moe.gguf()
    targets = tiny_moe.edit_targets.values()
    assert check_targets(base, tiny_moe.model_folder, targets, "qwen3moe") == []
    weights = {t.name: np.array(t.data) for t in gguf.GGUFReader(base).tensors}
    weights[EXPERTS_1] = weights[EXPERTS_1][[1, 0, *range(2, EXPERTS)]]
    write_moe_model(tmp_path / "swapped.gguf", weights)
    experts = next(t.name for t in targets if t.gguf == EXPERTS_1)
    assert check_targets(tmp_path / "swapped.gguf", tiny_moe.model_folder, targets, "qwen3moe") == [
        f"{EXPERTS_1} doesn't hold the same values as {experts}; the converter changed it"
    ]
    weights[EXPERTS_1] = weights[EXPERTS_1][:-1]
    write_moe_model(tmp_path / "fewer.gguf", weights)
    assert check_targets(tmp_path / "fewer.gguf", tiny_moe.model_folder, targets, "qwen3moe") == [
        f"{experts} is {[EXPERTS, HIDDEN, EXPERT_FFN]} but {EXPERTS_1} is {[EXPERTS - 1, HIDDEN, EXPERT_FFN]}"
    ]


def test_adapters_hold_each_layers_experts_as_one_3d_tensor(tiny_moe):
    made = tiny_moe.adapters()
    adapter = made["per-layer"][EXPERTS_1]
    tensors = {t.name: list(t.shape) for t in gguf.GGUFReader(adapter).tensors}
    assert tensors == {
        f"{EXPERTS_1}.lora_a": [EXPERT_FFN, 1, EXPERTS],
        f"{EXPERTS_1}.lora_b": [1, HIDDEN, EXPERTS],
    }
    assert set(made["per-layer"]) == {t.gguf for t in tiny_moe.edit_targets.values()}


@pytest.mark.parametrize(
    ("row_norm", "edit"),
    [("pre", EDIT), ("none", Edit("trial", None, EDIT.kernels)), ("full", Edit("trial", None, EDIT.kernels))],
    ids=["pre, one layer", "none, per layer", "full, per layer"],
)
def test_the_export_answers_as_the_model_with_the_edits_adapters(tiny_moe, row_norm, edit):
    tiny_moe.settings.edit.row_norm = row_norm
    shipped = export.export(tiny_moe, edit)
    base = Measurement(tiny_moe, None, "base").measure()
    adapters = Measurement(tiny_moe, edit, "adapters").measure()
    merged = Measurement(tiny_moe, edit, "merged", exported=True).measure()
    assert kl_between(adapters, merged) < 1e-5
    assert kl_between(base, merged) > 1e-3
    held = export.kept(tiny_moe, shipped, export.deltas(tiny_moe, edit))
    assert EXPERTS_1 in held and all(share == pytest.approx(1, abs=1e-4) for share in held.values())


def test_the_least_any_expert_keeps_is_its_layers_share(tiny_moe, tmp_path):
    shipped = export.export(tiny_moe, EDIT)
    deltas = export.deltas(tiny_moe, EDIT)
    weights = {t.name: np.array(t.data) for t in gguf.GGUFReader(shipped).tensors}
    base, _ = tiny_moe.gguf()
    unedited = {t.name: np.array(t.data) for t in gguf.GGUFReader(base).tensors}
    weights[EXPERTS_1][3] = unedited[EXPERTS_1][3]
    write_moe_model(tmp_path / "one-expert-lost.gguf", weights)
    held = export.kept(tiny_moe, tmp_path / "one-expert-lost.gguf", deltas)
    assert held[EXPERTS_1] == pytest.approx(0, abs=1e-6)
    assert all(share == pytest.approx(1, abs=1e-4) for name, share in held.items() if name != EXPERTS_1)


def test_inspect_passes_a_model_with_experts(tiny_moe):
    rows = _target_rows(str(tiny_moe.model_folder), HIDDEN, LAYERS, "qwen3moe", None)
    assert ("Experts", "ok", f"{EXPERTS} experts in each of {LAYERS} layers, all edited along the layer's "
            "direction at its MLP strength") in rows  # fmt: skip
    names = next(detail for row, status, detail in rows if row == "GGUF names" and status == "ok")
    assert (
        "model.layers.0.mlp.experts.down_proj" in names
        and "-> blk.0.ffn_down_exps.weight, a layer's" in names
    )


@pytest.mark.parametrize("row_norm", ["pre", "none", "full"])
def test_each_exported_expert_is_its_own_weight_edited_alone(tiny_moe, row_norm):
    """Checked weight by weight, as an expert the measured tokens never reach answers the same edited
    or not."""
    tiny_moe.settings.edit.row_norm = row_norm
    shipped = export.export(tiny_moe, EDIT)
    stored = {t.name: convert.stored(t) for t in gguf.GGUFReader(shipped).tensors}
    directions = tiny_moe.edit_directions(EDIT.name, EDIT.direction_layer)
    files = ModelFiles(tiny_moe.model_folder)
    for target in tiny_moe.edit_targets.values():
        strength = EDIT.strength(target.writers[0])
        experts = files.read(target.names, target.stacked).float()
        for e, expert in enumerate(experts if experts.dim() == 3 else []):
            b, a = edit_factors(expert, directions[target.layer], strength, RowNorm(row_norm))
            torch.testing.assert_close(stored[target.gguf][e], expert + b @ a, atol=1e-5, rtol=1e-5)


def to_a_float_type(base, destination, quantization, log, types=None):
    """llama-quantize given a float type: every matrix in that type, whatever types asks."""
    weights = {tensor.name: convert.stored(tensor).numpy() for tensor in gguf.GGUFReader(base).tensors}
    kinds = {name: TYPE[quantization] for name, values in weights.items() if values.ndim >= 2}
    write_moe_model(destination, weights, TEMPLATE, kinds)


def test_a_float_deployment_stores_experts_16_bits_round_away_in_f32(tiny_moe, monkeypatch):
    weights = tiny_moe.model_folder / "model.safetensors"
    save_file({name: tensor.to(torch.bfloat16) for name, tensor in load_file(weights).items()}, weights)
    monkeypatch.setattr(
        run.convert,
        "to_gguf",
        lambda folder, destination, log: moe_gguf_from_folder(folder, destination, TEMPLATE, TYPE.BF16),
    )
    monkeypatch.setattr(run.convert, "quantize", to_a_float_type)
    tiny_moe.settings.deployment.quantization = "BF16"
    shipped = export.export(tiny_moe, TINY)
    edited = {tiny_moe.edit_targets[name].gguf for name in export.deltas(tiny_moe, TINY)}
    assert EXPERTS_1 in edited and {convert.tensor_types(shipped)[name] for name in edited} == {"F32"}
    held = export.kept(tiny_moe, shipped, export.deltas(tiny_moe, TINY))
    assert all(share == pytest.approx(1, abs=1e-6) for share in held.values())


def folder_with_experts(tmp_path, shape) -> str:
    """A model folder's headers with fused experts of this shape after a Qwen3-MoE's attention."""
    folder = tmp_path / "fused"
    folder.mkdir()
    tensors = {
        f"model.layers.{i}.self_attn.o_proj.weight": torch.zeros(HIDDEN, HIDDEN) for i in range(LAYERS)
    }
    tensors |= {f"model.layers.{i}.mlp.experts.down_proj": torch.zeros(shape) for i in range(LAYERS)}
    save_file(tensors, folder / "model.safetensors")
    return str(folder)


@pytest.mark.parametrize(
    "shape", [(EXPERTS, EXPERT_FFN, HIDDEN), (EXPERTS, HIDDEN, HIDDEN)], ids=["hidden last", "square"]
)
def test_inspect_fails_experts_it_cant_tell_the_layout_of(tmp_path, shape):
    rows = _target_rows(folder_with_experts(tmp_path, shape), HIDDEN, LAYERS, "qwen3moe", None)
    [(status, detail)] = [(status, detail) for row, status, detail in rows if row == "Experts"]
    assert status == "fail" and "[experts, in, hidden], or square" in detail
    assert not any(row == "GGUF names" for row, _, _ in rows)


def test_a_run_refuses_experts_it_cant_tell_the_layout_of(tiny_moe, tmp_path):
    folder = Path(folder_with_experts(tmp_path, (EXPERTS, EXPERT_FFN, HIDDEN)))
    shutil.copy(folder / "model.safetensors", tiny_moe.model_folder / "model.safetensors")
    with pytest.raises(RunError, match="no longer passes candor inspect"):
        _ = tiny_moe.shape


def saved_elsewhere(tiny_moe) -> Path:
    """A copy of the tiny MoE with EDIT in its weights, each expert written back where the model keeps
    it, as another tool would save it."""
    folder = tiny_moe.folder.parent / "saved"
    shutil.copytree(tiny_moe.model_folder, folder)
    weights = load_file(folder / "model.safetensors")
    files = ModelFiles(tiny_moe.model_folder)
    for name, change in export.deltas(tiny_moe, EDIT).items():
        target = tiny_moe.edit_targets[name]
        edited = export.edited(files.read(target.names, target.stacked), change)
        for i, weight in enumerate(target.names):
            weights[weight] = (edited[i] if target.stacked else edited).to(weights[weight].dtype)
    save_file(weights, folder / "model.safetensors")
    return folder


@pytest.mark.parametrize("given", ["folder", "gguf"])
def test_a_moe_edited_elsewhere_ships_and_measures_as_the_same_edit_of_candors(tiny_moe, given):
    base, _ = tiny_moe.gguf()
    deltas = export.deltas(tiny_moe, EDIT)
    if given == "folder":
        path = saved_elsewhere(tiny_moe)
    else:
        path = tiny_moe.folder.parent / "saved.gguf"
        written = {
            tiny_moe.edit_targets[n].gguf: partial(export._written, tiny_moe, n, c) for n, c in deltas.items()
        }
        export._rewritten(base, path, written, export.F32)
    assert set(export._taken(tiny_moe, path)) == set(deltas)
    export.export_outside(tiny_moe, "saved", path)
    export.export(tiny_moe, EDIT)
    theirs = Measurement(tiny_moe, None, "saved", outside="saved").measure()
    ours = Measurement(tiny_moe, EDIT, "merged", exported=True).measure()
    assert kl_between(ours, theirs) < 1e-5


def test_a_heretic_trial_isnt_rebuilt_on_a_model_with_experts(tiny_moe):
    folder = on_disk(tiny_moe)
    journal = write_journal(tiny_moe.folder.parent / "heretic.jsonl", 0.25)
    result = CliRunner(env={"COLUMNS": "250"}).invoke(
        app, ["export", str(folder), "--heretic", str(journal), "--trial", "1"]
    )
    assert result.exit_code == 1
    assert "whether heretic edited the model's experts" in " ".join(result.output.split())
