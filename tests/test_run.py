import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gguf
import pytest
import torch
import torch.nn.functional as F
from datasets.exceptions import DatasetNotFoundError
from transformers import AutoModelForCausalLM, Qwen4ExpForCausalLM, Qwen4ExpTextConfig
from transformers.integrations.accelerate import _get_device_map, expand_device_map

from candor import config, directions, files, residuals, run, targets
from candor.edit import RowNorm, edit_factors, rank_one
from candor.engine import LlamaServer
from candor.kernel import Kernel
from candor.measure import Edit
from candor.run import Run, RunError
from candor.weights import ModelFiles

from .custom_code import write_custom_code_model
from .helpers import (
    FLASH_NEXT,
    HIDDEN,
    LAYERS,
    LLAMA_SERVER,
    TEMPLATE,
    TINY_QWEN4EXP,
    adapter_factors,
    fake_everything_but_the_stages,
    fake_residuals,
    llama,
    pinned,
    residuals_of,
    short,
    tiny_qwen4exp,
    write_llama,
    write_qwen4exp,
)
from .tiny_gguf import random_weights, write_model

GGUF = {"gguf/model-BF16.gguf", "gguf/model-Q4_K_M.gguf"}
ADAPTERS = {"adapters/per-layer", "adapters/layer-0", "adapters/layer-1"}


@pytest.fixture
def directions_by_seed():
    """Directions the fake directions stage returns, changeable to stand for a real change upstream."""
    state = {"seed": 0}

    def found() -> torch.Tensor:
        generator = torch.Generator().manual_seed(state["seed"])
        return F.normalize(torch.randn(LAYERS, HIDDEN, generator=generator), dim=-1)

    return state, found


@pytest.fixture
def current(tmp_path, model, monkeypatch, directions_by_seed):
    """A run whose directions stage is faked: these tests are about adapters and records."""
    state, found = directions_by_seed
    settings = config.RunConfig(model=config.Model(source=str(model)), measure=short())
    folder = tmp_path / "runs" / "test"
    folder.mkdir(parents=True)
    messages = []

    def fake_directions(self, answering=False):
        self.inputs["directions"] = f"seed-{state['seed']}"
        return found()

    monkeypatch.setattr(Run, "directions", fake_directions)
    return Run(folder, settings, report=messages.append), messages


def rebuilt(messages: list[str]) -> set[str]:
    return {m.removeprefix("Rebuilding ").split(":")[0] for m in messages if m.startswith("Rebuilding ")}


def test_one_adapter_per_target_for_each_direction_set(current):
    made = current[0].adapters()
    assert sorted(made) == ["layer-0", "layer-1", "per-layer"]
    for paths in made.values():
        assert sorted(paths) == [
            f"blk.{i}.{kind}.weight" for i in range(2) for kind in ("attn_output", "ffn_down")
        ]
        assert all(path.exists() for path in paths.values())


def test_each_weight_is_read_once_for_every_adapter_set(current, monkeypatch):
    read, real = [], ModelFiles.tensor
    monkeypatch.setattr(ModelFiles, "tensor", lambda self, name: read.append(name) or real(self, name))
    made = current[0].adapters()
    # 4 targets, for 3 sets.
    assert len(made) == 3 and sorted(read) == sorted(set(read)) and len(read) == 4


@pytest.mark.parametrize("row_norm", ["pre", "none"])
def test_adapters_hold_the_edit_for_their_layers_direction(current, model, directions_by_seed, row_norm):
    this, _ = current
    this.settings.edit.row_norm = row_norm
    made = this.adapters()
    found = directions_by_seed[1]()
    weight = ModelFiles(model).tensor("model.layers.1.mlp.down_proj.weight")
    for set_name, direction in (("per-layer", found[1]), ("layer-0", found[0])):
        b, a = rank_one(weight, direction, RowNorm(row_norm))
        stored_b, stored_a = adapter_factors(made[set_name]["blk.1.ffn_down.weight"])
        torch.testing.assert_close(stored_a, a)
        torch.testing.assert_close(stored_b, b)


def test_a_full_edit_has_one_adapter_built_at_its_strengths_and_no_shared_ones(
    current, model, directions_by_seed
):
    this, messages = current
    this.settings.edit.row_norm = "full"
    assert this.adapters() == {}
    edit = Edit("e", None, {"mlp": Kernel(1.3, 1.0, 0.0, 0.5)})
    path, inputs = this.edit_adapter(this.path("scores", "edits", "e"), edit)
    assert inputs["row_norm"] == "full" and inputs["edit"] == edit.record()
    tensors = {t.name for t in gguf.GGUFReader(path).tensors}
    # Only the weights the edit changes: the MLP's at layer 1, where its strength peaks.
    assert tensors == {"blk.1.ffn_down.weight.lora_a", "blk.1.ffn_down.weight.lora_b"}
    weight = ModelFiles(model).tensor("model.layers.1.mlp.down_proj.weight")
    b, a = edit_factors(weight, directions_by_seed[1]()[1], 1.3, RowNorm.FULL)
    stored_b, stored_a = adapter_factors(path)
    torch.testing.assert_close(stored_b @ stored_a, b @ a)
    assert stored_a.shape[0] == 3
    stamp = path.stat().st_mtime_ns
    messages.clear()
    assert this.edit_adapter(this.path("scores", "edits", "e"), edit)[0] == path
    assert path.stat().st_mtime_ns == stamp and messages == []


def test_a_run_with_a_full_edit_has_no_shared_adapters_to_count(tiny):
    tiny.settings.edit.row_norm = "full"
    done = tiny.prepare()
    assert "adapters" not in done and done["prompts"] > 0


def test_a_run_says_when_an_export_wont_fit_before_the_search_starts(tiny, monkeypatch):
    said = []
    tiny.report = said.append
    tiny.prepare()
    warning = "Exporting an edit will need about"
    assert not any(line.startswith(warning) for line in said)
    monkeypatch.setattr(run.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(0, 0, 1000))
    tiny.prepare()
    assert any(
        line.startswith(warning) and line.endswith("make room before the search ends") for line in said
    )


def test_unchanged_adapters_are_reused_untouched(current):
    this, messages = current
    first = this.adapters()
    stamps = {p: p.stat().st_mtime_ns for paths in first.values() for p in paths.values()}
    messages.clear()
    this.adapters()
    assert {p: p.stat().st_mtime_ns for p in stamps} == stamps
    assert messages == []


def test_a_changed_setting_rebuilds_the_adapters_it_affects(current):
    this, messages = current
    before = this.adapters()["layer-0"]["blk.0.attn_output.weight"]
    old_b, _ = adapter_factors(before)
    this.settings.edit.row_norm = "none"
    after = this.adapters()["layer-0"]["blk.0.attn_output.weight"]
    new_b, _ = adapter_factors(after)
    assert not torch.allclose(old_b, new_b)
    assert any("Rebuilding adapters/layer-0" in m for m in messages)


def test_new_directions_rebuild_every_adapter(current, directions_by_seed):
    this, messages = current
    this.adapters()
    directions_by_seed[0]["seed"] = 1
    messages.clear()
    this.adapters()
    assert sum("Rebuilding" in m for m in messages) == 3


def test_direction_sets_follow_the_settings_and_drop_old_ones(current):
    this, _ = current
    this.adapters()
    this.settings.search.direction_layer = (1.0, 1.0)
    made = this.adapters()
    assert sorted(made) == ["layer-1", "per-layer"]
    kept = ["layer-1", "layer-1.inputs.json", "per-layer", "per-layer.inputs.json"]
    assert sorted(p.name for p in this.path("adapters").iterdir()) == kept


def test_a_deleted_result_is_reported_as_missing(current):
    this, messages = current
    this.adapters()
    shutil.rmtree(this.path("adapters", "layer-0"))
    messages.clear()
    this.adapters()
    assert messages.count("Rebuilding adapters/layer-0: it is missing") == 1
    assert rebuilt(messages) == {"adapters/layer-0"}


def test_leftovers_from_a_crash_are_never_trusted(current):
    this, _ = current
    this.adapters()
    shutil.rmtree(this.path("adapters", "layer-1"))
    leftover = this.path("adapters", "layer-1.building")
    leftover.mkdir()
    (leftover / "half-written.gguf").write_bytes(b"x")
    made = this.adapters()
    assert not leftover.exists()
    assert all(p.exists() for p in made["layer-1"].values())


def test_a_result_without_its_record_is_rebuilt(current):
    this, messages = current
    this.adapters()
    (this.path("adapters", "per-layer.inputs.json")).unlink()
    this.adapters()
    assert any("Rebuilding adapters/per-layer" in m for m in messages)


def test_a_result_is_on_the_disk_before_its_record(current, monkeypatch):
    this, _ = current
    events = []
    sync, rename, replace = run.sync, Path.rename, Path.replace

    def recording_sync(path):
        events.append(("sync", path.name))
        sync(path)

    def recording(move):
        def moved(path, target):
            events.append(("rename", path.name))
            return move(path, target)

        return moved

    monkeypatch.setattr(run, "sync", recording_sync)
    monkeypatch.setattr(files, "sync", recording_sync)
    monkeypatch.setattr(Path, "rename", recording(rename))
    monkeypatch.setattr(Path, "replace", recording(replace))
    this.adapters()
    order = [
        ("sync", "per-layer.building"),
        ("rename", "per-layer.building"),
        ("sync", "per-layer.inputs.json.building"),
        ("rename", "per-layer.inputs.json.building"),
    ]
    assert [event for event in events if event in order] == order


def test_the_fingerprint_covers_the_files_a_run_reads_and_nothing_else(current, model, tmp_path):
    this, _ = current

    def fingerprint():
        del this.model_fingerprint
        return this.model_fingerprint

    before = this.model_fingerprint
    (model / ".DS_Store").write_bytes(b"finder")
    (model / "README.md").write_text("notes")
    (model / "LICENSE").write_text("terms")
    (model / "modeling_custom.py").write_text("code")
    assert fingerprint() == before
    moved = tmp_path / "elsewhere"
    model.rename(moved)
    this.settings.model.source = str(moved)
    del this.model_folder
    assert fingerprint() == before
    this.settings.model.trust_remote_code = True
    assert fingerprint() != before
    this.settings.model.trust_remote_code = False
    os.utime(moved / "config.json", ns=(1, 1))
    assert fingerprint() != before


def test_a_missing_local_model_is_named(current, model):
    this, _ = current
    shutil.rmtree(model)
    with pytest.raises(RunError, match=f"the model folder {model} is gone"):
        this.gguf()


def test_a_hub_id_is_never_read_from_a_folder_of_the_same_name(current, monkeypatch, tmp_path):
    this, _ = current
    (tmp_path / "org" / "model").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    this.settings.model.source = "org/model"
    this.settings.model.revision = "abc123"
    calls = []

    def download(repo, revision, allow_patterns, ignore_patterns=None, dry_run=False):
        calls.append((repo, revision, dry_run))
        return [] if dry_run else str(tmp_path / "snapshot")

    monkeypatch.setattr(run, "snapshot_download", download)
    monkeypatch.setattr(run.source, "tensor_shapes", lambda model, revision=None: {"w": [4, 4]})
    assert this.model_folder == tmp_path / "snapshot"
    assert calls == [("org/model", "abc123", True), ("org/model", "abc123", False)]


@pytest.mark.parametrize(("trusted", "python"), [(False, False), (True, True)])
def test_a_models_own_python_is_downloaded_only_when_trusted(current, monkeypatch, tmp_path, trusted, python):
    this, _ = current
    this.settings.model.source = "org/model"
    this.settings.model.trust_remote_code = trusted
    patterns, ignored = [], []

    def download(repo, revision, allow_patterns, ignore_patterns=None, dry_run=False):
        patterns.append(allow_patterns)
        ignored.append(ignore_patterns)
        return [] if dry_run else str(tmp_path)

    monkeypatch.setattr(run, "snapshot_download", download)
    monkeypatch.setattr(run.source, "tensor_shapes", lambda model, revision=None: {"w": [4, 4]})
    assert this.model_folder == tmp_path
    assert all(("*.py" in p) == python for p in patterns)
    # Neither the size check nor the download takes Mistral's own copy of the weights.
    assert ignored == [run.NOT_MODEL_FILES] * 2


def test_a_download_that_would_not_fit_stops_before_it_starts(current, monkeypatch, tmp_path):
    this, _ = current
    this.settings.model.source = "org/model"
    calls = []

    def download(repo, revision, allow_patterns, ignore_patterns=None, dry_run=False):
        calls.append(dry_run)
        planned = [(3000, True), (5000, False), (None, True)]
        if dry_run:
            return [SimpleNamespace(file_size=size, will_download=new) for size, new in planned]
        return str(tmp_path)

    asked = []

    def disk_usage(path):
        asked.append(path)
        return shutil._ntuple_diskusage(0, 0, free)

    monkeypatch.setattr(run, "snapshot_download", download)
    monkeypatch.setattr(run.constants, "HF_HUB_CACHE", str(tmp_path / "not" / "made" / "yet"))
    monkeypatch.setattr(run.shutil, "disk_usage", disk_usage)
    this.path("gguf").mkdir()  # converted already
    free = 2999
    with pytest.raises(RunError, match="downloading the model needs"):
        _ = this.model_folder
    assert calls == [True]
    assert asked == [tmp_path]
    free = 3000
    assert this.model_folder == tmp_path
    assert calls == [True, True, False]


def test_a_download_with_no_room_left_to_convert_stops_before_it_starts(current, monkeypatch, tmp_path):
    this, _ = current
    this.settings.model.source = "org/model"
    monkeypatch.setattr(
        run,
        "snapshot_download",
        lambda *args, dry_run=False, **kwargs: (
            [SimpleNamespace(file_size=3000, will_download=True)] if dry_run else str(tmp_path)
        ),
    )
    monkeypatch.setattr(run.source, "tensor_shapes", lambda model, revision=None: {"w": [100, 100]})
    monkeypatch.setattr(run.constants, "HF_HUB_CACHE", str(tmp_path))
    # Room for the download, and for converting it into a 16-bit and a quantised copy, on the one disk.
    needed = 3000 + 1.1 * 10_000 * (2 + config.QUANTIZATIONS["Q4_K_M"] / 8)
    monkeypatch.setattr(
        run.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(0, 0, math.floor(needed))
    )
    with pytest.raises(RunError, match="downloading and converting the model needs"):
        _ = this.model_folder
    monkeypatch.setattr(run.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(0, 0, math.ceil(needed)))
    assert this.model_folder == tmp_path


@pytest.mark.parametrize(("deployed", "copies"), [("BF16", 2), ("F16", 4)])
def test_a_download_deployed_in_16_bits_needs_room_for_a_second_copy_only_of_another_type(
    current, monkeypatch, tmp_path, deployed, copies
):
    this, _ = current
    this.settings.model.source = "org/model"
    this.settings.deployment.quantization = deployed
    monkeypatch.setattr(
        run,
        "snapshot_download",
        lambda *args, dry_run=False, **kwargs: (
            [SimpleNamespace(file_size=3000, will_download=True)] if dry_run else str(tmp_path)
        ),
    )
    # A BF16 checkpoint, as its headers on the Hub say.
    headers = {"w": {"dtype": "BF16", "shape": [100, 100]}, "norm": {"dtype": "F32", "shape": [100]}}
    monkeypatch.setattr(run.source, "tensor_headers", lambda model, revision=None: headers)
    monkeypatch.setattr(run.constants, "HF_HUB_CACHE", str(tmp_path))
    needed = 3000 + 1.1 * 10_100 * copies
    monkeypatch.setattr(
        run.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(0, 0, math.floor(needed) - 1)
    )
    with pytest.raises(RunError, match="downloading and converting the model needs"):
        _ = this.model_folder
    monkeypatch.setattr(run.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(0, 0, math.ceil(needed)))
    assert this.model_folder == tmp_path


def test_mistrals_own_copy_of_the_weights_is_never_downloaded():
    from huggingface_hub.utils import filter_repo_objects

    files = ["consolidated.safetensors", "model-00001-of-00002.safetensors", "params.json", "config.json"]
    kept = filter_repo_objects(files, allow_patterns=run.MODEL_FILES, ignore_patterns=run.NOT_MODEL_FILES)
    assert list(kept) == ["model-00001-of-00002.safetensors", "params.json", "config.json"]


@pytest.fixture
def fake_conversion(monkeypatch):
    calls = []

    def to_gguf(folder, destination, log):
        calls.append("base")
        destination.write_bytes(b"base")

    def quantize(base, destination, quantization, log):
        calls.append(quantization)
        destination.write_bytes(quantization.encode())

    monkeypatch.setattr(run.convert, "to_gguf", to_gguf)
    monkeypatch.setattr(run.convert, "quantize", quantize)
    monkeypatch.setattr(run.convert, "check_targets", lambda *_: [])
    return calls


WEIGHTS = sum(math.prod(shape) for shape in llama().values())


@pytest.mark.parametrize(
    ("first", "then", "bytes_per_weight"),
    [
        (None, "Q4_K_M", 2 + 4.9 / 8),
        (None, "BF16", 2),
        ("Q4_K_M", "Q8_0", 8.5 / 8),
    ],
)
def test_conversion_needs_room_for_each_file_it_writes(
    current, fake_conversion, monkeypatch, first, then, bytes_per_weight
):
    this, _ = current
    if first:
        this.settings.deployment.quantization = first
        this.gguf()
    this.settings.deployment.quantization = then
    needed = 1.1 * WEIGHTS * bytes_per_weight
    monkeypatch.setattr(
        run.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(0, 0, math.floor(needed) - 1)
    )
    with pytest.raises(RunError, match="GB free"):
        this.gguf()
    monkeypatch.setattr(
        run.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(0, 0, math.ceil(needed) + 1)
    )
    this.gguf()


def test_gguf_is_converted_once_and_requantised_only_when_needed(current, fake_conversion):
    this, _ = current
    this.gguf()
    this.gguf()
    assert fake_conversion == ["base", "Q4_K_M"]
    this.settings.deployment.quantization = "Q8_0"
    this.gguf()
    assert fake_conversion == ["base", "Q4_K_M", "Q8_0"]


def test_a_deleted_gguf_is_rebuilt_not_trusted(current, fake_conversion):
    this, _ = current
    _, quantized = this.gguf()
    quantized.unlink()
    this.gguf()
    assert fake_conversion == ["base", "Q4_K_M", "Q4_K_M"]


def test_a_gguf_that_changed_a_target_stops_the_run_and_leaves_nothing(current, fake_conversion, monkeypatch):
    this, _ = current
    monkeypatch.setattr(run.convert, "check_targets", lambda *_: ["blk.0.attn_output.weight changed"])
    with pytest.raises(RunError, match="changed"):
        this.gguf()
    assert sorted(p.name for p in this.path("gguf").iterdir()) == []


def test_every_stage_runs_under_the_runs_lock(current, monkeypatch):
    this, _ = current
    seen = []

    def adapters(self):
        try:
            with self.locked():
                seen.append("unlocked")
        except RunError as error:
            seen.append(str(error))
        raise KeyboardInterrupt

    monkeypatch.setattr(Run, "adapters", adapters)
    with pytest.raises(KeyboardInterrupt):
        this.prepare()
    assert seen == [f"another candor command is using {this.folder}"]
    with this.locked():
        pass


def test_reading_residuals_must_fit_the_gpu_before_anything_is_converted(
    current, fake_conversion, monkeypatch
):
    this, _ = current
    needed = 2 * WEIGHTS * 1.1
    monkeypatch.setattr(run.residuals, "device_and_dtype", lambda: ("mps", torch.float16))
    monkeypatch.setattr(run.residuals, "memory_budget", lambda device, spread: math.floor(needed) - 1)
    with pytest.raises(RunError, match=r"needs about 0\.0 GB of GPU memory .* run it on a pod"):
        this.prepare()
    assert fake_conversion == []
    monkeypatch.setattr(run.residuals, "memory_budget", lambda device, spread: math.ceil(needed))
    this.check_memory()


def test_a_hub_models_memory_is_checked_without_downloading_it(current, monkeypatch):
    this, _ = current
    this.settings.model.source = "org/model"
    this.settings.model.revision = "abc123"
    asked = []

    def shapes(model, revision=None):
        asked.append((model, revision))
        return {"w": [100_000, 10_000]}

    monkeypatch.setattr(run.source, "tensor_shapes", shapes)
    monkeypatch.setattr(run, "snapshot_download", lambda *_, **__: pytest.fail("downloaded"))
    monkeypatch.setattr(run.residuals, "device_and_dtype", lambda: ("cuda", torch.bfloat16))
    monkeypatch.setattr(run.residuals, "memory_budget", lambda device, spread: 2_000_000)
    with pytest.raises(RunError, match=r"needs about 2\.2 GB"):
        this.check_memory()
    assert asked == [("org/model", "abc123")]


def test_reading_residuals_may_take_what_every_gpu_has_free(current, monkeypatch):
    this, _ = current
    needed = 2 * WEIGHTS * 1.1
    monkeypatch.setattr(run.residuals, "device_and_dtype", lambda: ("cuda", torch.bfloat16))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda gpu=None: (math.ceil(needed / 2), 2 * WEIGHTS))
    this.check_memory()
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RunError, match="run it on a pod"):
        this.check_memory()


def test_a_gpu_torch_cannot_use_stops_the_run(current, monkeypatch):
    this, _ = current
    monkeypatch.setattr(run.residuals, "device_and_dtype", lambda: ("cpu", torch.float32))
    monkeypatch.setattr(
        run.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None
    )
    with pytest.raises(RunError, match="can't use this machine's NVIDIA GPU"):
        this.check_memory()


def test_directions_read_on_a_pod_are_used_without_the_up_front_memory_check(current, monkeypatch):
    this, _ = current
    checked = []
    monkeypatch.setattr(Run, "check_memory", lambda self: checked.append(True))
    monkeypatch.setattr(Run, "adapters", lambda self: {"per-layer": {"t": Path("t.gguf")}})
    this.path("directions.pt").write_bytes(b"read on a pod")
    record = run.record_of(this.path("directions.pt"))
    record.write_text(json.dumps({"code": {"directions.pt": run.STAGES["directions.pt"]}}))
    this.path("prompts.json").write_text(json.dumps({"harmful": [[1]]}))
    assert this.prepare()["adapters"] == 1 and checked == []
    # Directions read by older code are read again, so the memory that takes is checked first.
    record.write_text(json.dumps({"code": {"directions.pt": run.STAGES["directions.pt"] - 1}}))
    this.check_residuals()
    assert checked == [True]


def test_memory_is_checked_again_before_residuals_are_read(tmp_path, model, monkeypatch):
    folder = tmp_path / "runs" / "test"
    folder.mkdir(parents=True)
    this = Run(folder, config.RunConfig(model=config.Model(source=str(model))), report=lambda _: None)
    loads = fake_residuals(monkeypatch)
    monkeypatch.setattr(Run, "prompt_ids", lambda self: {"harmful": [[1]], "harmless": [[2]]})

    def too_big(self):
        raise RunError("too big")

    monkeypatch.setattr(Run, "check_memory", too_big)
    with pytest.raises(RunError, match="too big"):
        this.directions()
    assert loads.models == []


@pytest.mark.parametrize("lost", ["directions.pt", "directions_answering.pt"])
def test_directions_lost_after_they_were_read_are_never_read_again(tmp_path, model, monkeypatch, lost):
    this, _ = thinking_run(tmp_path, model, monkeypatch)
    loads = fake_residuals(monkeypatch)
    this.directions()
    this.path(lost).unlink()
    # Refused before the memory check, which a Mac too small for the model would fail first.
    monkeypatch.setattr(Run, "check_memory", lambda self: pytest.fail("checked the memory"))
    for check in (this.check_residuals, this.directions):
        with pytest.raises(RunError, match=rf"{lost} is missing but its record isn't.*: restore it"):
            check()
    assert len(loads.models) == 1


def test_a_model_that_needs_its_own_code_must_be_trusted_before_anything_is_downloaded(current, monkeypatch):
    this, _ = current
    this.settings.model.source = "org/model"
    this.settings.model.revision = "abc123"
    asked = []

    def read_config(model, revision=None):
        asked.append((model, revision))
        return {"model_type": "made_up", "auto_map": {"AutoModelForCausalLM": "modeling_made_up.Model"}}

    monkeypatch.setattr(run.source, "read_config", read_config)
    monkeypatch.setattr(run, "snapshot_download", lambda *_, **__: pytest.fail("downloaded"))
    with pytest.raises(RunError, match=r"model\.trust_remote_code = true"):
        this.prepare()
    assert asked == [("org/model", "abc123")]
    this.settings.model.trust_remote_code = True
    this.check_remote_code()


def test_the_kv_cache_type_reaches_llama_server(current, monkeypatch):
    this, _ = current
    monkeypatch.setattr(run.toolchain, "binaries", lambda: Path("/bin"))
    this.settings.deployment.kv_cache = "q8_0"
    assert this.server(Path("m.gguf"), context_per_slot=64, parallel=1).extra == [
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "-fa",
        "on",
    ]
    this.settings.deployment.kv_cache = "f16"
    assert this.server(Path("m.gguf"), context_per_slot=64, parallel=1).extra == [
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
    ]


def test_on_cuda_the_server_must_run_on_the_gpus(current, monkeypatch):
    this, _ = current
    monkeypatch.setattr(run.toolchain, "binaries", lambda: Path("/bin"))
    assert "--device" not in this.server(Path("m.gguf"), context_per_slot=64, parallel=1).extra
    monkeypatch.setattr(run.toolchain, "on_cuda", lambda: True)
    monkeypatch.setattr(run.toolchain, "_check_gpu", lambda folder: ("CUDA0", "CUDA1"))
    extra = this.server(Path("m.gguf"), context_per_slot=64, parallel=1).extra
    assert extra[extra.index("--device") + 1] == "CUDA0,CUDA1"


def test_a_sampled_run_samples_on_the_gpu(current, monkeypatch):
    this, _ = current
    monkeypatch.setattr(run.toolchain, "binaries", lambda: Path("/bin"))
    assert "--backend-sampling" not in this.server(Path("m.gguf"), context_per_slot=64, parallel=1).extra
    this.settings.sampling = config.Sampling(temperature=1.0)
    assert this.server(Path("m.gguf"), context_per_slot=64, parallel=1).extra[-1] == "--backend-sampling"


def test_overflowing_activations_stop_the_run(tmp_path, model, monkeypatch):
    this = Run(tmp_path, config.RunConfig(model=config.Model(source=str(model))), report=lambda _: None)
    monkeypatch.setattr(Run, "prompt_ids", lambda self: {"harmful": [[1]], "harmless": [[1]]})
    fake_residuals(monkeypatch)
    monkeypatch.setattr(
        run.residuals, "last_token_residuals", lambda *_: torch.full((1, 3, HIDDEN), float("nan"))
    )
    with pytest.raises(RunError, match="overflowed"):
        this.directions()
    assert not this.path("directions.pt").exists()


@pytest.mark.parametrize("trusted", [False, True])
def test_the_model_is_loaded_straight_onto_the_device(tmp_path, model, monkeypatch, trusted):
    settings = config.RunConfig(model=config.Model(source=str(model), trust_remote_code=trusted))
    this = Run(tmp_path, settings, report=lambda _: None)
    monkeypatch.setattr(Run, "prompt_ids", lambda self: {"harmful": [[1]], "harmless": [[2]]})
    loads = fake_residuals(monkeypatch)
    monkeypatch.setattr(run.residuals, "device_and_dtype", lambda: ("mps", torch.float16))
    this.directions()
    assert loads.models == [{"dtype": torch.float16, "device_map": "mps", "trust_remote_code": trusted}]
    assert loads.tokenizers == [trusted]


@pytest.mark.parametrize("gpus", [1, 2])
def test_on_several_gpus_the_model_is_spread_across_them_and_only_them(tmp_path, model, monkeypatch, gpus):
    messages = []
    this = Run(tmp_path, config.RunConfig(model=config.Model(source=str(model))), report=messages.append)
    monkeypatch.setattr(Run, "prompt_ids", lambda self: {"harmful": [[1]], "harmless": [[2]]})
    loads = fake_residuals(monkeypatch)
    free = {gpu: 80_000_000 - gpu for gpu in range(gpus)}
    monkeypatch.setattr(run.residuals, "device_and_dtype", lambda: ("cuda", torch.bfloat16))
    monkeypatch.setattr(run.residuals, "free_on_each_gpu", lambda: free)
    # The GPUs transformers used, which needn't be all of them.
    monkeypatch.setattr(run.residuals, "gpus_used", lambda model: [1])
    this.directions()
    allowed = {gpu: int(room / 1.1) for gpu, room in free.items()}
    placed = {"device_map": "auto", "max_memory": allowed} if gpus > 1 else {"device_map": "cuda"}
    assert loads.models == [{"dtype": torch.bfloat16, "trust_remote_code": False, **placed}]
    assert ("Reading residuals on GPUs 1" if gpus > 1 else "Reading residuals on cuda") in messages


def test_flash_next_spread_across_gpus_leaves_each_room_and_only_its_n_gram_embedding_off_them(monkeypatch):
    """Placed by transformers' own device-map inference, on the meta device, so no weight is made."""
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(Qwen4ExpTextConfig(**FLASH_NEXT)).bfloat16()
    free = {gpu: 95 * 2**30 for gpu in range(5)}
    monkeypatch.setattr(run.residuals, "free_on_each_gpu", lambda: free)
    device_map = _get_device_map(model, "auto", run.residuals.placement("cuda")["max_memory"], None)
    tensors = dict(model.named_parameters()) | dict(model.named_buffers())
    placed = expand_device_map(device_map, list(tensors))
    filled = Counter()
    for name, where in placed.items():
        filled[where] += tensors[name].numel() * tensors[name].element_size()
    # A tenth of what each GPU has free is left for a batch's activations.
    assert all(filled[gpu] <= room / 1.1 for gpu, room in free.items())
    # Kept on the CPU by transformers, as no one GPU holds it, and only read from there a row at a time.
    off = [name for name, where in placed.items() if where not in free]
    assert off == ["model.layers.1.ple.ple_embedding.ngram_embedding.weight"]


def test_a_machine_without_the_memory_to_join_the_n_gram_embedding_is_refused_before_loading(
    tmp_path, monkeypatch
):
    folder = write_qwen4exp(tmp_path / "qwen4exp", tiny_qwen4exp())
    this = Run(tmp_path, config.RunConfig(model=config.Model(source=str(folder))), report=lambda _: None)
    held = run.residuals.held_on_the_cpu(str(folder), None)
    # Twice the table, as its shards are joined, and a tenth more.
    needed = 2 * held * 1.1
    monkeypatch.setattr(run.residuals, "system_memory", lambda: int(needed) - 1)
    with pytest.raises(
        RunError, match="of this machine's memory, where transformers joins the weights it keeps"
    ):
        this.check_memory()
    monkeypatch.setattr(run.residuals, "system_memory", lambda: int(needed) + 1)
    this.check_memory()


def test_residuals_are_never_read_from_a_model_partly_on_the_disk(tmp_path, model, monkeypatch):
    this = Run(tmp_path, config.RunConfig(model=config.Model(source=str(model))), report=lambda _: None)
    monkeypatch.setattr(Run, "prompt_ids", lambda self: {"harmful": [[1]], "harmless": [[2]]})
    fake_residuals(monkeypatch)
    monkeypatch.setattr(run.residuals, "off_the_gpus", lambda model: ["model.layers.1", "model.layers.2"])
    with pytest.raises(RunError, match=r"put 2 of its modules, model\.layers\.1 first, on the CPU or disk"):
        this.directions()
    assert not this.path("directions.pt").exists()


def test_a_model_with_hyper_connections_has_a_direction_of_the_hidden_size_at_each_layer(
    tmp_path, monkeypatch
):
    folder = write_qwen4exp(tmp_path / "qwen4exp", tiny_qwen4exp())
    this = Run(tmp_path, config.RunConfig(model=config.Model(source=str(folder))), report=lambda _: None)
    monkeypatch.setattr(
        Run,
        "prompt_ids",
        lambda self: {"harmful": [[1, 7, 3, 8], [1, 9, 4]], "harmless": [[1, 5, 6, 11, 12]]},
    )
    found = this.directions()
    assert found.shape == (TINY_QWEN4EXP["num_hidden_layers"], HIDDEN)
    assert torch.isfinite(found).all()


def test_a_model_with_its_own_code_is_read_only_when_trusted(tmp_path, monkeypatch):
    folder = write_custom_code_model(tmp_path / "custom")
    settings = config.RunConfig(model=config.Model(source=str(folder)))
    this = Run(tmp_path, settings, report=lambda _: None)
    monkeypatch.setattr(
        Run, "prompt_ids", lambda self: {"harmful": [[1, 2, 3], [1, 4]], "harmless": [[1, 5]]}
    )
    with pytest.raises(ValueError, match="custom code"):
        this.directions()
    settings.model.trust_remote_code = True
    found = this.directions()
    assert found.shape == (LAYERS, HIDDEN)
    assert torch.isfinite(found).all()


def test_a_model_whose_code_was_written_for_transformers_4_is_read(tmp_path, monkeypatch):
    folder = write_custom_code_model(tmp_path / "custom", transformers_4=True)
    with pytest.raises(AttributeError):
        AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True)
    settings = config.RunConfig(model=config.Model(source=str(folder), trust_remote_code=True))
    this = Run(tmp_path, settings, report=lambda _: None)
    monkeypatch.setattr(
        Run, "prompt_ids", lambda self: {"harmful": [[1, 2, 3], [1, 4]], "harmless": [[1, 5]]}
    )
    found = this.directions()
    assert found.shape == (LAYERS, HIDDEN) and torch.isfinite(found).all()
    loaded = residuals.load_model(folder, True)
    assert loaded.lm_head.weight is loaded.model.embedding.weight
    # Left padding changes no residual, so the mask it makes the way transformers 4 did is right.
    torch.testing.assert_close(
        residuals.last_token_residuals(loaded, [[1, 2, 3, 4], [1, 5]], pad_id=0)[1],
        residuals.last_token_residuals(loaded, [[1, 5]], pad_id=0)[0],
        atol=1e-5,
        rtol=1e-4,
    )


def test_directions_follow_the_prompts_they_read_and_nothing_else(tmp_path, model, monkeypatch):
    this = Run(tmp_path, config.RunConfig(model=config.Model(source=str(model))), report=lambda _: None)
    ids = {"harmful": [[1, 2]], "harmless": [[3, 4]], "harmful_test": [[5]], "harmless_test": [[6]]}
    monkeypatch.setattr(Run, "prompt_ids", lambda self: ids)
    loads = fake_residuals(monkeypatch)
    this.directions()
    ids["harmful_test"] = [[7, 8, 9]]
    this.directions()
    assert len(loads.models) == 1
    ids["harmful"] = [[1, 2, 3]]
    this.directions()
    assert len(loads.models) == 2


def test_new_directions_rebuild_the_adapters_built_from_them(tmp_path, model, monkeypatch):
    messages = []
    this = Run(tmp_path, config.RunConfig(model=config.Model(source=str(model))), report=messages.append)
    ids = {"harmful": [[1, 2]], "harmless": [[3, 4]]}
    monkeypatch.setattr(Run, "prompt_ids", lambda self: ids)
    fake_residuals(monkeypatch)
    this.adapters()
    ids["harmful"] = [[1, 2, 3]]
    this.adapters()
    assert rebuilt(messages) == {"directions.pt"} | ADAPTERS


THINKS = config.Deployment(think_close="\n</think>\n\n", thinking_budget=8, thinking_budget_message="")


def thinking_run(tmp_path, model, monkeypatch, report=lambda _: None) -> tuple[Run, dict[str, Any]]:
    """A run of a thinking model whose prompt ids and residuals are faked: its prompts, and the same
    prompts with an empty thinking block closed after them."""
    settings = config.RunConfig(model=config.Model(source=str(model)), deployment=THINKS)
    ids = {
        "harmful": [[1, 2]],
        "harmless": [[3, 4]],
        "harmful_answering": [[1, 2, 9]],
        "harmless_answering": [[3, 4, 9]],
    }
    monkeypatch.setattr(Run, "prompt_ids", lambda self: ids)
    return Run(tmp_path, settings, report=report), ids


def test_a_thinking_model_reads_directions_where_its_thinking_and_its_answer_start_in_one_load(
    tmp_path, model, monkeypatch
):
    this, ids = thinking_run(tmp_path, model, monkeypatch)
    loads = fake_residuals(monkeypatch)
    start, answer = this.directions(), this.directions(answering=True)
    assert len(loads.models) == 1
    read = {name: residuals_of(ids[name]) for name in ids}
    assert torch.equal(start, directions.difference_of_means(read["harmful"], read["harmless"]))
    assert torch.equal(
        answer, directions.difference_of_means(read["harmful_answering"], read["harmless_answering"])
    )
    ids["harmful_answering"] = [[1, 2, 8]]
    assert not torch.equal(this.directions(answering=True), answer)
    assert len(loads.models) == 2 and torch.equal(this.directions(), start)


def test_only_a_thinking_model_has_directions_where_its_answer_starts(tmp_path, model):
    this = Run(tmp_path, config.RunConfig(model=config.Model(source=str(model))), report=lambda _: None)
    with pytest.raises(RunError, match="only a thinking model"):
        this.directions(answering=True)


def test_a_thinking_models_adapters_are_read_at_both_places_and_rebuilt_apart(tmp_path, model, monkeypatch):
    messages = []
    this, ids = thinking_run(tmp_path, model, monkeypatch, messages.append)
    fake_residuals(monkeypatch)
    made = this.adapters()
    assert set(made) == {
        f"{prefix}{name}" for prefix in ("", run.ANSWER_SETS) for name in ("per-layer", "layer-0", "layer-1")
    }
    start, answer = (
        adapter_factors(made["layer-1"]["blk.0.attn_output.weight"]),
        adapter_factors(made["answer-layer-1"]["blk.0.attn_output.weight"]),
    )
    assert not torch.equal(start[0], answer[0])
    messages.clear()
    ids["harmless_answering"] = [[3, 4, 8]]
    this.adapters()
    assert rebuilt(messages) == {"directions_answering.pt"} | {
        "adapters/answer-" + a.split("/")[1] for a in ADAPTERS
    }


def test_a_new_version_of_the_answer_start_directions_rebuilds_their_adapters(tmp_path, model, monkeypatch):
    messages = []
    this, _ = thinking_run(tmp_path, model, monkeypatch, messages.append)
    fake_residuals(monkeypatch)
    this.adapters()
    messages.clear()
    monkeypatch.setitem(run.STAGES, "directions_answering.pt", run.STAGES["directions_answering.pt"] + 1)
    this.adapters()
    assert rebuilt(messages) == {"directions_answering.pt"} | {
        "adapters/answer-" + a.split("/")[1] for a in ADAPTERS
    }


@pytest.mark.parametrize("stage", ["adapters", "measurements", "scores", "exports"])
def test_the_answer_start_directions_dont_depend_on_the_stages_they_dont_use(
    tmp_path, model, monkeypatch, stage
):
    this, _ = thinking_run(tmp_path, model, monkeypatch)
    loads = fake_residuals(monkeypatch)
    this.adapters()
    made = this.adapter_inputs(answering=True)
    monkeypatch.setitem(run.STAGES, stage, run.STAGES[stage] + 1)
    this.directions(answering=True)
    assert len(loads.models) == 1 and this.adapter_inputs(answering=True) == made
    monkeypatch.setitem(run.STAGES, "directions.pt", run.STAGES["directions.pt"] + 1)
    this.directions(answering=True)
    assert len(loads.models) == 2 and this.adapter_inputs(answering=True) != made


def test_a_change_where_the_thinking_starts_rebuilds_only_whats_read_there(tmp_path, model, monkeypatch):
    messages = []
    this, ids = thinking_run(tmp_path, model, monkeypatch, messages.append)
    fake_residuals(monkeypatch)
    this.adapters()
    messages.clear()
    ids["harmful"] = [[1, 2, 7]]
    this.adapters()
    assert rebuilt(messages) == {"directions.pt"} | ADAPTERS


IGNORES_THE_DATE = (
    "<s>Today: {{ strftime_now('%d %b %Y') }}\n{% for m in messages %}{{ m['content'] }}{% endfor %}"
)


@pytest.fixture
def tiny_run(current, monkeypatch):
    """A run whose model is a tiny random GGUF, for stages that need a real llama-server."""
    if not LLAMA_SERVER.exists():
        pytest.skip("no llama-server")
    this, messages = current
    monkeypatch.setenv("CANDOR_LLAMA_BIN", str(LLAMA_SERVER.parent))
    prompts = {"value": ["first prompt", "second"]}
    monkeypatch.setattr(run, "load_prompts", lambda spec: [f"{p} {spec.split}" for p in prompts["value"]])

    def use(template: str):
        tiny = this.folder / "tiny.gguf"
        write_model(tiny, random_weights(), chat_template=template)

        def fake_gguf(self):
            self.inputs["base"] = template
            return tiny, tiny

        monkeypatch.setattr(Run, "gguf", fake_gguf)

    return this, messages, use, prompts


def test_prompt_ids_take_the_pinned_date(tiny_run):
    this, _, use, _ = tiny_run
    use(TEMPLATE)
    this.settings.model.date = "01 Jan 2020"
    first = this.prompt_ids()
    this.settings.model.date = "28 Feb 2021"
    second = this.prompt_ids()
    assert {name: len(v) for name, v in second.items()} == dict.fromkeys(run.PROMPT_SETS, 2)
    assert first["harmful"] != second["harmful"]


def test_prompts_are_rendered_with_the_runs_chat_template_settings(tiny_run):
    this, _, use, _ = tiny_run
    use(TEMPLATE.replace("assistant:", "assistant {{ effort | default('high') }}:"))
    first = this.prompt_ids()
    record = run.record_of(this.path("prompts.json"))
    # A run without the setting keeps the record it had before there was one.
    assert "chat_template_kwargs" not in json.loads(record.read_text())
    this.settings.deployment.chat_template_kwargs = {"effort": "low"}
    second = this.prompt_ids()
    assert first["harmful"] != second["harmful"]
    assert json.loads(record.read_text())["chat_template_kwargs"] == {"effort": "low"}
    this.settings.deployment.chat_template_kwargs = {"effort": "high"}
    assert this.prompt_ids() == first
    this.settings.model.date = "01 Jan 2020"
    assert this.template_kwargs() == {"effort": "high", "date_string": "01 Jan 2020"}
    assert this.template_kwargs(run.DATE_PROBE) == {"effort": "high", "date_string": run.DATE_PROBE}


def test_a_template_that_ignores_the_pinned_date_is_caught_even_today(tiny_run):
    this, _, use, _ = tiny_run
    use(IGNORES_THE_DATE)
    this.settings.model.date = config.today_in("%d %b %Y")
    with pytest.raises(RunError, match="ignores the pinned date"):
        this.prompt_ids()
    assert not this.path("prompts.json").exists()


@pytest.mark.parametrize(("slot", "servers"), [(run.TOKENIZING_SLOT, 1), (8, 2)])
def test_ids_that_disagree_with_the_chat_endpoint_stop_the_run(tiny_run, monkeypatch, slot, servers):
    this, _, use, _ = tiny_run
    use(TEMPLATE)
    started = []
    server = Run.server
    monkeypatch.setattr(Run, "server", lambda *args, **kwargs: started.append(1) or server(*args, **kwargs))
    real = LlamaServer.chat_prompt_length
    monkeypatch.setattr(LlamaServer, "chat_prompt_length", lambda *args: real(*args) + 1)
    # Prompts that fit the tokenizing server's slot are checked on it; longer ones get a server of their own.
    monkeypatch.setattr(run, "TOKENIZING_SLOT", slot)
    with pytest.raises(RunError, match="don't match its chat endpoint"):
        this.prompt_ids()
    assert not this.path("prompts.json").exists() and len(started) == servers


def test_prompts_too_long_for_the_model_are_named(tiny_run):
    this, _, use, prompts = tiny_run
    use(TEMPLATE)
    prompts["value"] = ["w0 " * 100]
    with pytest.raises(
        RunError,
        match=r"model's 256 tokens with their answers: 1 harmful, 1 harmless, 1 harmful_test, "
        "1 harmless_test, 1 harmful_held_out, 1 harmless_held_out, 1 maths$",
    ):
        this.prompt_ids()


def test_prompts_are_read_at_their_pinned_commit(monkeypatch):
    calls = []

    def load_dataset(dataset, subset, split, revision):
        calls.append((dataset, subset, split, revision))
        return {"text": ["a", "b"]}

    monkeypatch.setattr(run, "load_dataset", load_dataset)
    spec = config.PromptSet(dataset="org/prompts", subset="main", split="train[:2]", revision="abc123")
    assert run.load_prompts(spec) == ["a", "b"]
    assert calls == [("org/prompts", "main", "train[:2]", "abc123")]


def test_excluded_rows_are_left_out(monkeypatch):
    rows = {
        "prompt": ["one", "two", "three", "four"],
        "source": ["kept", "AdvBench", "kept", "other"],
        "kind": ["a", "a", "b", "a"],
    }
    monkeypatch.setattr(run, "load_dataset", lambda *args, **kwargs: rows)
    spec = config.PromptSet(
        dataset="org/prompts", split="train", column="prompt", exclude={"source": ["AdvBench"], "kind": ["b"]}
    )
    assert run.load_prompts(spec) == ["one", "four"]


def test_an_exclusion_that_matches_no_row_stops_the_run_without_showing_any_row(monkeypatch):
    rows = {"prompt": ["secret one", "secret two"], "source": ["AdvBench", "custom"]}
    monkeypatch.setattr(run, "load_dataset", lambda *args, **kwargs: rows)
    exclude = {"source": ["Advbench"]}
    spec = config.PromptSet(dataset="org/prompts", split="train", column="prompt", exclude=exclude)
    with pytest.raises(RunError, match="org/prompts has no rows whose source is Advbench") as error:
        run.load_prompts(spec)
    assert "secret" not in str(error.value)


def test_a_gated_dataset_says_how_to_get_access(monkeypatch):
    def gated(*args, **kwargs):
        raise DatasetNotFoundError("Dataset 'org/prompts' is a gated dataset on the Hub.")

    monkeypatch.setattr(run, "load_dataset", gated)
    with pytest.raises(RunError, match=r"accept its terms at https://huggingface\.co/datasets/org/prompts"):
        run.load_prompts(config.PromptSet(dataset="org/prompts", split="train"))


def test_each_maths_question_ends_with_the_instruction_and_nothing_else_does(current, monkeypatch):
    this, _ = current
    monkeypatch.setattr(run, "load_prompts", lambda spec: [f"question from {spec.dataset} {spec.split}"])
    this.settings.measure.maths_instruction = "Show your working."
    texts, _ = this.prompt_texts()
    assert texts["maths"] == ["question from openai/gsm8k test\n\nShow your working."]
    others = [text for name in run.PROMPT_SETS if name != "maths" for text in texts[name]]
    assert len(others) == len(run.PROMPT_SETS) - 1
    assert not any("working" in text for text in others)


def test_held_out_sets_lose_prompts_seen_elsewhere_or_twice():
    texts = {name: [f"only in {name}"] for name in run.PROMPT_SETS}
    texts["harmful"] += ["Make a **thing**"]
    texts["harmless_test"] += ["Plan a picnic"]
    texts["harmful_held_out"] = ["make a  thing", "new one", "New One", "unseen"]
    texts["harmless_held_out"] = ["plan a picnic", "plan a walk"]
    dropped = run.drop_repeats(texts)
    assert dropped == {"harmful_held_out": 2, "harmless_held_out": 1}
    assert texts["harmful_held_out"] == ["new one", "unseen"]
    assert texts["harmless_held_out"] == ["plan a walk"]
    assert texts["harmful"] == ["only in harmful", "Make a **thing**"]


def test_a_held_out_set_emptied_by_repeats_is_named(tiny_run, monkeypatch):
    this, _, use, _ = tiny_run
    use(TEMPLATE)
    monkeypatch.setattr(run, "load_prompts", lambda spec: ["the same prompt"])
    with pytest.raises(RunError, match=r"harmful_held_out prompt set .* every one of its prompts is also in"):
        this.prompt_ids()


def test_the_held_out_counts_are_reported(tiny_run):
    this, messages, use, _ = tiny_run
    use(TEMPLATE)
    this.prompt_ids()
    held_out = "Holding out 2 harmful and 2 harmless prompts from the search"
    assert f"{held_out} (repeats of other prompts dropped: 0)" in messages


def test_a_damaged_record_means_rebuild_not_crash(current):
    this, messages = current
    this.adapters()
    this.path("adapters", "per-layer.inputs.json").write_text('{"half')
    this.adapters()
    assert any("Rebuilding adapters/per-layer" in m for m in messages)


def test_a_long_server_log_is_kept_as_the_last_one_when_a_server_starts(current, monkeypatch):
    this, _ = current
    monkeypatch.setattr(run.toolchain, "binaries", lambda: Path("/bin"))
    log = this.path("logs", "server.log")
    log.parent.mkdir()
    log.write_bytes(b"x" * 10)
    this.server(Path("m.gguf"), context_per_slot=64, parallel=1)
    assert log.exists() and not log.with_name("server.log.1").exists()
    monkeypatch.setattr(run, "LOG_BYTES", 5)
    this.server(Path("m.gguf"), context_per_slot=64, parallel=1)
    assert not log.exists() and log.with_name("server.log.1").read_bytes() == b"x" * 10


def test_old_gguf_files_are_cleared_before_the_disk_check(current, fake_conversion, monkeypatch):
    this, messages = current
    this.gguf()
    this.settings.deployment.quantization = "Q8_0"
    leftover = this.path("gguf", "model-BF16.gguf.building")
    leftover.write_bytes(b"half")

    def disk_usage(_):
        return shutil._ntuple_diskusage(0, 0, 0 if leftover.exists() else 10**12)

    monkeypatch.setattr(run.shutil, "disk_usage", disk_usage)
    this.gguf()
    assert sorted(p.name for p in this.path("gguf").iterdir()) == [
        "model-BF16.gguf",
        "model-BF16.gguf.inputs.json",
        "model-Q8_0.gguf",
        "model-Q8_0.gguf.inputs.json",
    ]
    assert any("Removing gguf/model-Q4_K_M.gguf" in m for m in messages)


def test_changing_only_the_quantisation_keeps_the_prompt_ids(current, fake_conversion):
    this, _ = current
    this.gguf()
    before = this.inputs["base"]
    this.settings.deployment.quantization = "Q8_0"
    this.gguf()
    assert this.inputs["base"] == before


ANSWERED = ["harmful_test", "harmful_held_out", "harmless_held_out", "maths"]


@pytest.mark.parametrize("answered", ANSWERED)
def test_lengths_count_answers_only_for_the_prompts_the_model_answers(current, answered):
    this, _ = current
    limit = 256
    this.settings.scoring.answer_tokens = 100
    this.settings.measure.answer_tokens = 100
    this.settings.measure.maths_answer_tokens = 100
    fits = {name: [[0] * (150 if name in ANSWERED else 200)] for name in run.PROMPT_SETS}
    this._check_lengths(fits)
    with pytest.raises(RunError, match=f"model's {limit} tokens with their answers: 1 {answered}$"):
        this._check_lengths(fits | {answered: [[0] * 200]})


def test_kl_sets_of_a_thinking_model_need_room_for_its_whole_thinking(current):
    this, _ = current
    this.settings.deployment = THINKS.model_copy()
    room = this.thinking_room()
    this.settings.scoring.answer_tokens = 1
    this.settings.measure.answer_tokens = 1
    this.settings.measure.maths_answer_tokens = 1
    fits = {name: [[0] * (256 - room - 1)] for name in run.PROMPT_SETS}
    this._check_lengths(fits)
    with pytest.raises(RunError, match=r"with their answers: 1 harmless_test$"):
        this._check_lengths(fits | {"harmless_test": [[0] * (256 - room)]})


def test_raising_the_answer_length_later_is_still_checked(tiny_run):
    this, _, use, _ = tiny_run
    use(TEMPLATE)
    this.prompt_ids()
    this.settings.scoring.answer_tokens = 250
    with pytest.raises(RunError, match="too long"):
        this.prompt_ids()


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """A model and a run carried through every stage, made once and copied by each test that uses it."""
    if not LLAMA_SERVER.exists():
        pytest.skip("no llama-server")
    root = tmp_path_factory.mktemp("prepared")
    write_model(root / "tiny.gguf", random_weights(), chat_template=TEMPLATE)
    model = write_llama(root / "model")
    (root / "run").mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        fake_everything_but_the_stages(monkeypatch, root / "tiny.gguf")
        Run(root / "run", pinned(model), report=lambda _: None).prepare()
    return root


@pytest.fixture
def pipeline(prepared, tmp_path, monkeypatch):
    """A copy of the prepared run, and a way to prepare it again as a new candor command would."""
    root = tmp_path / "copy"
    shutil.copytree(prepared, root)
    fake_everything_but_the_stages(monkeypatch, root / "tiny.gguf")
    settings = pinned(root / "model")
    messages = []

    def again() -> list[str]:
        """Prepare the run again and say what was rebuilt."""
        messages.clear()
        Run(root / "run", settings, report=messages.append).prepare()
        return sorted(rebuilt(messages))

    return settings, again


def touch_the_weights(settings, monkeypatch):
    os.utime(Path(settings.model.source) / "model.safetensors", ns=(1, 1))


def rename_a_target(settings, monkeypatch):
    original = targets.gguf_name
    monkeypatch.setattr(run.targets, "gguf_name", lambda *args: f"{original(*args)}.renamed")


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (touch_the_weights, GGUF | {"prompts.json", "directions.pt"} | ADAPTERS),
        (lambda s, m: m.setattr(run, "LLAMA_CPP_RELEASE", "b1"), GGUF | {"prompts.json"}),
        (lambda s, m: setattr(s.model, "date", "28 Feb 2021"), {"prompts.json", "directions.pt"} | ADAPTERS),
        (
            lambda s, m: setattr(s.deployment, "system_prompt", "Be brief."),
            {"prompts.json", "directions.pt"} | ADAPTERS,
        ),
        (lambda s, m: setattr(s.prompts.harmful, "revision", "abc123"), {"prompts.json"}),
        (lambda s, m: setattr(s.measure, "maths_instruction", "Show your working."), {"prompts.json"}),
        (lambda s, m: setattr(s.model, "trust_remote_code", True), {"directions.pt"} | ADAPTERS),
        (lambda s, m: setattr(s.edit, "row_norm", "none"), ADAPTERS),
        (rename_a_target, ADAPTERS),
    ],
    ids=[
        "model files",
        "llama.cpp",
        "date",
        "system prompt",
        "prompt commit",
        "maths instruction",
        "remote code",
        "row norm",
        "targets",
    ],
)
def test_a_change_rebuilds_what_it_affects_and_nothing_else(pipeline, monkeypatch, change, expected):
    settings, again = pipeline
    assert again() == []
    change(settings, monkeypatch)
    assert again() == sorted(expected)


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("gguf", GGUF | {"prompts.json", "directions.pt"} | ADAPTERS),
        ("prompts.json", {"prompts.json", "directions.pt"} | ADAPTERS),
        ("directions.pt", {"directions.pt"} | ADAPTERS),
        ("adapters", ADAPTERS),
    ],
)
def test_a_new_version_of_a_stage_rebuilds_it_and_everything_after_it(pipeline, monkeypatch, stage, expected):
    _, again = pipeline
    monkeypatch.setitem(run.STAGES, stage, run.STAGES[stage] + 1)
    assert again() == sorted(expected)


def test_prepare_runs_each_stage_once(pipeline, monkeypatch):
    _, again = pipeline
    calls = []
    for stage in ("gguf", "prompt_ids", "directions", "adapters"):
        real = getattr(Run, stage)

        def counted(self, *args, real=real, stage=stage):
            calls.append(stage)
            return real(self, *args)

        monkeypatch.setattr(Run, stage, counted)
    again()
    assert calls == ["adapters", "directions", "prompt_ids", "gguf"]
