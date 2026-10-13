import json
import math
import os
import shutil
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gguf
import pytest
import torch
import torch.nn.functional as F
from datasets.exceptions import DatasetNotFoundError
from safetensors.torch import save_file

from candor import config, directions, run, targets
from candor.edit import RowNorm, rank_one
from candor.engine import LlamaServer
from candor.run import Run, RunError
from candor.weights import ModelFiles

from .custom_code import write_custom_code_model
from .test_engine import LLAMA_SERVER
from .test_targets import llama
from .tiny_gguf import random_weights, write_model

HIDDEN, LAYERS = 64, 2
GGUF = {"gguf/model-BF16.gguf", "gguf/model-Q4_K_M.gguf"}
ADAPTERS = {"adapters/per-layer", "adapters/layer-0", "adapters/layer-1"}


def write_llama(folder: Path) -> Path:
    folder.mkdir()
    torch.manual_seed(7)
    save_file({name: torch.randn(shape) for name, shape in llama().items()}, folder / "model.safetensors")
    model_config = {"architectures": ["LlamaForCausalLM"], "hidden_size": HIDDEN, "num_hidden_layers": LAYERS}
    (folder / "config.json").write_text(json.dumps(model_config | {"max_position_embeddings": 256}))
    return folder


@pytest.fixture
def model(tmp_path):
    return write_llama(tmp_path / "model")


@pytest.fixture
def directions_by_seed():
    """Directions the fake directions stage returns, changeable to stand for a real change upstream."""
    state = {"seed": 0}

    def found() -> torch.Tensor:
        generator = torch.Generator().manual_seed(state["seed"])
        return F.normalize(torch.randn(LAYERS + 1, HIDDEN, generator=generator), dim=-1)

    return state, found


def short() -> config.Measure:
    """Measure settings that fit the tiny test models' 256 tokens of context."""
    return config.Measure(answer_tokens=16, maths_answer_tokens=16, maths_instruction="Answer.")


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


class Loaded(torch.nn.Module):
    """Stands in for a model loaded by transformers when the residuals are faked."""


def residuals_of(prompts: list[list[int]]) -> torch.Tensor:
    """The residuals fake_residuals gives for these prompts."""
    generator = torch.Generator().manual_seed(zlib.crc32(json.dumps(prompts).encode()))
    return torch.randn(len(prompts), LAYERS + 1, HIDDEN, generator=generator)


def fake_residuals(monkeypatch) -> SimpleNamespace:
    """Residuals fixed by the prompts they are read for. Returns how each model and tokenizer was loaded."""
    loads = SimpleNamespace(models=[], tokenizers=[])

    def residuals(model, prompts, pad_id):
        return residuals_of(prompts)

    def load_model(folder, **kwargs):
        loads.models.append(kwargs)
        return Loaded()

    def pad_token_id_for(folder, trust_remote_code):
        loads.tokenizers.append(trust_remote_code)
        return 0

    monkeypatch.setattr(run.residuals, "last_token_residuals", residuals)
    monkeypatch.setattr(run.residuals, "pad_token_id_for", pad_token_id_for)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", load_model)
    return loads


def adapter_factors(path):
    tensors = {t.name: torch.from_numpy(t.data.copy()) for t in gguf.GGUFReader(path).tensors}
    a = next(v for k, v in tensors.items() if k.endswith("lora_a"))
    b = next(v for k, v in tensors.items() if k.endswith("lora_b"))
    return b, a


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


@pytest.mark.parametrize("row_norm", ["pre", "none"])
def test_adapters_hold_the_edit_for_their_layers_direction(current, model, directions_by_seed, row_norm):
    this, _ = current
    this.settings.edit.row_norm = row_norm
    made = this.adapters()
    found = directions_by_seed[1]()
    weight = ModelFiles(model).tensor("model.layers.1.mlp.down_proj.weight")
    for set_name, direction in (("per-layer", found[2]), ("layer-0", found[1])):
        b, a = rank_one(weight, direction, RowNorm(row_norm))
        stored_b, stored_a = adapter_factors(made[set_name]["blk.1.ffn_down.weight"])
        torch.testing.assert_close(stored_a, a)
        torch.testing.assert_close(stored_b, b)


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
    sync, rename = run.sync, Path.rename

    def recording_sync(path):
        events.append(("sync", path.name))
        sync(path)

    def recording_rename(path, target):
        events.append(("rename", path.name))
        return rename(path, target)

    monkeypatch.setattr(run, "sync", recording_sync)
    monkeypatch.setattr(Path, "rename", recording_rename)
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

    def download(repo, revision, allow_patterns, dry_run=False):
        calls.append((repo, revision, dry_run))
        return [] if dry_run else str(tmp_path / "snapshot")

    monkeypatch.setattr(run, "snapshot_download", download)
    assert this.model_folder == tmp_path / "snapshot"
    assert calls == [("org/model", "abc123", True), ("org/model", "abc123", False)]


@pytest.mark.parametrize(("trusted", "python"), [(False, False), (True, True)])
def test_a_models_own_python_is_downloaded_only_when_trusted(current, monkeypatch, tmp_path, trusted, python):
    this, _ = current
    this.settings.model.source = "org/model"
    this.settings.model.trust_remote_code = trusted
    patterns = []

    def download(repo, revision, allow_patterns, dry_run=False):
        patterns.append(allow_patterns)
        return [] if dry_run else str(tmp_path)

    monkeypatch.setattr(run, "snapshot_download", download)
    assert this.model_folder == tmp_path
    assert all(("*.py" in p) == python for p in patterns)


def test_a_download_that_would_not_fit_stops_before_it_starts(current, monkeypatch, tmp_path):
    this, _ = current
    this.settings.model.source = "org/model"
    calls = []

    def download(repo, revision, allow_patterns, dry_run=False):
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
    free = 2999
    with pytest.raises(RunError, match="downloading the model needs"):
        _ = this.model_folder
    assert calls == [True]
    assert asked == [tmp_path]
    free = 3000
    assert this.model_folder == tmp_path
    assert calls == [True, True, False]


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
    monkeypatch.setattr(run.residuals, "memory_budget", lambda device: math.floor(needed) - 1)
    with pytest.raises(RunError, match=r"needs about 0\.0 GB of GPU memory .* run it on a pod"):
        this.prepare()
    assert fake_conversion == []
    monkeypatch.setattr(run.residuals, "memory_budget", lambda device: math.ceil(needed))
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
    monkeypatch.setattr(run.residuals, "memory_budget", lambda device: 2_000_000)
    with pytest.raises(RunError, match=r"needs about 2\.2 GB"):
        this.check_memory()
    assert asked == [("org/model", "abc123")]


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
    monkeypatch.setattr(Run, "check_memory", lambda self: pytest.fail("checked the memory"))
    monkeypatch.setattr(Run, "adapters", lambda self: {"per-layer": {"t": Path("t.gguf")}})
    this.path("directions.pt").write_bytes(b"read on a pod")
    this.path("prompts.json").write_text(json.dumps({"harmful": [[1]]}))
    assert this.prepare()["adapters"] == 1


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
    assert this.server(Path("m.gguf")).extra == [
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "-fa",
        "on",
    ]
    this.settings.deployment.kv_cache = "f16"
    assert this.server(Path("m.gguf")).extra == ["--cache-type-k", "f16", "--cache-type-v", "f16"]


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
    assert found.shape == (LAYERS + 1, HIDDEN)
    assert torch.isfinite(found).all()


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


TEMPLATE = (
    "{%- if not date_string is defined %}{%- set date_string = strftime_now('%d %b %Y') %}{%- endif %}"
    "<s>Today: {{ date_string }}\n{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant:{% endif %}"
)
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


def test_a_template_that_ignores_the_pinned_date_is_caught_even_today(tiny_run):
    this, _, use, _ = tiny_run
    use(IGNORES_THE_DATE)
    this.settings.model.date = config.today_in("%d %b %Y")
    with pytest.raises(RunError, match="ignores the pinned date"):
        this.prompt_ids()
    assert not this.path("prompts.json").exists()


def test_ids_that_disagree_with_the_chat_endpoint_stop_the_run(tiny_run, monkeypatch):
    this, _, use, _ = tiny_run
    use(TEMPLATE)
    real = LlamaServer.chat_prompt_length
    monkeypatch.setattr(LlamaServer, "chat_prompt_length", lambda *args: real(*args) + 1)
    with pytest.raises(RunError, match="don't match its chat endpoint"):
        this.prompt_ids()
    assert not this.path("prompts.json").exists()


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


def test_raising_the_answer_length_later_is_still_checked(tiny_run):
    this, _, use, _ = tiny_run
    use(TEMPLATE)
    this.prompt_ids()
    this.settings.scoring.answer_tokens = 250
    with pytest.raises(RunError, match="too long"):
        this.prompt_ids()


@pytest.mark.skipif(not hasattr(run.fcntl, "F_FULLFSYNC"), reason="only macOS has F_FULLFSYNC")
def test_results_are_fully_flushed_on_macos(tmp_path, monkeypatch):
    calls = []
    real = run.fcntl.fcntl

    def recording(descriptor, command, *args):
        calls.append(command)
        return real(descriptor, command, *args)

    monkeypatch.setattr(run.fcntl, "fcntl", recording)
    (tmp_path / "result").write_text("x")
    run.sync(tmp_path / "result")
    assert calls.count(run.fcntl.F_FULLFSYNC) >= 2


def fake_everything_but_the_stages(monkeypatch, tiny: Path) -> None:
    """The converter copies a tiny random GGUF, and residuals are random but fixed by their prompts."""
    monkeypatch.setenv("CANDOR_LLAMA_BIN", str(LLAMA_SERVER.parent))
    monkeypatch.setattr(
        run.convert, "to_gguf", lambda folder, destination, log: shutil.copy(tiny, destination)
    )
    monkeypatch.setattr(run.convert, "quantize", lambda base, destination, *_: shutil.copy(base, destination))
    monkeypatch.setattr(run.convert, "check_targets", lambda *_: [])
    monkeypatch.setattr(run, "load_prompts", lambda spec: [f"first prompt {spec.split}", "second"])
    fake_residuals(monkeypatch)


def pinned(model: Path) -> config.RunConfig:
    return config.RunConfig(model=config.Model(source=str(model), date="01 Jan 2020"), measure=short())


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
