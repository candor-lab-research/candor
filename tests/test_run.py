import json
import os
import shutil
import threading

import gguf
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from candor import config, run
from candor.edit import RowNorm, rank_one
from candor.run import Run, RunError
from candor.weights import ModelFiles

from .test_engine import LLAMA_SERVER
from .test_targets import llama
from .tiny_gguf import random_weights, write_model

HIDDEN, LAYERS = 64, 2


@pytest.fixture
def model(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    torch.manual_seed(7)
    save_file({name: torch.randn(shape) for name, shape in llama().items()}, folder / "model.safetensors")
    model_config = {"architectures": ["LlamaForCausalLM"], "hidden_size": HIDDEN, "num_hidden_layers": LAYERS}
    (folder / "config.json").write_text(json.dumps(model_config | {"max_position_embeddings": 256}))
    return folder


@pytest.fixture
def directions_by_seed():
    """Directions the fake directions stage returns, changeable to stand for a real change upstream."""
    state = {"seed": 0}

    def found() -> torch.Tensor:
        generator = torch.Generator().manual_seed(state["seed"])
        return F.normalize(torch.randn(LAYERS + 1, HIDDEN, generator=generator), dim=-1)

    return state, found


@pytest.fixture
def current(tmp_path, model, monkeypatch, directions_by_seed):
    """A run whose directions stage is faked: these tests are about adapters and records."""
    state, found = directions_by_seed
    settings = config.RunConfig(model=config.Model(source=str(model)))
    folder = tmp_path / "runs" / "test"
    folder.mkdir(parents=True)
    messages = []

    def fake_directions(self):
        self.inputs["directions"] = f"seed-{state['seed']}"
        return found()

    monkeypatch.setattr(Run, "directions", fake_directions)
    return Run(folder, settings, report=messages.append), messages


def adapter_factors(path):
    tensors = {t.name: torch.from_numpy(t.data.copy()) for t in gguf.GGUFReader(path).tensors}
    a = next(v for k, v in tensors.items() if k.endswith("lora_a"))
    b = next(v for k, v in tensors.items() if k.endswith("lora_b"))
    return b, a


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
    assert sorted(p.name for p in this.path("adapters").iterdir()) == [
        "layer-1", "layer-1.inputs.json", "per-layer", "per-layer.inputs.json",
    ]  # fmt: skip


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


def test_hidden_files_in_the_model_folder_change_nothing(current, model):
    this, _ = current
    before = this.model_fingerprint
    (model / ".DS_Store").write_bytes(b"finder")
    del this.model_fingerprint
    assert this.model_fingerprint == before
    os.utime(model / "config.json", ns=(1, 1))
    del this.model_fingerprint
    assert this.model_fingerprint != before


def test_conversion_stops_early_without_enough_disk(current, monkeypatch):
    monkeypatch.setattr(run.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(0, 0, 1000))
    with pytest.raises(RunError, match="GB free"):
        current[0].gguf()


@pytest.fixture
def fake_conversion(monkeypatch):
    calls = []

    def to_bf16(folder, destination, log):
        calls.append("bf16")
        destination.write_bytes(b"bf16")

    def quantize(bf16, destination, quantization, log):
        calls.append(quantization)
        destination.write_bytes(quantization.encode())

    monkeypatch.setattr(run.convert, "to_bf16", to_bf16)
    monkeypatch.setattr(run.convert, "quantize", quantize)
    monkeypatch.setattr(run.convert, "check_targets", lambda *_: [])
    return calls


def test_gguf_is_converted_once_and_requantised_only_when_needed(current, fake_conversion):
    this, _ = current
    this.gguf()
    this.gguf()
    assert fake_conversion == ["bf16", "Q4_K_M"]
    this.settings.deployment.quantization = "Q8_0"
    this.gguf()
    assert fake_conversion == ["bf16", "Q4_K_M", "Q8_0"]


def test_a_deleted_gguf_is_rebuilt_not_trusted(current, fake_conversion):
    this, _ = current
    _, quantized = this.gguf()
    quantized.unlink()
    this.gguf()
    assert fake_conversion == ["bf16", "Q4_K_M", "Q4_K_M"]


def test_a_gguf_that_changed_a_target_stops_the_run_and_leaves_nothing(current, fake_conversion, monkeypatch):
    this, _ = current
    monkeypatch.setattr(run.convert, "check_targets", lambda *_: ["blk.0.attn_output.weight changed"])
    with pytest.raises(RunError, match="changed"):
        this.gguf()
    assert sorted(p.name for p in this.path("gguf").iterdir()) == []


def test_two_commands_cannot_use_one_run_at_once(current):
    this, _ = current
    entered, release = threading.Event(), threading.Event()

    def hold():
        with this.locked():
            entered.set()
            release.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    entered.wait(5)
    try:
        with pytest.raises(RunError, match="another candor command"), this.locked():
            pass
    finally:
        release.set()
        holder.join()


def test_overflowing_activations_stop_the_run(tmp_path, model, monkeypatch):
    this = Run(tmp_path, config.RunConfig(model=config.Model(source=str(model))), report=lambda _: None)
    monkeypatch.setattr(Run, "prompt_ids", lambda self: {"harmful": [[1]], "harmless": [[1]]})
    this.inputs["prompts"] = "p"
    monkeypatch.setattr(
        run.residuals, "last_token_residuals", lambda *_: torch.full((1, 3, HIDDEN), float("nan"))
    )
    monkeypatch.setattr(run.reference, "pad_token_id_for", lambda _: 0)

    class Loaded(torch.nn.Module):
        def to(self, *_):
            return self

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *_, **__: Loaded())
    with pytest.raises(RunError, match="overflowed"):
        this.directions()
    assert not this.path("directions.pt").exists()


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
    monkeypatch.setattr(run.PromptSet, "load", lambda self: prompts["value"])

    def use(template: str):
        tiny = this.folder / "tiny.gguf"
        write_model(tiny, random_weights(), chat_template=template)

        def fake_gguf(self):
            self.inputs["gguf"] = template
            return tiny, tiny

        monkeypatch.setattr(Run, "gguf", fake_gguf)

    return this, messages, use, prompts


def test_prompt_ids_take_the_pinned_date(tiny_run):
    this, _, use, _ = tiny_run
    use(TEMPLATE)
    this.settings.model.date = config.today_in("%d %b %Y")
    ids = this.prompt_ids()
    assert {name: len(v) for name, v in ids.items()} == dict.fromkeys(run.PROMPT_SETS, 2)


def test_a_template_that_ignores_the_pinned_date_is_caught_even_today(tiny_run):
    this, _, use, _ = tiny_run
    use(IGNORES_THE_DATE)
    this.settings.model.date = config.today_in("%d %b %Y")
    with pytest.raises(RunError, match="ignores the pinned date"):
        this.prompt_ids()
    assert not this.path("prompts.json").exists()


def test_a_changed_system_prompt_rebuilds_the_prompt_ids(tiny_run):
    this, messages, use, _ = tiny_run
    use(TEMPLATE)
    before = this.prompt_ids()
    this.settings.deployment.system_prompt = "Be brief."
    after = this.prompt_ids()
    assert any("Rebuilding prompts.json" in m for m in messages)
    assert len(after["harmful"][0]) > len(before["harmful"][0])


def test_prompts_too_long_for_the_model_are_named(tiny_run):
    this, _, use, prompts = tiny_run
    use(TEMPLATE)
    prompts["value"] = ["w0 " * 100]
    with pytest.raises(RunError, match="don't fit the model's 256 tokens: 1 harmful"):
        this.prompt_ids()
