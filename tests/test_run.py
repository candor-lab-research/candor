import json
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
    (folder / "config.json").write_text(
        json.dumps(
            {"architectures": ["LlamaForCausalLM"], "hidden_size": HIDDEN, "num_hidden_layers": LAYERS}
        )
    )
    return folder


@pytest.fixture
def current(tmp_path, model, monkeypatch):
    settings = config.RunConfig(model=config.Model(source=str(model)))
    folder = tmp_path / "runs" / "test"
    folder.mkdir(parents=True)
    found = F.normalize(torch.randn(LAYERS + 1, HIDDEN), dim=-1)
    monkeypatch.setattr(Run, "directions", lambda self: found)
    return Run(folder, settings, report=lambda _: None)


def test_one_adapter_per_target_for_each_direction_set(current):
    made = current.adapters()
    assert sorted(made) == ["layer-0", "layer-1", "per-layer"]
    for paths in made.values():
        assert sorted(paths) == [
            f"blk.{i}.{kind}.weight" for i in range(2) for kind in ("attn_output", "ffn_down")
        ]
        assert all(path.exists() for path in paths.values())


def test_adapters_hold_the_edit_for_their_layers_direction(current, model):
    made = current.adapters()
    found = current.directions()
    weight = ModelFiles(model).tensor("model.layers.1.mlp.down_proj.weight")
    for set_name, direction in (("per-layer", found[2]), ("layer-0", found[1])):
        b, a = rank_one(weight, direction, RowNorm.PRE)
        reader = gguf.GGUFReader(made[set_name]["blk.1.ffn_down.weight"])
        tensors = {t.name: t.data for t in reader.tensors}
        torch.testing.assert_close(torch.from_numpy(tensors["blk.1.ffn_down.weight.lora_a"].copy()), a)
        torch.testing.assert_close(torch.from_numpy(tensors["blk.1.ffn_down.weight.lora_b"].copy()), b)


def test_finished_adapter_sets_are_reused_and_leftovers_replaced(current):
    first = current.adapters()
    stamp = first["layer-0"]["blk.0.attn_output.weight"].stat().st_mtime_ns
    leftover = current.path("adapters", ".layer-1.partial")
    shutil.rmtree(current.path("adapters", "layer-1"))
    leftover.mkdir()
    (leftover / "half-written.gguf").write_bytes(b"x")
    second = current.adapters()
    assert second["layer-0"]["blk.0.attn_output.weight"].stat().st_mtime_ns == stamp
    assert not leftover.exists()
    assert sorted(p.name for p in current.path("adapters", "layer-1").iterdir()) == sorted(
        p.name for p in second["layer-1"].values()
    )


def test_conversion_stops_early_without_enough_disk(current, monkeypatch):
    monkeypatch.setattr(run.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(0, 0, 1000))
    with pytest.raises(RunError, match="GB free"):
        current.gguf()


def test_a_deleted_gguf_is_rebuilt_not_trusted(current, monkeypatch):
    out = current.path("gguf")
    out.mkdir()
    (out / "checked").write_text("{}")
    calls = []

    def fake_convert(folder, out_dir, quantization, log):
        calls.append(quantization)
        for name in ("model-BF16.gguf", f"model-{quantization}.gguf"):
            (out_dir / name).write_bytes(b"gguf")
        return out_dir / "model-BF16.gguf", out_dir / f"model-{quantization}.gguf"

    monkeypatch.setattr(run.convert, "to_gguf", fake_convert)
    monkeypatch.setattr(run.convert, "check_targets", lambda *_: [])
    current.gguf()
    assert calls == ["Q4_K_M"]
    current.gguf()
    assert calls == ["Q4_K_M"]


def test_a_gguf_that_changed_a_target_stops_the_run(current, monkeypatch):
    def fake_convert(folder, out_dir, quantization, log):
        return out_dir / "a.gguf", out_dir / "b.gguf"

    monkeypatch.setattr(run.convert, "to_gguf", fake_convert)
    monkeypatch.setattr(run.convert, "check_targets", lambda *_: ["blk.0.attn_output.weight changed"])
    with pytest.raises(RunError, match="changed"):
        current.gguf()
    assert not current.path("gguf", "checked").exists()


def test_a_finished_result_is_never_replaced(tmp_path):
    (tmp_path / "final").write_text("kept")
    (tmp_path / "partial").write_text("new")
    with pytest.raises(RunError):
        run._replace(tmp_path / "partial", tmp_path / "final")
    assert (tmp_path / "final").read_text() == "kept"


DATED_TEMPLATE = (
    "{%- if not date_string is defined %}{%- set date_string = strftime_now('%d %b %Y') %}{%- endif %}"
    "<s>Today: {{ date_string }}\n{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant:{% endif %}"
)


@pytest.mark.skipif(not LLAMA_SERVER.exists(), reason="no llama-server")
@pytest.mark.parametrize(
    ("template", "works"),
    [(DATED_TEMPLATE, True), (DATED_TEMPLATE.replace("{{ date_string }}", "today"), False)],
)
def test_prompt_ids_use_the_pinned_date(current, monkeypatch, template, works):
    tiny = current.folder / "tiny.gguf"
    write_model(tiny, random_weights(), chat_template=template)
    current.settings.model.date = "09 Oct 2026"
    monkeypatch.setattr(Run, "gguf", lambda self: (tiny, tiny))
    monkeypatch.setattr(run.PromptSet, "load", lambda self: ["first prompt", "second"])
    monkeypatch.setenv("CANDOR_LLAMA_BIN", str(LLAMA_SERVER.parent))
    if works:
        ids = current.prompt_ids()
        assert {name: len(v) for name, v in ids.items()} == dict.fromkeys(run.PROMPT_SETS, 2)
    else:
        with pytest.raises(RunError, match="doesn't appear"):
            current.prompt_ids()
        assert not current.path("prompts.json").exists()
