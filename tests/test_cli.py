import json

import pytest
import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from candor import config
from candor.cli import app

from .test_checks import needs_node
from .test_targets import llama
from .tiny_tokenizer import DATED, PINNABLE, WITH_BOS, tokenizer

HIDDEN = 64


def model_folder(
    tmp_path, extra: dict[str, list[int]] | None = None, config=None, template: str | None = WITH_BOS
):
    folder = tmp_path / "model"
    folder.mkdir()
    shapes = llama() | (extra or {})
    save_file({name: torch.zeros(shape) for name, shape in shapes.items()}, folder / "model.safetensors")
    config = config or {"architectures": ["LlamaForCausalLM"], "hidden_size": HIDDEN, "num_hidden_layers": 2}
    (folder / "config.json").write_text(json.dumps(config))
    tokenizer(template, adds_bos=True).save_pretrained(folder)
    return folder


def run(folder):
    return CliRunner(env={"COLUMNS": "250"}).invoke(app, ["inspect", str(folder)])


def test_inspect_passes_a_llama_style_model_and_warns_about_heretic(tmp_path):
    result = run(model_folder(tmp_path))
    assert result.exit_code == 0, result.output
    assert "4 writers, every layer covered" in result.output
    assert "blk.0.attn_output.weight" in result.output
    assert "heretic 2" in result.output
    assert "fail" not in result.output


def test_inspect_fails_on_a_writer_it_cannot_place(tmp_path):
    result = run(model_folder(tmp_path, {"model.layers.1.mlp.mystery.weight": [HIDDEN, 128]}))
    assert result.exit_code == 1
    assert "model.layers.1.mlp.mystery.weight" in result.output


def test_a_base_model_fails_cleanly(tmp_path):
    result = run(model_folder(tmp_path, template=None))
    assert result.exit_code == 1
    assert "base model" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_a_missing_folder_or_repo_cannot_be_inspected(tmp_path):
    result = run(tmp_path / "does-not-exist")
    assert result.exit_code == 3
    assert "couldn't read" in result.output


def test_a_config_without_sizes_cannot_be_inspected(tmp_path):
    result = run(model_folder(tmp_path, config={"architectures": ["LlamaForCausalLM"], "text_config": {}}))
    assert result.exit_code == 3
    assert "hidden size and layer count" in result.output


def test_settings_under_text_config_are_found(tmp_path):
    nested = {
        "architectures": ["LlamaForCausalLM"],
        "text_config": {"hidden_size": HIDDEN, "num_hidden_layers": 2},
    }
    result = run(model_folder(tmp_path, config=nested))
    assert result.exit_code == 0, result.output


def test_a_template_that_cannot_render_one_message_fails_cleanly(tmp_path):
    result = run(model_folder(tmp_path, template="{{ raise_exception('broken') }}"))
    assert result.exit_code == 1
    assert "can't render a one-message chat" in result.output


def test_a_quantised_checkpoint_fails(tmp_path):
    scales = {f"model.layers.{i}.mlp.down_proj.weight_scale": [HIDDEN, 1] for i in range(2)}
    result = run(model_folder(tmp_path, scales))
    assert result.exit_code == 1
    assert "quantisation scales" in result.output


def init(tmp_path, folder, name="first"):
    return CliRunner(env={"COLUMNS": "250"}).invoke(
        app, ["init", name, "--model", str(folder), "--runs", str(tmp_path / "runs")]
    )


def test_init_writes_settings_that_load(tmp_path):
    folder = model_folder(tmp_path)
    result = init(tmp_path, folder)
    assert result.exit_code == 0, result.output
    settings = config.load(tmp_path / "runs" / "first")
    assert settings.model.source == str(folder.resolve())
    assert settings.model.date is None
    assert (tmp_path / "runs" / "first" / "inspect.json").exists()


def test_init_pins_the_date_when_the_template_can_take_one(tmp_path):
    result = init(tmp_path, model_folder(tmp_path, template=PINNABLE))
    assert result.exit_code == 0, result.output
    assert config.load(tmp_path / "runs" / "first").model.date == config.today_in("%d %b %Y")


def test_init_warns_when_the_date_cannot_be_pinned(tmp_path):
    result = init(tmp_path, model_folder(tmp_path, template=DATED))
    assert result.exit_code == 0, result.output
    assert "finish each run on the day it starts" in result.output
    assert config.load(tmp_path / "runs" / "first").model.date is None


def test_init_refuses_a_model_candor_cannot_edit(tmp_path):
    result = init(tmp_path, model_folder(tmp_path, {"model.layers.1.mlp.mystery.weight": [HIDDEN, 128]}))
    assert result.exit_code == 1
    assert "Nothing was written" in result.output
    assert not (tmp_path / "runs" / "first").exists()


def test_init_never_overwrites_a_run(tmp_path):
    folder = model_folder(tmp_path)
    assert init(tmp_path, folder).exit_code == 0
    before = (tmp_path / "runs" / "first" / "candor.toml").read_text()
    result = init(tmp_path, folder)
    assert result.exit_code == 1
    assert "already exists" in result.output
    assert (tmp_path / "runs" / "first" / "candor.toml").read_text() == before


@needs_node
def test_init_goes_ahead_when_only_bionic_cannot_read_the_template(tmp_path):
    template = "{% if messages[0]['content'] is sameas true %}x{% endif %}{{ messages[0]['content'] }}"
    result = init(tmp_path, model_folder(tmp_path, template=template))
    assert result.exit_code == 0, result.output
    assert "Bionic can't read this model's chat template" in result.output


def test_init_pins_the_date_in_the_templates_own_format(tmp_path):
    template = PINNABLE.replace("%d %b %Y", "%Y-%m-%d")
    result = init(tmp_path, model_folder(tmp_path, template=template))
    assert result.exit_code == 0, result.output
    assert config.load(tmp_path / "runs" / "first").model.date == config.today_in("%Y-%m-%d")


def test_a_local_model_has_no_revision(tmp_path):
    assert init(tmp_path, model_folder(tmp_path)).exit_code == 0
    assert config.load(tmp_path / "runs" / "first").model.revision is None


@pytest.mark.parametrize("name", ["../escape", "a/b", "..", "-x"])
def test_init_rejects_names_that_are_not_one_plain_folder(tmp_path, name):
    result = init(tmp_path, model_folder(tmp_path), name=name)
    assert result.exit_code == 2
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "runs").exists()


def test_init_expands_home_in_the_runs_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    folder = model_folder(tmp_path)
    result = CliRunner(env={"COLUMNS": "250"}).invoke(
        app, ["init", "first", "--model", str(folder), "--runs", "~/runs"]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "home" / "runs" / "first" / "candor.toml").exists()
    assert not (tmp_path / "~").exists()


def test_a_failure_while_writing_exits_3_and_leaves_nothing(tmp_path, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(config, "render", broken)
    result = init(tmp_path, model_folder(tmp_path))
    assert result.exit_code == 3
    assert "disk full" in result.output
    assert not (tmp_path / "runs" / "first").exists()
