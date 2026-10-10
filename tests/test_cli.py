import json

import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from candor.cli import app

from .test_targets import llama
from .tiny_tokenizer import WITH_BOS, tokenizer

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
    assert result.exit_code == 2
    assert "couldn't read" in result.output


def test_a_config_without_sizes_cannot_be_inspected(tmp_path):
    result = run(model_folder(tmp_path, config={"architectures": ["LlamaForCausalLM"], "text_config": {}}))
    assert result.exit_code == 2
    assert "hidden size and layer count" in result.output


def test_settings_under_text_config_are_found(tmp_path):
    nested = {
        "architectures": ["LlamaForCausalLM"],
        "text_config": {"hidden_size": HIDDEN, "num_hidden_layers": 2},
    }
    result = run(model_folder(tmp_path, config=nested))
    assert result.exit_code == 0, result.output
