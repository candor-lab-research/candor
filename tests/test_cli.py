import json

import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from candor.cli import app

from .test_targets import llama
from .tiny_tokenizer import WITH_BOS, tokenizer

HIDDEN = 64


def model_folder(tmp_path, extra: dict[str, list[int]] | None = None):
    folder = tmp_path / "model"
    folder.mkdir()
    shapes = llama() | (extra or {})
    save_file({name: torch.zeros(shape) for name, shape in shapes.items()}, folder / "model.safetensors")
    config = {"model_type": "llama", "hidden_size": HIDDEN, "num_hidden_layers": 2}
    (folder / "config.json").write_text(json.dumps(config))
    tokenizer(WITH_BOS, adds_bos=True).save_pretrained(folder)
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
