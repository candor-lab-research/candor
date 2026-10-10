"""End to end on a small real model with the real converter. Run with CANDOR_E2E=1 (downloads ~270 MB
the first time, and sets up llama.cpp's converter)."""

import os
import socket
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from candor import convert, source, targets, toolchain
from candor.engine import LlamaServer

pytestmark = pytest.mark.skipif(os.environ.get("CANDOR_E2E") != "1", reason="set CANDOR_E2E=1 to run")
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
CHATS = [
    [{"role": "user", "content": "Name three primary colours."}],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 17 times 23?"},
    ],
    [{"role": "user", "content": "A haiku about the sea, with an emoji \U0001f30a."}],
    [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "How are you?"},
    ],
]


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    folder = Path(snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"]))
    out = tmp_path_factory.mktemp("smol")
    bf16, quantized = convert.to_gguf(folder, out, "Q4_K_M", out / "convert.log")
    return folder, bf16, quantized, out


def test_every_edit_target_reaches_the_gguf_unchanged(converted):
    folder, bf16, _, _ = converted
    config = source.read_config(str(folder))
    layers = config["num_hidden_layers"]
    found = targets.find_targets(source.tensor_shapes(str(folder)), config["hidden_size"], layers)
    arch = targets.gguf_architecture(config)
    assert found.ok and arch is not None
    assert convert.check_targets(bf16, folder, found.writers, arch, layers) == []


def test_chat_ids_match_real_chat_requests_and_transformers(converted):
    folder, _, quantized, out = converted
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    binary = toolchain.binaries() / "llama-server"
    with LlamaServer(
        binary, quantized, out / "server.log", parallel=2, context_per_slot=256, port=port
    ) as server:
        ids = server.chat_ids(CHATS, {})
        lengths = [server.chat_prompt_length(chat, {}) for chat in CHATS]
    tokenizer = AutoTokenizer.from_pretrained(folder)
    assert [len(i) for i in ids] == lengths
    assert ids == [
        tokenizer.apply_chat_template(c, add_generation_prompt=True, return_dict=False) for c in CHATS
    ]


def test_init_and_run_prepare_every_stage_and_resume(tmp_path):
    from typer.testing import CliRunner

    from candor.cli import app

    runner = CliRunner(env={"COLUMNS": "250"})
    created = runner.invoke(app, ["init", "smol", "--model", MODEL, "--runs", str(tmp_path)])
    assert created.exit_code == 0, created.output
    first = runner.invoke(app, ["run", str(tmp_path / "smol")])
    assert first.exit_code == 0, first.output
    assert "60 edit targets" in first.output and "1000 prompts" in first.output
    stamp = (tmp_path / "smol" / "directions.pt").stat().st_mtime_ns
    again = runner.invoke(app, ["run", str(tmp_path / "smol")])
    assert again.exit_code == 0, again.output
    assert "Converting" not in again.output and "Reading residuals" not in again.output
    assert (tmp_path / "smol" / "directions.pt").stat().st_mtime_ns == stamp
