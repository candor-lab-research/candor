"""End to end on a small real model with the real converter. Run with CANDOR_E2E=1 (downloads ~270 MB
the first time, and sets up llama.cpp's converter)."""

import json
import os
import socket
from pathlib import Path

import gguf
import pytest
import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

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
    bf16, quantized = out / "model-BF16.gguf", out / "model-Q4_K_M.gguf"
    convert.to_bf16(folder, bf16, out / "convert.log")
    convert.quantize(bf16, quantized, "Q4_K_M", out / "convert.log")
    return folder, bf16, quantized, out


def test_every_edit_target_reaches_the_gguf_unchanged(converted):
    folder, bf16, _, _ = converted
    assert converted_targets(folder, bf16) == []


def converted_targets(folder: Path, model_gguf: Path) -> list[str]:
    config = source.read_config(str(folder))
    text = source.text_config(config)
    layers = text["num_hidden_layers"]
    found = targets.find_targets(source.tensor_shapes(str(folder)), text["hidden_size"], layers)
    arch = targets.gguf_architecture(config)
    assert found.ok and arch is not None
    return convert.check_targets(model_gguf, folder, found.writers, arch, layers)


def test_a_model_whose_tokenizer_transformers_5_saved_converts_with_the_same_vocabulary(converted, tmp_path):
    smol, smol_gguf, _, _ = converted
    folder, out = tmp_path / "tiny", tmp_path / "out"
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=49152, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4
    )
    LlamaForCausalLM(config).bfloat16().save_pretrained(folder)
    PreTrainedTokenizerFast.from_pretrained(smol).save_pretrained(folder)
    saved = json.loads((folder / "tokenizer_config.json").read_text())
    assert saved["tokenizer_class"] == "TokenizersBackend"
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    out.mkdir()

    convert.to_bf16(folder, out / "model.gguf", out / "convert.log")

    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before
    assert sorted(path.name for path in out.iterdir()) == ["convert.log", "model.gguf"]
    assert converted_targets(folder, out / "model.gguf") == []
    ours, reference = gguf.GGUFReader(out / "model.gguf").fields, gguf.GGUFReader(smol_gguf).fields
    for key in (
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.chat_template",
    ):
        assert ours[key].contents() == reference[key].contents(), key


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


def test_init_and_run_prepare_every_stage_resume_and_follow_settings(tmp_path):
    from typer.testing import CliRunner

    from candor.cli import app

    runner = CliRunner(env={"COLUMNS": "250"})
    run_folder = tmp_path / "smol"
    created = runner.invoke(app, ["init", "smol", "--model", MODEL, "--runs", str(tmp_path)])
    assert created.exit_code == 0, created.output
    first = runner.invoke(app, ["run", str(run_folder)])
    assert first.exit_code == 0, first.output
    assert "60 edit targets" in first.output and "1000 prompts" in first.output

    results = [
        run_folder / "gguf" / "model-BF16.gguf",
        run_folder / "prompts.json",
        run_folder / "directions.pt",
    ]
    results += sorted((run_folder / "adapters").rglob("*.gguf"))
    stamps = {path: path.stat().st_mtime_ns for path in results}
    again = runner.invoke(app, ["run", str(run_folder)])
    assert again.exit_code == 0, again.output
    assert "Rebuilding" not in again.output and "Writing adapters" not in again.output
    assert {path: path.stat().st_mtime_ns for path in stamps} == stamps

    settings = (run_folder / "candor.toml").read_text()
    (run_folder / "candor.toml").write_text(
        settings.replace('system_prompt = ""', 'system_prompt = "Be brief."')
    )
    changed = runner.invoke(app, ["run", str(run_folder)])
    assert changed.exit_code == 0, changed.output
    assert "Rebuilding prompts.json" in changed.output
    assert "Rebuilding directions.pt" in changed.output
    assert "Converting" not in changed.output
    assert (run_folder / "gguf" / "model-BF16.gguf").stat().st_mtime_ns == stamps[results[0]]
