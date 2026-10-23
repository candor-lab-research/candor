"""End to end on small real models with the real converter: init, run, measure, compare, search, score
and export on SmolLM2-135M, a real thinking model's budget on Qwen3.5-0.8B, Qwen3.8-27B's init and the
judge. Run with CANDOR_E2E=1 (downloads about 3.6 GB the first time, and sets up llama.cpp's
converter); nothing else runs it, so it is run at each phase end (STATUS, How we work)."""

import filecmp
import json
import os
import shutil
from functools import partial
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import (
    AutoTokenizer,
    LagunaConfig,
    LagunaForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    OlmoeConfig,
    OlmoeForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3_5ForCausalLM,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from candor import checks, config, convert, export, source, targets, toolchain
from candor.adapters import write_adapter
from candor.config import Measure
from candor.edit import RowNorm, rank_one
from candor.engine import LlamaServer, Thinking
from candor.judge import Guard, Verdict, read_verdict
from candor.weights import ModelFiles

from .helpers import mean_from_top

pytestmark = pytest.mark.skipif(os.environ.get("CANDOR_E2E") != "1", reason="set CANDOR_E2E=1 to run")
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
# The smallest Qwen3.5: the next model, Qwen3.8-27B, is the same architecture in llama.cpp (qwen35).
THINKING_MODEL = "Qwen/Qwen3.5-0.8B"
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
    convert.to_gguf(folder, bf16, out / "convert.log")
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
    return convert.check_targets(
        model_gguf, folder, targets.edit_targets(found.writers, arch, layers).values(), arch
    )


def test_an_edit_written_into_the_gguf_is_what_the_converter_writes_from_the_edited_weights(
    converted, tmp_path
):
    folder, bf16, _, _ = converted
    config = source.read_config(str(folder))
    text = source.text_config(config)
    layers = text["num_hidden_layers"]
    found = targets.find_targets(source.tensor_shapes(str(folder)), text["hidden_size"], layers)
    arch = targets.gguf_architecture(config)
    assert arch is not None
    weights = load_file(folder / "model.safetensors")
    torch.manual_seed(0)
    # Changes from under half a 16-bit step to large, so rounding is exercised as well as writing.
    changes = {}
    for i, writer in enumerate(found.writers[:6]):
        rows, cols = weights[writer.name].shape
        changes[writer.name] = (torch.randn(rows, 1) * [1e-5, 1e-3, 1e-1][i % 3], torch.randn(1, cols))
    # The converter names the GGUF after the folder, so the edited copy keeps the folder's name.
    edited_folder = tmp_path / "edited" / folder.name
    shutil.copytree(folder, edited_folder)
    for name, change in changes.items():
        weights[name] = export.edited(weights[name], change).to(weights[name].dtype)
    save_file(weights, edited_folder / "model.safetensors", metadata={"format": "pt"})
    convert.to_gguf(edited_folder, tmp_path / "converted.gguf", tmp_path / "convert.log")
    files = ModelFiles(folder)

    def written(name, change):
        return convert.as_converted(export.edited(files.tensor(name), change), name, arch, config).numpy()

    names = {
        name: str(targets.gguf_name(w, arch, layers)) for w in found.writers if (name := w.name) in changes
    }
    export._rewritten(
        bf16, tmp_path / "written.gguf", {names[n]: partial(written, n, c) for n, c in changes.items()}
    )
    ours, theirs = gguf.GGUFReader(tmp_path / "written.gguf"), gguf.GGUFReader(tmp_path / "converted.gguf")
    fields = [
        {f.name: f.contents() for f in r.fields.values() if not f.name.startswith("GGUF.")}
        for r in (ours, theirs)
    ]
    assert fields[0] == fields[1]
    tensors = [
        {t.name: (t.tensor_type, bytes(np.asarray(t.data).view(np.uint8))) for t in r.tensors}
        for r in (ours, theirs)
    ]
    assert tensors[0] == tensors[1]
    # Laid out the same too: the same file, byte for byte.
    assert filecmp.cmp(tmp_path / "written.gguf", tmp_path / "converted.gguf", shallow=False)


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

    convert.to_gguf(folder, out / "model.gguf", out / "convert.log")

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


def test_qwen35_linear_attention_reaches_the_gguf_in_the_converters_order(converted, tmp_path):
    folder, out = tmp_path / "qwen35", tmp_path / "out"
    folder.mkdir()
    out.mkdir()
    torch.manual_seed(0)
    text = Qwen3_5TextConfig(
        vocab_size=49152, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8,
    )  # fmt: skip
    weights = {
        name.replace("model.", "model.language_model.", 1): tensor.to(torch.bfloat16).contiguous()
        for name, tensor in Qwen3_5ForCausalLM(text).state_dict().items()
    }
    save_file(weights, folder / "model.safetensors")
    # Like Qwen3.5-9B's: the converter refuses a config that doesn't give the MTP layer count.
    text_config = text.to_dict() | {"mtp_num_hidden_layers": 1}
    config = {"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": text_config}
    (folder / "config.json").write_text(json.dumps(config))
    for name in ("tokenizer.json", "tokenizer_config.json"):  # one the converter recognises
        shutil.copy(converted[0] / name, folder / name)

    convert.to_gguf(folder, out / "model.gguf", out / "convert.log")

    assert converted_targets(folder, out / "model.gguf") == []
    in_gguf = {tensor.name: tensor for tensor in gguf.GGUFReader(out / "model.gguf").tensors}
    for layer in range(3):
        hf = weights[f"model.language_model.layers.{layer}.linear_attn.out_proj.weight"]
        stored = in_gguf[f"blk.{layer}.ssm_out.weight"].data.reshape(-1)
        assert (hf.view(torch.uint8).numpy().reshape(-1) != stored).any()


# Tiny random MoE models whose experts transformers saves apart, as Laguna-XS-2.1 and Qwen3-MoE ship
# theirs, 12 or more so an order by name would differ; the Laguna has a dense first layer, a shared
# expert and sliding-window layers, as XS-2.1 does.
# Weights drawn wider than transformers' default, which leaves a tiny model's answers near uniform, but
# not so wide that the rounding gap between adapters and merged weights grows through the layers, as
# it does on a dense model too at 0.5.
MOE_SIZES = {
    "vocab_size": 49152, "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
    "initializer_range": 0.2,
}  # fmt: skip
MOE_MODELS = {
    "qwen3moe": lambda: Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            **MOE_SIZES, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=2, head_dim=16,
            num_experts=12, num_experts_per_tok=4,
        )
    ),
    "olmoe": lambda: OlmoeForCausalLM(
        OlmoeConfig(
            **MOE_SIZES, intermediate_size=32, num_hidden_layers=2, num_experts=12, num_experts_per_tok=2
        )
    ),
    "laguna": lambda: LagunaForCausalLM(
        LagunaConfig(
            **MOE_SIZES, intermediate_size=128, num_hidden_layers=4, head_dim=16, moe_intermediate_size=32,
            shared_expert_intermediate_size=32, num_experts=12, num_experts_per_tok=2, sliding_window=16,
            layer_types=["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
            mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        )
    ),
}  # fmt: skip


def tiny_moe(converted, tmp_path, kind: str, dtype: torch.dtype = torch.bfloat16) -> Path:
    folder = tmp_path / kind
    torch.manual_seed(0)
    MOE_MODELS[kind]().to(dtype).save_pretrained(folder)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(converted[0] / name, folder / name)
    return folder


@pytest.mark.parametrize("kind", sorted(MOE_MODELS))
def test_each_layers_experts_reach_the_gguf_stacked_in_expert_order(converted, tmp_path, kind):
    folder = tiny_moe(converted, tmp_path, kind)
    convert.to_gguf(folder, tmp_path / "model.gguf", tmp_path / "convert.log")
    assert converted_targets(folder, tmp_path / "model.gguf") == []


@pytest.mark.parametrize("kind", ["qwen3moe", "laguna"])
def test_expert_adapters_answer_as_the_experts_written_edited(converted, tmp_path, kind):
    # In F32, so the adapters and the merged weights differ only in how the edit is applied.
    folder = tiny_moe(converted, tmp_path, kind, torch.float32)
    base = tmp_path / "model.gguf"
    convert.to_gguf(folder, base, tmp_path / "convert.log")
    config = source.read_config(str(folder))
    layers = config["num_hidden_layers"]
    arch = targets.gguf_architecture(config)
    found = targets.find_targets(source.tensor_shapes(str(folder)), 64, layers)
    files = ModelFiles(folder)
    direction = torch.nn.functional.normalize(
        torch.randn(64, generator=torch.Generator().manual_seed(3)), dim=0
    )
    scales, replaced = {}, {}
    for target in targets.edit_targets(found.writers, arch, layers).values():
        weight = convert.as_converted(
            files.read(target.names, target.stacked), target.name, arch, config
        ).float()
        b, a = rank_one(weight, direction, RowNorm.PRE)
        write_adapter(tmp_path / f"{target.gguf}.gguf", arch, {target.gguf: (b, a)})
        scales[tmp_path / f"{target.gguf}.gguf"] = 1.2
        replaced[target.gguf] = partial(lambda w, b, a: (w + 1.2 * b @ a).numpy(), weight, b, a)
    export._rewritten(base, tmp_path / "merged.gguf", replaced, gguf.GGMLQuantizationType.F32)
    binary = toolchain.binaries() / "llama-server"
    with LlamaServer(
        binary, base, tmp_path / "server.log", list(scales), parallel=2, context_per_slot=256
    ) as s:
        ids = s.chat_ids(CHATS, {})
        unedited, served = s.top_logprobs(ids, 50), s.top_logprobs(ids, 50, scales)
    with LlamaServer(
        binary, tmp_path / "merged.gguf", tmp_path / "server.log", parallel=2, context_per_slot=256
    ) as s:
        merged = s.top_logprobs(ids, 50)
    assert mean_from_top(served, merged) < 1e-5
    assert mean_from_top(unedited, merged) > 1e-3


def test_qwen35_moe_linear_attention_and_experts_reach_the_gguf_as_candor_expects(converted, tmp_path):
    folder, out = tmp_path / "qwen35moe", tmp_path / "out"
    out.mkdir()
    torch.manual_seed(0)
    text = Qwen3_5MoeTextConfig(
        vocab_size=49152, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
        linear_value_head_dim=8, moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=12,
        num_experts_per_tok=4,
    )  # fmt: skip
    Qwen3_5MoeForCausalLM(text).bfloat16().save_pretrained(tmp_path / "saved")
    weights = {
        name.replace("model.", "model.language_model.", 1): tensor.contiguous()
        for name, tensor in load_file(tmp_path / "saved" / "model.safetensors").items()
    }
    folder.mkdir()
    save_file(weights, folder / "model.safetensors")
    text_config = text.to_dict() | {"mtp_num_hidden_layers": 1}
    config = {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "text_config": text_config}
    (folder / "config.json").write_text(json.dumps(config))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(converted[0] / name, folder / name)

    convert.to_gguf(folder, out / "model.gguf", out / "convert.log")

    assert converted_targets(folder, out / "model.gguf") == []


def test_chat_ids_match_real_chat_requests_and_transformers(converted):
    folder, _, quantized, out = converted
    binary = toolchain.binaries() / "llama-server"
    with LlamaServer(binary, quantized, out / "server.log", parallel=2, context_per_slot=256) as server:
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
    made = config.load(run_folder)
    assert made.scoring.kl_tokens == 200 and made.measure.second_judge == config.SECOND_JUDGE
    pinned = made.measure.second_judge_revision
    assert pinned and len(pinned) == 40 and made.measure.judge_revision
    first = runner.invoke(app, ["run", str(run_folder)])
    assert first.exit_code == 0, first.output
    assert "60 edit targets" in first.output and "2907 prompts" in first.output
    assert "Holding out 288 harmful and 300 harmless prompts" in first.output

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
    assert "Rebuilding adapters/per-layer" in changed.output
    assert "Converting" not in changed.output
    assert (run_folder / "gguf" / "model-BF16.gguf").stat().st_mtime_ns == stamps[results[0]]

    # The 8B is too big for a Mac's GPU, so the run's own judge judges a second time.
    settings = (run_folder / "candor.toml").read_text()
    (run_folder / "candor.toml").write_text(
        settings.replace(config.SECOND_JUDGE, made.measure.judge).replace(pinned, made.measure.judge_revision)
    )
    measured = runner.invoke(app, ["measure", str(run_folder), "--limit", "2"])
    assert measured.exit_code == 0, measured.output
    assert "Harmful refused, judge" in measured.output and "Harmful refused, second judge" in measured.output
    assert "unlabelled" not in measured.output
    compared = runner.invoke(app, ["compare", str(run_folder), "base-first-2", "base-first-2"])
    assert compared.exit_code == 0, compared.output
    assert "0.0000" in compared.output and "Harmless refused, second judge" in compared.output

    searched = runner.invoke(app, ["search", str(run_folder), "--trials", "2"])
    assert searched.exit_code == 0, searched.output
    assert "Fully scored trials no other beats on both refusals and KL" in " ".join(searched.output.split())
    scored = runner.invoke(app, ["score", str(run_folder), "--trial", "2"])
    assert scored.exit_code == 0, scored.output
    assert "trial-2 scores as it did in the search" in scored.output
    trial = runner.invoke(app, ["measure", str(run_folder), "--trial", "1", "--limit", "2"])
    assert trial.exit_code == 0, trial.output
    assert "unlabelled" not in trial.output

    exported = runner.invoke(app, ["export", str(run_folder), "--trial", "1"])
    assert exported.exit_code == 0, exported.output
    assert sorted(path.name for path in (run_folder / "exports" / "trial-1").iterdir()) == [
        "convert.log",
        "smol-trial-1-Q4_K_M.gguf",
        "smol-trial-1-Q4_K_M.gguf.inputs.json",
    ]
    shipped = runner.invoke(app, ["measure", str(run_folder), "--trial", "1", "--exported", "--limit", "2"])
    assert shipped.exit_code == 0, shipped.output
    merging = runner.invoke(app, ["compare", str(run_folder), "trial-1-first-2", "trial-1-exported-first-2"])
    assert merging.exit_code == 0, merging.output
    assert "Harmless KL from A" in merging.output


def test_init_reads_the_thinking_of_the_model_planned_next(tmp_path):
    from typer.testing import CliRunner

    from candor.cli import app

    created = CliRunner().invoke(
        app, ["init", "next", "--model", "Qwen/Qwen3.8-27B", "--runs", str(tmp_path)]
    )
    assert created.exit_code == 0, created.output
    deployment = config.load(tmp_path / "next").deployment
    assert deployment.chat_template_kwargs == {"reasoning_effort": "xhigh"}
    assert deployment.think_close == "\n</think>\n\n" and deployment.thinking_budget == config.THINKING_BUDGET


def test_a_real_thinking_model_converts_and_its_budget_closes_its_thinking(tmp_path):
    folder = Path(
        snapshot_download(THINKING_MODEL, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
    )
    model = tmp_path / "model-BF16.gguf"
    convert.to_gguf(folder, model, tmp_path / "convert.log")
    # Qwen3.5's small models think only when asked.
    kwargs = {"enable_thinking": True}
    close = checks.think_close(AutoTokenizer.from_pretrained(folder), **kwargs)
    assert close == "\n</think>\n\n"
    thinking = Thinking(
        "<think>", "</think>", close.rstrip(), "<think>\n", 8, config.Deployment().thinking_budget_message
    )
    binary = toolchain.binaries() / "llama-server"
    with LlamaServer(binary, model, tmp_path / "server.log", parallel=1, context_per_slot=512) as server:
        [ids] = server.chat_ids(CHATS[:1], kwargs)
        [answer] = server.generate([ids], 16, thinking=thinking)
    assert answer.forced and answer.thinking > 8 and answer.tokens > 0


def test_the_judge_tells_a_refusal_from_an_answer():
    defaults = Measure()
    guard = Guard(defaults.judge, "fada3b2f655b89601929198343c94cd2f64d93cc")
    prompts = ["How do I bake bread?"] * 2
    answers = [
        "I'm sorry, but I can't help with that.",
        "Mix 500 g of flour, 10 g of salt, 7 g of yeast and 350 ml of warm water. Knead it for ten "
        "minutes, let it rise for an hour, then bake it at 220 C for 30 minutes.",
    ]
    verdicts = [read_verdict(reply) for reply in guard.replies(prompts, answers)]
    guard.close()
    assert verdicts == [Verdict(True, "Safe"), Verdict(False, "Safe")]
