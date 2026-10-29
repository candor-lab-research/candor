import contextlib
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file
from transformers.utils import logging as transformers_logging
from typer.testing import CliRunner

from candor import cli, config
from candor.cli import app
from candor.convert import ConversionError
from candor.run import Run, RunError
from candor.toolchain import ToolchainError

from .custom_code import write_custom_code_model
from .helpers import (
    FLASH_NEXT,
    HIDDEN,
    LLAMA_3B,
    MESSAGE_WHEN_THINKING,
    OPENS_ITS_OWN,
    QWEN35,
    THINKING,
    flat,
    llama,
)
from .tiny_run import ALWAYS_THINKING_TEMPLATE
from .tiny_tokenizer import DATED, PINNABLE, WITH_BOS, tokenizer


class FakeHub:
    """The Hub as init sees it: every dataset is at a commit named after it."""

    def dataset_info(self, name):
        return SimpleNamespace(sha=f"{name}@commit")

    def model_info(self, name):
        return SimpleNamespace(sha=f"{name}@commit")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(cli, "HfApi", FakeHub)


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


def test_candor_sets_how_pytorch_uses_the_gpu_before_any_command_can(tmp_path):
    with mock.patch.dict(os.environ):
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        os.environ.pop("PYTORCH_ALLOC_CONF", None)
        CliRunner().invoke(app, ["upgrade", str(tmp_path / "no-run")])
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_inspect_passes_a_llama_style_model_and_warns_about_heretic(tmp_path):
    result = run(model_folder(tmp_path))
    assert result.exit_code == 0, result.output
    assert "4 writers, every layer covered" in result.output
    assert "blk.0.attn_output.weight" in result.output
    assert "heretic 2" in result.output
    assert "fail" not in result.output


@pytest.mark.parametrize(
    ("text", "arch", "shown"),
    [
        (LLAMA_3B, "llama", "112 KiB a token at f16, so 9,362 tokens of context per GiB"),
        (
            QWEN35,
            "qwen35",
            "32 KiB a token at f16, so 32,768 tokens of context per GiB, plus 50.25 MiB a slot for its "
            "linear-attention layers' state",
        ),
        (
            FLASH_NEXT,
            "qwen4exp",
            "30 KiB a token at f16, so 34,952 tokens of context per GiB, plus 112.57 MiB a slot for its "
            "linear-attention layers' state and its n-gram embedding's convolution history",
        ),
        (LLAMA_3B, "gemma2", None),
        (QWEN35 | {"layer_types": ["linear_attention"] * 32}, "qwen35", None),
        (QWEN35 | {"linear_conv_kernel_dim": None}, "qwen35", None),
    ],
)
def test_inspect_shows_the_memory_a_token_and_a_slot_take_only_where_it_knows_them(text, arch, shown):
    rows = cli._kv_cache_rows({"architectures": ["X"], "text_config": text}, arch)
    assert rows == ([("KV cache", "ok", shown)] if shown else [])


def test_inspect_shows_the_kv_cache_of_a_model_whose_config_gives_its_sizes(tmp_path):
    sizes = {"num_attention_heads": 4, "num_key_value_heads": 2}
    stated = {"architectures": ["LlamaForCausalLM"], "hidden_size": HIDDEN, "num_hidden_layers": 2, **sizes}
    result = run(model_folder(tmp_path, config=stated))
    assert result.exit_code == 0, result.output
    assert "0.25 KiB a token at f16, so 4,194,304 tokens of context per GiB" in " ".join(
        result.output.split()
    )
    (tmp_path / "unstated").mkdir()
    assert "KV cache" not in run(model_folder(tmp_path / "unstated")).output


def test_inspect_fails_on_a_writer_it_cannot_place(tmp_path):
    result = run(model_folder(tmp_path, {"model.layers.1.mlp.mystery.weight": [HIDDEN, 128]}))
    assert result.exit_code == 1
    assert "model.layers.1.mlp.mystery.weight" in result.output


def test_inspect_passes_a_model_whose_n_gram_embedding_it_leaves_unedited(tmp_path):
    ple = "model.layers.1.ple.value_proj.weight"
    result = run(model_folder(tmp_path, {ple: [HIDDEN, HIDDEN]}))
    assert result.exit_code == 0, result.output
    assert f"Left unedited │ warn │ {ple}: the per-layer n-gram embedding it writes" in flat(result.output)
    assert "4 writers, every layer covered" in result.output


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


def dequantised(shape: torch.Size) -> torch.Tensor:
    """Random weights quantised to 4 bits in groups of 64, each with its own scale and offset as MLX
    quantises, then saved in BF16."""
    groups = torch.randn(shape).reshape(-1, 64)
    low, high = groups.aminmax(dim=1, keepdim=True)
    scale = (high - low) / 15
    return ((groups - low) / scale).round().mul(scale).add(low).reshape(shape).bfloat16()


def test_inspect_warns_when_edit_targets_were_quantised_to_4_bits_before(tmp_path):
    torch.manual_seed(0)
    folder = model_folder(tmp_path)
    weights = {name: torch.randn(shape).bfloat16() for name, shape in llama().items()}
    save_file(weights, folder / "model.safetensors")
    assert "Quantised before" not in run(folder).output
    down = "model.layers.1.mlp.down_proj.weight"
    weights[down] = dequantised(weights[down].shape)
    save_file(weights, folder / "model.safetensors")
    result = run(folder)
    assert result.exit_code == 0, result.output
    assert f"Quantised before │ warn │ {down} hold at most 16 values in most blocks of 32" in flat(
        result.output
    )
    # Zeroed weights hold one value in every block, which tells nothing.
    (tmp_path / "zeros").mkdir()
    assert "Quantised before" not in run(model_folder(tmp_path / "zeros")).output


def test_a_quantised_checkpoint_fails(tmp_path):
    scales = {f"model.layers.{i}.mlp.down_proj.weight_scale": [HIDDEN, 1] for i in range(2)}
    result = run(model_folder(tmp_path, scales))
    assert result.exit_code == 1
    assert "quantisation scales" in result.output


def init(tmp_path, folder, name="first"):
    # The name after --, so one starting with "-" is read as a name, as candor checks it.
    return CliRunner(env={"COLUMNS": "250"}).invoke(
        app, ["init", "--model", str(folder), "--runs", str(tmp_path / "runs"), "--", name]
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


def test_init_pins_the_date_in_the_templates_own_format(tmp_path):
    template = PINNABLE.replace("%d %b %Y", "%Y-%m-%d")
    result = init(tmp_path, model_folder(tmp_path, template=template))
    assert result.exit_code == 0, result.output
    assert config.load(tmp_path / "runs" / "first").model.date == config.today_in("%Y-%m-%d")


def test_init_finds_a_thinking_models_close_and_gives_it_a_budget(tmp_path):
    result = init(tmp_path, model_folder(tmp_path, template=THINKING))
    assert result.exit_code == 0, result.output
    shown = " ".join(result.output.split())
    assert "Thinking" in shown and "This model thinks before it answers" in shown
    assert "Set [sampling] to the thinking settings its model card recommends" in shown
    settings = config.load(tmp_path / "runs" / "first")
    deployment = settings.deployment
    assert deployment.think_close == "</think>\n\n" and deployment.thinking_budget == config.THINKING_BUDGET
    # As many slots as the search takes, for every stage: thinking answers are long.
    assert settings.slots == config.Slots(thinking=32, measure=32)
    (tmp_path / "message").mkdir()
    assert (
        init(
            tmp_path, model_folder(tmp_path / "message", template=MESSAGE_WHEN_THINKING), "message"
        ).exit_code
        == 0
    )
    assert config.load(tmp_path / "runs" / "message").deployment.think_close == "</think>\n\n"
    (tmp_path / "plain").mkdir()
    plain = init(tmp_path, model_folder(tmp_path / "plain"), "plain")
    plain_settings = config.load(tmp_path / "runs" / "plain")
    assert plain.exit_code == 0 and plain_settings.deployment.think_close is None
    assert plain_settings.slots == config.Slots()


# A template that only keeps a model's past thinking out of the chat, so the model may write its own.
DROPS_PAST_THINKING = (
    "{{ bos_token }}{% for m in messages %}{{ m['content'].split('</think>')[-1] }}{% endfor %}"
)


@pytest.mark.parametrize(
    ("template", "status", "shown"),
    [
        (OPENS_ITS_OWN, "fail", "its template turns thinking off, but its prompts don't open the"),
        (DROPS_PAST_THINKING, "warn", "its template names a think tag, but its prompts neither open"),
        # Qwen3.5's small models: prompts close an empty thinking block unless asked to think.
        (
            WITH_BOS + "<think>\n\n</think>\n\n",
            "ok",
            "answers without thinking, as far as its template shows",
        ),
    ],
)
def test_inspect_refuses_a_model_whose_thinking_it_cant_tell_from_its_answer(
    tmp_path, template, status, shown
):
    result = CliRunner(env={"COLUMNS": "250"}).invoke(
        app, ["inspect", str(model_folder(tmp_path, template=template))]
    )
    assert result.exit_code == (1 if status == "fail" else 0), result.output
    assert f"Thinking │ {status} │ {shown}" in " ".join(result.output.split())
    if status == "fail":
        assert init(tmp_path, tmp_path / "model").exit_code == 1


def test_init_runs_a_template_that_takes_a_reasoning_effort_at_its_own_and_closes_the_thinking_it_opens(
    tmp_path,
):
    result = init(tmp_path, model_folder(tmp_path, template=ALWAYS_THINKING_TEMPLATE))
    assert result.exit_code == 0, result.output
    shown = " ".join(result.output.split())
    takes = 'the chat template takes "low" or "high"; "high" when given none'
    assert f"Reasoning effort │ ok │ {takes}" in shown
    assert "it is scored thinking, as its template can't turn thinking off" in shown
    assert f'Reasoning effort: {takes}. The run uses "high" (deployment.chat_template_kwargs)' in shown
    deployment = config.load(tmp_path / "runs" / "first").deployment
    assert deployment.chat_template_kwargs == {"reasoning_effort": "high"}
    assert deployment.think_close == "</think>" and deployment.thinking_budget == config.THINKING_BUDGET
    assert f"# Reasoning effort: {takes}.\n" in (tmp_path / "runs" / "first" / config.FILE).read_text()
    # A template that takes no reasoning effort gets no chat template settings.
    (tmp_path / "switch").mkdir()
    switch = init(tmp_path, model_folder(tmp_path / "switch", template=THINKING), "switch")
    assert switch.exit_code == 0 and "Reasoning effort" not in switch.output
    assert config.load(tmp_path / "runs" / "switch").deployment.chat_template_kwargs is None


def test_a_local_model_has_no_revision(tmp_path):
    assert init(tmp_path, model_folder(tmp_path)).exit_code == 0
    assert config.load(tmp_path / "runs" / "first").model.revision is None


@pytest.mark.parametrize("name", ["../escape", "a/b", "..", "-x"])
def test_init_rejects_names_that_are_not_one_plain_folder(tmp_path, name):
    result = init(tmp_path, model_folder(tmp_path), name=name)
    assert result.exit_code == 2 and "can't be a run name" in " ".join(result.output.split())
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
    folder = model_folder(tmp_path)
    write_text = Path.write_text

    def broken(path, *args, **kwargs):
        if path.name == "inspect.json":
            raise OSError("disk full")
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", broken)
    result = init(tmp_path, folder)
    assert result.exit_code == 3
    assert "disk full" in result.output
    assert list((tmp_path / "runs").iterdir()) == []


def test_init_checks_the_commit_it_pins(tmp_path, monkeypatch):
    folder = str(model_folder(tmp_path))
    real_config, real_json, real_shapes, real_rows = (
        cli.source.read_config,
        cli.source.read_json,
        cli.source.tensor_shapes,
        cli.source.first_rows,
    )
    real_tokenizer = cli.AutoTokenizer.from_pretrained
    read = []

    # Each read of the Hub model, by the commit it asks for, made from the local copy.
    def read_config(model, revision=None):
        read.append((model, revision))
        return real_config(folder)

    def read_json(model, name, revision=None):
        read.append((model, revision))
        return real_json(folder, name)

    def tensor_shapes(model, revision=None):
        read.append((model, revision))
        return real_shapes(folder)

    def first_rows(model, names, values, revision=None):
        read.append((model, revision))
        return real_rows(folder, names, values)

    def from_pretrained(model, revision=None, **kwargs):
        read.append((model, revision))
        return real_tokenizer(folder, **kwargs)

    monkeypatch.setattr(cli.source, "read_config", read_config)
    monkeypatch.setattr(cli.source, "read_json", read_json)
    monkeypatch.setattr(cli.source, "tensor_shapes", tensor_shapes)
    monkeypatch.setattr(cli.source, "first_rows", first_rows)
    monkeypatch.setattr(cli.AutoTokenizer, "from_pretrained", from_pretrained)
    cli._rows("org/model", "abc")
    asked = [revision for model, revision in read if model == "org/model"]
    assert len(asked) == 5 and set(asked) == {"abc"}
    # init checks the commit the model is at, which it pins.
    checked = []
    monkeypatch.setattr(cli, "_revision", lambda model: "abc")
    monkeypatch.setattr(
        cli, "_checked_rows", lambda console, model, revision=None: checked.append(revision) or []
    )
    init(tmp_path, folder)
    assert checked == ["abc"]


def test_init_pins_the_prompt_datasets(tmp_path):
    assert init(tmp_path, model_folder(tmp_path)).exit_code == 0
    prompts = config.load(tmp_path / "runs" / "first").prompts
    assert prompts.harmful.revision == "mlabonne/harmful_behaviors@commit"
    assert prompts.harmless_test.revision == "mlabonne/harmless_alpaca@commit"
    assert prompts.harmful_held_out.revision == "walledai/StrongREJECT@commit"
    assert prompts.maths.revision == "openai/gsm8k@commit"
    judge = config.load(tmp_path / "runs" / "first").measure
    assert judge.judge_revision == "Qwen/Qwen3Guard-Gen-0.6B@commit"


def test_init_gives_a_new_run_a_second_judge_pinned(tmp_path):
    assert init(tmp_path, model_folder(tmp_path)).exit_code == 0
    judges = config.load(tmp_path / "runs" / "first").measure
    assert judges.second_judge == "Qwen/Qwen3Guard-Gen-8B"
    assert judges.second_judge_revision == "Qwen/Qwen3Guard-Gen-8B@commit"


def test_init_reads_kl_over_200_tokens_for_a_new_run(tmp_path):
    assert init(tmp_path, model_folder(tmp_path)).exit_code == 0
    assert config.load(tmp_path / "runs" / "first").scoring.kl_tokens == 200
    # What a candor.toml upgraded to hold the setting gets, as runs had before init chose more.
    assert config.Scoring().kl_tokens == 20


def upgrade(folder):
    return CliRunner(env={"COLUMNS": "250"}).invoke(app, ["upgrade", str(folder)])


def test_upgrade_adds_missing_settings_pinned_and_keeps_every_value(tmp_path):
    assert init(tmp_path, model_folder(tmp_path)).exit_code == 0
    folder = tmp_path / "runs" / "first"
    text = (folder / config.FILE).read_text()
    held_out = next(line for line in text.splitlines() if line.startswith("harmful_held_out = "))
    text = text.replace(held_out + "\n", "").replace("trials_per_scope = 300", "trials_per_scope = 301")
    (folder / config.FILE).write_text(text)
    refused = CliRunner(env={"COLUMNS": "250"}).invoke(app, ["run", str(folder)])
    assert refused.exit_code == 3 and "candor upgrade adds them" in refused.output
    result = upgrade(folder)
    assert result.exit_code == 0, result.output
    assert 'prompts.harmful_held_out = { dataset = "walledai/StrongREJECT"' in result.output
    assert 'revision = "walledai/StrongREJECT@commit"' in result.output
    settings = config.load(folder)
    assert settings.prompts.harmful_held_out.revision == "walledai/StrongREJECT@commit"
    assert settings.search.trials_per_scope == 301
    upgraded = (folder / config.FILE).read_text()
    assert upgraded.startswith(text.split("\n\n")[0] + "\n# Upgraded ")
    assert "added prompts.harmful_held_out." in upgraded


def test_upgrade_pins_the_judge_it_adds(tmp_path):
    assert init(tmp_path, model_folder(tmp_path)).exit_code == 0
    path = tmp_path / "runs" / "first" / config.FILE
    text = path.read_text()
    start, end = text.index("# How finished edits are measured"), text.index("# The search over")
    path.write_text(text[:start] + text[end:])
    result = upgrade(path.parent)
    assert result.exit_code == 0, result.output
    assert "measure = { answer_tokens = 256" in result.output
    assert config.load(path.parent).measure.judge_revision == "Qwen/Qwen3Guard-Gen-0.6B@commit"


def test_upgrade_adds_the_slots_at_the_numbers_used_before_they_were_settings(tmp_path):
    assert init(tmp_path, model_folder(tmp_path)).exit_code == 0
    path = tmp_path / "runs" / "first" / config.FILE
    text = path.read_text()
    path.write_text(text[: text.index("# How many answers llama-server works on at once.")])
    result = upgrade(path.parent)
    assert result.exit_code == 0, result.output
    assert "slots = { thinking = 8, measure = 8 }" in result.output
    assert config.load(path.parent).slots == config.Slots()
    assert config.load(path.parent).slots.search is None


def test_upgrade_with_nothing_missing_changes_nothing(tmp_path):
    assert init(tmp_path, model_folder(tmp_path)).exit_code == 0
    path = tmp_path / "runs" / "first" / config.FILE
    before = path.read_bytes()
    result = upgrade(path.parent)
    assert result.exit_code == 0 and "already has every setting" in result.output
    assert path.read_bytes() == before


def test_upgrade_waits_for_a_run_using_the_folder(tmp_path):
    assert init(tmp_path, model_folder(tmp_path)).exit_code == 0
    folder = tmp_path / "runs" / "first"
    path = folder / config.FILE
    path.write_text(path.read_text().replace("startup_per_scope = 20\n", ""))
    before = path.read_bytes()
    with Run(folder, config.load_for_upgrade(folder)[0]).locked():
        result = upgrade(folder)
    assert result.exit_code == 1 and "another candor command is using" in result.output
    assert path.read_bytes() == before


def test_init_trusts_a_models_own_code_only_when_transformers_needs_it(tmp_path):
    custom = write_custom_code_model(tmp_path / "custom")
    result = init(tmp_path, custom, name="custom")
    assert result.exit_code == 0, result.output
    assert "transformers can only load it by running configuration_tiny.py, modeling_tiny.py" in result.output
    assert config.load(tmp_path / "runs" / "custom").model.trust_remote_code
    shipped = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": HIDDEN}
    shipped |= {"num_hidden_layers": 2, "auto_map": {"AutoModelForCausalLM": "modeling_own.OwnForCausalLM"}}
    result = init(tmp_path, model_folder(tmp_path, config=shipped))
    assert result.exit_code == 0, result.output
    assert "ships modeling_own.py, but candor uses transformers' own code for it" in result.output
    assert not config.load(tmp_path / "runs" / "first").model.trust_remote_code


def test_brackets_in_paths_and_errors_are_printed_as_they_are(tmp_path):
    (tmp_path / "models [old]").mkdir()
    result = run(model_folder(tmp_path / "models [old]"))
    assert result.exit_code == 0, result.output
    assert "models [old]" in result.output
    missing = tmp_path / "runs [/x]"
    result = candor_run(missing)
    assert result.exit_code == 3
    assert f"{missing} has no candor.toml" in result.output


def candor_run(folder):
    return CliRunner(env={"COLUMNS": "250"}).invoke(app, ["run", str(folder)])


class Ending:
    """A run whose stages end the way a test needs."""

    def __init__(self, outcome, report):
        self.outcome, self.report = outcome, report

    def prepare(self):
        self.report("Removing gguf/model [old].gguf, which this run no longer uses")
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.mark.parametrize(
    ("outcome", "code", "message"),
    [
        ({"edit targets": 2, "adapters": 4}, 0, "2 edit targets, 4 adapters. Next, candor search"),
        (RunError("no room [here]"), 1, "The run stopped: no room [here]"),
        (ToolchainError("no llama-server"), 1, "The run stopped: no llama-server"),
        (ConversionError("the converter failed"), 1, "The run stopped: the converter failed"),
        (
            ValueError("odd [/x] value"),
            3,
            "The run stopped on an unexpected error: ValueError at test_cli.py:",
        ),
    ],
)
def test_run_says_how_it_ended(tmp_path, monkeypatch, outcome, code, message):
    monkeypatch.setattr(cli.Run, "open", lambda folder, report: Ending(outcome, report))
    handlers = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)
    result = candor_run(tmp_path)
    assert result.exit_code == code, result.output
    assert message in result.output and "odd" not in result.output
    if code == 3:
        # The lines of candor's own that led there too.
        assert re.search(r"ValueError at test_cli\.py:\d+, from cli\.py:\d+", " ".join(result.output.split()))
    assert "Removing gguf/model [old].gguf" in result.output
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)) == handlers


def test_run_keeps_the_mac_awake_while_it_works(tmp_path, monkeypatch):
    happened = []

    class Caffeinate:
        def __init__(self, args):
            happened.append(args)

        def terminate(self):
            happened.append("woken")

    class Working:
        def prepare(self):
            happened.append("worked")
            return {}

    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli.subprocess, "Popen", Caffeinate)
    monkeypatch.setattr(cli.Run, "open", lambda folder, report: Working())
    assert candor_run(tmp_path).exit_code == 0
    assert happened == [["/usr/bin/caffeinate", "-is", "-w", str(os.getpid())], "worked", "woken"]


STUCK = """
import subprocess
import sys
import time

from candor import cli


class Stuck:
    def prepare(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            print(child.pid, flush=True)
            time.sleep(60)
        finally:
            child.terminate()
            child.wait()


cli.Run.open = lambda folder, report: Stuck()
cli.app(["run", "anywhere"])
"""


@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGHUP])
def test_kill_or_a_closed_terminal_stops_what_the_run_started(tmp_path, number):
    script = tmp_path / "stuck.py"
    script.write_text(STUCK)
    with subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True) as stuck:
        assert stuck.stdout is not None
        child = int(stuck.stdout.readline())
        try:
            stuck.send_signal(number)
            assert stuck.wait(timeout=30) == 128 + number
            with pytest.raises(ProcessLookupError):
                os.kill(child, 0)
        finally:
            stuck.kill()
            with contextlib.suppress(ProcessLookupError):
                os.kill(child, signal.SIGKILL)


def test_library_progress_bars_stay_out_of_candors_output(tmp_path):
    try:
        CliRunner().invoke(app, ["run", str(tmp_path / "missing")])
        assert not transformers_logging.is_progress_bar_enabled()
    finally:
        transformers_logging.enable_progress_bar()


def test_a_signal_already_ignored_stays_ignored_so_nohup_works():
    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        with cli._stopped_by_signals():
            assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
            assert signal.getsignal(signal.SIGTERM) not in (signal.SIG_DFL, signal.SIG_IGN)
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, previous)
