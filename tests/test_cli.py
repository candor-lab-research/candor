import contextlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
from .test_checks import needs_node
from .test_targets import llama
from .tiny_tokenizer import DATED, PINNABLE, WITH_BOS, tokenizer

HIDDEN = 64


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


def test_init_pins_the_prompt_datasets(tmp_path):
    assert init(tmp_path, model_folder(tmp_path)).exit_code == 0
    prompts = config.load(tmp_path / "runs" / "first").prompts
    assert prompts.harmful.revision == "mlabonne/harmful_behaviors@commit"
    assert prompts.harmless_test.revision == "mlabonne/harmless_alpaca@commit"
    assert prompts.harmful_held_out.revision == "walledai/StrongREJECT@commit"
    assert prompts.maths.revision == "openai/gsm8k@commit"
    judge = config.load(tmp_path / "runs" / "first").measure
    assert judge.judge_revision == "Qwen/Qwen3Guard-Gen-0.6B@commit"


def upgrade(folder):
    return CliRunner(env={"COLUMNS": "250"}).invoke(app, ["upgrade", str(folder)])


def test_upgrade_adds_missing_settings_pinned_and_keeps_every_value(tmp_path):
    assert init(tmp_path, model_folder(tmp_path)).exit_code == 0
    folder = tmp_path / "runs" / "first"
    text = (folder / config.FILE).read_text()
    held_out = next(line for line in text.splitlines() if line.startswith("harmful_held_out = "))
    text = text.replace(held_out + "\n", "").replace("trials = 200", "trials = 201")
    (folder / config.FILE).write_text(text)
    refused = CliRunner(env={"COLUMNS": "250"}).invoke(app, ["run", str(folder)])
    assert refused.exit_code == 3 and "candor upgrade adds them" in refused.output
    result = upgrade(folder)
    assert result.exit_code == 0, result.output
    assert 'prompts.harmful_held_out = { dataset = "walledai/StrongREJECT"' in result.output
    assert 'revision = "walledai/StrongREJECT@commit"' in result.output
    settings = config.load(folder)
    assert settings.prompts.harmful_held_out.revision == "walledai/StrongREJECT@commit"
    assert settings.search.trials == 201
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
    path.write_text(path.read_text().replace("startup_trials = 60\n", ""))
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
            "The run stopped on an unexpected error: ValueError: odd [/x] value",
        ),
    ],
)
def test_run_says_how_it_ended(tmp_path, monkeypatch, outcome, code, message):
    monkeypatch.setattr(cli.Run, "open", lambda folder, report: Ending(outcome, report))
    handlers = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)
    result = candor_run(tmp_path)
    assert result.exit_code == code, result.output
    assert message in result.output
    assert "Removing gguf/model [old].gguf" in result.output
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)) == handlers


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
