import json
import os
import shlex
import signal
import socket
import sys
import time
from datetime import date

import pytest
from typer.testing import CliRunner

from candor import config, serve
from candor.cli import app
from candor.run import Run

from .test_engine import stub
from .test_export import converting  # noqa: F401
from .test_run import TEMPLATE
from .test_search import write_journal
from .tiny_gguf import random_weights, write_model


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def stopped_by_ctrl_c(monkeypatch):
    """serve.start as it is, but the wait for Ctrl-C ends at once, as if it were pressed."""
    started = []
    real = serve.start

    def start(args, log):
        process = real(args, log)
        waiting = process.wait

        def wait(timeout=None):
            if timeout is None:
                raise KeyboardInterrupt
            return waiting(timeout)

        process.wait = wait
        started.append(process)
        return process

    monkeypatch.setattr(serve, "start", start)
    return started


def cli_folder(tiny):
    config.write(tiny.folder.parent, "cli", tiny.settings, "made by a test")
    return tiny.folder.parent / "cli"


def test_the_server_runs_at_the_runs_settings(tiny, monkeypatch):
    model = tiny.folder / "model.gguf"
    args = serve.arguments(tiny, model, 8080, 4096)
    assert args[1:3] == ["-m", str(model)] and args[args.index("-c") + 1] == "4096"
    assert args[args.index("--cache-type-k") + 1] == "f16" and "-fa" not in args
    assert "--chat-template-kwargs" not in args  # the only setting is the pinned date
    tiny.settings.deployment.kv_cache = "q8_0"
    monkeypatch.setattr(Run, "template_kwargs", lambda self: {"date_string": "x", "enable_thinking": False})
    args = serve.arguments(tiny, model, 8080, 4096)
    assert args[args.index("-fa") + 1] == "on"
    assert json.loads(args[args.index("--chat-template-kwargs") + 1]) == {"enable_thinking": False}


def test_the_server_samples_and_thinks_as_measured(tiny):
    model = tiny.folder / "model.gguf"
    flags = serve.arguments(tiny, model, 8080, 4096)
    assert flags[flags.index("--temp") + 1] == "0.0" and "--reasoning-budget" not in flags
    tiny.settings.sampling = config.Sampling(temperature=1.0, top_k=20, min_p=0.0, presence_penalty=1.5)
    tiny.settings.deployment = config.Deployment(think_close="</think>", thinking_budget=300)
    flags = serve.arguments(tiny, model, 8080, 4096)
    set_to = {flag: flags[flags.index(flag) + 1] for flag in flags if flag.startswith("--")}
    assert set_to["--temp"] == "1.0" and set_to["--top-k"] == "20" and set_to["--top-p"] == "0.95"
    assert set_to["--min-p"] == "0.0" and set_to["--presence-penalty"] == "1.5"
    assert set_to["--repeat-penalty"] == "1.0" and set_to["--reasoning-budget"] == "300"
    assert set_to["--reasoning-budget-message"] == tiny.settings.deployment.thinking_budget_message


def test_serve_shows_the_mode_then_serves_until_ctrl_c(tiny, stopped_by_ctrl_c):
    tiny.settings.deployment.system_prompt = "Be brief."
    folder = cli_folder(tiny)
    port = free_port()
    result = CliRunner(env={"COLUMNS": "250"}).invoke(app, ["serve", str(folder), "--port", str(port)])
    assert result.exit_code == 0, result.output
    shown = result.output
    assert f"Serving the unedited model: {folder / 'gguf' / 'model-Q4_K_M.gguf'}" in shown
    assert "KV cache f16, 256 tokens of context, chat template settings: none" in shown  # the model's limit
    assert "set the system message to the one measured, between the lines:\n---\nBe brief.\n---" in shown
    assert "keep the reasoning menu on Default" in shown and "sampled as measured, greedily." in shown
    assert "turn off every tool under Tools" in shown and "leave the sampling settings empty" in shown
    command = shlex.split(next(line for line in shown.splitlines() if "--cache-type-v f16" in line))
    assert command[1:3] == ["-m", str(folder / "gguf" / "model-Q4_K_M.gguf")]
    assert f"Today: {date.today().strftime('%d %b %Y')}\nsystem: Be brief.\nuser: Hello\nassistant:" in shown
    assert f"Chat at http://127.0.0.1:{port}" in shown and shown.rstrip().endswith("Stopped.")
    assert stopped_by_ctrl_c[0].poll() is not None
    assert (folder / "private" / "logs" / "serve.log").exists()


def test_serve_runs_a_trials_export_and_only_a_current_one(tiny, converting, stopped_by_ctrl_c):  # noqa: F811
    folder = cli_folder(tiny)
    journal = write_journal(tiny.folder.parent / "heretic.jsonl", 0.25)
    runner = CliRunner(env={"COLUMNS": "250"})
    picked = ["--heretic", str(journal), "--trial", "1", "--port", str(free_port())]
    early = runner.invoke(app, ["serve", str(folder), *picked])
    assert early.exit_code == 1 and "heretic-1 has no export matching this run" in early.output
    assert runner.invoke(app, ["export", str(folder), *picked[:4]]).exit_code == 0
    served = runner.invoke(app, ["serve", str(folder), *picked])
    assert served.exit_code == 0, served.output
    assert (
        f"Serving heretic-1: {folder / 'exports' / 'heretic-1' / 'cli-heretic-1-Q4_K_M.gguf'}"
        in served.output
    )
    assert "leave the system message empty, as measured" in served.output


def test_the_sample_shows_the_servers_own_template_settings(tiny, tmp_path, monkeypatch, stopped_by_ctrl_c):
    # A request with no settings of its own, as the chat page sends on Default, gets the server's.
    write_model(tmp_path / "tiny.gguf", random_weights(), chat_template="mode={{ mode }}\n" + TEMPLATE)
    monkeypatch.setattr(Run, "template_kwargs", lambda self, date=None: {"date_string": "x", "mode": "deep"})
    folder = cli_folder(tiny)
    result = CliRunner(env={"COLUMNS": "250"}).invoke(app, ["serve", str(folder), "--port", str(free_port())])
    assert result.exit_code == 0, result.output
    assert 'chat template settings: {"mode": "deep"}' in result.output
    # TEMPLATE opens with {%- , which trims the newline before it.
    assert f"mode=deep<s>Today: {date.today().strftime('%d %b %Y')}\nuser: Hello" in result.output


ONE_TOOL_SERVER = """
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    result = {"tools": [{"name": "say", "description": "Says it.", "inputSchema": {"type": "object"}}]}
    if request["method"] == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""


def test_serve_offers_the_tools_of_mcp_servers_and_stops_if_none_start(tiny, tmp_path, stopped_by_ctrl_c):
    (tmp_path / "one.py").write_text(ONE_TOOL_SERVER)
    one = {"command": sys.executable, "args": [str(tmp_path / "one.py")]}
    broken = {"command": "false"}
    (tmp_path / "mcp.json").write_text(json.dumps({"mcpServers": {"one": one, "broken": broken}}))
    (tmp_path / "broken.json").write_text(json.dumps({"mcpServers": {"broken": broken}}))
    folder = cli_folder(tiny)
    runner = CliRunner(env={"COLUMNS": "250"})
    serving = ["serve", str(folder), "--port", str(free_port()), "--mcp"]
    result = runner.invoke(app, [*serving, str(tmp_path / "mcp.json")])
    assert result.exit_code == 0, result.output
    assert f"--mcp-servers-config {tmp_path / 'mcp.json'}" in result.output
    assert f"under Tools, keep Server on (one_say, from {tmp_path / 'mcp.json'};" in result.output
    assert "turn off every tool" not in result.output
    failed = runner.invoke(app, [*serving, str(tmp_path / "broken.json")])
    assert failed.exit_code == 1 and f"no MCP server in {tmp_path / 'broken.json'}" in failed.output
    assert stopped_by_ctrl_c[1].poll() is not None
    missing = runner.invoke(app, [*serving, str(tmp_path / "missing.json")])
    assert missing.exit_code == 2 and "is not a file" in missing.output


def test_the_server_renders_with_the_runs_chat_template_settings(tiny, tmp_path, stopped_by_ctrl_c):
    effort = "effort={{ reasoning_effort | default('high') }}\n"
    write_model(tmp_path / "tiny.gguf", random_weights(), chat_template=effort + TEMPLATE)
    settings = {"reasoning_effort": "low", "short": True}
    tiny.settings.deployment.chat_template_kwargs = settings
    args = serve.arguments(tiny, tiny.folder / "model.gguf", 8080, 4096)
    # The pinned date is left out: apps write today's.
    assert json.loads(args[args.index("--chat-template-kwargs") + 1]) == settings
    folder = cli_folder(tiny)
    result = CliRunner(env={"COLUMNS": "250"}).invoke(app, ["serve", str(folder), "--port", str(free_port())])
    assert result.exit_code == 0, result.output
    assert 'chat template settings: {"reasoning_effort": "low", "short": true}' in result.output
    assert f"effort=low<s>Today: {date.today().strftime('%d %b %Y')}\nuser: Hello" in result.output


def test_a_server_that_exits_or_hangs_while_starting_is_stopped_and_reported(tmp_path, monkeypatch):
    stopped = []
    real = serve.stop
    monkeypatch.setattr(serve, "stop", lambda process: (stopped.append(process), real(process)))
    (tmp_path / "exits").mkdir()
    (tmp_path / "hangs").mkdir()
    exits = [str(stub(tmp_path / "exits", "exit 3")), "--port", str(free_port())]
    with pytest.raises(serve.ServeError, match="llama-server stopped with code 3"):
        serve.start(exits, tmp_path / "serve.log")
    hangs = [str(stub(tmp_path / "hangs", "exec sleep 60")), "--port", str(free_port())]
    with pytest.raises(serve.ServeError, match="didn't start within 1 s"):
        serve.start(hangs, tmp_path / "serve.log", timeout=1)
    assert len(stopped) == 2 and all(process.poll() is not None for process in stopped)


def test_sigterm_while_serving_stops_the_server(tiny, monkeypatch):
    started = []
    real = serve.start

    def start(args, log):
        process = real(args, log)
        started.append(process)
        waiting = process.wait

        def wait(timeout=None):
            if timeout is None:
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(10)  # the signal ends this first
            return waiting(timeout)

        process.wait = wait
        return process

    monkeypatch.setattr(serve, "start", start)
    folder = cli_folder(tiny)
    result = CliRunner().invoke(app, ["serve", str(folder), "--port", str(free_port())])
    assert result.exit_code == 128 + signal.SIGTERM
    assert started[0].poll() is not None


def test_a_port_in_use_is_refused_before_starting(tiny):
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        args = serve.arguments(tiny, tiny.folder / "model.gguf", port, 256)
        with pytest.raises(serve.ServeError, match=f"port {port} is already in use"):
            serve.start(args, tiny.folder / "serve.log")
