"""Run a model in llama-server as candor measures it, with llama-server's chat page, and show the mode
it runs in before anyone chats with it."""

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import gguf
import httpx

from . import toolchain
from .engine import port_in_use
from .run import Run


class ServeError(RuntimeError):
    pass


def chat_settings(run: Run) -> dict[str, Any]:
    """The chat template settings measurements use, such as a thinking mode, except the pinned date:
    apps write today's, and so does the server."""
    return {key: value for key, value in run.template_kwargs().items() if key != "date_string"}


def context_limit(model: Path) -> int | None:
    """The context the model was trained for, as the GGUF served records it, so serving never needs the
    original weights."""
    fields = gguf.GGUFReader(model).fields
    arch = fields.get("general.architecture")
    limit = fields.get(f"{arch.contents()}.context_length") if arch else None
    return limit.contents() if limit else None


def arguments(run: Run, model: Path, port: int, context: int, mcp: Path | None = None) -> list[str]:
    args = [
        str(toolchain.binaries() / "llama-server"), "-m", str(model), "--host", "127.0.0.1",
        "--port", str(port), "-ngl", "999", "-np", "1", "-c", str(context), *run.cache_flags(),
    ]  # fmt: skip
    if settings := chat_settings(run):
        args += ["--chat-template-kwargs", json.dumps(settings)]
    if mcp is not None:
        args += ["--mcp-servers-config", str(mcp)]
    return args + sampling_flags(run) + thinking_flags(run)


def sampling_flags(run: Run) -> list[str]:
    """The server's sampling defaults, which every chat request the page sends without its own takes:
    the run's, as measurements sample."""
    sampling = run.settings.sampling
    flags = {
        "--temp": sampling.temperature,
        "--top-k": sampling.top_k,
        "--top-p": sampling.top_p,
        "--min-p": sampling.min_p,
        "--presence-penalty": sampling.presence_penalty,
        "--repeat-penalty": sampling.repeat_penalty,
    }
    return [str(item) for flag, value in flags.items() for item in (flag, value)]


def thinking_flags(run: Run) -> list[str]:
    """A thinking model's budget and the message it is made to write when the budget runs out, which
    the chat endpoint applies to every request that brings no budget of its own."""
    deployment = run.settings.deployment
    if deployment.thinking_budget is None:
        return []
    return [
        "--reasoning-budget",
        str(deployment.thinking_budget),
        "--reasoning-budget-message",
        deployment.thinking_budget_message,
    ]


def start(args: list[str], log: Path, timeout: float = 600) -> subprocess.Popen:
    """llama-server, once it answers; its output goes to log."""
    port = int(args[args.index("--port") + 1])
    if port_in_use(port):
        raise ServeError(f"port {port} is already in use; choose another with --port")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as out:
        process = subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + timeout
        while not _healthy(port):
            if process.poll() is not None:
                raise ServeError(f"llama-server stopped with code {process.returncode}; see {log}")
            if time.monotonic() > deadline:
                raise ServeError(f"llama-server didn't start within {timeout:.0f} s; see {log}")
            time.sleep(0.5)
    except BaseException:
        stop(process)
        raise
    return process


def _healthy(port: int) -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200
    except httpx.TransportError:
        return False


def stop(process: subprocess.Popen, timeout: float = 30) -> None:
    process.terminate()
    try:
        process.wait(timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def mcp_tools(port: int) -> list[str]:
    """The tools the server's MCP servers listed when it started, by the names the model sees."""
    reply = httpx.get(f"http://127.0.0.1:{port}/tools", timeout=30)
    reply.raise_for_status()
    return [tool["tool"] for tool in reply.json() if tool["type"] == "mcp"]


def sample(run: Run, port: int) -> str:
    """How the server's chat template renders a made-up first message with the server's own settings,
    as a chat page request that adds none gets it, ready for the answer."""
    body = {"messages": run.chats(["Hello"])[0]}
    reply = httpx.post(f"http://127.0.0.1:{port}/apply-template", json=body, timeout=30)
    reply.raise_for_status()
    return reply.json()["prompt"]
