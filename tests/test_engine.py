"""llama-server with per-layer adapters, on a tiny random model.

The tests that need a real llama-server use the one candor would: CANDOR_LLAMA_BIN's, or the one
in candor's own cache. They never download it, and are skipped when neither is there.
"""

import csv
import os
import random
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from candor import engine, kl, toolchain
from candor.adapters import write_adapter
from candor.edit import RowNorm, rank_one
from candor.engine import MAX_ARGUMENT, LlamaServer, ServerError

from .tiny_gguf import FIRST_WORD, HIDDEN, SPECIAL, random_weights, write_model


def installed_server() -> Path:
    if folder := os.environ.get("CANDOR_LLAMA_BIN"):
        return Path(folder).expanduser() / "llama-server"
    return toolchain.cache() / toolchain.this_platform() / "bin" / "llama-server"


LLAMA_SERVER = installed_server()

PROMPTS = [[1, FIRST_WORD + i, FIRST_WORD + 2 * i + 1, FIRST_WORD + 3] for i in range(6)]
STRENGTHS = {"blk.0.attn_output.weight": 1.5, "blk.1.ffn_down.weight": 0.8}


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    if not LLAMA_SERVER.exists():
        pytest.skip(f"no llama-server at {LLAMA_SERVER}; run candor to install it or set CANDOR_LLAMA_BIN")
    toolchain._check(LLAMA_SERVER.parent)
    folder = tmp_path_factory.mktemp("engine")
    weights = random_weights()
    write_model(folder / "base.gguf", weights)
    torch.manual_seed(5)
    direction = F.normalize(torch.randn(HIDDEN), dim=0)
    merged = dict(weights)
    scales = {}
    for name, strength in STRENGTHS.items():
        b, a = rank_one(torch.from_numpy(weights[name]), direction, RowNorm.PRE)
        path = folder / f"{name}.gguf"
        write_adapter(path, "llama", {name: (b, a)})
        scales[path] = strength
        merged[name] = weights[name] + strength * (b @ a).numpy()
    write_model(folder / "merged.gguf", merged)
    unused = folder / "unused.gguf"
    name = "blk.1.attn_output.weight"
    write_adapter(
        unused, "llama", {name: rank_one(torch.from_numpy(weights[name]), -direction, RowNorm.NONE)}
    )
    return folder, scales, unused


def server(folder: Path, model: str, adapters=()) -> LlamaServer:
    return LlamaServer(
        LLAMA_SERVER, folder / model, folder / "server.log", adapters, parallel=2, context_per_slot=64
    )


def stub(folder: Path, script: str) -> Path:
    """A shell script standing in for llama-server."""
    path = folder / "llama-server"
    path.write_text(f"#!/bin/sh\n{script}\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def stop_later():
    """Stops the servers given to it when the test ends, even one that started when it shouldn't."""
    servers: list[LlamaServer] = []
    yield servers.append
    for s in servers:
        s.stop()


def assert_same(a: list[dict[int, float]], b: list[dict[int, float]]) -> None:
    assert kl.mean_from_top(a, b) < 1e-4
    assert [max(x, key=x.__getitem__) for x in a] == [max(y, key=y.__getitem__) for y in b]


@pytest.fixture(scope="module")
def references(files):
    folder, _, _ = files
    with server(folder, "base.gguf") as base, server(folder, "merged.gguf") as merged:
        return base.top_logprobs(PROMPTS, 50), merged.top_logprobs(PROMPTS, 50)


def test_adapters_are_off_unless_a_request_names_them(files, references):
    folder, scales, unused = files
    base, _ = references
    with server(folder, "base.gguf", [*scales, unused]) as s:
        assert set(s.adapter_ids) == {*scales, unused}
        assert all(adapter["scale"] == 0 for adapter in s.client.get("/lora-adapters").json())
        assert_same(s.top_logprobs(PROMPTS, 50), base)


def test_adapters_at_their_strengths_match_the_merged_model(files, references):
    folder, scales, unused = files
    base, merged = references
    with server(folder, "base.gguf", [*scales, unused]) as s:
        edited = s.top_logprobs(PROMPTS, 50, scales)
    assert_same(edited, merged)
    assert kl.mean_from_top(base, merged) > 1e-3


def test_adapters_load_from_folders_with_commas_and_quotes(files, references, tmp_path):
    folder, scales, _ = files
    _, merged = references
    moved = {}
    for i, (path, strength) in enumerate(scales.items()):
        target = tmp_path / "runs, old" / f'set,"{i}"' / path.name
        target.parent.mkdir(parents=True)
        shutil.copy(path, target)
        moved[target] = strength
    with server(folder, "base.gguf", list(moved)) as s:
        assert set(s.adapter_ids) == set(moved)
        assert_same(s.top_logprobs(PROMPTS, 50, moved), merged)


def test_adapters_left_on_by_default_are_refused(files, monkeypatch, stop_later):
    folder, scales, _ = files
    s = server(folder, "base.gguf", list(scales))
    stop_later(s)
    call = s._call

    def ignoring_new_scales(method, endpoint, *args, **kwargs):
        if method == "POST" and endpoint == "/lora-adapters":
            return []
        return call(method, endpoint, *args, **kwargs)

    monkeypatch.setattr(s, "_call", ignoring_new_scales)
    with pytest.raises(ServerError, match="default scale to 0"):
        s.start()
    assert s.process is not None and s.process.poll() is not None


def test_cached_prompts_are_not_shared_between_adapter_settings(files, references):
    folder, scales, _ = files
    base, merged = references
    with server(folder, "base.gguf", list(scales)) as s:
        s.generate(PROMPTS, 8)
        assert_same(s.top_logprobs(PROMPTS, 50, scales), merged)
        s.generate(PROMPTS, 8, scales)
        assert_same(s.top_logprobs(PROMPTS, 50), base)


def test_the_prompt_cache_and_context_checkpoints_are_off(files, tmp_path):
    folder, _, _ = files
    log = tmp_path / "server.log"
    with LlamaServer(LLAMA_SERVER, folder / "base.gguf", log, parallel=1, extra=["-lv", "4"]):
        pass
    text = log.read_text()
    assert "prompt cache is disabled" in text and "context checkpoints disabled" in text


def test_an_answer_cut_short_by_the_context_is_refused(files):
    folder, _, _ = files
    prompt = PROMPTS[1]  # the tiny model never ends this one, so it runs until the slot is full
    with server(folder, "base.gguf") as s:
        room = s.client.get("/props").json()["default_generation_settings"]["n_ctx"] - len(prompt)
        [answer] = s.generate([prompt], room)
        assert not answer.finished and answer.tokens == room
        with pytest.raises(ServerError, match=f"a slot of at least {len(prompt) + room + 1} tokens"):
            s.generate([prompt], room + 1)


def test_a_busy_port_is_refused(files):
    folder, _, _ = files
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        s = LlamaServer(LLAMA_SERVER, folder / "base.gguf", folder / "server.log", port=busy.getsockname()[1])
        with pytest.raises(ServerError, match="already in use"):
            s.start()
        assert s.process is None


def test_a_server_serving_another_model_is_refused(files, tmp_path, monkeypatch, stop_later):
    folder, _, _ = files
    with server(folder, "base.gguf") as running:
        monkeypatch.setattr(engine, "_port_in_use", lambda _: False)
        sleeper = stub(tmp_path, "exec sleep 60")
        other = LlamaServer(sleeper, folder / "merged.gguf", tmp_path / "server.log", port=running.port)
        stop_later(other)
        with pytest.raises(ServerError, match="is serving"):
            other.start()
        assert other.process is not None and other.process.poll() is not None


def test_a_port_that_never_answers_cannot_hold_up_the_start(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "_port_in_use", lambda _: False)
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen()
        sleeper, port = stub(tmp_path, "exec sleep 60"), silent.getsockname()[1]
        s = LlamaServer(sleeper, tmp_path / "model.gguf", tmp_path / "server.log", port=port)
        with pytest.raises(ServerError, match="wasn't ready after 1 s"):
            s.start(timeout=1)


def test_a_failed_start_leaves_no_server_running(files, monkeypatch, stop_later):
    folder, scales, _ = files
    s = server(folder, "base.gguf", list(scales))
    stop_later(s)

    def fail() -> None:
        raise ServerError("injected")

    monkeypatch.setattr(s, "_zero_default_scales", fail)
    with pytest.raises(ServerError, match="injected"):
        s.start()
    assert s.process is not None and s.process.poll() is not None


def test_a_server_that_dies_is_reported_with_its_log(files):
    folder, _, _ = files
    with server(folder, "base.gguf") as s:
        assert s.process is not None
        s.process.kill()
        s.process.wait()
        with pytest.raises(ServerError, match=r"didn't answer /completion .* code -9, see .*server\.log"):
            s.generate(PROMPTS, 8)


def test_a_server_that_ignores_terminate_is_killed(tmp_path):
    binary = stub(tmp_path, "trap '' TERM\necho ready\nexec sleep 60")
    s = LlamaServer(binary, tmp_path / "model.gguf", tmp_path / "server.log")
    with subprocess.Popen([binary], stdout=subprocess.PIPE, text=True) as process:
        s.process = process
        try:
            assert process.stdout is not None and process.stdout.readline() == "ready\n"
            s.stop(timeout=0.5)
            assert process.returncode == -signal.SIGKILL
        finally:
            process.kill()


def test_thousands_of_adapters_fit_in_one_argument(tmp_path):
    # K2-Horizon's 72 edit targets for every direction set, in a long run folder on a pod.
    root = tmp_path / "workspace" / "candor" / "runs" / "k2-horizon-high-effort" / "adapters"
    root.mkdir(parents=True)
    adapters = [
        root / f"layer-{s}" / f"blk.{layer}.{target}.weight.gguf"
        for s in range(37)
        for layer in range(36)
        for target in ("attn_output", "ffn_down")
    ]
    binary = stub(tmp_path, f'pwd > "{tmp_path}/cwd"\nprintf "%s\\n" "$@" > "{tmp_path}/args"\nexit 1')
    s = LlamaServer(binary, tmp_path / "model.gguf", tmp_path / "server.log", adapters)
    with pytest.raises(ServerError, match="exited with code 1"):
        s.start()
    args = (tmp_path / "args").read_text().splitlines()
    assert max(len(arg.encode()) for arg in args) < MAX_ARGUMENT
    workdir = Path((tmp_path / "cwd").read_text().strip())
    listed = next(csv.reader([args[args.index("--lora") + 1]]))
    assert [(workdir / path).resolve() for path in listed] == adapters


def test_more_adapters_than_one_argument_holds_are_refused(tmp_path):
    adapters = [tmp_path / "set" / f"{i:05d}-{'x' * 40}.gguf" for i in range(3000)]
    s = LlamaServer(tmp_path / "llama-server", tmp_path / "model.gguf", tmp_path / "server.log", adapters)
    with pytest.raises(ServerError, match="more than Linux allows in one argument"):
        s.start()
    assert s.process is None


def test_every_prompt_gets_first_token_probabilities(files):
    folder, _, _ = files
    rng = random.Random(0)
    prompts = [[1, *(rng.randrange(FIRST_WORD - 256, FIRST_WORD + 29) for _ in range(5))] for _ in range(60)]
    with server(folder, "base.gguf") as s:
        assert all(s.top_logprobs(prompts, 20))


def test_a_banned_token_does_not_change_the_reported_probabilities(files):
    folder, _, _ = files
    with server(folder, "base.gguf") as s:
        for prompt in PROMPTS:
            payload = {"prompt": prompt, "n_predict": 1, "n_probs": 20, "temperature": 0.0}
            if probabilities := s._complete(payload).get("completion_probabilities"):
                plain = probabilities[0]
                break
        else:
            pytest.fail("every prompt's first token was an incomplete UTF-8 sequence")
        partial_bytes = range(len(SPECIAL) + 0x80, len(SPECIAL) + 0x100)
        bias = [[token, False] for token in [plain["id"], *partial_bytes]]
        banned = s._complete({**payload, "logit_bias": bias})["completion_probabilities"][0]
    assert banned["id"] != plain["id"]
    assert banned["top_logprobs"] == plain["top_logprobs"]


def test_chat_ids_are_what_a_real_chat_request_uses(files):
    folder, _, _ = files
    chats = [
        [{"role": "user", "content": "hi"}],
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello there"}],
    ]
    with server(folder, "base.gguf") as s:
        ids = s.chat_ids(chats, {})
        assert [len(i) for i in ids] == [s.chat_prompt_length(chat, {}) for chat in chats]


def test_each_answer_is_reported_as_it_finishes(files):
    folder, _, _ = files
    finished = []
    with server(folder, "base.gguf") as s:
        answers = s.generate(PROMPTS, 4, done=lambda: finished.append(1))
        s.top_logprobs(PROMPTS, 5, done=lambda: finished.append(2))
    assert finished == [1] * len(PROMPTS) + [2] * len(PROMPTS)
    assert all(a.tokens <= 4 for a in answers)
    assert all(a.finished == (a.tokens < 4) for a in answers)


def answered(payload):
    return {
        "content": str(payload["prompt"][0]),
        "tokens_predicted": 1,
        "truncated": False,
        "stop_type": "eos",
    }


def test_answers_come_in_order_and_closing_early_cancels_those_not_started(monkeypatch):
    s = LlamaServer(LLAMA_SERVER, Path("model.gguf"), Path("server.log"), parallel=2)
    sent, done = [], []

    def complete(payload):
        sent.append(payload["prompt"])
        time.sleep(0.05 if payload["prompt"][0] % 2 == 0 else 0.02)  # odd ones finish first
        done.append(payload["prompt"])
        return answered(payload)

    monkeypatch.setattr(s, "_complete", complete)
    answering = s.answers([[i] for i in range(40)], 4)
    assert [next(answering).text for _ in range(3)] == ["0", "1", "2"]
    answering.close()
    started = len(sent)
    time.sleep(0.1)
    assert len(sent) == started < 20
    assert sorted(done) == sorted(sent)


def test_a_failed_answer_stops_those_still_queued(monkeypatch):
    s = LlamaServer(LLAMA_SERVER, Path("model.gguf"), Path("server.log"), parallel=2)
    sent = []

    def complete(payload):
        sent.append(payload)
        time.sleep(0.02)
        if len(sent) == 3:
            raise ServerError("injected")
        return answered(payload)

    monkeypatch.setattr(s, "_complete", complete)
    with pytest.raises(ServerError, match="injected"):
        list(s.answers([[i] for i in range(40)], 4))
    time.sleep(0.1)
    assert len(sent) < 20


def test_a_failure_stops_the_requests_still_queued(monkeypatch):
    s = LlamaServer(LLAMA_SERVER, Path("model.gguf"), Path("server.log"), parallel=2)
    sent = []

    def complete(payload):
        sent.append(payload)
        time.sleep(0.02)
        return {}

    def done():
        raise KeyboardInterrupt

    monkeypatch.setattr(s, "_complete", complete)
    with pytest.raises(KeyboardInterrupt):
        s._complete_all([{}] * 40, done)
    time.sleep(0.1)
    assert len(sent) <= 4
