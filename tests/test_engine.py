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
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from candor import engine, kl, toolchain
from candor.adapters import write_adapter
from candor.edit import RowNorm, rank_one
from candor.engine import MAX_ARGUMENT, Answer, LlamaServer, ServerError, Thinking
from candor.run import seeds as run_seeds

from .tiny_gguf import (
    END_THINK,
    FIRST_WORD,
    HIDDEN,
    SPECIAL,
    THINK,
    THINKING_TOKENS,
    random_weights,
    write_model,
)


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


def test_a_slots_context_changes_neither_answers_nor_first_tokens(files):
    # The search sizes each slot for the larger of what its requests need, as the size changes nothing.
    # One slot, as several can batch requests differently from run to run.
    folder, scales, _ = files
    sampled = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0}
    found = []
    for context in (64, 1024):
        with LlamaServer(
            LLAMA_SERVER, folder / "base.gguf", folder / "server.log", list(scales), 1, context
        ) as s:
            found.append(
                (
                    s.generate(PROMPTS, 8, scales),
                    s.generate(PROMPTS, 8, scales, None, sampled, run_seeds(0, len(PROMPTS))),
                    s.top_logprobs(PROMPTS, 50, scales),
                )
            )
    assert found[0] == found[1]


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
        monkeypatch.setattr(engine, "port_in_use", lambda _: False)
        sleeper = stub(tmp_path, "exec sleep 60")
        other = LlamaServer(sleeper, folder / "merged.gguf", tmp_path / "server.log", port=running.port)
        stop_later(other)
        with pytest.raises(ServerError, match="is serving"):
            other.start()
        assert other.process is not None and other.process.poll() is not None


def test_a_port_that_never_answers_cannot_hold_up_the_start(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "port_in_use", lambda _: False)
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


def test_adapters_loaded_through_short_links_apply_as_by_their_own_paths(files, references, monkeypatch):
    folder, scales, unused = files
    _, merged = references
    # Short enough that the adapters' own names don't fit, and their numbers do.
    monkeypatch.setattr(engine, "MAX_ARGUMENT", 40)
    with server(folder, "base.gguf", [*scales, unused]) as s:
        assert s.links is not None and set(s.adapter_ids) == {*scales, unused}
        assert_same(s.top_logprobs(PROMPTS, 50, scales), merged)


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
    with subprocess.Popen([binary], stdout=subprocess.PIPE, text=True) as process:
        try:
            assert process.stdout is not None and process.stdout.readline() == "ready\n"
            engine.terminate(process, timeout=0.5)
            assert process.returncode == -signal.SIGKILL
        finally:
            process.kill()


def test_a_starting_server_is_asked_often_whether_its_ready(tmp_path, monkeypatch):
    answers = iter([False, False, True])
    slept = []
    monkeypatch.setattr(engine, "_healthy", lambda port: next(answers))
    monkeypatch.setattr(engine.time, "sleep", slept.append)
    process = engine.launch([str(stub(tmp_path, "exec sleep 60")), "--port", "1"], tmp_path / "server.log")
    process.kill()
    process.wait()
    # A tiny model is ready in a few hundredths of a second, and a test suite starts hundreds.
    assert len(slept) == 2 and max(slept) <= 0.05


def test_a_server_whose_gpu_has_failed_says_so(tmp_path):
    binary = stub(
        tmp_path, "echo 'error while handling argument \"--device\": invalid device: CUDA0'\nexit 1"
    )
    s = LlamaServer(binary, tmp_path / "model.gguf", tmp_path / "server.log")
    with pytest.raises(
        ServerError, match="can't see CUDA0, the GPU it was started on, so the GPU has failed"
    ):
        s.start()


# What llama.cpp logs when a KV cache doesn't fit on the GPU, as K2-Horizon's search did on an A40.
NO_ROOM = (
    "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 61488.00 MiB on device 0: cudaMalloc failed: "
    "out of memory"
)
KV_FAILED = "llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache"


@pytest.mark.parametrize(
    ("logged", "message"),
    [
        (
            [NO_ROOM, "alloc_tensor_range: failed to allocate CUDA0 buffer of size 64474841088", KV_FAILED],
            r"llama-server couldn't allocate its KV cache for 28 slots of 15,616 tokens: it asked for a "
            r"buffer of 61,488 MiB, more GPU memory than it could get; lower slots\.search in candor\.toml$",
        ),
        (
            ["ggml_metal_buffer_init: error: failed to allocate buffer, size =  4096.00 MiB", KV_FAILED],
            "28 slots of 15,616 tokens: it asked for a buffer of 4,096 MiB",
        ),
        # A hybrid model's recurrent state, which doesn't grow with the context.
        (
            [NO_ROOM.replace("61488.00", "1608.00"), KV_FAILED.replace("kv cache", "rs cache")],
            r"couldn't allocate its recurrent state for 28 slots: it asked for a buffer of 1,608 MiB, more "
            r"GPU memory than it could get; lower slots\.search",
        ),
        # A buffer that failed after the cache is not the cache's.
        ([NO_ROOM, KV_FAILED, NO_ROOM.replace("61488.00", "512.00")], "asked for a buffer of 61,488 MiB"),
        # The model's weights didn't fit, not its KV cache.
        ([NO_ROOM, "llama_model_load: error loading model: unable to allocate CUDA0 buffer"], "code 1, see"),
        ([], "exited with code 1, see"),
    ],
)
def test_a_kv_cache_that_doesnt_fit_says_how_much_it_asked_for_and_what_to_lower(tmp_path, logged, message):
    log = tmp_path / "server.log"
    # An earlier start's failure, which this start mustn't take for its own.
    log.write_text(f"{NO_ROOM}\n{KV_FAILED}\n")
    binary = stub(tmp_path, "".join(f"echo '{line}'\n" for line in logged) + "exit 1")
    s = LlamaServer(
        binary,
        tmp_path / "model.gguf",
        log,
        parallel=28,
        context_per_slot=15616,
        slots_setting="slots.search in candor.toml",
    )
    with pytest.raises(ServerError, match=message):
        s.start()


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
    # They fit by their paths from the folder holding them, so no links are made.
    assert workdir == root.resolve() and s.links is None


def test_a_thinking_model_of_36_layers_scored_with_adapters_of_its_own_starts_through_short_links(tmp_path):
    # K2-Horizon's 72 edit targets for each of its 40 direction sets, both places directions are read,
    # and the set candor score builds for a heretic trial's direction between two layers: as paths from
    # the run folder they pass the limit.
    root = tmp_path / "root" / "candor" / "runs" / "k2"
    sets = [
        root / "adapters" / f"{place}{name}"
        for place in ("", "answer-")
        for name in ["per-layer", *(f"layer-{i}" for i in range(14, 33))]
    ]
    sets.append(root / "scores" / f"answer-layer-{20.123456789012346!r}")
    adapters = [
        folder / f"blk.{layer}.{target}.weight.gguf"
        for folder in sets
        for layer in range(36)
        for target in ("attn_output", "ffn_down")
    ]
    listed = ",".join(f'"{path.relative_to(root)}"' for path in adapters)
    assert len(listed.encode()) >= MAX_ARGUMENT
    for path in adapters:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    report = f'pwd > "{tmp_path}/cwd"\nprintf "%s\\n" "$@" > "{tmp_path}/args"\n'
    # Read in one process: a readlink for each of the thousands of links takes seconds.
    listing = "import os; print(*(f + chr(9) + os.readlink(f) for f in os.listdir()), sep=chr(10))"
    report += f'"{sys.executable}" -c "{listing}" > "{tmp_path}/links"\nexit 1'
    binary = stub(tmp_path, report)
    s = LlamaServer(binary, tmp_path / "model.gguf", tmp_path / "server.log", adapters)
    with pytest.raises(ServerError, match="exited with code 1"):
        s.start()
    args = (tmp_path / "args").read_text().splitlines()
    assert max(len(arg.encode()) for arg in args) < MAX_ARGUMENT
    links = dict(line.split("\t") for line in (tmp_path / "links").read_text().splitlines())
    names = next(csv.reader([args[args.index("--lora") + 1]]))
    assert [Path(links[name]) for name in names] == [path.resolve() for path in adapters]
    # The links are gone once the server is stopped, and the adapters are where they were.
    assert not Path((tmp_path / "cwd").read_text().strip()).exists() and s.links is None
    assert all(path.exists() for path in adapters)


def test_a_thinking_model_of_64_layers_with_3_targets_a_layer_starts_through_links_by_number(tmp_path):
    # Per-layer and one-layer direction sets at both places directions are read (68), and 17 more from
    # candor score, each with 3 edit targets in each of 64 layers: 16,320 adapters.
    root = tmp_path / "runs" / "r"
    adapters = [
        root / "adapters" / f"set-{s}" / f"blk.{layer}.{target}.weight.gguf"
        for s in range(85)
        for layer in range(64)
        for target in ("attn_output", "ffn_down", "ssm_out")
    ]
    binary = stub(
        tmp_path, f'printf "%s\\n" "$@" > "{tmp_path}/args"\nls | wc -l > "{tmp_path}/links"\nexit 1'
    )
    s = LlamaServer(binary, tmp_path / "model.gguf", tmp_path / "server.log", adapters)
    with pytest.raises(ServerError, match="exited with code 1"):
        s.start()
    args = (tmp_path / "args").read_text().splitlines()
    listed = args[args.index("--lora") + 1]
    assert len(listed.encode()) < MAX_ARGUMENT
    assert listed.split(",") == [str(i) for i in range(len(adapters))]
    assert int((tmp_path / "links").read_text()) == len(adapters)


def test_more_adapters_than_one_argument_holds_even_by_number_are_refused(tmp_path):
    adapters = [tmp_path / "set" / f"{i:05d}-{'x' * 40}.gguf" for i in range(24000)]
    s = LlamaServer(tmp_path / "llama-server", tmp_path / "model.gguf", tmp_path / "server.log", adapters)
    with pytest.raises(ServerError, match="more than Linux allows in one argument"):
        s.start()
    assert s.process is None and s.links is None


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

    def complete(payload, requests=None):
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

    def complete(payload, requests=None):
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

    def complete(payload, requests=None):
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


def test_a_failed_batch_of_first_tokens_stops_every_request_of_it(monkeypatch):
    s = LlamaServer(LLAMA_SERVER, Path("model.gguf"), Path("server.log"), parallel=2)
    given = []

    def complete(payload, requests=None):
        given.append(requests)
        if len(given) == 2:
            raise ServerError("injected")
        time.sleep(0.05)
        return {}

    monkeypatch.setattr(s, "_complete", complete)
    with pytest.raises(ServerError, match="injected"):
        s._complete_all([{}] * 4)
    # One set of requests for the batch, stopped, so the one still running is closed.
    assert given[0] is not None and all(requests is given[0] for requests in given) and given[0].stopped


# Sampling that keeps every token, so seeds decide the answers.
SAMPLED = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0}
NEWLINE = len(SPECIAL) + 0x0A
THINKING_PROMPTS = [[*p, THINKING_TOKENS.index(THINK), NEWLINE] for p in PROMPTS]


def one_slot(folder: Path, model: str) -> LlamaServer:
    """A server answering one prompt at a time, so no other answer in the batch can tip a near tie."""
    return LlamaServer(LLAMA_SERVER, folder / model, folder / "server.log", parallel=1, context_per_slot=96)


@pytest.mark.parametrize("batch", ["generate", "thoughts"])
def test_a_server_stopped_while_answering_stops_at_once(tmp_path, monkeypatch, batch):
    # Big enough that a long answer takes seconds.
    weights = random_weights(hidden=512, ffn=2048, layers=8)
    write_model(tmp_path / "slow.gguf", weights, context=8192, thinking=True)
    s = LlamaServer(
        LLAMA_SERVER, tmp_path / "slow.gguf", tmp_path / "server.log", parallel=2, context_per_slot=8192
    )
    real_terminate, real_complete = engine.terminate, s._complete
    monkeypatch.setattr(engine, "terminate", lambda process: real_terminate(process, timeout=10))
    sent = []

    def complete(payload, requests=None):
        sent.append(payload)
        if len(sent) == 2:
            # Stopped, as by Ctrl-C, while the other answer is well under way.
            time.sleep(0.5)
            raise KeyboardInterrupt
        return real_complete(payload, requests)

    monkeypatch.setattr(s, "_complete", complete)
    long = Thinking(THINK, END_THINK, END_THINK, THINK, 6000, "")
    with pytest.raises(KeyboardInterrupt), s:
        if batch == "generate":
            s.generate(PROMPTS[:2], 6000)
        else:
            s.thoughts(PROMPTS[:2], engine.GREEDY, [0, 0], long)
    # llama-server waits for the answer it is working on to be closed before it stops: not killed
    # after the wait, as it would be were that answer still open.
    assert s.process is not None and s.process.returncode == 0


def test_sampled_answers_repeat_with_their_seeds(files):
    folder, _, _ = files
    seven, eight = [7] * len(PROMPTS), [8] * len(PROMPTS)
    with one_slot(folder, "base.gguf") as s:
        first = [a.text for a in s.generate(PROMPTS, 8, sampling=SAMPLED, seeds=seven)]
        again = [a.text for a in s.generate(PROMPTS, 8, sampling=SAMPLED, seeds=seven)]
        other = [a.text for a in s.generate(PROMPTS, 8, sampling=SAMPLED, seeds=eight)]
        greedy = [[a.text for a in s.generate(PROMPTS, 8, seeds=seeds)] for seeds in (seven, eight)]
    assert first == again and first != other
    assert greedy[0] == greedy[1]


@pytest.mark.parametrize("close", [END_THINK, f"\n{END_THINK}"], ids=["tag", "newline and tag"])
def test_a_thinking_model_is_made_to_stop_at_its_budget_then_answers(files, close):
    folder, _, _ = files
    write_model(folder / "thinking.gguf", random_weights(), thinking=True)
    thinking = Thinking(THINK, END_THINK, close, f"{THINK}\n", 6, "w1w2")
    with one_slot(folder, "thinking.gguf") as s:
        # Made to stop, the model writes the message and the close, tokenized apart, as the server does.
        forced = s.tokenize("w1w2") + s.tokenize(close)
        # The budget also counts the open's tokens after the start tag, as the server tokenizes it.
        free = 6 - (len(s.tokenize(f"{THINK}\n")) - 1)
        answers = s.generate(THINKING_PROMPTS, 5, sampling=SAMPLED, seeds=list(range(6)), thinking=thinking)
    made_to_stop = [a for a in answers if a.forced]
    # The thinking's length leaves out the end tag, and a model made to stop may finish a character first.
    least = free + len(forced) - 1
    assert len(made_to_stop) >= 4 and all(least <= a.thinking <= least + 8 for a in made_to_stop)
    assert all(a.thinking < free for a in answers if not a.forced)
    assert all(a.tokens <= 5 for a in answers) and any(a.text for a in answers)


def servers_own_budget(
    s: LlamaServer, prompt: list[int], settings: dict, thinking: Thinking
) -> list[int] | None:
    """The thinking llama-server's own reasoning budget gives, up to and with the end tag, as candor
    asked for it before it held the model to its budget itself; None where the server was still waiting
    for a character to finish when the room candor gave it ran out, which candor then refused."""
    end = s.token_of(thinking.end)
    forced = s.tokenize(thinking.message) + s.tokenize(thinking.close)
    payload = {
        "prompt": prompt,
        "n_predict": thinking.budget + len(forced) + 8,
        "reasoning_budget_tokens": thinking.budget,
        "reasoning_budget_start_tag": thinking.start,
        "reasoning_budget_end_tags": list(dict.fromkeys([thinking.close, thinking.end])),
        "reasoning_budget_message": thinking.message,
        "generation_prompt": thinking.opened,
        "preserved_tokens": [thinking.start, thinking.end],
        "stop": [thinking.end],
        "return_tokens": True,
        **settings,
    }
    result = s._complete(payload)
    tokens = result["tokens"]
    if end in tokens:
        return tokens[: tokens.index(end) + 1]
    return None if result["stop_type"] == "limit" else tokens


@pytest.mark.parametrize("budget", [2, 5, 9, 30])
def test_candor_holds_a_model_to_its_budget_exactly_as_llama_server_does(files, budget):
    folder, _, _ = files
    write_model(folder / "thinking.gguf", random_weights(), thinking=True)
    thinking = Thinking(THINK, END_THINK, f"\n{END_THINK}", f"{THINK}\n", budget, "w1w2")
    made_to_stop = closed = finished_a_character = 0
    with one_slot(folder, "thinking.gguf") as s:
        freely = max(budget - (len(s.tokenize(thinking.opened)) - 1), 0)
        forced_tokens = len(s.tokenize(thinking.message) + s.tokenize(thinking.close))
        for seed, prompt in enumerate(THINKING_PROMPTS):
            for sampling in (engine.GREEDY, SAMPLED):
                settings = {**sampling, "seed": seed, "lora": []}
                ours, ended, forced = s._think(prompt, settings, thinking)
                theirs = servers_own_budget(s, prompt, settings, thinking)
                finishing = forced and len(ours) > freely + forced_tokens
                if theirs is None:
                    # The tiny random model can write broken characters without end; candor makes it
                    # stop after a few tokens.
                    assert forced and len(ours) == freely + engine.UNFINISHED_CHARACTER + forced_tokens
                elif sampling == engine.GREEDY or not finishing:
                    assert ours == theirs
                    finished_a_character += finishing
                else:
                    # Tokens drawn to finish a character come from a request of their own, so with
                    # sampling only the thinking before them is drawn the same.
                    assert ours[:freely] == theirs[:freely]
                made_to_stop += forced
                closed += ended and not forced
    if budget <= 9:
        assert made_to_stop
    if budget in (5, 9):
        assert finished_a_character
    if budget == 30:
        assert closed


def thinking_server(
    monkeypatch, thought: dict, answer: dict | None = None, *more: dict, broken: frozenset = frozenset()
) -> tuple[LlamaServer, list]:
    """A server whose model thinks the given tokens, then answers, recording each request; any more
    replies come between, and the tokens in broken end part way through a character."""
    replies = [thought, *more, answer or {}]
    s = LlamaServer(LLAMA_SERVER, Path("model.gguf"), Path("server.log"))
    words = {END_THINK: [99], "": [], "stop": [8, 9], THINK: [98], f"{THINK}\n": [98, 10]}
    monkeypatch.setattr(s, "tokenize", lambda text: words[text])
    monkeypatch.setattr(s, "_whole", lambda token: token not in broken)
    asked = []

    def complete(payload, requests=None):
        asked.append(payload)
        reply = replies[len(asked) - 1]
        return {"truncated": False, "tokens_predicted": len(reply.get("tokens", [0, 0])), **reply}

    monkeypatch.setattr(s, "_complete", complete)
    return s, asked


def test_the_answer_goes_on_from_where_the_thinking_ended(monkeypatch):
    thought = {"tokens": [5, 6, 99, 7], "content": "", "stop_type": "word"}
    s, asked = thinking_server(monkeypatch, thought, {"content": "the answer", "stop_type": "eos"})
    thinking = Thinking(THINK, END_THINK, END_THINK, f"{THINK}\n", 50, "")
    [answer] = s.generate([[1, 2]], 9, sampling=SAMPLED, seeds=[3], thinking=thinking)
    assert answer == Answer("the answer", 2, True, 2, False)
    first, then = asked
    # candor holds the model to its budget, so the server samples on the GPU: the open's newline counts.
    assert first["n_predict"] == 49 and first["stop"] == [END_THINK]
    assert not any(key.startswith("reasoning_budget") for key in first)
    assert then["prompt"] == [1, 2, 5, 6, 99] and then["n_predict"] == 9
    assert first["seed"] == then["seed"] == 3 and first["temperature"] == then["temperature"] == 1.0


def test_thinking_made_to_stop_by_its_budget_is_reported(monkeypatch):
    thought = {"tokens": [5, 6, 7], "content": "", "stop_type": "limit"}
    s, asked = thinking_server(monkeypatch, thought, {"content": "late", "stop_type": "limit"})
    [answer] = s.generate([[1]], 9, thinking=Thinking(THINK, END_THINK, END_THINK, THINK, 3, "stop"))
    assert answer == Answer("late", 2, False, 5, True)
    # Made to write the message and the close, then answers after them.
    assert asked[0]["n_predict"] == 3 and asked[1]["prompt"] == [1, 5, 6, 7, 8, 9, 99]
    # The same tokens closed one token sooner were the model's own close, before its budget ran out.
    early = {"tokens": [5, 8, 9, 99], "content": "", "stop_type": "word"}
    s, _ = thinking_server(monkeypatch, early, {"content": "late", "stop_type": "limit"})
    [answer] = s.generate([[1]], 9, thinking=Thinking(THINK, END_THINK, END_THINK, THINK, 3, "stop"))
    assert answer == Answer("late", 2, False, 3, False)


def test_a_model_that_ends_its_turn_just_as_its_budget_runs_out_isnt_made_to_answer(monkeypatch):
    thinking = Thinking(THINK, END_THINK, END_THINK, THINK, 3, "stop")
    s, asked = thinking_server(monkeypatch, {"tokens": [5, 6, 7], "content": "", "stop_type": "eos"})
    assert s.generate([[1]], 9, thinking=thinking) == [Answer("", 0, False, 3)] and len(asked) == 1
    # Nor when it ends its turn while it finishes a character: the server keeps that token, last, which
    # ends no character.
    thought = {"tokens": [5, 6, 7], "content": "", "stop_type": "limit"}
    ending = {"tokens": [11, 13], "content": "", "stop_type": "eos"}
    s, asked = thinking_server(monkeypatch, thought, None, ending, broken=frozenset({7, 11}))
    assert s.generate([[1]], 9, thinking=thinking) == [Answer("", 0, False, 5)] and len(asked) == 2


def test_a_model_that_never_closes_its_thinking_gives_no_answer(monkeypatch):
    s, asked = thinking_server(monkeypatch, {"tokens": [5, 6], "content": "", "stop_type": "eos"})
    [answer] = s.generate([[1]], 9, thinking=Thinking(THINK, END_THINK, END_THINK, THINK, 50, ""))
    assert answer == Answer("", 0, False, 2) and len(asked) == 1


def test_only_thinking_the_model_ended_by_itself_is_kept_whole(monkeypatch):
    thinking = Thinking(THINK, END_THINK, END_THINK, THINK, 3, "stop")
    for reply, kept in [
        # Closed by the model before its budget: kept up to and with the end tag, and not answered.
        ({"tokens": [5, 8, 9, 99, 4], "content": "", "stop_type": "word"}, [5, 8, 9, 99]),
        # Made to stop by its budget, or never closed: nothing to read KL after.
        ({"tokens": [5, 6, 7], "content": "", "stop_type": "limit"}, None),
        ({"tokens": [5, 6], "content": "", "stop_type": "eos"}, None),
    ]:
        s, asked = thinking_server(monkeypatch, reply)
        assert s.thoughts([[1]], SAMPLED, [3], thinking) == [kept]
        [payload] = asked
        assert payload["seed"] == 3 and payload["temperature"] == 1.0 and payload["stop"] == [END_THINK]


def test_a_model_part_way_through_a_character_at_its_budget_finishes_it_first(monkeypatch):
    thinking = Thinking(THINK, END_THINK, END_THINK, THINK, 3, "stop")
    thought = {"tokens": [5, 6, 7], "content": "", "stop_type": "limit"}
    answer = {"content": "late", "stop_type": "limit"}
    # Finished in one request, cut after the first token that ends a whole character.
    more = {"tokens": [11, 12, 4, 4], "content": "", "stop_type": "limit"}
    s, asked = thinking_server(monkeypatch, thought, answer, more, broken=frozenset({7, 11}))
    assert s.generate([[1]], 9, thinking=thinking) == [Answer("late", 2, False, 7, True)]
    assert [a["n_predict"] for a in asked] == [3, engine.UNFINISHED_CHARACTER, 9]
    assert asked[1]["prompt"] == [1, 5, 6, 7] and asked[2]["prompt"] == [1, 5, 6, 7, 11, 12, 8, 9, 99]
    # A model that closes its thinking while it finishes the character closed it itself.
    closing = {"tokens": [11, 99], "content": "", "stop_type": "word"}
    s, asked = thinking_server(monkeypatch, thought, answer, closing, broken=frozenset({7, 11}))
    assert s.generate([[1]], 9, thinking=thinking) == [Answer("late", 2, False, 4, False)]
    # One that never finishes it is made to stop anyway.
    endless = {"tokens": [11] * engine.UNFINISHED_CHARACTER, "content": "", "stop_type": "limit"}
    s, asked = thinking_server(monkeypatch, thought, answer, endless, broken=frozenset({7, 11}))
    [made] = s.generate([[1]], 9, thinking=thinking)
    assert made.forced and made.thinking == 3 + engine.UNFINISHED_CHARACTER + 2


def test_each_prompt_has_a_seed_of_its_own_that_never_changes():
    # A resumed search records only the base seed, so the seeds made from it must never change.
    assert run_seeds(0, 3) == [2893166218, 4011020074, 2468915676]
    assert run_seeds(1, 2) == [2791857979, 3602223452]


def test_a_request_no_one_wants_any_more_is_dropped_at_once(files, tmp_path):
    # Big enough that a long answer takes seconds.
    write_model(tmp_path / "slow.gguf", random_weights(hidden=512, ffn=2048, layers=8), context=8192)
    s = LlamaServer(
        LLAMA_SERVER, tmp_path / "slow.gguf", tmp_path / "server.log", parallel=1, context_per_slot=8192
    )
    requests = engine.Requests()
    with s, ThreadPoolExecutor(1) as pool:
        running = pool.submit(s._answer_one, PROMPTS[1], 6000, [], engine.GREEDY, 0, None, requests)
        time.sleep(0.5)
        logged = s.log.stat().st_size
        requests.stop()
        with pytest.raises(engine.Stopped):
            running.result(timeout=2)
        # The server dropped it too, once nothing else finished for a second: the next answer waited
        # for that, not for the rest of it.
        s.generate([PROMPTS[0]], 4)
        with s.log.open("rb") as log:
            log.seek(logged)
            assert b"cancel task" in log.read()
        # A request asked for once the answers are stopped is never sent.
        started = time.monotonic()
        with pytest.raises(engine.Stopped):
            s._answer_one(PROMPTS[1], 6000, [], engine.GREEDY, 0, None, requests)
        assert time.monotonic() - started < 1


def test_a_thinking_model_whose_answer_no_one_wants_any_more_doesnt_answer(monkeypatch):
    s = LlamaServer(LLAMA_SERVER, Path("model.gguf"), Path("server.log"), parallel=2)
    monkeypatch.setattr(s, "tokenize", lambda text: {END_THINK: [99], "": []}.get(text, [98, 10]))
    asked = {"thinking": 0, "answering": 0}

    def complete(payload, requests=None):
        if requests is not None and requests.stopped:
            raise engine.Stopped
        if "stop" in payload:
            asked["thinking"] += 1
            time.sleep(0.05)
            return {
                "tokens": [5, 99],
                "content": "",
                "stop_type": "word",
                "truncated": False,
                "tokens_predicted": 2,
            }
        asked["answering"] += 1
        return {"content": "x", "stop_type": "eos", "truncated": False, "tokens_predicted": 1}

    monkeypatch.setattr(s, "_complete", complete)
    thinking = Thinking(THINK, END_THINK, END_THINK, THINK, 50, "")
    answering = s.answers([[i] for i in range(20)], 4, thinking=thinking)
    next(answering)
    answering.close()
    assert asked["thinking"] >= 3 and asked["answering"] < asked["thinking"]
