"""llama-server with per-layer adapters, on a tiny random model. Skipped when no llama-server is found."""

import os
import random
import socket
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from candor import kl
from candor.adapters import write_adapter
from candor.edit import RowNorm, rank_one
from candor.engine import LlamaServer, ServerError

from .tiny_gguf import FIRST_WORD, HIDDEN, SPECIAL, random_weights, write_model

LLAMA_SERVER = Path(
    os.environ.get("CANDOR_LLAMA_SERVER", "~/Desktop/tune/llama.cpp/llama-b11524/llama-server")
).expanduser()
pytestmark = pytest.mark.skipif(not LLAMA_SERVER.exists(), reason=f"no llama-server at {LLAMA_SERVER}")

PROMPTS = [[1, FIRST_WORD + i, FIRST_WORD + 2 * i + 1, FIRST_WORD + 3] for i in range(6)]
STRENGTHS = {"blk.0.attn_output.weight": 1.5, "blk.1.ffn_down.weight": 0.8}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def files(tmp_path_factory):
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
        LLAMA_SERVER, folder / model, folder / "server.log", adapters, parallel=2, context_per_slot=64,
        port=free_port(),
    )  # fmt: skip


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
        assert all(adapter["scale"] == 0 for adapter in s.client.get("/lora-adapters").json())
        assert_same(s.top_logprobs(PROMPTS, 50), base)


def test_adapters_at_their_strengths_match_the_merged_model(files, references):
    folder, scales, unused = files
    base, merged = references
    with server(folder, "base.gguf", [*scales, unused]) as s:
        edited = s.top_logprobs(PROMPTS, 50, scales)
    assert_same(edited, merged)
    assert kl.mean_from_top(base, merged) > 1e-3


def test_cached_prompts_are_not_shared_between_adapter_settings(files, references):
    folder, scales, _ = files
    base, merged = references
    with server(folder, "base.gguf", list(scales)) as s:
        s.generate(PROMPTS, 8)
        assert_same(s.top_logprobs(PROMPTS, 50, scales), merged)
        s.generate(PROMPTS, 8, scales)
        assert_same(s.top_logprobs(PROMPTS, 50), base)


def test_a_busy_port_is_refused(files):
    folder, _, _ = files
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        s = LlamaServer(LLAMA_SERVER, folder / "base.gguf", folder / "server.log", port=busy.getsockname()[1])
        with pytest.raises(ServerError, match="already in use"):
            s.start()
        assert s.process is None


def test_a_failed_start_leaves_no_server_running(files, monkeypatch):
    folder, scales, _ = files
    s = server(folder, "base.gguf", list(scales))

    def fail() -> None:
        raise ServerError("injected")

    monkeypatch.setattr(s, "_zero_default_scales", fail)
    with pytest.raises(ServerError, match="injected"):
        s.start()
    assert s.process is not None and s.process.poll() is not None


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
