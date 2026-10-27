import copy
import math
import os
from unittest import mock

import pytest
import torch
from accelerate import dispatch_model
from transformers import LlamaForCausalLM

from candor import residuals, source
from candor.residuals import (
    gpus_used,
    held_on_the_cpu,
    last_token_residuals,
    memory_budget,
    off_the_gpus,
    placement,
    system_memory,
)

from .helpers import HIDDEN, tiny_qwen4exp, write_llama, write_qwen4exp

SHORT = [1, 5, 9, 14]
LONG = [1, 7, 3, 8, 2, 11, 6, 4, 10]


def test_a_watermark_of_0_leaves_what_the_apple_gpu_is_recommended_to_take(monkeypatch):
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 1000)
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
    assert memory_budget("mps") == 700
    # PyTorch reads 0 as no limit, and its out-of-memory message suggests it.
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    assert memory_budget("mps") == 1000


@pytest.mark.parametrize("own", [None, "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"])
def test_pytorch_may_give_a_freed_models_cuda_memory_back_unless_told_otherwise(own):
    with mock.patch.dict(os.environ):
        for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
            os.environ.pop(name, None)
        if own:
            os.environ[own] = "max_split_size_mb:512"
        residuals.protect_memory()
        if own:
            assert os.environ[own] == "max_split_size_mb:512"
            assert set(os.environ) & {"PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"} == {own}
        else:
            assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_a_model_falls_short_of_a_gpu_by_its_16_bit_weights_and_a_tenth(monkeypatch):
    monkeypatch.setattr(source, "tensor_shapes", lambda model, revision=None: {"w": [1000, 100]})
    needed = 2 * 100_000 * 1.1
    monkeypatch.setattr(residuals, "memory_budget", lambda device, spread: math.floor(needed))
    assert residuals.shortfall("org/model", "abc", "mps") == (needed, math.floor(needed))
    monkeypatch.setattr(residuals, "memory_budget", lambda device, spread: math.ceil(needed))
    assert residuals.shortfall("org/model", "abc", "cuda") is None
    # The CPU's limit isn't known, so nothing is read.
    monkeypatch.setattr(residuals, "memory_budget", lambda device, spread: None)
    monkeypatch.setattr(source, "tensor_shapes", lambda *_: pytest.fail("read the headers"))
    assert residuals.shortfall("org/model", "abc", "cpu") is None


@pytest.fixture
def two_gpus(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda gpu=None: ({None: 30, 0: 30, 1: 50}[gpu], 100))


def test_a_model_spread_across_gpus_may_take_what_they_have_free_together(two_gpus):
    assert memory_budget("cuda") == 30
    assert memory_budget("cuda", spread=True) == 80


def test_several_gpus_take_the_model_between_them_and_nothing_else_does(two_gpus, monkeypatch):
    # A tenth less than each has free, for a batch's activations.
    assert placement("cuda") == {"device_map": "auto", "max_memory": {0: 27, 1: 45}}
    assert placement("mps") == {"device_map": "mps"}
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert placement("cuda") == {"device_map": "cuda"}


def test_modules_put_on_the_cpu_or_disk_are_found(tiny_llama):
    assert off_the_gpus(tiny_llama) == []
    spread = copy.deepcopy(tiny_llama)
    spread.hf_device_map = {"model.embed_tokens": 0, "model.layers.0": 1, "model.layers.1": "disk"}
    assert off_the_gpus(spread) == ["model.layers.1"]


def test_the_gpus_holding_a_model_are_found(tiny_llama):
    assert gpus_used(tiny_llama) == []
    spread = copy.deepcopy(tiny_llama)
    spread.hf_device_map = {
        "model.embed_tokens": 3,
        "model.layers.0": 1,
        "model.layers.1": 3,
        "lm_head": "disk",
    }
    assert gpus_used(spread) == [1, 3]


def test_only_what_transformers_keeps_off_the_gpus_is_held_on_the_cpu(tmp_path):
    folder = str(write_qwen4exp(tmp_path / "qwen4exp", tiny_qwen4exp()))
    # The n-gram embedding's 4096 rows of 16, in 16 bits.
    assert held_on_the_cpu(folder, None) == 2 * 4096 * 16
    assert held_on_the_cpu(str(write_llama(tmp_path / "llama")), None) == 0


@pytest.mark.parametrize(
    ("limits", "free"),
    [
        (["max", "9223372036854771712"], 8 * 2**30),
        (["4294967296"], 4 * 2**30),
        (["max", "2147483648"], 2 * 2**30),
        ([], 8 * 2**30),
    ],
    ids=["no limit", "a container's limit", "a container's limit under cgroup v1", "no cgroup"],
)
def test_the_memory_free_here_is_the_kernels_or_the_containers_limit(tmp_path, monkeypatch, limits, free):
    (tmp_path / "meminfo").write_text("MemTotal:       16777216 kB\nMemAvailable:    8388608 kB\n")
    paths = [tmp_path / f"limit-{i}" for i in range(2)]
    for path, limit in zip(paths, limits, strict=False):
        path.write_text(f"{limit}\n")
    monkeypatch.setattr(residuals, "MEMINFO", tmp_path / "meminfo")
    monkeypatch.setattr(residuals, "MEMORY_LIMITS", tuple(paths))
    assert system_memory() == free
    monkeypatch.setattr(residuals, "MEMINFO", tmp_path / "no-meminfo")
    assert system_memory() is None


def test_shape(tiny_llama):
    residuals = last_token_residuals(tiny_llama, [SHORT, LONG], pad_id=0)
    assert residuals.shape == (2, 4, 32)
    assert residuals.dtype == torch.float32


def test_only_the_last_tokens_are_kept_while_the_model_runs(tiny_llama, monkeypatch):
    kept, stack = [], torch.stack

    def stacked(tensors, dim=0):
        kept.extend(t.untyped_storage().nbytes() for t in tensors)
        return stack(tensors, dim=dim)

    monkeypatch.setattr(torch, "stack", stacked)
    last_token_residuals(tiny_llama, [SHORT, LONG], pad_id=0)
    # Each layer's capture holds its 2 prompts' last tokens alone, not the layer's whole output.
    assert kept == [2 * 32 * 4] * 4


def test_left_padding_does_not_change_residuals(tiny_llama):
    alone = last_token_residuals(tiny_llama, [SHORT], pad_id=0)
    padded = last_token_residuals(tiny_llama, [LONG, SHORT], pad_id=0)[1:]
    torch.testing.assert_close(padded, alone, atol=1e-5, rtol=1e-4)


def test_matches_hidden_states_before_the_final_norm(tiny_llama):
    residuals = last_token_residuals(tiny_llama, [LONG], pad_id=0)[0]
    with torch.no_grad():
        hidden = tiny_llama(torch.tensor([LONG]), output_hidden_states=True).hidden_states
    for index in range(len(hidden) - 1):
        torch.testing.assert_close(residuals[index], hidden[index][0, -1])
    # The last layer's output, which hidden_states only gives after the final norm.
    with torch.no_grad():
        torch.testing.assert_close(tiny_llama.model.norm(residuals[-1]), hidden[-1][0, -1])
    assert not torch.allclose(residuals[-1], hidden[-1][0, -1])


def test_prompts_go_where_the_embedding_is(tiny_llama, monkeypatch):
    """On several GPUs the model's first parameter needn't be where its embedding is."""
    elsewhere = torch.nn.Parameter(torch.zeros(1, device="meta"))
    monkeypatch.setattr(LlamaForCausalLM, "parameters", lambda self, recurse=True: iter([elsewhere]))
    assert last_token_residuals(tiny_llama, [SHORT], pad_id=0).shape == (1, 4, 32)


GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None


@pytest.mark.skipif(GPU is None, reason="needs a GPU")
def test_a_model_spread_across_devices_reads_as_on_one(tiny_llama):
    assert GPU
    spread = copy.deepcopy(tiny_llama)
    on_the_gpu = ["model.layers.1", "model.layers.2", "model.norm", "model.rotary_emb", "lm_head"]
    # Run where placed, as on GPUs alone, rather than moved to the GPU as it runs; accelerate compares
    # main_device with "cpu", so it takes the name.
    dispatch_model(
        spread,
        {"model.embed_tokens": "cpu", "model.layers.0": "cpu", **dict.fromkeys(on_the_gpu, GPU)},
        main_device="cpu",  # pyright: ignore[reportArgumentType]
    )
    found = last_token_residuals(spread, [SHORT, LONG], pad_id=0)
    torch.testing.assert_close(found, last_token_residuals(tiny_llama, [SHORT, LONG], pad_id=0))


def test_a_model_with_hyper_connections_reads_the_mean_of_its_streams():
    model = tiny_qwen4exp()
    found = last_token_residuals(model, [LONG, SHORT], pad_id=0)
    assert found.shape == (2, 5, HIDDEN)
    alone = last_token_residuals(model, [SHORT], pad_id=0)[0]
    torch.testing.assert_close(found[1], alone, atol=1e-5, rtol=1e-4)
    with torch.no_grad():
        hidden = model(torch.tensor([LONG]), output_hidden_states=True).hidden_states
    # Each layer's input, its 4 streams side by side, and then the streams mixed for the head.
    means = torch.stack([states[0, -1].unflatten(-1, (4, HIDDEN)).mean(0) for states in hidden[:-1]])
    torch.testing.assert_close(found[0, :-1], means)
