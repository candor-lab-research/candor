import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from candor import toolchain

from .tiny_run import tiny_run


@pytest.fixture(scope="session")
def tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(config).eval()


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    return tiny_run(tmp_path, monkeypatch)


@pytest.fixture(autouse=True)
def no_tools_from_the_shell(monkeypatch):
    """A CANDOR_MCP set where the tests run would give every serve tools."""
    monkeypatch.delenv("CANDOR_MCP", raising=False)


@pytest.fixture(autouse=True)
def not_on_cuda(monkeypatch):
    """Tests run llama-server as on the Mac, where it names no GPU, wherever they run."""
    monkeypatch.setattr(toolchain, "on_cuda", lambda: False)
