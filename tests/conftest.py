import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from candor import convert, export, toolchain
from candor.convert import check_targets

from .helpers import gguf_from_folder, write_llama
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


@pytest.fixture
def model(tmp_path):
    return write_llama(tmp_path / "model")


@pytest.fixture
def converting(tiny, monkeypatch):
    """The run's GGUF made from the tiny model's own weights, as the converter would, and every export
    written, by where it goes."""
    monkeypatch.setattr(
        convert,
        "to_gguf",
        lambda folder, destination, log: gguf_from_folder(
            folder, destination, log, thinks=bool(tiny.settings.deployment.think_close)
        ),
    )
    monkeypatch.setattr(convert, "check_targets", check_targets)
    shipped = []
    real = export._ship
    monkeypatch.setattr(
        export, "_ship", lambda run, path, *args: shipped.append(path) or real(run, path, *args)
    )
    return shipped
