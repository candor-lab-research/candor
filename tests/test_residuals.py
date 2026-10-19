import torch

from candor.residuals import last_token_residuals, memory_budget

SHORT = [1, 5, 9, 14]
LONG = [1, 7, 3, 8, 2, 11, 6, 4, 10]


def test_a_watermark_of_0_leaves_what_the_apple_gpu_is_recommended_to_take(monkeypatch):
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 1000)
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
    assert memory_budget("mps") == 700
    # PyTorch reads 0 as no limit, and its out-of-memory message suggests it.
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    assert memory_budget("mps") == 1000


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
