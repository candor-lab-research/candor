import torch

from candor.reference import first_token_logprobs
from candor.residuals import last_token_residuals

SHORT = [1, 5, 9, 14]
LONG = [1, 7, 3, 8, 2, 11, 6, 4, 10]


def test_shape(tiny_llama):
    residuals = last_token_residuals(tiny_llama, [SHORT, LONG], pad_id=0)
    assert residuals.shape == (2, 4, 32)
    assert residuals.dtype == torch.float32


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


def test_left_padding_does_not_change_logprobs(tiny_llama):
    alone = first_token_logprobs(tiny_llama, [SHORT], pad_id=0)
    padded = first_token_logprobs(tiny_llama, [LONG, SHORT], pad_id=0)[1:]
    torch.testing.assert_close(padded, alone, atol=1e-5, rtol=1e-4)
