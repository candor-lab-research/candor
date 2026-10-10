"""Plain transformers versions of what the engine measures, used to check the engine."""

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .residuals import left_pad


def pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if not isinstance(pad, int):
        raise ValueError("the tokenizer has neither a pad token nor a single end-of-sequence token")
    return pad


def _batches(prompts: list[list[int]], pad_id: int, batch_size: int, device: torch.device):
    for start in range(0, len(prompts), batch_size):
        ids, mask = left_pad(prompts[start : start + batch_size], pad_id)
        yield ids.to(device), mask.to(device)


@torch.inference_mode()
def first_token_logprobs(
    model: PreTrainedModel, prompts: list[list[int]], pad_id: int, batch_size: int = 32
) -> Tensor:
    """Full-vocabulary log probabilities of the first answer token, shape (prompts, vocab)."""
    device = next(model.parameters()).device
    rows = []
    for ids, mask in _batches(prompts, pad_id, batch_size, device):
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        logits = model(
            input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, logits_to_keep=1
        ).logits[:, -1, :]
        rows.append(F.log_softmax(logits.float(), dim=-1).cpu())
    return torch.cat(rows)


@torch.inference_mode()
def greedy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[list[int]],
    max_tokens: int,
    batch_size: int = 50,
) -> list[str]:
    device = next(model.parameters()).device
    pad_id = pad_token_id(tokenizer)
    answers = []
    for ids, mask in _batches(prompts, pad_id, batch_size, device):
        output = model.generate(  # pyright: ignore[reportCallIssue]
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=pad_id,
        )
        answers += tokenizer.batch_decode(output[:, ids.shape[1] :], skip_special_tokens=True)
    return answers
