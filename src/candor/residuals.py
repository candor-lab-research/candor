"""Read the residual stream with PyTorch."""

from typing import cast

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel


def left_pad(batch: list[list[int]], pad_id: int) -> tuple[Tensor, Tensor]:
    width = max(map(len, batch))
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, tokens in enumerate(batch):
        ids[row, width - len(tokens) :] = torch.tensor(tokens)
        mask[row, width - len(tokens) :] = 1
    return ids, mask


@torch.inference_mode()
def last_token_residuals(
    model: PreTrainedModel, prompts: list[list[int]], pad_id: int, batch_size: int = 32
) -> Tensor:
    """Residuals at each prompt's last token, shape (prompts, layers + 1, hidden), float32 on the CPU.

    Index 0 is the embedding output and index i + 1 the output of layer i, before any final norm.
    They are read with forward hooks rather than output_hidden_states, which some remote code breaks.
    """
    decoder = model.get_decoder()
    modules = [decoder.get_input_embeddings(), *cast(nn.ModuleList, decoder.layers)]
    captured: dict[int, Tensor] = {}

    def capture(index: int):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[index] = hidden[:, -1, :].float()

        return hook

    handles = [module.register_forward_hook(capture(i)) for i, module in enumerate(modules)]
    device = next(model.parameters()).device
    chunks = []
    try:
        for start in range(0, len(prompts), batch_size):
            ids, mask = left_pad(prompts[start : start + batch_size], pad_id)
            positions = (mask.cumsum(-1) - 1).clamp(min=0)
            decoder(
                input_ids=ids.to(device),
                attention_mask=mask.to(device),
                position_ids=positions.to(device),
                use_cache=False,
            )
            chunks.append(torch.stack([captured[i] for i in range(len(modules))], dim=1).cpu())
    finally:
        for handle in handles:
            handle.remove()
    return torch.cat(chunks)
