"""Read the residual stream with PyTorch, on the fastest device here."""

import gc
import math
import os
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, PreTrainedModel

from . import source


def device_and_dtype() -> tuple[str, torch.dtype]:
    """The fastest device here, with the dtype that loads reliably on it (float16 on Apple GPUs)."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def protect_memory() -> None:
    """Defaults that stop PyTorch on an Apple GPU taking more memory than the machine has, which
    freezes it, and a loader deadlock on Metal; and expandable segments on CUDA, without which a freed
    model's memory stays reserved (all 50 GiB of a 27B's, measured) and a llama-server started next
    can't have it. Torch reads these once, the first time it uses the GPU, so candor sets them as it
    starts."""
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    # Torch 2.11 also reads the newer name, and only the first of the two it finds.
    if not {"PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"} & set(os.environ):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def memory_budget(device: str) -> int | None:
    """The bytes a model may take on the device: on an Apple GPU, the share protect_memory allows, or
    all the GPU is recommended to take for a ratio of 0, which PyTorch reads as no limit; on CUDA,
    what is free now. None on the CPU."""
    if device == "mps":
        ratio = float(os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]) or 1.0
        return int(torch.mps.recommended_max_memory() * ratio)
    if device == "cuda":
        return torch.cuda.mem_get_info()[0]
    return None


def shortfall(model: str, revision: str | None, device: str) -> tuple[float, int] | None:
    """The bytes a model needs on the device, its weights in 16 bits and a tenth more for a batch's
    activations, and the bytes the device allows, when that is too few; None when it fits, or on the
    CPU, whose limit isn't known. Only the model's file headers are read, and only for a GPU."""
    protect_memory()
    budget = memory_budget(device)
    if budget is None:
        return None
    weights = sum(math.prod(shape) for shape in source.tensor_shapes(model, revision).values())
    needed = 2 * weights * 1.1
    return (needed, budget) if needed > budget else None


def free_memory(device: str) -> None:
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def pad_token_id_for(model_folder: Path, trust_remote_code: bool) -> int:
    tokenizer = AutoTokenizer.from_pretrained(model_folder, trust_remote_code=trust_remote_code)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if not isinstance(pad, int):
        raise ValueError("the tokenizer has neither a pad token nor a single end-of-sequence token")
    return pad


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
    They are read with forward hooks rather than output_hidden_states, which some remote code breaks
    (K2-Horizon's returns none).
    """
    decoder = model.get_decoder()
    modules = [decoder.get_input_embeddings(), *cast(nn.ModuleList, decoder.layers)]
    captured: dict[int, Tensor] = {}

    def capture(index: int):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # A copy, so the layer's whole output isn't kept alive by a view of its last token.
            captured[index] = hidden[:, -1, :].to(torch.float32, copy=True)

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
