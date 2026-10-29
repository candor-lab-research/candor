"""Read the residual stream with PyTorch, on the fastest device here."""

import gc
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, cast

import torch
import transformers
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, masking_utils
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from . import source

MEMINFO = Path("/proc/meminfo")
# A container's memory limit under cgroup v2 and v1: "max", or a number v1 makes huge, when it has none.
MEMORY_LIMITS = (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))


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


def memory_budget(device: str, spread: bool = False) -> int | None:
    """The bytes a model may take on the device: on an Apple GPU, the share protect_memory allows, or
    all the GPU is recommended to take for a ratio of 0, which PyTorch reads as no limit; on CUDA,
    what is free now, on every GPU together for a model spread across them. None on the CPU."""
    if device == "mps":
        ratio = float(os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]) or 1.0
        return int(torch.mps.recommended_max_memory() * ratio)
    if device == "cuda" and spread:
        return sum(free_on_each_gpu().values())
    if device == "cuda":
        return torch.cuda.mem_get_info()[0]
    return None


def free_on_each_gpu() -> dict[int, int]:
    """The bytes free now on each CUDA GPU, by its index."""
    return {gpu: torch.cuda.mem_get_info(gpu)[0] for gpu in range(torch.cuda.device_count())}


def placement(device: str) -> dict[str, Any]:
    """Where transformers puts a model to read its residuals: all on the device, or with several CUDA
    GPUs spread across them, each filled to a tenth less than it has free, the room shortfall leaves for
    a batch's activations. The CPU is given nothing, so a layer that still doesn't fit goes to the disk,
    which off_the_gpus finds, but weights the model's class won't place stay on the CPU when no one GPU
    can hold them (see held_on_the_cpu)."""
    free = free_on_each_gpu() if device == "cuda" else {}
    if len(free) < 2:
        return {"device_map": device}
    return {"device_map": "auto", "max_memory": {gpu: int(room / 1.1) for gpu, room in free.items()}}


def load_model(model_folder: Path, trust_remote_code: bool, **settings: Any) -> PreTrainedModel:
    """The model as transformers loads it with these settings, its remote code made to run as transformers
    4 ran it where it was written for that, as Spark-X's was: the weights it lists as tied given as the
    mapping transformers 5 takes, from its output embedding to its input one, both named by the class
    built on the meta device, which holds no weights; and the mask makers it calls taking the arguments
    transformers 4 gave them."""
    reference = source.read_config(str(model_folder)).get("auto_map", {}).get("AutoModelForCausalLM")
    if not trust_remote_code or reference is None:
        return AutoModelForCausalLM.from_pretrained(
            model_folder, trust_remote_code=trust_remote_code, **settings
        )
    model_config = AutoConfig.from_pretrained(model_folder, trust_remote_code=True)
    loaded = get_class_from_dynamic_module(reference, model_folder)
    if isinstance(loaded._tied_weights_keys, list) and getattr(model_config, "tie_word_embeddings", False):
        with torch.device("meta"):
            shell = type(loaded.__name__, (loaded,), {"_tied_weights_keys": None})(model_config)
        names = {weight: name for name, weight in shell.named_parameters(remove_duplicate=False)}
        output, embedding = shell.get_output_embeddings(), shell.get_input_embeddings()
        loaded._tied_weights_keys = {names[output.weight]: names[embedding.weight]} if output else None
    module = sys.modules[loaded.__module__]
    for maker in ("create_causal_mask", "create_sliding_window_causal_mask"):
        if getattr(module, maker, None) is getattr(masking_utils, maker):
            setattr(module, maker, _as_transformers_4(getattr(masking_utils, maker)))
    return loaded.from_pretrained(model_folder, config=model_config, trust_remote_code=True, **settings)


def _as_transformers_4(make: Any) -> Any:
    """A mask maker that also takes what transformers 4 passed it: input_embeds, since renamed, and
    cache_position, which transformers 5 reads from the positions."""

    def made(
        *args: Any, input_embeds: Tensor | None = None, cache_position: Tensor | None = None, **kwargs: Any
    ):
        if input_embeds is not None:
            kwargs["inputs_embeds"] = input_embeds
        return make(*args, **kwargs)

    return made


def off_the_gpus(model: PreTrainedModel) -> list[str]:
    """The modules of a model spread across GPUs that were put on the CPU or disk instead."""
    placed = getattr(model, "hf_device_map", None) or {}
    return [name for name, where in placed.items() if where in ("cpu", "disk")]


def gpus_used(model: PreTrainedModel) -> list[int]:
    """The CUDA GPUs that hold a model's weights, by index."""
    placed = getattr(model, "hf_device_map", None)
    if placed is None:
        return sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
    return sorted({where for where in placed.values() if isinstance(where, int)})


def held_on_the_cpu(model: str, revision: str | None) -> int:
    """The bytes, in 16 bits, of the weights transformers keeps on the CPU when no one GPU can hold them,
    as the model's class names them in _no_placement_params: Qwen4-Exp's n-gram embedding table, which
    it also joins there from the checkpoint's shards whatever the placement. Read from the file headers,
    and only for a model with such weights."""
    name = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(source.read_config(model, revision).get("model_type"))
    kept = getattr(getattr(transformers, name, None), "_no_placement_params", None) if name else None
    modules = [f".{param.removesuffix('.weight')}." for param in kept or []]
    if not modules:
        return 0
    shapes = source.tensor_shapes(model, revision)
    return sum(2 * math.prod(shape) for tensor, shape in shapes.items() if any(m in tensor for m in modules))


def system_memory() -> int | None:
    """The bytes of memory a program here may take, on Linux: what the kernel counts available, or the
    container's limit when that is less, as inside a container /proc/meminfo describes the whole host.
    Only the limit at the root of the container's cgroup view is read, so one set on a parent cgroup isn't
    seen. None elsewhere, or where the kernel doesn't say what is available."""
    try:
        meminfo = MEMINFO.read_text()
    except OSError:
        return None
    available = re.search(r"^MemAvailable:\s+(\d+) kB$", meminfo, re.MULTILINE)
    if available is None:
        return None
    limits = [int(available[1]) * 1024]
    for path in MEMORY_LIMITS:
        try:
            limit = path.read_text().strip()
        except OSError:
            continue
        if limit.isdigit():
            limits.append(int(limit))
    return min(limits)


def memory_shortfall(model: str, revision: str | None) -> tuple[float, int] | None:
    """The bytes of memory loading the model takes, twice what transformers holds on the CPU, as it
    joins those weights from their shards there, and a tenth more, as shortfall leaves, for the copies
    the other weights pass through and the program itself; and the bytes this machine allows, when too
    few. None when they fit, or where what it allows isn't known."""
    held = held_on_the_cpu(model, revision)
    allowed = system_memory() if held else None
    needed = 2 * held * 1.1
    return (needed, allowed) if allowed is not None and needed > allowed else None


def shortfall(
    model: str, revision: str | None, device: str, spread: bool = False
) -> tuple[float, int] | None:
    """The bytes a model needs on the device, its weights in 16 bits and a tenth more for a batch's
    activations, and the bytes the device allows, when that is too few; None when it fits, or on the
    CPU, whose limit isn't known. With spread, the model goes across every GPU, as placement puts it.
    Only the model's file headers are read, and only for a GPU."""
    protect_memory()
    budget = memory_budget(device, spread)
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
    """Residuals at each prompt's last token, shape (prompts, layers, hidden), float32 on the CPU.

    Index i is the output of layer i, before any final norm. They are read with forward hooks rather
    than output_hidden_states, which some remote code breaks (K2-Horizon's returns none). A model with
    hyper-connections, as Qwen4-Exp has, carries several streams of the hidden size from layer to
    layer, each the embedding plus some multiple of each block's output and what the n-gram embedding
    writes into it; its residual is their mean, as heretic reads it. Prompts are batched longest first,
    so each batch pads to a width near its own, and one too long for the memory fails at once.
    """
    decoder = model.get_decoder()
    layers = cast(nn.ModuleList, decoder.layers)
    text = model.config.get_text_config()
    streams, width = getattr(text, "hc_count", 1), text.hidden_size
    captured: dict[int, Tensor] = {}

    def capture(index: int):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            last = hidden[:, -1, :]
            if streams > 1 and last.shape[-1] == streams * width:
                captured[index] = last.float().unflatten(-1, (streams, width)).mean(-2)
            else:
                # A copy, so the layer's whole output isn't kept alive by a view of its last token.
                captured[index] = last.to(torch.float32, copy=True)

        return hook

    handles = [layer.register_forward_hook(capture(i)) for i, layer in enumerate(layers)]
    # Spread across GPUs, the model's first parameter needn't be on the embedding's, and every later
    # layer moves what it is given to its own.
    device = next(decoder.get_input_embeddings().parameters()).device
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    chunks = []
    try:
        for start in range(0, len(prompts), batch_size):
            ids, mask = left_pad([prompts[i] for i in order[start : start + batch_size]], pad_id)
            positions = (mask.cumsum(-1) - 1).clamp(min=0)
            decoder(
                input_ids=ids.to(device),
                attention_mask=mask.to(device),
                position_ids=positions.to(device),
                use_cache=False,
            )
            chunks.append(torch.stack([captured[i].cpu() for i in range(len(layers))], dim=1))
    finally:
        for handle in handles:
            handle.remove()
    return torch.cat(chunks)[torch.tensor(order).argsort()]
