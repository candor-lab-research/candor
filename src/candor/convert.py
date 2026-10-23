"""Turn a Hugging Face model into the GGUF files llama.cpp runs, and check the edit targets survived.

The adapters are applied to the GGUF's weights, so every target must reach the GGUF exactly as
candor expects it: same name, same shape, same values. llama.cpp's converter reorders some tensors
for some architectures. The one reorder of an edit target it is known to make is reproduced by
as_converted, which both the check and the adapters use; check_targets catches anything else
before anything is scored.
"""

import json
import re
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch
from torch import Tensor

from . import source, targets, toolchain
from .weights import ModelFiles

STORED_AS = {gguf.GGMLQuantizationType.BF16: torch.bfloat16, gguf.GGMLQuantizationType.F16: torch.float16}


class ConversionError(RuntimeError):
    pass


def _run(program: str, args: list, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as out:
        result = subprocess.run([str(a) for a in args], stdout=out, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise ConversionError(f"{program} failed with code {result.returncode}, see {log}")


def float_type(model_folder: Path) -> str:
    """The GGUF type a model folder converts to without rounding (see source.float_type)."""
    return source.float_type(str(model_folder))


def _readable_by_the_converter(model_folder: Path, staging: Path) -> Path:
    """The folder to hand the converter, whose transformers 4 can't read two things transformers 5
    writes into tokenizer_config.json.

    A fast tokenizer's class is saved as TokenizersBackend, which is PreTrainedTokenizerFast in
    transformers 4, and extra special tokens are saved as a list, which transformers 4 calls
    additional_special_tokens. Such a model is converted from a folder of links to its files with
    only those settings renamed. The folder keeps the model folder's name, which the converter falls
    back on to name the GGUF, and the model folder itself is never touched.
    """
    tokenizer_config = model_folder / "tokenizer_config.json"
    if not tokenizer_config.exists():
        return model_folder
    settings = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    readable = dict(settings)
    if readable.get("tokenizer_class") == "TokenizersBackend":
        readable["tokenizer_class"] = "PreTrainedTokenizerFast"
    if isinstance(readable.get("extra_special_tokens"), list):
        readable["additional_special_tokens"] = readable.pop("extra_special_tokens")
    if readable == settings:
        return model_folder
    staged = staging / model_folder.name
    staged.mkdir()
    for item in model_folder.iterdir():
        if item.name != tokenizer_config.name:
            (staged / item.name).symlink_to(item.resolve())
    (staged / tokenizer_config.name).write_text(json.dumps(readable, indent=2), encoding="utf-8")
    return staged


def to_gguf(model_folder: Path, destination: Path, log: Path) -> None:
    """Convert a Hugging Face model folder to an unquantised GGUF of its float_type at destination."""
    python, script = toolchain.converter()
    outtype = float_type(model_folder).lower()
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        folder = _readable_by_the_converter(model_folder, Path(staging))
        _run(
            "llama.cpp's converter",
            [python, script, folder, "--outtype", outtype, "--outfile", destination],
            log,
        )


def quantize(
    base: Path, destination: Path, quantization: str, log: Path, types: dict[str, str] | None = None
) -> None:
    """Quantise an unquantised GGUF to destination; types gives some tensors, by GGUF name, a type
    of their own."""
    args: list = [toolchain.binaries() / "llama-quantize"]
    if types:
        # llama-quantize matches each pattern as a regex anywhere in a tensor's name, lower-cased.
        listed = destination.with_name(destination.name + ".types")
        listed.write_text(
            "".join(f"^{re.escape(name)}$={kind.lower()}\n" for name, kind in sorted(types.items()))
        )
        args += ["--tensor-type-file", listed]
    _run("llama-quantize", [*args, base, destination, quantization], log)


def tensor_types(model_gguf: Path) -> dict[str, str]:
    """Each tensor's type in a GGUF, by name."""
    return {tensor.name: tensor.tensor_type.name for tensor in gguf.GGUFReader(model_gguf).tensors}


def as_converted(weight: Tensor, name: str, arch: str, config: dict[str, Any]) -> Tensor:
    """The weight as llama.cpp b11524's converter writes it into the GGUF.

    For Qwen3.5 and Qwen3.5-MoE with more linear-attention value heads than key heads, the converter
    stores the value heads tiled across key heads instead of grouped by them
    (_LinearAttentionVReorderBase in conversion/qwen.py), so it reorders the input columns of
    linear_attn.out_proj to match. The rows, which are what the edit changes, keep their order. Every
    other target is written as is, a layer's experts among them.
    """
    order = _column_order(name, arch, config)
    return weight if order is None else weight[:, order]


def as_original(weight: Tensor, name: str, arch: str, config: dict[str, Any]) -> Tensor:
    """A weight read from a GGUF, as the model stores it: as_converted undone."""
    order = _column_order(name, arch, config)
    return weight if order is None else weight[:, torch.argsort(order)]


def _column_order(name: str, arch: str, config: dict[str, Any]) -> Tensor | None:
    """The order the converter writes a weight's input columns in, for a weight it reorders."""
    hparams = config | config.get("text_config", {})
    keys, values = hparams.get("linear_num_key_heads", 0), hparams.get("linear_num_value_heads", 0)
    if arch not in ("qwen35", "qwen35moe") or "linear_attn." not in name or ".out_proj." not in name:
        return None
    if keys <= 0 or values <= 0 or keys == values:
        return None
    head = hparams["linear_value_head_dim"]
    return torch.arange(values * head).reshape(keys, values // keys, head).transpose(0, 1).reshape(-1)


def stored(tensor: gguf.ReaderTensor) -> Tensor:
    """A GGUF tensor's values in float32, laid out as torch lays out the model's weight."""
    values = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
    return torch.from_numpy(np.array(values, dtype=np.float32)).reshape(list(reversed(tensor.shape.tolist())))


def check_targets(
    model_gguf: Path, model_folder: Path, edit_targets: Iterable[targets.Target], arch: str
) -> list[str]:
    """Problems with how the edit targets arrived in the GGUF; empty when each is as_converted says,
    a layer's experts stacked in expert order.

    Weights stored in the GGUF's own type must match byte for byte. Others are compared within bf16
    rounding.
    """
    reader = gguf.GGUFReader(model_gguf)
    in_gguf = {tensor.name: tensor for tensor in reader.tensors}
    files = ModelFiles(model_folder)
    config = json.loads((model_folder / "config.json").read_text())
    problems = []
    for target in edit_targets:
        name = target.gguf
        tensor = in_gguf.get(name)
        if tensor is None:
            problems.append(f"{target.name} has no tensor {name} in the GGUF")
            continue
        expected = as_converted(files.read(target.names, target.stacked), target.name, arch, config)
        shape = list(reversed(tensor.shape.tolist()))
        if shape != list(expected.shape):
            problems.append(f"{target.name} is {list(expected.shape)} but {name} is {shape}")
            continue
        if STORED_AS.get(tensor.tensor_type) == expected.dtype:
            same = np.array_equal(
                expected.contiguous().view(torch.uint8).numpy().reshape(-1),
                np.asarray(tensor.data).view(np.uint8).reshape(-1),
            )
        else:
            same = np.allclose(stored(tensor).numpy(), expected.float().numpy(), rtol=1 / 128, atol=1e-6)
        if not same:
            problems.append(f"{name} doesn't hold the same values as {target.name}; the converter changed it")
    return problems
