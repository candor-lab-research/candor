"""Turn a Hugging Face model into the GGUF files llama.cpp runs, and check the edit targets survived.

The adapters are built from the Hugging Face weights but applied to the GGUF's, so every target
must reach the GGUF unchanged: same name, same shape, same values. llama.cpp's converter
reorders some tensors for some architectures; check_targets catches that before anything is
scored.
"""

import subprocess
from pathlib import Path

import gguf
import numpy as np
import torch

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
    """The GGUF type a model converts to without rounding: F16 when every weight matrix is stored
    as F16, otherwise BF16, which also holds the range of F32 weights."""
    stored = {info["dtype"] for info in source.local_headers(model_folder).values() if len(info["shape"]) > 1}
    return "F16" if stored == {"F16"} else "BF16"


def to_bf16(model_folder: Path, destination: Path, log: Path) -> None:
    """Convert a Hugging Face model folder to an unquantised GGUF of its float_type at destination."""
    python, script = toolchain.converter()
    outtype = float_type(model_folder).lower()
    _run(
        "llama.cpp's converter",
        [python, script, model_folder, "--outtype", outtype, "--outfile", destination],
        log,
    )


def quantize(base: Path, destination: Path, quantization: str, log: Path) -> None:
    """Quantise an unquantised GGUF to destination."""
    _run("llama-quantize", [toolchain.binaries() / "llama-quantize", base, destination, quantization], log)


def _as_float(tensor: gguf.ReaderTensor) -> np.ndarray:
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        raw = np.ascontiguousarray(tensor.data).view(np.uint16).astype(np.uint32) << 16
        return raw.view(np.float32).reshape(list(reversed(tensor.shape.tolist())))
    return np.asarray(tensor.data, dtype=np.float32).reshape(list(reversed(tensor.shape.tolist())))


def check_targets(
    model_gguf: Path, model_folder: Path, writers: list[targets.Writer], arch: str, layers: int
) -> list[str]:
    """Problems with how the edit targets arrived in the GGUF; empty when every one is identical.

    Weights stored in the GGUF's own type must match byte for byte. Others are compared within bf16
    rounding.
    """
    reader = gguf.GGUFReader(model_gguf)
    in_gguf = {tensor.name: tensor for tensor in reader.tensors}
    files = ModelFiles(model_folder)
    problems = []
    for writer in writers:
        name = targets.gguf_name(writer, arch, layers)
        tensor = in_gguf.get(name or "")
        if tensor is None:
            problems.append(f"{writer.name} has no tensor {name} in the GGUF")
            continue
        original = files.tensor(writer.name)
        shape = list(reversed(tensor.shape.tolist()))
        if shape != list(original.shape):
            problems.append(f"{writer.name} is {list(original.shape)} but {name} is {shape}")
            continue
        if STORED_AS.get(tensor.tensor_type) == original.dtype:
            same = np.array_equal(
                original.view(torch.uint8).numpy().reshape(-1),
                np.asarray(tensor.data).view(np.uint8).reshape(-1),
            )
        else:
            same = np.allclose(_as_float(tensor), original.float().numpy(), rtol=1 / 128, atol=1e-6)
        if not same:
            problems.append(f"{name} doesn't hold the same values as {writer.name}; the converter changed it")
    return problems
