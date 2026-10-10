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

from . import targets, toolchain
from .weights import ModelFiles


class ConversionError(RuntimeError):
    pass


def _run(args: list, log: Path) -> None:
    with log.open("ab") as out:
        result = subprocess.run([str(a) for a in args], stdout=out, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise ConversionError(f"{Path(str(args[1])).name} failed with code {result.returncode}, see {log}")


def to_gguf(model_folder: Path, out: Path, quantization: str, log: Path) -> tuple[Path, Path]:
    """The model as bf16 GGUF and in the run's quantisation. Existing files are reused."""
    out.mkdir(parents=True, exist_ok=True)
    bf16 = out / "model-BF16.gguf"
    if not bf16.exists():
        python, script = toolchain.converter()
        partial = bf16.with_suffix(".partial")
        _run([python, script, model_folder, "--outtype", "bf16", "--outfile", partial], log)
        partial.rename(bf16)
    if quantization == "BF16":
        return bf16, bf16
    quantized = out / f"model-{quantization}.gguf"
    if not quantized.exists():
        partial = quantized.with_suffix(".partial")
        _run([toolchain.binaries() / "llama-quantize", bf16, partial, quantization], log)
        partial.rename(quantized)
    return bf16, quantized


def _as_float(tensor: gguf.ReaderTensor) -> np.ndarray:
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        raw = np.ascontiguousarray(tensor.data).view(np.uint16).astype(np.uint32) << 16
        return raw.view(np.float32).reshape(list(reversed(tensor.shape.tolist())))
    return np.asarray(tensor.data, dtype=np.float32).reshape(list(reversed(tensor.shape.tolist())))


def check_targets(
    bf16_gguf: Path, model_folder: Path, writers: list[targets.Writer], arch: str, layers: int
) -> list[str]:
    """Problems with how the edit targets arrived in the GGUF; empty when every one is identical.

    Weights stored as bf16 must match byte for byte. Others are compared within bf16 rounding.
    """
    reader = gguf.GGUFReader(bf16_gguf)
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
        if original.dtype == torch.bfloat16 and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            same = np.array_equal(
                original.view(torch.int16).numpy().view(np.uint8).reshape(-1),
                np.asarray(tensor.data).reshape(-1),
            )
        else:
            same = np.allclose(_as_float(tensor), original.float().numpy(), rtol=1 / 128, atol=1e-6)
        if not same:
            problems.append(f"{name} doesn't hold the same values as {writer.name}; the converter changed it")
    return problems
