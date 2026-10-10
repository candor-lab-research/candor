"""The stages of a run, each saving its result in the run folder so a run can stop and resume.

    model -> GGUF -> prompt ids -> directions -> adapters -> (search, next)

Every result is written under a temporary name and renamed into place when complete, so a crash
never leaves a half-made result. Beside each result is a record of what it was built from: the
settings it used and the records of the stages before it. A result is reused only when that
record still matches, so editing the settings rebuilds what they affect and everything after it.
"""

import fcntl
import hashlib
import json
import math
import os
import shutil
import socket
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars

from . import config, convert, directions, reference, residuals, source, targets, toolchain
from .adapters import write_adapter
from .edit import RowNorm, rank_one
from .engine import LLAMA_CPP_RELEASE, LlamaServer
from .prompts import PromptSet
from .weights import ModelFiles

MODEL_FILES = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.tiktoken"]
PROMPT_SETS = ("harmful", "harmless", "harmful_test", "harmless_test")
BITS_PER_WEIGHT = {
    "Q4_K_S": 4.6,
    "Q4_K_M": 4.9,
    "Q5_K_S": 5.5,
    "Q5_K_M": 5.7,
    "Q6_K": 6.6,
    "Q8_0": 8.5,
    "F16": 16,
}
DATE_PROBE = "31 Dec 1999"
CHECKED_PER_SET = 3


class RunError(RuntimeError):
    pass


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def _record(output: Path) -> Path:
    return output.with_name(output.name + ".inputs.json")


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _read_record(record: Path) -> Any:
    try:
        return json.loads(record.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _sync(path: Path) -> None:
    """Flush a file, or every file in a folder, and the folder entries, to the disk."""
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    for file in files:
        with file.open("rb") as handle:
            os.fsync(handle.fileno())
    for folder in {path.parent, *(f.parent for f in files)}:
        descriptor = os.open(folder, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class Run:
    folder: Path
    settings: config.RunConfig
    report: Callable[[str], None] = print
    inputs: dict[str, str] = field(default_factory=dict)

    @classmethod
    def open(cls, folder: Path, report: Callable[[str], None] = print) -> "Run":
        return cls(folder, config.load(folder), report)

    def path(self, *parts: str) -> Path:
        return self.folder.joinpath(*parts)

    @contextmanager
    def locked(self) -> Iterator[None]:
        with self.path(".lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RunError(f"another candor command is using {self.folder}") from None
            yield

    def _fresh(self, output: Path, inputs: dict[str, Any]) -> bool:
        """Whether output exists and was built from exactly these inputs."""
        record = _record(output)
        expected = json.loads(json.dumps(inputs))
        if output.exists() and _read_record(record) == expected:
            return True
        if output.exists() or record.exists():
            self.report(f"Rebuilding {output.relative_to(self.folder)}: what it was built from has changed")
        return False

    def _commit(self, partial: Path, output: Path, inputs: dict[str, Any]) -> None:
        """Replace output with the finished partial and record what it was built from. The output is
        on the disk before its record is, so a crash or power cut at any point leaves either no record
        or a record that matches, never a record trusting a half-written output."""
        record = _record(output)
        record.unlink(missing_ok=True)
        _remove(output)
        _sync(partial)
        partial.rename(output)
        written = record.with_name(record.name + ".building")
        written.write_text(json.dumps(inputs, sort_keys=True))
        _sync(written)
        written.rename(record)
        _sync(record)

    # Model

    @cached_property
    def model_folder(self) -> Path:
        model = self.settings.model
        if source.is_local(model.source):
            return Path(model.source).expanduser()
        disable_progress_bars()
        return Path(snapshot_download(model.source, revision=model.revision, allow_patterns=MODEL_FILES))

    @cached_property
    def model_fingerprint(self) -> str:
        """Local folders by their files' names, sizes and times; Hub models by their commit."""
        folder = self.model_folder
        if source.is_local(self.settings.model.source):
            listed = [p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")]
            files = sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in listed)
            return _digest({"folder": str(folder.resolve()), "files": files})
        return _digest({"repo": self.settings.model.source, "commit": folder.name})

    @cached_property
    def context_limit(self) -> int | None:
        text = source.text_config(source.read_config(str(self.model_folder)))
        limit = text.get("max_position_embeddings")
        return limit if isinstance(limit, int) else None

    @cached_property
    def shape(self) -> tuple[int, int, str, list[targets.Writer]]:
        model_config = source.read_config(str(self.model_folder))
        text = source.text_config(model_config)
        hidden, layers = text["hidden_size"], text["num_hidden_layers"]
        arch = targets.gguf_architecture(model_config)
        found = targets.find_targets(source.tensor_shapes(str(self.model_folder)), hidden, layers)
        if arch is None or not found.ok or found.experts or found.quantized:
            raise RunError("the model no longer passes candor inspect; run it to see why")
        return hidden, layers, arch, found.writers

    # GGUF

    def _check_disk(self, bf16_needed: bool, quantization: str) -> None:
        shapes = source.tensor_shapes(str(self.model_folder))
        weights = sum(math.prod(shape) for shape in shapes.values())
        needed = 1.1 * weights * ((2 if bf16_needed else 0) + BITS_PER_WEIGHT.get(quantization, 0) / 8)
        free = shutil.disk_usage(self.folder).free
        if free < needed:
            raise RunError(
                f"converting needs about {needed / 1e9:.1f} GB free here; there is {free / 1e9:.1f} GB"
            )

    def gguf(self) -> tuple[Path, Path]:
        """The model as bf16 GGUF, every edit target checked, and in the deployment quantisation."""
        out = self.path("gguf")
        out.mkdir(exist_ok=True)
        quantization = self.settings.deployment.quantization
        bf16 = out / "model-BF16.gguf"
        bf16_inputs = {"model": self.model_fingerprint, "llama.cpp": LLAMA_CPP_RELEASE}
        quantized = bf16 if quantization == "BF16" else out / f"model-{quantization}.gguf"
        quantized_inputs = {"bf16": _digest(bf16_inputs), "quantization": quantization}
        for leftover in out.iterdir():
            kept = {bf16.name, quantized.name, _record(bf16).name, _record(quantized).name, "convert.log"}
            if leftover.name not in kept:
                self.report(f"Removing {leftover.relative_to(self.folder)}, which this run no longer uses")
                _remove(leftover)
        bf16_fresh = self._fresh(bf16, bf16_inputs)
        quantized_fresh = bf16_fresh if quantized == bf16 else self._fresh(quantized, quantized_inputs)
        if not (bf16_fresh and quantized_fresh):
            self._check_disk(not bf16_fresh, "BF16" if quantized_fresh else quantization)
        if not bf16_fresh:
            self.report("Converting to GGUF")
            partial = out / "model-BF16.gguf.building"
            convert.to_bf16(self.model_folder, partial, out / "convert.log")
            _, layers, arch, writers = self.shape
            if problems := convert.check_targets(partial, self.model_folder, writers, arch, layers):
                partial.unlink()
                raise RunError(
                    "the GGUF doesn't hold the edit targets unchanged:\n  " + "\n  ".join(problems[:5])
                )
            self._commit(partial, bf16, bf16_inputs)
        if quantized != bf16 and not quantized_fresh:
            self.report(f"Quantising to {quantization}")
            partial = out / f"model-{quantization}.gguf.building"
            convert.quantize(bf16, partial, quantization, out / "convert.log")
            self._commit(partial, quantized, quantized_inputs)
        # Quantising keeps the vocabulary and chat template, so prompt ids depend on the bf16 file alone.
        self.inputs["bf16"] = _digest(bf16_inputs)
        return bf16, quantized

    def server(
        self, model: Path, adapters: Sequence[Path] = (), context_per_slot: int = 8192, parallel: int = 4
    ) -> LlamaServer:
        cache = self.settings.deployment.kv_cache
        extra = ["--cache-type-k", cache, "--cache-type-v", cache]
        if cache not in {"f32", "f16", "bf16"}:
            extra += ["-fa", "on"]  # llama.cpp refuses a quantised V cache without flash attention
        self.path("logs").mkdir(exist_ok=True)
        binary = toolchain.binaries() / "llama-server"
        log = self.path("logs", "server.log")
        return LlamaServer(binary, model, log, adapters, parallel, context_per_slot, _free_port(), extra)

    # Prompt ids

    def template_kwargs(self, date: str | None = None) -> dict[str, str]:
        date = date or self.settings.model.date
        return {"date_string": date} if date else {}

    def chats(self, prompts: list[str]) -> list[list[dict[str, str]]]:
        system = self.settings.deployment.system_prompt
        head = [{"role": "system", "content": system}] if system else []
        return [[*head, {"role": "user", "content": prompt}] for prompt in prompts]

    def _check_lengths(self, ids: dict[str, list[list[int]]]) -> None:
        """Harmful test prompts must fit with a whole answer; the rest need room for one token."""
        limit, answer = self.context_limit, self.settings.scoring.answer_tokens
        if not limit:
            return
        room = {name: answer if name == "harmful_test" else 1 for name in PROMPT_SETS}
        too_long = {name: sum(len(p) + room[name] > limit for p in ids[name]) for name in PROMPT_SETS}
        if any(too_long.values()):
            counts = ", ".join(f"{count} {name}" for name, count in too_long.items() if count)
            raise RunError(
                f"prompts too long for the model's {limit} tokens (answers get {answer}): {counts}"
            )

    def prompt_ids(self) -> dict[str, list[list[int]]]:
        """Each prompt set as the exact token ids llama-server feeds the model for that chat."""
        _, quantized = self.gguf()
        path = self.path("prompts.json")
        settings = self.settings
        inputs = {
            "bf16": self.inputs["bf16"],
            "system_prompt": settings.deployment.system_prompt,
            "date": settings.model.date,
            "prompts": settings.prompts.model_dump(),
        }
        if self._fresh(path, inputs):
            ids = json.loads(path.read_text())
            self._check_lengths(ids)
            return ids
        texts = {}
        for name in PROMPT_SETS:
            spec = getattr(settings.prompts, name)
            texts[name] = PromptSet(spec.dataset, spec.split, spec.column).load()
            if not texts[name]:
                raise RunError(f"the {name} prompt set ({spec.dataset}, {spec.split}) is empty")
        self.report("Tokenizing prompts with llama-server")
        kwargs = self.template_kwargs()
        with self.server(quantized, context_per_slot=2048, parallel=2) as server:
            if settings.model.date:
                body = {
                    "messages": self.chats(["x"])[0],
                    "chat_template_kwargs": self.template_kwargs(DATE_PROBE),
                }
                rendered = server.client.post("/apply-template", json=body)
                rendered.raise_for_status()
                if DATE_PROBE not in rendered.json()["prompt"]:
                    raise RunError(
                        "the chat template ignores the pinned date, so prompts would still change daily"
                    )
            ids = {name: server.chat_ids(self.chats(texts[name]), kwargs) for name in PROMPT_SETS}
        self._check_lengths(ids)
        sampled = {name: ids[name][:CHECKED_PER_SET] for name in PROMPT_SETS}
        slot = max(len(p) for prompts in sampled.values() for p in prompts) + 16
        with self.server(quantized, context_per_slot=slot, parallel=1) as server:
            for name in PROMPT_SETS:
                expected = [
                    server.chat_prompt_length(chat, kwargs)
                    for chat in self.chats(texts[name][:CHECKED_PER_SET])
                ]
                if [len(i) for i in sampled[name]] != expected:
                    raise RunError(f"llama-server's ids for the {name} prompts don't match its chat endpoint")
        partial = path.with_name("prompts.json.building")
        partial.write_text(json.dumps(ids))
        self._commit(partial, path, inputs)
        return ids

    # Directions

    def directions(self) -> torch.Tensor:
        """One refusal direction per residual index, from transformers fed the run's exact prompt ids."""
        ids = self.prompt_ids()
        path = self.path("directions.pt")
        inputs = {"ids": _digest([ids["harmful"], ids["harmless"]]), "model": self.model_fingerprint}
        self.inputs["directions"] = _digest(inputs)
        if self._fresh(path, inputs):
            return torch.load(path)
        reference.protect_memory()
        device, dtype = reference.device_and_dtype()
        self.report(f"Reading residuals on {device}")
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(self.model_folder, dtype=dtype)
        model.to(device).eval()  # pyright: ignore[reportArgumentType]
        pad_id = reference.pad_token_id_for(self.model_folder)
        harmful = residuals.last_token_residuals(model, ids["harmful"], pad_id)
        harmless = residuals.last_token_residuals(model, ids["harmless"], pad_id)
        del model
        reference.free_memory(device)
        if not (torch.isfinite(harmful).all() and torch.isfinite(harmless).all()):
            raise RunError(
                f"the model's activations overflowed in {dtype} on {device}; run on a GPU in bfloat16"
            )
        found = directions.difference_of_means(harmful, harmless)
        if not torch.isfinite(found).all():
            raise RunError("some refusal directions aren't finite numbers")
        partial = path.with_name("directions.pt.building")
        torch.save(found, partial)
        self._commit(partial, path, inputs)
        return found

    # Adapters

    def direction_sets(self, found: torch.Tensor, layers: int) -> dict[str, torch.Tensor]:
        """The candidate directions: each layer's own, or one layer's for every layer.

        found[0] is the embedding output and found[i + 1] the output of layer i.
        """
        low, high = self.settings.search.direction_layer
        last = layers - 1
        sets = {"per-layer": found[1:]}
        for layer in range(round(low * last), round(high * last) + 1):
            sets[f"layer-{layer}"] = found[layer + 1].expand(layers, -1)
        return sets

    def adapters(self) -> dict[str, dict[str, Path]]:
        """For each direction set, one rank-1 adapter per edit target, named by its GGUF tensor."""
        found = self.directions()
        _, layers, arch, writers = self.shape
        names = {w.name: targets.gguf_name(w, arch, layers) for w in writers}
        if unmapped := [name for name, mapped in names.items() if mapped is None]:
            raise RunError(f"no llama.cpp name for {', '.join(unmapped[:3])}")
        row_norm = RowNorm(self.settings.edit.row_norm)
        inputs = {
            "directions": self.inputs["directions"],
            "row_norm": row_norm.value,
            "targets": sorted([w.name, str(names[w.name])] for w in writers),
        }
        root = self.path("adapters")
        root.mkdir(exist_ok=True)
        sets = self.direction_sets(found, layers)
        for leftover in root.iterdir():
            if leftover.name.removesuffix(".inputs.json") not in sets:
                _remove(leftover)
        files = None
        made = {}
        for set_name, set_directions in sets.items():
            folder = root / set_name
            if not self._fresh(folder, inputs):
                files = files or ModelFiles(self.model_folder)
                self.report(f"Writing adapters for {set_name}")
                partial = root / f"{set_name}.building"
                _remove(partial)
                partial.mkdir()
                for writer in writers:
                    name = str(names[writer.name])
                    factors = rank_one(files.tensor(writer.name), set_directions[writer.layer], row_norm)
                    write_adapter(partial / f"{name}.gguf", arch, {name: factors})
                self._commit(partial, folder, inputs)
            made[set_name] = {str(names[w.name]): folder / f"{names[w.name]}.gguf" for w in writers}
        return made

    def prepare(self) -> dict[str, int]:
        """Every stage up to the search. Results still matching their settings are reused."""
        with self.locked():
            ids = self.prompt_ids()
            found = self.directions()
            made = self.adapters()
        return {
            "edit targets": len(next(iter(made.values()))),
            "direction sets": len(made),
            "adapters": sum(map(len, made.values())),
            "prompts": sum(map(len, ids.values())),
            "directions": len(found),
        }
