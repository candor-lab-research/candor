"""The stages of a run, each saving its result in the run folder so a run can stop and resume.

Every stage writes under a temporary name and renames it into place when complete, so a crash
never leaves a half-made result that a later run would trust.

    model -> GGUF -> prompt ids -> directions -> adapters -> (search, next)
"""

import json
import shutil
import socket
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars

from . import config, convert, directions, reference, residuals, source, targets, toolchain
from .adapters import write_adapter
from .edit import RowNorm, rank_one
from .engine import LlamaServer
from .prompts import PromptSet
from .weights import ModelFiles

MODEL_FILES = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.tiktoken"]
PROMPT_SETS = ("harmful", "harmless", "harmful_test", "harmless_test")


class RunError(RuntimeError):
    pass


def _replace(partial: Path, final: Path) -> None:
    """Move a finished result into place, replacing nothing: the final name must be free."""
    if final.exists():
        raise RunError(f"{final} appeared while it was being made")
    partial.rename(final)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class Run:
    folder: Path
    settings: config.RunConfig
    report: Callable[[str], None] = print

    @classmethod
    def open(cls, folder: Path, report: Callable[[str], None] = print) -> "Run":
        return cls(folder, config.load(folder), report)

    def path(self, *parts: str) -> Path:
        return self.folder.joinpath(*parts)

    # Model

    @cached_property
    def model_folder(self) -> Path:
        model = self.settings.model
        if source.is_local(model.source):
            return Path(model.source).expanduser()
        disable_progress_bars()
        return Path(snapshot_download(model.source, revision=model.revision, allow_patterns=MODEL_FILES))

    def model_shape(self) -> tuple[dict, int, int, str, list[targets.Writer]]:
        folder = self.model_folder
        model_config = source.read_config(str(folder))
        text = source.text_config(model_config)
        hidden, layers = text["hidden_size"], text["num_hidden_layers"]
        arch = targets.gguf_architecture(model_config)
        found = targets.find_targets(source.tensor_shapes(str(folder)), hidden, layers)
        if arch is None or not found.ok or found.experts or found.quantized:
            raise RunError("the model no longer passes candor inspect; run it to see why")
        return model_config, hidden, layers, arch, found.writers

    # GGUF

    def gguf(self) -> tuple[Path, Path]:
        """The model as bf16 GGUF and in the deployment quantisation, with every edit target checked."""
        out = self.path("gguf")
        quantization = self.settings.deployment.quantization
        bf16 = out / "model-BF16.gguf"
        quantized = bf16 if quantization == "BF16" else out / f"model-{quantization}.gguf"
        done = out / "checked"
        if done.exists() and bf16.exists() and quantized.exists():
            return bf16, quantized
        done.unlink(missing_ok=True)
        folder = self.model_folder
        weights = sum(folder.joinpath(name).stat().st_size for name in source.weight_files(folder))
        needed, free = 1.5 * weights, shutil.disk_usage(self.folder).free
        if free < needed:
            raise RunError(
                f"converting needs about {needed / 1e9:.0f} GB free here; there is {free / 1e9:.0f} GB"
            )
        self.report("Converting to GGUF")
        bf16, quantized = convert.to_gguf(folder, out, quantization, out / "convert.log")
        _, _, layers, arch, writers = self.model_shape()
        if problems := convert.check_targets(bf16, folder, writers, arch, layers):
            raise RunError(
                "the GGUF doesn't hold the edit targets unchanged:\n  " + "\n  ".join(problems[:5])
            )
        done.write_text(json.dumps({"targets": len(writers)}))
        return bf16, quantized

    def server(self, model: Path, adapters: Sequence[Path] = ()) -> LlamaServer:
        cache = self.settings.deployment.kv_cache
        extra = ["--cache-type-k", cache, "--cache-type-v", cache]
        if cache not in {"f32", "f16", "bf16"}:
            extra += ["-fa", "on"]  # llama.cpp refuses a quantised V cache without flash attention
        self.path("logs").mkdir(exist_ok=True)
        binary = toolchain.binaries() / "llama-server"
        log = self.path("logs", "server.log")
        return LlamaServer(
            binary, model, log, adapters, parallel=16, context_per_slot=1024, port=_free_port(), extra=extra
        )

    # Prompt ids

    def template_kwargs(self) -> dict[str, str]:
        date = self.settings.model.date
        return {"date_string": date} if date else {}

    def chats(self, prompts: list[str]) -> list[list[dict[str, str]]]:
        system = self.settings.deployment.system_prompt
        head = [{"role": "system", "content": system}] if system else []
        return [[*head, {"role": "user", "content": prompt}] for prompt in prompts]

    def prompt_ids(self) -> dict[str, list[list[int]]]:
        """Each prompt set as the exact token ids llama-server feeds the model for that chat."""
        path = self.path("prompts.json")
        if path.exists():
            return json.loads(path.read_text())
        _, quantized = self.gguf()
        texts = {}
        for name in PROMPT_SETS:
            spec = getattr(self.settings.prompts, name)
            texts[name] = PromptSet(spec.dataset, spec.split, spec.column).load()
            if not texts[name]:
                raise RunError(f"the {name} prompt set ({spec.dataset}, {spec.split}) is empty")
        self.report("Tokenizing prompts with llama-server")
        kwargs = self.template_kwargs()
        with self.server(quantized) as server:
            if date := kwargs.get("date_string"):
                body = {"messages": self.chats(["x"])[0], "chat_template_kwargs": kwargs}
                rendered = server.client.post("/apply-template", json=body)
                rendered.raise_for_status()
                if date not in rendered.json()["prompt"]:
                    raise RunError(f"the pinned date {date!r} doesn't appear in the rendered prompt")
            ids = {name: server.chat_ids(self.chats(texts[name]), kwargs) for name in PROMPT_SETS}
            sample = self.chats(texts["harmless_test"][:3])
            if [len(i) for i in ids["harmless_test"][:3]] != [
                server.chat_prompt_length(c, kwargs) for c in sample
            ]:
                raise RunError("llama-server's prompt ids don't match what its chat endpoint uses")
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(ids))
        _replace(partial, path)
        return ids

    # Directions

    def directions(self) -> torch.Tensor:
        """One refusal direction per residual index, from transformers fed the run's exact prompt ids."""
        path = self.path("directions.pt")
        if path.exists():
            return torch.load(path)
        ids = self.prompt_ids()
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
        found = directions.difference_of_means(harmful, harmless)
        partial = path.with_suffix(".partial")
        torch.save(found, partial)
        _replace(partial, path)
        return found

    # Adapters

    def direction_sets(self, layers: int) -> dict[str, torch.Tensor]:
        """The candidate directions: each layer's own, or one layer's for every layer."""
        found = self.directions()
        low, high = self.settings.search.direction_layer
        last = layers - 1
        sets = {"per-layer": found[1:]}
        for layer in range(round(low * last), round(high * last) + 1):
            sets[f"layer-{layer}"] = found[layer + 1].expand(layers, -1)
        return sets

    def adapters(self) -> dict[str, dict[str, Path]]:
        """For each direction set, one rank-1 adapter per edit target, named by its GGUF tensor."""
        _, _, layers, arch, writers = self.model_shape()
        names = {w.name: targets.gguf_name(w, arch, layers) for w in writers}
        if unmapped := [name for name, mapped in names.items() if mapped is None]:
            raise RunError(f"no llama.cpp name for {', '.join(unmapped[:3])}")
        files = ModelFiles(self.model_folder)
        row_norm = RowNorm(self.settings.edit.row_norm)
        root = self.path("adapters")
        root.mkdir(exist_ok=True)
        made = {}
        for set_name, set_directions in self.direction_sets(layers).items():
            folder = root / set_name
            if not folder.exists():
                partial = root / f".{set_name}.partial"
                shutil.rmtree(partial, ignore_errors=True)
                partial.mkdir()
                for writer in writers:
                    factors = rank_one(files.tensor(writer.name), set_directions[writer.layer], row_norm)
                    name = str(names[writer.name])
                    write_adapter(partial / f"{name}.gguf", arch, {name: factors})
                _replace(partial, folder)
            made[set_name] = {str(names[w.name]): folder / f"{names[w.name]}.gguf" for w in writers}
        return made

    def prepare(self) -> dict[str, int]:
        """Every stage up to the search. Finished stages are reused."""
        self.gguf()
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
