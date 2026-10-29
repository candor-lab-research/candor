"""The stages of a run, each saving its result in the run folder so a run can stop and resume.

    model -> GGUF -> prompt ids -> directions -> adapters -> (search, next)

Every result is written under a temporary name and renamed into place when complete, so a crash
never leaves a half-made result. Beside each result is a record of what it was built from: the
settings it used, the records of the stages before it and the version of candor's code for each
stage. A result is reused only when that record still matches, so editing the settings, or a new
version of a stage, rebuilds what they affect and everything after it.
"""

import fcntl
import hashlib
import json
import math
import re
import shutil
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from fnmatch import fnmatch
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import constants, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from . import checks, config, convert, directions, residuals, source, targets, toolchain
from .adapters import write_adapter
from .edit import RowNorm, edit_factors, full_device, rank_one
from .engine import LLAMA_CPP_RELEASE, LlamaServer, Thinking, seed_for
from .files import sync, write_atomic
from .judge import normalise
from .weights import ModelFiles

if TYPE_CHECKING:
    from .measure import Edit

MODEL_FILES = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.tiktoken"]
# Mistral's own format beside the transformers one: the same weights again, which nothing here reads.
NOT_MODEL_FILES = ["consolidated*.safetensors"]
# A llama-server log past this many bytes is kept as server.log.1 when the next server starts.
LOG_BYTES = 50_000_000
PROMPT_SETS = tuple(config.Prompts.model_fields)
# Prompt sets the search never sees, kept for comparing finished edits.
HELD_OUT = ("harmful_held_out", "harmless_held_out")
# The prompt sets KL is read on. For a thinking model it is read where the answer starts, after the
# unedited model's own thinking (thinking/).
KL_SETS = ("harmless_test", "harmless_held_out")
# For a thinking model, these sets are also kept with an empty thinking block closed after the prompt,
# under their names and ANSWERING: directions are read there as well as where the thinking starts,
# candor measure reads KL there beside the KL the search aims at, as heretic read KL, and the search may
# score its test prompts there with thinking off.
CLOSED_SETS = ("harmful", "harmless", *KL_SETS, "harmful_test")
ANSWERING = "_answering"
# The names of a thinking model's adapter sets read where the answer starts begin with this.
ANSWER_SETS = "answer-"
DATE_PROBE = "31 Dec 1999"
CHECKED_PER_SET = 3
# The slot of the server that tokenizes the prompts, which also checks a few of them against the chat
# endpoint when they fit.
TOKENIZING_SLOT = 2048
# The version of each stage's code, in stage order. Bump a stage's number in any commit that changes
# the results it builds. Each record holds the numbers of its own stage and every stage before it, so
# results built by older code are rebuilt along with everything after them, unless DEPENDS names the
# stages it is built from. scores holds adapters built only to score edits: candor score's for directions
# the search has none for, and a "full" edit's own adapter for each edit or search trial scored; exports
# holds the GGUFs candor export ships. Neither is built from a measurement. Stages added later
# go last, so adding one rebuilds nothing already made: a thinking model's directions where its answer
# starts, and its unedited thinking, after which KL is read.
STAGES = {
    "gguf": 1,
    "prompts.json": 1,
    "directions.pt": 2,
    "adapters": 1,
    "measurements": 2,
    "scores": 1,
    "exports": 3,
    "directions_answering.pt": 2,
    "thinking": 2,
}
# The stages an edit is built from.
EDITED = ("gguf", "prompts.json", "directions.pt", "adapters")
DEPENDS = {
    "scores": EDITED,
    "exports": EDITED,
    "directions_answering.pt": ("gguf", "prompts.json", "directions.pt"),
    "thinking": ("gguf", "prompts.json"),
}


class RunError(RuntimeError):
    pass


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def record_of(output: Path) -> Path:
    return output.with_name(output.name + ".inputs.json")


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def read_record(record: Path) -> Any:
    try:
        return json.loads(record.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def load_prompts(spec: config.PromptSet) -> list[str]:
    try:
        rows = load_dataset(spec.dataset, spec.subset, split=spec.split, revision=spec.revision)
    except DatasetNotFoundError as error:
        raise RunError(
            f"{error} If it is gated, accept its terms at https://huggingface.co/datasets/{spec.dataset} "
            "and log in with hf auth login."
        ) from None
    texts = list(rows[spec.column])
    keep = [True] * len(texts)
    for column, excluded in (spec.exclude or {}).items():
        labels = list(rows[column])
        if unmatched := set(excluded) - set(labels):
            raise RunError(f"{spec.dataset} has no rows whose {column} is {', '.join(sorted(unmatched))}")
        keep = [kept and label not in excluded for kept, label in zip(keep, labels, strict=True)]
    return [text for text, kept in zip(texts, keep, strict=True) if kept]


def seeds(seed: int, count: int) -> list[int]:
    """A seed of its own for each of count prompts, made from seed: the same prompt always gets the
    same one, whatever edit answers it."""
    return [seed_for(seed, str(i)) for i in range(count)]


def _after(closed: list[int], end: int) -> list[int]:
    """The tokens a prompt with its thinking closed holds after the thinking's end tag: the text the
    template writes between the thinking and the answer."""
    return closed[len(closed) - closed[::-1].index(end) :]


def answering(settings: config.RunConfig) -> dict[str, Any]:
    """How the model answers, for the records of what its answers depend on: its sampling, the seed
    left out, unless it answers greedily as by default, and a thinking model's thinking. Nothing for a
    greedy model that doesn't think, so the records of runs made before candor sampled still match."""
    found: dict[str, Any] = {}
    if settings.sampling != config.Sampling(seed=settings.sampling.seed):
        found["sampling"] = settings.sampling.model_dump(exclude={"seed"})
    if thinking := thinking_settings(settings.deployment):
        found["thinking"] = thinking
    return found


def thinking_settings(deployment: config.Deployment) -> dict[str, Any] | None:
    """How a thinking model's answers are capped, for the records of what they depend on."""
    if deployment.think_close is None:
        return None
    return {
        "close": deployment.think_close,
        "budget": deployment.thinking_budget,
        "message": deployment.thinking_budget_message,
    }


def every_tenth(report: Callable[[str], None], what: str, total: int, done: int = 0) -> Callable[[], None]:
    """A callback for each of total as it is done, after the done already are, that reports how many are
    at every tenth of them, so a long wait shows it moves where no progress bar can be seen."""
    count = done

    def one() -> None:
        nonlocal count
        count += 1
        if count * 10 // total > (count - 1) * 10 // total:
            report(f"{what}: {count} of {total}")

    return one


def duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60} h {minutes % 60} min" if minutes >= 60 else f"{max(minutes, 1)} min"


def drop_repeats(texts: dict[str, list[str]]) -> dict[str, int]:
    """Remove from each held-out set every prompt that is also in another set, or earlier in its own,
    so the held-out sets only hold prompts the search never saw. Returns how many each set lost."""
    seen = {normalise(text) for name, prompts in texts.items() if name not in HELD_OUT for text in prompts}
    dropped = {}
    for name in HELD_OUT:
        kept = []
        for text in texts[name]:
            if (key := normalise(text)) not in seen:
                seen.add(key)
                kept.append(text)
        dropped[name] = len(texts[name]) - len(kept)
        texts[name] = kept
    return dropped


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

    def _with_code(self, output: Path, inputs: dict[str, Any]) -> dict[str, Any]:
        """The inputs, and the code versions of the output's stage and the stages it depends on: every
        stage before it, unless DEPENDS names them."""
        stages = list(STAGES)
        stage = output.relative_to(self.folder).parts[0]
        before = DEPENDS.get(stage, stages[: stages.index(stage)])
        return {**inputs, "code": {name: STAGES[name] for name in (*before, stage)}}

    def record_for(self, output: Path, inputs: dict[str, Any]) -> dict[str, Any]:
        """The record output gets when built from these inputs by this version of candor."""
        return json.loads(json.dumps(self._with_code(output, inputs)))

    def current(self, output: Path, inputs: dict[str, Any]) -> bool:
        """Whether output exists and was built from exactly these inputs, by this version of candor."""
        return output.exists() and read_record(record_of(output)) == self.record_for(output, inputs)

    def fresh(self, output: Path, inputs: dict[str, Any]) -> bool:
        """Whether output is current, saying why it will be rebuilt when it isn't."""
        if self.current(output, inputs):
            return True
        record = record_of(output)
        if output.exists() or record.exists():
            why = "what it was built from has changed" if output.exists() else "it is missing"
            self.report(f"Rebuilding {output.relative_to(self.folder)}: {why}")
        return False

    def discard(self, output: Path) -> None:
        """Remove output and its record, the record first."""
        record_of(output).unlink(missing_ok=True)
        _remove(output)

    def commit(self, partial: Path, output: Path, inputs: dict[str, Any]) -> None:
        """Replace output with the finished partial and record what it was built from. The output is
        on the disk before its record is, so a crash or power cut at any point leaves either no record
        or a record that matches, never a record trusting a half-written output."""
        record = record_of(output)
        record.unlink(missing_ok=True)
        _remove(output)
        sync(partial)
        partial.rename(output)
        write_atomic(record, json.dumps(self._with_code(output, inputs), sort_keys=True))

    # Model

    @property
    def model_files(self) -> list[str]:
        """The model's files a run reads, as download patterns: its own Python only when trusted."""
        return MODEL_FILES + (["*.py"] if self.settings.model.trust_remote_code else [])

    @cached_property
    def model_folder(self) -> Path:
        model = self.settings.model
        if source.is_local(model.source):
            folder = Path(model.source).expanduser()
            if not folder.is_dir():
                raise RunError(f"the model folder {folder} is gone; move it back or edit model.source")
            return folder
        self._check_download()
        return Path(
            snapshot_download(
                model.source,
                revision=model.revision,
                allow_patterns=self.model_files,
                ignore_patterns=NOT_MODEL_FILES,
            )
        )

    def _check_download(self) -> None:
        """Stop before downloading unless there is room for the download, and for converting the model
        too while the run has no GGUF of it, where that goes on the same disk."""
        model = self.settings.model
        planned = snapshot_download(
            model.source,
            revision=model.revision,
            allow_patterns=self.model_files,
            ignore_patterns=NOT_MODEL_FILES,
            dry_run=True,
        )
        needed = sum(file.file_size or 0 for file in planned if file.will_download)
        cache = Path(constants.HF_HUB_CACHE)
        cache = next(p for p in [cache, *cache.parents] if p.exists())
        what = "downloading the model"
        if not self.path("gguf").exists() and cache.stat().st_dev == self.folder.stat().st_dev:
            # A 16-bit deployment needs no copy beyond the converted one when it is the model's own type.
            deployed = self.settings.deployment.quantization
            quantized = deployed not in ("F16", "BF16") or deployed != source.float_type(
                self._model_without_download, model.revision
            )
            needed += self._room(1, quantized, self._model_without_download)
            what = "downloading and converting the model"
        free = shutil.disk_usage(cache).free
        if free < needed:
            raise RunError(
                f"{what} needs {needed / 1e9:.1f} GB free in {cache}; there is {free / 1e9:.1f} GB"
            )

    @cached_property
    def float_type(self) -> str:
        """The GGUF type the model converts to without rounding (see source.float_type), read from the
        weights here, or from a Hub model's headers on the Hub while it isn't downloaded."""
        model = self.settings.model
        if source.is_local(model.source):
            return source.float_type(str(self.model_folder))
        try:
            downloaded = snapshot_download(
                model.source,
                revision=model.revision,
                allow_patterns=self.model_files,
                ignore_patterns=NOT_MODEL_FILES,
                local_files_only=True,
            )
            # A folder holding only the config, as the checks fetch it, raises as no weights.
            return source.float_type(downloaded)
        except (LocalEntryNotFoundError, OSError, ValueError):
            return source.float_type(model.source, model.revision)

    @cached_property
    def model_fingerprint(self) -> str:
        """Local folders by the names, sizes and times of the files a run reads; Hub models by their
        commit, found without downloading them when the run pins it. Nothing else, so editing a README
        or moving the folder rebuilds nothing."""
        model = self.settings.model
        if not source.is_local(model.source):
            # A download is kept under the commit it is at.
            pinned = model.revision is not None and re.fullmatch(r"[0-9a-f]{40}", model.revision)
            commit = model.revision if pinned else self.model_folder.name
            return digest({"repo": model.source, "commit": commit})
        folder = self.model_folder
        read = [
            p
            for p in folder.iterdir()
            if p.is_file()
            and not p.name.startswith(".")
            and any(fnmatch(p.name, pattern) for pattern in self.model_files)
        ]
        return digest(sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in read))

    @cached_property
    def context_limit(self) -> int | None:
        text = source.text_config(source.read_config(str(self.model_folder)))
        limit = text.get("max_position_embeddings")
        return limit if isinstance(limit, int) else None

    @cached_property
    def layers(self) -> int:
        """The model's layer count from its config alone, so a trial's edit can be rebuilt on a machine
        without the full-size files."""
        model = self.settings.model
        found = source.sizes(source.read_config(self._model_without_download, model.revision))
        if found is None:
            raise RunError("the model no longer passes candor inspect; run it to see why")
        return found[1]

    @cached_property
    def shape(self) -> tuple[int, int, str, list[targets.Writer]]:
        model_config = source.read_config(str(self.model_folder))
        sizes = source.sizes(model_config)
        arch = targets.gguf_architecture(model_config)
        if sizes is None or arch is None:
            raise RunError("the model no longer passes candor inspect; run it to see why")
        hidden, layers = sizes
        found = targets.find_targets(source.tensor_shapes(str(self.model_folder)), hidden, layers)
        if not found.ok or found.hidden_last or found.quantized:
            raise RunError("the model no longer passes candor inspect; run it to see why")
        return hidden, layers, arch, found.writers

    def _device(self) -> tuple[str, torch.dtype]:
        device, dtype = residuals.device_and_dtype()
        if device == "cpu" and shutil.which("nvidia-smi"):
            raise RunError(
                f"torch {torch.__version__} can't use this machine's NVIDIA GPU, likely because its driver "
                f"is older than CUDA {torch.version.cuda} needs; reading residuals on the CPU instead "
                "would be very slow"
            )
        return device, dtype

    @property
    def _model_without_download(self) -> str:
        """Where to read the model's config and headers before deciding to download it."""
        model = self.settings.model
        return str(self.model_folder) if source.is_local(model.source) else model.source

    def check_remote_code(self) -> None:
        model = self.settings.model
        if not model.trust_remote_code and checks.needs_remote_code(
            source.read_config(self._model_without_download, model.revision)
        ):
            raise RunError(
                "transformers can load this model only with the Python files it ships; set "
                "model.trust_remote_code = true in candor.toml to run them"
            )

    def check_memory(self) -> None:
        """Stop if this machine's GPUs can't hold the model for reading residuals, or its memory can't
        hold what transformers loads on the CPU, which would have the model killed partway through."""
        device, _ = self._device()
        model = self.settings.model
        if short := residuals.shortfall(self._model_without_download, model.revision, device, spread=True):
            needed, budget = short
            raise RunError(
                f"reading residuals needs about {needed / 1e9:.1f} GB of GPU memory and this machine "
                f"allows {budget / 1e9:.1f} GB; run it on a pod"
            )
        if short := residuals.memory_shortfall(self._model_without_download, model.revision):
            needed, allowed = short
            raise RunError(
                f"loading the model needs about {needed / 1e9:.1f} GB of this machine's memory, where "
                "transformers joins the weights it keeps off the GPUs, such as an n-gram embedding table, "
                f"from their shards, and it allows {allowed / 1e9:.1f} GB; run it on a pod with more"
            )

    def check_judge(self, model: str, revision: str | None) -> None:
        """Stop before loading a judge this machine's GPU can't hold, as one too big freezes a Mac. A
        judge already downloaded is sized from the Hub's cache, so judging offline still works."""
        device, _ = residuals.device_and_dtype()
        if short := residuals.shortfall(str(source.downloaded(model, revision) or model), revision, device):
            needed, budget = short
            raise RunError(
                f"the judge {model} needs about {needed / 1e9:.1f} GB of GPU memory and this machine "
                f"allows {budget / 1e9:.1f} GB; judge on a pod, or set a smaller one in [measure]"
            )

    # GGUF

    def _room(self, copies: int, quantized_needed: bool, where: str, extra: int = 0) -> float:
        """The bytes this many 16-bit copies of the model, a quantised one and extra bytes take, with
        a tenth to spare; the model read from where."""
        weights = sum(
            math.prod(shape) for shape in source.tensor_shapes(where, self.settings.model.revision).values()
        )
        quantized = config.QUANTIZATIONS[self.settings.deployment.quantization] / 8 if quantized_needed else 0
        return 1.1 * (weights * (2 * copies + quantized) + extra)

    def check_disk(self, copies: int, quantized_needed: bool, extra: int = 0) -> None:
        """Stop unless there is room for this many 16-bit copies of the model, a quantised one, and
        extra bytes."""
        needed = self._room(copies, quantized_needed, str(self.model_folder), extra)
        free = shutil.disk_usage(self.folder).free
        if free < needed:
            raise RunError(
                f"converting needs about {needed / 1e9:.1f} GB free here; there is {free / 1e9:.1f} GB"
            )

    def _planned_gguf(self) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
        """Where the model's unquantised and deployed GGUFs go, and what each is built from."""
        out = self.path("gguf")
        quantization = self.settings.deployment.quantization
        base_type = self.float_type
        base = out / f"model-{base_type}.gguf"
        base_inputs = {"model": self.model_fingerprint, "llama.cpp": LLAMA_CPP_RELEASE}
        quantized = base if quantization == base_type else out / f"model-{quantization}.gguf"
        return base, base_inputs, quantized, {"base": digest(base_inputs), "quantization": quantization}

    def model_file(self) -> str:
        """What the deployed GGUF is built from, without building it."""
        return digest(self._planned_gguf()[3])

    def gguf(self) -> tuple[Path, Path]:
        """The model as an unquantised GGUF of its own float type, every edit target checked, and in
        the deployment quantisation."""
        out = self.path("gguf")
        out.mkdir(exist_ok=True)
        base, base_inputs, quantized, quantized_inputs = self._planned_gguf()
        quantization = self.settings.deployment.quantization
        for leftover in out.iterdir():
            kept = {base.name, quantized.name, record_of(base).name, record_of(quantized).name, "convert.log"}
            if leftover.name not in kept:
                self.report(f"Removing {leftover.relative_to(self.folder)}, which this run no longer uses")
                _remove(leftover)
        base_fresh = self.fresh(base, base_inputs)
        quantized_fresh = base_fresh if quantized == base else self.fresh(quantized, quantized_inputs)
        if not (base_fresh and quantized_fresh):
            self.check_disk(0 if base_fresh else 1, quantized != base and not quantized_fresh)
        if not base_fresh:
            self.report("Converting to GGUF")
            partial = base.with_name(base.name + ".building")
            convert.to_gguf(self.model_folder, partial, out / "convert.log")
            _, _, arch, _ = self.shape
            if problems := convert.check_targets(
                partial, self.model_folder, self.edit_targets.values(), arch
            ):
                partial.unlink()
                raise RunError(
                    "the GGUF doesn't hold the edit targets unchanged:\n  " + "\n  ".join(problems[:5])
                )
            self.commit(partial, base, base_inputs)
        if quantized != base and not quantized_fresh:
            self.report(f"Quantising to {quantization}")
            partial = quantized.with_name(quantized.name + ".building")
            convert.quantize(base, partial, quantization, out / "convert.log")
            self.commit(partial, quantized, quantized_inputs)
        # Quantising keeps the vocabulary and chat template, so prompt ids depend on the base file alone.
        self.inputs["base"] = digest(base_inputs)
        self.inputs["quantized"] = digest(quantized_inputs)
        return base, quantized

    def server(
        self,
        model: Path,
        adapters: Sequence[Path] = (),
        *,
        context_per_slot: int,
        parallel: int,
        slots_setting: str | None = None,
    ) -> LlamaServer:
        binaries = toolchain.binaries()
        extra = self.cache_flags() + toolchain.device_flags(binaries)
        if self.settings.sampling.temperature > 0:
            # Sampling on the GPU, which a request falls back from where it can't, such as when it
            # asks for first-token probabilities: the same samplers, many times faster for a large
            # vocabulary.
            extra.append("--backend-sampling")
        self.path("logs").mkdir(exist_ok=True)
        binary = binaries / "llama-server"
        log = self.path("logs", "server.log")
        if log.exists() and log.stat().st_size > LOG_BYTES:
            log.replace(log.with_name("server.log.1"))
        return LlamaServer(
            binary, model, log, adapters, parallel, context_per_slot, extra=extra, slots_setting=slots_setting
        )

    def machine(self) -> dict[str, Any]:
        """Where the run's llama-server runs (see toolchain.machine)."""
        return toolchain.machine(toolchain.binaries())

    def cache_flags(self) -> list[str]:
        """llama-server's flags for the KV cache type the run deploys with."""
        cache = self.settings.deployment.kv_cache
        flags = ["--cache-type-k", cache, "--cache-type-v", cache]
        if cache not in {"f32", "f16", "bf16"}:
            flags += ["-fa", "on"]  # llama.cpp refuses a quantised V cache without flash attention
        return flags

    # Prompt ids

    def template_kwargs(self, date: str | None = None) -> dict[str, Any]:
        """What the chat template renders every prompt with: the run's chat template settings, and the
        pinned date unless another is given."""
        date = date or self.settings.model.date
        settings = self.settings.deployment.chat_template_kwargs or {}
        return {**settings, "date_string": date} if date else dict(settings)

    def sampling(self) -> dict[str, float]:
        """How llama-server picks each token, as the run's settings say the model is used, seeds
        apart."""
        return self.settings.sampling.model_dump(exclude={"seed"})

    def thinking(self, server: LlamaServer, budget: int | None = None) -> Thinking | None:
        """How a thinking model thinks, with its budget, the deployment's unless given; None for a
        model that doesn't think. Every prompt must end by opening its thinking, or the budget would
        never start counting; and a prompt that ends by opening some thinking needs a think_close, or
        its thinking would be judged as its answer."""
        deployment = self.settings.deployment
        rendered = server.apply_template(self.chats(["x"])[0], self.template_kwargs())
        if deployment.think_close is None:
            if checks.OPENS_THINKING.search(rendered):
                raise RunError(
                    "every prompt ends by opening the model's thinking, but deployment.think_close isn't "
                    "set, so its thinking would be judged as its answer; set it, and a thinking_budget"
                )
            return None
        start, end = deployment.tags()
        opened = rendered[rendered.rfind(start) :] if start in rendered else ""
        if opened.strip() != start:
            raise RunError(
                f"the chat template doesn't end a prompt by opening the thinking with {start}, as "
                "deployment.think_close says it does"
            )
        budget = budget or deployment.thinking_budget
        assert budget is not None
        close = deployment.think_close.rstrip()
        return Thinking(start, end, close, opened, budget, deployment.thinking_budget_message)

    def check_thinking_off(self, server: LlamaServer) -> None:
        """A thinking model answers after an empty thinking block in use only where its chat template
        turns thinking off by writing that block, as Qwen's does. K2-Horizon's always opens the thinking,
        so answers after an empty block would score a way the model is never run."""
        chat, kwargs = self.chats(["x"])[0], self.template_kwargs()
        on = server.apply_template(chat, kwargs)
        off = server.apply_template(chat, {**kwargs, "enable_thinking": False})
        if off == on:
            raise RunError(
                "this model's chat template can't turn its thinking off, so the model is never used "
                "answering after an empty thinking block and a score with thinking off would measure a way "
                'no one runs it: keep scoring.thinking = "on"'
            )
        if not off.startswith(on):
            raise RunError(
                "this model's chat template changes more than the end of a prompt when it turns thinking "
                "off at these settings, as Qwen3.8's leaves out the system message of every reasoning "
                "effort but medium, so answers after an empty thinking block wouldn't be the model with "
                'its thinking off: keep scoring.thinking = "on"'
            )
        if off != on + (self.settings.deployment.think_close or ""):
            raise RunError(
                "deployment.think_close isn't what the chat template writes after a prompt with thinking "
                "off, so answers after it wouldn't be the model with its thinking off; set think_close to "
                "match"
            )

    def thinking_room(self, budget: int | None = None) -> int:
        """The most tokens a thinking model's thinking can take before its answer, the message it is
        made to write at the end of its budget (the deployment's unless given) included."""
        deployment = self.settings.deployment
        if deployment.think_close is None:
            return 0
        budget = budget or deployment.thinking_budget
        assert budget is not None
        # A token holds at least one byte, so the message takes at most its bytes. The 16 covers the close
        # and the engine.UNFINISHED_CHARACTER tokens a model may take to finish a character at the budget.
        return budget + len(deployment.thinking_budget_message.encode()) + 16

    def kl_ids(self, ids: dict[str, list[list[int]]], name: str, limit: int | None = None) -> list[list[int]]:
        """The first limit prompts of one of KL_SETS as KL is read on them: for a thinking model, after
        the unedited model's own thinking, where its answers start in use, leaving out those whose
        thinking didn't end by itself."""
        if not self.settings.deployment.think_close:
            return ids[name][:limit]
        return [prompt for prompt in self.thinking_traces()[name][:limit] if prompt is not None]

    def _thinking_inputs(self) -> dict[str, Any]:
        ids = self.prompt_ids()
        self.gguf()
        settings = self.settings
        return {
            "model": self.inputs["quantized"],
            "ids": digest([ids[name] for name in KL_SETS]),
            "kv_cache": settings.deployment.kv_cache,
            "seed": settings.sampling.seed,
            "slots": settings.slots.thinking,
            **answering(settings),
        }

    def thinking_key(self) -> str:
        """What the unedited model's thinking is sampled from, as a name. Sampled thinking can't be
        sampled again exactly, so each is kept under its own name: a change samples afresh and keeps
        the old, which going back finds again."""
        if self.settings.deployment.think_close is None:
            raise RunError("only a thinking model has thinking to read KL after")
        return digest(self._with_code(self.path("thinking"), self._thinking_inputs()))

    def thinking_traces(self) -> dict[str, list[list[int] | None]]:
        """For each of KL_SETS, each prompt followed by the unedited model's own thinking and the text its
        template writes after the thinking's end tag: where its answer starts in thinking-on use, so KL
        is read there. Sampled as the run deploys the model, each prompt with its seed, at the
        deployment's budget; None for a prompt whose thinking didn't end by itself, which KL leaves out.
        The thinking is the model's own text, so it is kept only in the private folder, as token ids."""
        key = self.thinking_key()
        record, private = (
            self.path("thinking", f"{key}.json"),
            self.path("private", "thinking", f"{key}.json"),
        )
        sampled = {name: private.with_name(f"{key}.{name}.json") for name in KL_SETS}
        if record.exists():
            made = json.loads(record.read_text())
            if private.exists() and digest(traces := json.loads(private.read_text())) == made["traces"]:
                # Sets kept along the way that a stop after the record, or a pull, left behind.
                for path in sampled.values():
                    path.unlink(missing_ok=True)
                return traces
            raise RunError(
                f"the unedited model's thinking that KL is read after is missing or changed in "
                f"{private.relative_to(self.folder)}, and sampling it again wouldn't give the same: restore "
                f"it, or delete {record.relative_to(self.folder)} to sample afresh, after which KL can't be "
                "compared with KL read before"
            )
        settings = self.settings
        ids = self.prompt_ids()
        _, quantized = self.gguf()
        # Each set is kept as soon as it is sampled, so an interruption loses only the set it stopped.
        private.parent.mkdir(parents=True, exist_ok=True)
        traces = {name: json.loads(path.read_text()) for name, path in sampled.items() if path.exists()}
        if traces:
            sets = " and ".join(name.replace("_", " ") for name in traces)
            self.report(f"Reusing the unedited model's thinking on the {sets} prompts, sampled before")
        if missing := [name for name in KL_SETS if name not in traces]:
            self.report("Thinking with the unedited model, to read KL where its answers start")
            room = max(len(p) for name in KL_SETS for p in ids[name]) + self.thinking_room() + 8
            slots, setting = settings.slots.thinking, "slots.thinking in candor.toml"
            with self.server(
                quantized, context_per_slot=room, parallel=slots, slots_setting=setting
            ) as server:
                thinking = self.thinking(server)
                assert thinking is not None
                end = server.token_of(thinking.end)
                for name in missing:
                    prompts = ids[name]
                    done = every_tenth(
                        self.report, f"Thought through {name.replace('_', ' ')} prompts", len(prompts)
                    )
                    thoughts = server.thoughts(
                        prompts, self.sampling(), seeds(settings.sampling.seed, len(prompts)), thinking, done
                    )
                    traces[name] = [
                        None if thought is None else prompt + thought + _after(closed, end)
                        for prompt, thought, closed in zip(
                            prompts, thoughts, ids[name + ANSWERING], strict=True
                        )
                    ]
                    if not any(traces[name]):
                        raise RunError(
                            f"the unedited model's thinking didn't end by itself on any of the "
                            f"{len(prompts)} {name.replace('_', ' ')} prompts, so KL can't be read after it; "
                            "raise deployment.thinking_budget"
                        )
                    write_atomic(sampled[name], json.dumps(traces[name]))
        traces = {name: traces[name] for name in KL_SETS}
        write_atomic(private, json.dumps(traces))
        made = {
            "traces": digest(traces),
            "kept": {name: sum(trace is not None for trace in traces[name]) for name in KL_SETS},
        }
        record.parent.mkdir(exist_ok=True)
        partial = record.with_name(f"{record.name}.building")
        partial.write_text(json.dumps(made))
        self.commit(partial, record, self._thinking_inputs())
        for path in sampled.values():
            path.unlink()
        return traces

    def chats(self, prompts: list[str]) -> list[list[dict[str, str]]]:
        system = self.settings.deployment.system_prompt
        head = [{"role": "system", "content": system}] if system else []
        return [[*head, {"role": "user", "content": prompt}] for prompt in prompts]

    @property
    def answer_tokens(self) -> dict[str, int]:
        """How long an answer to each prompt set the model answers in full may be."""
        settings = self.settings
        return {
            "harmful_test": settings.scoring.answer_tokens,
            "harmful_held_out": settings.measure.answer_tokens,
            "harmless_held_out": settings.measure.answer_tokens,
            "maths": settings.measure.maths_answer_tokens,
        }

    def _check_chat_ids(
        self, server: LlamaServer, ids: dict[str, list[list[int]]], texts: dict[str, list[str]]
    ) -> None:
        """That the ids of a few prompts from each set are as long as the chat endpoint's own."""
        for name in PROMPT_SETS:
            chats = self.chats(texts[name][:CHECKED_PER_SET])
            expected = [server.chat_prompt_length(chat, self.template_kwargs()) for chat in chats]
            if [len(i) for i in ids[name][:CHECKED_PER_SET]] != expected:
                raise RunError(f"llama-server's ids for the {name} prompts don't match its chat endpoint")

    def _check_lengths(self, ids: dict[str, list[list[int]]]) -> None:
        """Prompts the model answers must fit with whole thinking and a whole answer, and those KL is read
        on with whole thinking and one token; the rest need room for one token."""
        limit, thinking = self.context_limit, self.thinking_room()
        if not limit:
            return
        room = {name: thinking + 1 for name in KL_SETS}
        room |= {name: tokens + thinking for name, tokens in self.answer_tokens.items()}
        too_long = {name: sum(len(p) + room.get(name, 1) > limit for p in ids[name]) for name in PROMPT_SETS}
        if any(too_long.values()):
            counts = ", ".join(f"{count} {name}" for name, count in too_long.items() if count)
            raise RunError(f"prompts too long for the model's {limit} tokens with their answers: {counts}")

    def prompt_texts(self) -> tuple[dict[str, list[str]], dict[str, int]]:
        """Each prompt set's prompts as the model is asked them, and how many repeats each held-out
        set lost."""
        settings = self.settings
        texts = {name: load_prompts(getattr(settings.prompts, name)) for name in PROMPT_SETS}
        dropped = drop_repeats(texts)
        for name in PROMPT_SETS:
            if not texts[name]:
                spec = getattr(settings.prompts, name)
                why = ": every one of its prompts is also in another set" if dropped.get(name) else ""
                raise RunError(f"the {name} prompt set ({spec.dataset}, {spec.split}) is empty{why}")
        instruction = settings.measure.maths_instruction
        texts["maths"] = [f"{question}\n\n{instruction}" for question in texts["maths"]]
        return texts, dropped

    def searched_texts(self, name: str) -> list[str]:
        """One of the sets the search sees, as prompt_texts gives it, read from its dataset alone: only
        the held-out sets and maths are changed by the others."""
        assert name not in (*HELD_OUT, "maths")
        if not (texts := load_prompts(spec := getattr(self.settings.prompts, name))):
            raise RunError(f"the {name} prompt set ({spec.dataset}, {spec.split}) is empty")
        return texts

    def check_prompts(self, server: LlamaServer, name: str, texts: list[str], ids: list[list[int]]) -> None:
        """Prompts read again from their dataset must be the ones prompts.json was built from, before
        a judge reads them or answers are checked against their rows."""
        if server.chat_ids(self.chats(texts), self.template_kwargs()) != ids:
            raise RunError(
                f"the {name} prompts have changed since prompts.json was built; run candor run first"
            )

    def prompt_ids(self) -> dict[str, list[list[int]]]:
        """Each prompt set as the exact token ids llama-server feeds the model for that chat."""
        _, quantized = self.gguf()
        path = self.path("prompts.json")
        settings = self.settings
        inputs = {
            "base": self.inputs["base"],
            "system_prompt": settings.deployment.system_prompt,
            "date": settings.model.date,
            "prompts": settings.prompts.model_dump(),
            "maths_instruction": settings.measure.maths_instruction,
        }
        # Only when set, so the records of runs made before the setting are as they were.
        if template_settings := settings.deployment.chat_template_kwargs:
            inputs["chat_template_kwargs"] = template_settings
        close = settings.deployment.think_close
        if close:
            inputs["think_close"] = close
            inputs["closed"] = list(CLOSED_SETS)
        if self.fresh(path, inputs):
            ids = json.loads(path.read_text())
            self._check_lengths(ids)
            return ids
        texts, dropped = self.prompt_texts()
        self.report(
            f"Holding out {len(texts['harmful_held_out'])} harmful and {len(texts['harmless_held_out'])} "
            f"harmless prompts from the search (repeats of other prompts dropped: {sum(dropped.values())})"
        )
        self.report("Tokenizing prompts with llama-server")
        kwargs = self.template_kwargs()
        with self.server(quantized, context_per_slot=TOKENIZING_SLOT, parallel=2) as server:
            if settings.model.date:
                rendered = server.apply_template(self.chats(["x"])[0], self.template_kwargs(DATE_PROBE))
                if DATE_PROBE not in rendered:
                    raise RunError(
                        "the chat template ignores the pinned date, so prompts would still change daily"
                    )
            # A think_close the prompts don't open, such as one left from another reasoning effort, is
            # caught before residuals are read from them.
            self.thinking(server)
            ids = {name: server.chat_ids(self.chats(texts[name]), kwargs) for name in PROMPT_SETS}
            if close:
                for name in CLOSED_SETS:
                    ids[name + ANSWERING] = server.chat_ids(self.chats(texts[name]), kwargs, close)
            self._check_lengths(ids)
            slot = max(len(p) for name in PROMPT_SETS for p in ids[name][:CHECKED_PER_SET]) + 16
            if slot <= TOKENIZING_SLOT:
                self._check_chat_ids(server, ids, texts)
        if slot > TOKENIZING_SLOT:
            with self.server(quantized, context_per_slot=slot, parallel=1) as server:
                self._check_chat_ids(server, ids, texts)
        partial = path.with_name("prompts.json.building")
        partial.write_text(json.dumps(ids))
        self.commit(partial, path, inputs)
        return ids

    # Directions

    @property
    def positions(self) -> tuple[bool, ...]:
        """Where the run reads directions, as answering: at the end of the prompt, and for a thinking
        model also where its answer starts after an empty thinking block."""
        return (False, True) if self.settings.deployment.think_close else (False,)

    def _direction_plan(self, answering: bool) -> tuple[Path, dict[str, Any]]:
        ids = self.prompt_ids()
        closed = ANSWERING if answering else ""
        inputs = {
            "ids": digest([ids["harmful" + closed], ids["harmless" + closed]]),
            "model": self.model_fingerprint,
            "remote_code": self.settings.model.trust_remote_code,
        }
        return self.path(f"directions{closed}.pt"), inputs

    def check_residuals(self) -> None:
        """Whether this machine can read the model's residuals, checked up front so a model it can't
        isn't downloaded and converted first. Directions already read by this code, as on a pod, need
        nothing, but directions lost after they were read are refused (see _check_directions_kept)."""
        self._check_directions_kept()
        record = read_record(record_of(self.path("directions.pt"))) or {}
        if record.get("code", {}).get("directions.pt") != STAGES["directions.pt"]:
            self.check_memory()

    def _check_directions_kept(self) -> None:
        for answering in self.positions:
            path = self.path(f"directions{ANSWERING if answering else ''}.pt")
            if record_of(path).exists() and not path.exists():
                raise RunError(
                    f"{path.name} is missing but its record isn't, and residuals read again, on another "
                    "machine above all, wouldn't give exactly the directions the adapters and searches here "
                    "were built from: restore it, or build the run afresh in a new folder"
                )

    def directions(self, answering: bool = False) -> torch.Tensor:
        """One refusal direction per layer, read after it, from transformers fed the run's exact prompt ids:
        read at the end of each prompt or, with answering, where a thinking model's answer starts after
        an empty thinking block. A thinking model's are read at both in one pass."""
        if answering not in self.positions:
            raise RunError("only a thinking model has directions where its answer starts")
        plans = self.planned_directions()
        if stale := {position: plan for position, plan in plans.items() if not self.fresh(*plan)}:
            self._read_directions(stale)
        return torch.load(plans[answering][0])

    def planned_directions(self) -> dict[bool, tuple[Path, dict[str, Any]]]:
        """Where each position's directions go and what they are read from, named in the run's inputs
        as directions() names them but without reading any, so what is built from them can be checked
        first."""
        self._check_directions_kept()
        plans = {position: self._direction_plan(position) for position in self.positions}
        self.inputs["directions"] = digest(plans[False][1])
        if True in plans:
            # The stage comes after the adapters in STAGES, so its code version reaches them only here.
            self.inputs["answering directions"] = digest(self._with_code(*plans[True]))
        return plans

    def _read_directions(self, stale: dict[bool, tuple[Path, dict[str, Any]]]) -> None:
        """Read the residuals for each stale position, loading the model once, and save its directions."""
        self.check_memory()
        ids = self.prompt_ids()
        device, dtype = self._device()
        trusted = self.settings.model.trust_remote_code
        placement = residuals.placement(device)
        spread = "max_memory" in placement
        self.report("Loading the model across the GPUs" if spread else f"Reading residuals on {device}")
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            self.model_folder, dtype=dtype, trust_remote_code=trusted, **placement
        ).eval()
        if off := residuals.off_the_gpus(model):
            del model
            residuals.free_memory(device)
            raise RunError(
                f"the GPUs couldn't hold the whole model, so transformers put {len(off)} of its modules, "
                f"{off[0]} first, on the CPU or disk; run it on a pod with more GPU memory"
            )
        if spread:
            self.report(f"Reading residuals on GPUs {', '.join(map(str, residuals.gpus_used(model)))}")
        pad_id = residuals.pad_token_id_for(self.model_folder, trusted)
        read = {}
        for answering in stale:
            closed = ANSWERING if answering else ""
            read[answering] = [
                residuals.last_token_residuals(model, ids[name + closed], pad_id)
                for name in ("harmful", "harmless")
            ]
        del model
        residuals.free_memory(device)
        if not all(torch.isfinite(found).all() for pair in read.values() for found in pair):
            raise RunError(
                f"the model's activations overflowed in {dtype} on {device}; run on a GPU in bfloat16"
            )
        for answering, (path, inputs) in stale.items():
            partial = path.with_name(f"{path.name}.building")
            torch.save(directions.difference_of_means(*read[answering]), partial)
            self.commit(partial, path, inputs)

    def edit_directions(self, name: str, layer: float | None, answering: bool = False) -> torch.Tensor:
        """The direction an edit removes at each layer: each layer's own when layer is None, otherwise
        the one read after that layer, which may fall between two; read where the answer starts when
        answering."""
        found = self.directions(answering)
        _, layers, _, _ = self.shape
        if layer is None:
            return found
        if 0 <= layer <= layers - 1:
            return directions.after_layer(found, layer).expand(layers, -1)
        raise RunError(f"{name} reads its direction at layer {layer:g}; this model has {layers}")

    # Adapters

    def direction_sets(self, found: torch.Tensor, layers: int) -> dict[str, torch.Tensor]:
        """The candidate directions: each layer's own, or one layer's for every layer, found[i] being
        the one read after layer i."""
        low, high = self.settings.search.direction_layers(layers)
        sets = {"per-layer": found}
        for layer in range(low, high + 1):
            sets[f"layer-{layer}"] = found[layer].expand(layers, -1)
        return sets

    @cached_property
    def target_names(self) -> dict[str, str]:
        """Each writer's GGUF tensor name, by its name in the model; a layer's experts share theirs."""
        _, layers, arch, writers = self.shape
        names = {w.name: targets.gguf_name(w, arch, layers) for w in writers}
        if unmapped := [name for name, mapped in names.items() if mapped is None]:
            raise RunError(f"no llama.cpp name for {', '.join(unmapped[:3])}")
        return {name: str(mapped) for name, mapped in names.items()}

    @cached_property
    def edit_targets(self) -> dict[str, targets.Target]:
        """The weights an edit changes, by name: each writer on its own, or a layer's experts stacked as
        the GGUF holds them."""
        _, layers, arch, writers = self.shape
        try:
            return targets.edit_targets(writers, arch, layers)
        except ValueError as error:
            raise RunError(str(error)) from None

    def target_weight(
        self, files: ModelFiles, target: targets.Target, model_config: dict[str, Any]
    ) -> torch.Tensor:
        """The target's weight laid out as the GGUF stores it."""
        _, _, arch, _ = self.shape
        return convert.as_converted(files.read(target.names, target.stacked), target.name, arch, model_config)

    def adapter_inputs(self, answering: bool = False) -> dict[str, Any]:
        """What every adapter set read at one position is built from besides its directions; call after
        planned_directions() or directions()."""
        return {
            "directions": self.inputs["answering directions" if answering else "directions"],
            "row_norm": self.settings.edit.row_norm,
            "targets": sorted(self.target_names.items()),
        }

    def adapter_set(
        self, folder: Path, set_directions: torch.Tensor, inputs: dict[str, Any]
    ) -> dict[str, Path]:
        """One rank-1 adapter per edit target along its layer's direction in set_directions, named by its
        GGUF tensor. Written to folder unless it already holds them, built from these inputs."""
        return self.adapter_sets({folder: (set_directions, inputs)})[folder]

    def adapter_sets(
        self, sets: dict[Path, tuple[torch.Tensor, dict[str, Any]]]
    ) -> dict[Path, dict[str, Path]]:
        """adapter_set for each folder, its directions and inputs. Each weight is read once for every set
        that needs writing, rather than once a set."""
        _, _, arch, _ = self.shape
        edit_targets = self.edit_targets.values()
        stale = {folder: made for folder, made in sets.items() if not self.fresh(folder, made[1])}
        if stale:
            files = ModelFiles(self.model_folder)
            model_config = source.read_config(str(self.model_folder))
            row_norm = RowNorm(self.settings.edit.row_norm)
            [first, *more] = [str(folder.relative_to(self.folder)) for folder in stale]
            self.report(
                f"Writing adapters for {first}" + (f" and {len(more)} more direction sets" if more else "")
            )
            partials = {folder: folder.with_name(f"{folder.name}.building") for folder in stale}
            for partial in partials.values():
                _remove(partial)
                partial.mkdir(parents=True)
            for target in edit_targets:
                weight = self.target_weight(files, target, model_config)
                for folder, (set_directions, _) in stale.items():
                    factors = rank_one(weight, set_directions[target.layer], row_norm)
                    write_adapter(partials[folder] / f"{target.gguf}.gguf", arch, {target.gguf: factors})
            for folder, (_, inputs) in stale.items():
                self.commit(partials[folder], folder, inputs)
        return {folder: {t.gguf: folder / f"{t.gguf}.gguf" for t in edit_targets} for folder in sets}

    @property
    def linear(self) -> bool:
        """Whether an edit is linear in strength, so one adapter per edit target covers every strength;
        a "full" edit isn't, and has adapters built for each edit."""
        return RowNorm(self.settings.edit.row_norm) is not RowNorm.FULL

    def adapters(self) -> dict[str, dict[str, Path]]:
        """For each direction set, one rank-1 adapter per edit target, named by its GGUF tensor. The names
        of a thinking model's sets read where the answer starts begin with ANSWER_SETS. None for an edit
        that isn't linear in strength."""
        _, layers, _, _ = self.shape
        root = self.path("adapters")
        root.mkdir(exist_ok=True)
        sets = {}
        for answering in self.positions:
            found, inputs = self.directions(answering), self.adapter_inputs(answering)
            if not self.linear:
                continue
            prefix = ANSWER_SETS if answering else ""
            sets |= {
                prefix + name: (made, inputs) for name, made in self.direction_sets(found, layers).items()
            }
        for leftover in root.iterdir():
            if leftover.name.removesuffix(".inputs.json") not in sets:
                _remove(leftover)
        built = self.adapter_sets({root / name: made for name, made in sets.items()})
        return {name: built[root / name] for name in sets}

    def changes(self, edit: "Edit") -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Each weight the edit changes, by its edit target's name, and the change as factors (b, a), the
        strength folded in, laid out as the GGUF stores the weight: what its adapter holds and its
        export writes, built the same way for both (on one machine, the same numbers)."""
        found = self.edit_directions(edit.name, edit.direction_layer, edit.answering)
        files = ModelFiles(self.model_folder)
        model_config = source.read_config(str(self.model_folder))
        row_norm = RowNorm(self.settings.edit.row_norm)
        made = {
            target.name: edit_factors(
                self.target_weight(files, target, model_config), found[target.layer], strength, row_norm
            )
            for target in self.edit_targets.values()
            if (strength := edit.strength(target.writers[0]))
        }
        # A "full" change is built on the GPU, whose memory goes back before a llama-server needs it.
        residuals.free_memory(full_device())
        return made

    def edit_adapter(self, folder: Path, edit: "Edit") -> tuple[Path, dict[str, Any]]:
        """One adapter holding every change the edit makes, at its strengths, applied at scale 1, as an
        edit that isn't linear in strength needs; and what it is built from. Written to folder unless
        already there."""
        _, _, arch, _ = self.shape
        self.edit_directions(edit.name, edit.direction_layer, edit.answering)
        inputs = {**self.adapter_inputs(edit.answering), "edit": edit.record()}
        path = folder / "edit.gguf"
        if not self.fresh(path, inputs):
            factors = {self.edit_targets[name].gguf: change for name, change in self.changes(edit).items()}
            if not factors:
                raise RunError(f"{edit.name} changes no weight")
            folder.mkdir(parents=True, exist_ok=True)
            partial = path.with_name("edit.gguf.building")
            write_adapter(partial, arch, factors)
            self.commit(partial, path, inputs)
        return path, inputs

    def prepare(self) -> dict[str, int]:
        """Every stage up to the search. The last stage pulls in the ones before it, and results still
        matching their settings are reused."""
        with self.locked():
            self.check_remote_code()
            self.check_residuals()
            made = self.adapters()
            ids = json.loads(self.path("prompts.json").read_text())
            kept = self.thinking_traces() if self.settings.deployment.think_close else {}
            # Found at the end of a search instead, a pod's disk can't grow without a restart that wipes it.
            quantized = self.settings.deployment.quantization != convert.float_type(self.model_folder)
            export = self._room(1, quantized, str(self.model_folder))
            if (free := shutil.disk_usage(self.folder).free) < export:
                self.report(
                    f"Exporting an edit will need about {export / 1e9:.1f} GB free here, and there is "
                    f"{free / 1e9:.1f} GB: make room before the search ends"
                )
        # How many prompts of each set KL is read on, after the unedited model's thinking ended by itself.
        thought = {
            f"of {len(kept[name])} {name.replace('_', ' ')} prompts thought through": sum(
                t is not None for t in kept[name]
            )
            for name in kept
        }
        # A "full" edit builds each edit's own adapter when it is scored, so none are shared to count.
        shared = (
            {
                "edit targets": len(next(iter(made.values()))),
                "direction sets": len(made),
                "adapters": sum(map(len, made.values())),
            }
            if made
            else {}
        )
        # Each prompt once: a thinking model's sets also hold each prompt with its thinking closed.
        prompts = sum(len(found) for name, found in ids.items() if not name.endswith(ANSWERING))
        return {**shared, "prompts": prompts, **thought}
