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
import os
import re
import shutil
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from fnmatch import fnmatch
from functools import cached_property
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import constants, snapshot_download

from . import checks, config, convert, directions, residuals, source, targets, toolchain
from .adapters import write_adapter
from .edit import RowNorm, rank_one
from .engine import LLAMA_CPP_RELEASE, LlamaServer, Thinking
from .judge import normalise
from .weights import ModelFiles

MODEL_FILES = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.tiktoken"]
PROMPT_SETS = tuple(config.Prompts.model_fields)
# Prompt sets the search never sees, kept for comparing finished edits.
HELD_OUT = ("harmful_held_out", "harmless_held_out")
# The prompt sets KL is read on. For a thinking model it is read where the answer starts, after the
# unedited model's own thinking (thinking/).
KL_SETS = ("harmless_test", "harmless_held_out")
# For a thinking model, these sets are also kept with an empty thinking block closed after the prompt,
# under their names and ANSWERING: directions are read there as well as where the thinking starts, and
# KL there is kept beside the KL the search aims at, as heretic read KL.
CLOSED_SETS = ("harmful", "harmless", *KL_SETS)
ANSWERING = "_answering"
# The names of a thinking model's adapter sets read where the answer starts begin with this.
ANSWER_SETS = "answer-"
# How many answers the unedited model thinks at once for thinking/; sampled answers can shift with it.
THINKING_SLOTS = 8
# A sampled answer's seed is never llama.cpp's "pick one at random".
RANDOM_SEED = 2**32 - 1
# A prompt that ends by opening a thinking tag, such as Qwen3.5's <think> or K2-Horizon's <ifm|think>.
OPENS_THINKING = re.compile(r"<[^<>/]*think[^<>]*>\s*$", re.IGNORECASE)
DATE_PROBE = "31 Dec 1999"
CHECKED_PER_SET = 3
# The version of each stage's code, in stage order. Bump a stage's number in any commit that changes
# the results it builds. Each record holds the numbers of its own stage and every stage before it, so
# results built by older code are rebuilt along with everything after them. scores holds the adapters
# candor score builds for edits the search has none for, and exports the GGUFs candor export ships.
# Stages added later go last, so adding one rebuilds nothing already made: a thinking model's directions
# where its answer starts, and its unedited thinking, after which KL is read. Each of those holds the
# numbers of the stages in DEPENDS rather than of every stage before it.
STAGES = {
    "gguf": 1,
    "prompts.json": 1,
    "directions.pt": 1,
    "adapters": 1,
    "measurements": 1,
    "scores": 1,
    "exports": 2,
    "directions_answering.pt": 1,
    "thinking": 1,
}
DEPENDS = {
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


def _read_record(record: Path) -> Any:
    try:
        return json.loads(record.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _flush(descriptor: int) -> None:
    """On macOS a plain fsync can leave data in the drive's cache; F_FULLFSYNC waits for the disk."""
    if hasattr(fcntl, "F_FULLFSYNC"):
        try:
            fcntl.fcntl(descriptor, fcntl.F_FULLFSYNC)
            return
        except OSError:
            pass
    os.fsync(descriptor)


def sync(path: Path) -> None:
    """Flush a file, or every file in a folder, and the folder entries, to the disk."""
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    for item in [*files, *{path.parent, *(f.parent for f in files)}]:
        descriptor = os.open(item, os.O_RDONLY)
        try:
            _flush(descriptor)
        finally:
            os.close(descriptor)


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
    made = (hashlib.sha256(f"{seed}:{i}".encode()).digest() for i in range(count))
    return [int.from_bytes(digest[:4]) % RANDOM_SEED for digest in made]


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

    def current(self, output: Path, inputs: dict[str, Any]) -> bool:
        """Whether output exists and was built from exactly these inputs, by this version of candor."""
        expected = json.loads(json.dumps(self._with_code(output, inputs)))
        return output.exists() and _read_record(record_of(output)) == expected

    def fresh(self, output: Path, inputs: dict[str, Any]) -> bool:
        """Whether output is current, saying why it will be rebuilt when it isn't."""
        if self.current(output, inputs):
            return True
        record = record_of(output)
        if output.exists() or record.exists():
            why = "what it was built from has changed" if output.exists() else "it is missing"
            self.report(f"Rebuilding {output.relative_to(self.folder)}: {why}")
        return False

    def commit(self, partial: Path, output: Path, inputs: dict[str, Any]) -> None:
        """Replace output with the finished partial and record what it was built from. The output is
        on the disk before its record is, so a crash or power cut at any point leaves either no record
        or a record that matches, never a record trusting a half-written output."""
        record = record_of(output)
        record.unlink(missing_ok=True)
        _remove(output)
        sync(partial)
        partial.rename(output)
        written = record.with_name(record.name + ".building")
        written.write_text(json.dumps(self._with_code(output, inputs), sort_keys=True))
        sync(written)
        written.rename(record)
        sync(record)

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
        return Path(snapshot_download(model.source, revision=model.revision, allow_patterns=self.model_files))

    def _check_download(self) -> None:
        model = self.settings.model
        planned = snapshot_download(
            model.source, revision=model.revision, allow_patterns=self.model_files, dry_run=True
        )
        needed = sum(file.file_size or 0 for file in planned if file.will_download)
        cache = Path(constants.HF_HUB_CACHE)
        free = shutil.disk_usage(next(p for p in [cache, *cache.parents] if p.exists())).free
        if free < needed:
            raise RunError(
                f"downloading the model needs {needed / 1e9:.1f} GB free in {cache}; there is "
                f"{free / 1e9:.1f} GB"
            )

    @cached_property
    def model_fingerprint(self) -> str:
        """Local folders by the names, sizes and times of the files a run reads; Hub models by their
        commit. Nothing else, so editing a README or moving the folder rebuilds nothing."""
        folder = self.model_folder
        if not source.is_local(self.settings.model.source):
            return digest({"repo": self.settings.model.source, "commit": folder.name})
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
    def shape(self) -> tuple[int, int, str, list[targets.Writer]]:
        model_config = source.read_config(str(self.model_folder))
        sizes = source.sizes(model_config)
        arch = targets.gguf_architecture(model_config)
        if sizes is None or arch is None:
            raise RunError("the model no longer passes candor inspect; run it to see why")
        hidden, layers = sizes
        found = targets.find_targets(source.tensor_shapes(str(self.model_folder)), hidden, layers)
        if not found.ok or found.experts or found.quantized:
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
        """Stop if this machine's GPU can't hold the model for reading residuals: its weights in 16
        bits, and a tenth more for a batch's activations."""
        residuals.protect_memory()
        device, _ = self._device()
        budget = residuals.memory_budget(device)
        if budget is None:
            return
        model = self.settings.model
        where = self._model_without_download
        weights = sum(math.prod(shape) for shape in source.tensor_shapes(where, model.revision).values())
        needed = 2 * weights * 1.1
        if needed > budget:
            raise RunError(
                f"reading residuals needs about {needed / 1e9:.1f} GB of GPU memory and this machine "
                f"allows {budget / 1e9:.1f} GB; run it on a pod"
            )

    # GGUF

    def check_disk(self, copies: int, quantized_needed: bool, extra: int = 0) -> None:
        """Stop unless there is room for this many 16-bit copies of the model, a quantised one, and
        extra bytes."""
        shapes = source.tensor_shapes(str(self.model_folder))
        weights = sum(math.prod(shape) for shape in shapes.values())
        quantized = config.QUANTIZATIONS[self.settings.deployment.quantization] / 8 if quantized_needed else 0
        needed = 1.1 * (weights * (2 * copies + quantized) + extra)
        free = shutil.disk_usage(self.folder).free
        if free < needed:
            raise RunError(
                f"converting needs about {needed / 1e9:.1f} GB free here; there is {free / 1e9:.1f} GB"
            )

    def gguf(self) -> tuple[Path, Path]:
        """The model as an unquantised GGUF of its own float type, every edit target checked, and in
        the deployment quantisation."""
        out = self.path("gguf")
        out.mkdir(exist_ok=True)
        quantization = self.settings.deployment.quantization
        base_type = convert.float_type(self.model_folder)
        base = out / f"model-{base_type}.gguf"
        base_inputs = {"model": self.model_fingerprint, "llama.cpp": LLAMA_CPP_RELEASE}
        quantized = base if quantization == base_type else out / f"model-{quantization}.gguf"
        quantized_inputs = {"base": digest(base_inputs), "quantization": quantization}
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
            _, layers, arch, writers = self.shape
            if problems := convert.check_targets(partial, self.model_folder, writers, arch, layers):
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
        self, model: Path, adapters: Sequence[Path] = (), context_per_slot: int = 8192, parallel: int = 4
    ) -> LlamaServer:
        extra = self.cache_flags()
        self.path("logs").mkdir(exist_ok=True)
        binary = toolchain.binaries() / "llama-server"
        log = self.path("logs", "server.log")
        return LlamaServer(binary, model, log, adapters, parallel, context_per_slot, extra=extra)

    def cache_flags(self) -> list[str]:
        """llama-server's flags for the KV cache type the run deploys with."""
        cache = self.settings.deployment.kv_cache
        flags = ["--cache-type-k", cache, "--cache-type-v", cache]
        if cache not in {"f32", "f16", "bf16"}:
            flags += ["-fa", "on"]  # llama.cpp refuses a quantised V cache without flash attention
        return flags

    # Prompt ids

    def template_kwargs(self, date: str | None = None) -> dict[str, str]:
        date = date or self.settings.model.date
        return {"date_string": date} if date else {}

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
            if OPENS_THINKING.search(rendered):
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

    def thinking_room(self, budget: int | None = None) -> int:
        """The most tokens a thinking model's thinking can take before its answer, the message it is
        made to write at the end of its budget (the deployment's unless given) included."""
        deployment = self.settings.deployment
        if deployment.think_close is None:
            return 0
        budget = budget or deployment.thinking_budget
        assert budget is not None
        # A token holds at least one byte, and the engine allows a few more for the end tag.
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
            "slots": THINKING_SLOTS,
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
        if record.exists():
            made = json.loads(record.read_text())
            if private.exists() and digest(traces := json.loads(private.read_text())) == made["traces"]:
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
        self.report("Thinking with the unedited model, to read KL where its answers start")
        room = max(len(p) for name in KL_SETS for p in ids[name]) + self.thinking_room() + 8
        traces: dict[str, list[list[int] | None]] = {}
        with self.server(quantized, context_per_slot=room, parallel=THINKING_SLOTS) as server:
            thinking = self.thinking(server)
            assert thinking is not None
            end = server.token_of(thinking.end)
            for name in KL_SETS:
                prompts = ids[name]
                done = every_tenth(
                    self.report, f"Thought through {name.replace('_', ' ')} prompts", len(prompts)
                )
                thoughts = server.thoughts(
                    prompts, self.sampling(), seeds(settings.sampling.seed, len(prompts)), thinking, done
                )
                traces[name] = [
                    None if thought is None else prompt + thought + _after(closed, end)
                    for prompt, thought, closed in zip(prompts, thoughts, ids[name + ANSWERING], strict=True)
                ]
                if not any(traces[name]):
                    raise RunError(
                        f"the unedited model's thinking didn't end by itself on any of the {len(prompts)} "
                        f"{name.replace('_', ' ')} prompts, so KL can't be read after it; raise "
                        "deployment.thinking_budget"
                    )
        private.parent.mkdir(parents=True, exist_ok=True)
        building = private.with_name(f"{private.name}.building")
        building.write_text(json.dumps(traces))
        sync(building)
        building.replace(private)
        sync(private)
        made = {
            "traces": digest(traces),
            "kept": {name: sum(trace is not None for trace in traces[name]) for name in KL_SETS},
        }
        record.parent.mkdir(exist_ok=True)
        partial = record.with_name(f"{record.name}.building")
        partial.write_text(json.dumps(made))
        self.commit(partial, record, self._thinking_inputs())
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

    def _check_lengths(self, ids: dict[str, list[list[int]]]) -> None:
        """Prompts the model answers must fit with whole thinking and a whole answer; the rest need room
        for one token."""
        limit, thinking = self.context_limit, self.thinking_room()
        if not limit:
            return
        room = {name: tokens + thinking for name, tokens in self.answer_tokens.items()}
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
        with self.server(quantized, context_per_slot=2048, parallel=2) as server:
            if settings.model.date:
                rendered = server.apply_template(self.chats(["x"])[0], self.template_kwargs(DATE_PROBE))
                if DATE_PROBE not in rendered:
                    raise RunError(
                        "the chat template ignores the pinned date, so prompts would still change daily"
                    )
            ids = {name: server.chat_ids(self.chats(texts[name]), kwargs) for name in PROMPT_SETS}
            if close:
                for name in CLOSED_SETS:
                    ids[name + ANSWERING] = server.chat_ids(self.chats(texts[name]), kwargs, close)
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

    def directions(self, answering: bool = False) -> torch.Tensor:
        """One refusal direction per residual index, from transformers fed the run's exact prompt ids:
        read at the end of each prompt or, with answering, where a thinking model's answer starts after
        an empty thinking block. A thinking model's are read at both in one pass."""
        if answering not in self.positions:
            raise RunError("only a thinking model has directions where its answer starts")
        plans = {position: self._direction_plan(position) for position in self.positions}
        self.inputs["directions"] = digest(plans[False][1])
        if True in plans:
            # The stage comes after the adapters in STAGES, so its code version reaches them only here.
            self.inputs["answering directions"] = digest(self._with_code(*plans[True]))
        if stale := {position: plan for position, plan in plans.items() if not self.fresh(*plan)}:
            self._read_directions(stale)
        return torch.load(plans[answering][0])

    def _read_directions(self, stale: dict[bool, tuple[Path, dict[str, Any]]]) -> None:
        """Read the residuals for each stale position, loading the model once, and save its directions."""
        self.check_memory()
        ids = self.prompt_ids()
        device, dtype = self._device()
        trusted = self.settings.model.trust_remote_code
        self.report(f"Reading residuals on {device}")
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            self.model_folder, dtype=dtype, device_map=device, trust_remote_code=trusted
        ).eval()
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
            return found[1:]
        if 0 <= layer <= layers - 1:
            return directions.after_layer(found, layer).expand(layers, -1)
        raise RunError(f"{name} reads its direction at layer {layer:g}; this model has {layers}")

    # Adapters

    def direction_sets(self, found: torch.Tensor, layers: int) -> dict[str, torch.Tensor]:
        """The candidate directions: each layer's own, or one layer's for every layer.

        found[0] is the embedding output and found[i + 1] the output of layer i.
        """
        low, high = self.settings.search.direction_layers(layers)
        sets = {"per-layer": found[1:]}
        for layer in range(low, high + 1):
            sets[f"layer-{layer}"] = found[layer + 1].expand(layers, -1)
        return sets

    @cached_property
    def target_names(self) -> dict[str, str]:
        """Each edit target's GGUF tensor name, by its name in the model."""
        _, layers, arch, writers = self.shape
        names = {w.name: targets.gguf_name(w, arch, layers) for w in writers}
        if unmapped := [name for name, mapped in names.items() if mapped is None]:
            raise RunError(f"no llama.cpp name for {', '.join(unmapped[:3])}")
        return {name: str(mapped) for name, mapped in names.items()}

    def adapter_inputs(self, answering: bool = False) -> dict[str, Any]:
        """What every adapter set read at one position is built from besides its directions; call after
        directions()."""
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
        _, _, arch, writers = self.shape
        names = self.target_names
        if not self.fresh(folder, inputs):
            files = ModelFiles(self.model_folder)
            model_config = source.read_config(str(self.model_folder))
            row_norm = RowNorm(self.settings.edit.row_norm)
            self.report(f"Writing adapters for {folder.relative_to(self.folder)}")
            partial = folder.with_name(f"{folder.name}.building")
            _remove(partial)
            partial.mkdir(parents=True)
            for writer in writers:
                name = names[writer.name]
                weight = convert.as_converted(files.tensor(writer.name), writer.name, arch, model_config)
                factors = rank_one(weight, set_directions[writer.layer], row_norm)
                write_adapter(partial / f"{name}.gguf", arch, {name: factors})
            self.commit(partial, folder, inputs)
        return {names[w.name]: folder / f"{names[w.name]}.gguf" for w in writers}

    def adapters(self) -> dict[str, dict[str, Path]]:
        """For each direction set, one rank-1 adapter per edit target, named by its GGUF tensor. The names
        of a thinking model's sets read where the answer starts begin with ANSWER_SETS."""
        _, layers, _, _ = self.shape
        root = self.path("adapters")
        root.mkdir(exist_ok=True)
        sets = {}
        for answering in self.positions:
            found, inputs = self.directions(answering), self.adapter_inputs(answering)
            prefix = ANSWER_SETS if answering else ""
            sets |= {
                prefix + name: (made, inputs) for name, made in self.direction_sets(found, layers).items()
            }
        for leftover in root.iterdir():
            if leftover.name.removesuffix(".inputs.json") not in sets:
                _remove(leftover)
        return {name: self.adapter_set(root / name, made, inputs) for name, (made, inputs) in sets.items()}

    def prepare(self) -> dict[str, int]:
        """Every stage up to the search. The last stage pulls in the ones before it, and results still
        matching their settings are reused."""
        with self.locked():
            self.check_remote_code()
            # Checked again before reading residuals; up front only so a model this machine can't
            # read isn't downloaded and converted first. Directions read on a pod skip it.
            if not self.path("directions.pt").exists():
                self.check_memory()
            made = self.adapters()
            ids = json.loads(self.path("prompts.json").read_text())
            kept = self.thinking_traces() if self.settings.deployment.think_close else {}
        # How many prompts of each set KL is read on, after the unedited model's thinking ended by itself.
        thought = {
            f"of {len(kept[name])} {name.replace('_', ' ')} prompts thought through": sum(
                t is not None for t in kept[name]
            )
            for name in kept
        }
        return {
            "edit targets": len(next(iter(made.values()))),
            "direction sets": len(made),
            "adapters": sum(map(len, made.values())),
            "prompts": sum(map(len, ids.values())),
            **thought,
        }
