"""Measure a finished edit on prompts the search never sees, and compare two measurements.

The model answers the held-out prompts and the maths questions in llama-server, from the quantised
GGUF the run deploys. The judge labels each held-out answer, with keyword counts kept beside it, and
the first-token probabilities on the harmless held-out prompts are kept for KL divergence. Answers
and the judge's replies are written only to the run's private folder. A measurement keeps numbers,
one per prompt, so two measurements of the same prompts can be compared prompt by prompt.
"""

import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import heretic, kl
from .directions import after_layer
from .engine import LlamaServer
from .judge import Guard, is_refusal, judged_refusal, read_verdict
from .kernel import Kernel
from .run import HELD_OUT, Run, RunError, digest, load_prompts, sync
from .targets import Writer

SETS = ("harmful_held_out", "harmless_held_out", "maths")
LABELS = {
    "harmful_held_out": "held-out harmful prompts",
    "harmless_held_out": "held-out harmless prompts",
    "maths": "maths questions",
}
# Short enough for progress bars to fit an 80-column terminal.
SHORT_LABELS = {"harmful_held_out": "Harmful", "harmless_held_out": "Harmless", "maths": "Maths"}
DEFAULT_SLOTS = 8
# How the judge's replies become labels. Bump it in any commit that changes that, so saved labels are
# judged again (the answers aren't).
COUNTING = 2
# "Answer: $1,234.50" and the like; failing that, an answer's last number is taken as its answer.
STATED = re.compile(r"answer\W*?\$?\s*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Starts a progress bar with a description and a total, and returns what to call as each item is done.
Progress = Callable[[str, int], Callable[[], None]]


def no_progress(_description: str, _total: int) -> Callable[[], None]:
    return lambda: None


@dataclass(frozen=True)
class Edit:
    """A finished edit: the refusal direction every layer uses (None: each layer its own) and how
    strongly each component is edited at each layer."""

    name: str
    direction_layer: float | None
    kernels: dict[str, Kernel]

    def strengths(self, writers: list[Writer], names: dict[str, str]) -> dict[str, float]:
        """Each edit target's strength, by its GGUF tensor name."""
        return {
            names[w.name]: self.kernels[w.component].at(w.layer) if w.component in self.kernels else 0.0
            for w in writers
        }

    def record(self) -> dict[str, Any]:
        kernels = {component: asdict(kernel) for component, kernel in sorted(self.kernels.items())}
        return {"direction_layer": self.direction_layer, "kernels": kernels}


def heretic_trial(journal_path: Path, number: int, row_norm: str) -> Edit:
    """A trial from a heretic study, to be rebuilt with this run's own directions."""
    try:
        journal = heretic.read_journal(journal_path)
    except (OSError, ValueError, KeyError) as error:
        raise RunError(
            f"candor couldn't read {journal_path} as a heretic study ({type(error).__name__})"
        ) from None
    trial = journal.trials.get(number)
    if trial is None:
        raise RunError(f"{journal_path.name} has no finished trial {number}")
    settings = journal.settings
    problems = []
    if settings.get("row_normalization") != row_norm:
        problems.append(
            f'it normalised rows as "{settings.get("row_normalization")}" and this run uses '
            f'edit.row_norm = "{row_norm}"'
        )
    if settings.get("winsorization_quantile", 1.0) < 1:
        problems.append("it winsorised the residuals, which candor doesn't")
    if settings.get("orthogonalize_direction") is not True:
        problems.append("it doesn't record orthogonalising its directions, as candor does")
    if problems:
        raise RunError(f"trial {number} can't be rebuilt like for like: " + "; ".join(problems))
    return Edit(f"heretic-{number}", trial.direction_layer, trial.kernels)


def edit_adapters(run: Run, edit: Edit, folder: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    """An edit's own adapters, one per edit target along the direction it reads, written to folder
    unless already there, by GGUF tensor name; and what they are built from."""
    if not run.path("directions.pt").exists():
        run.check_memory()
    found = run.directions()
    _, layers, _, _ = run.shape
    layer = edit.direction_layer
    if layer is None:
        directions = found[1:]
    elif 0 <= layer <= layers - 1:
        directions = after_layer(found, layer).expand(layers, -1)
    else:
        raise RunError(f"{edit.name} reads its direction at layer {layer:g}; this model has {layers}")
    inputs = {**run.adapter_inputs(), "direction_layer": layer}
    return run.adapter_set(folder, directions, inputs), inputs


def gold_number(answer: str) -> float:
    """The number a GSM8K answer ends with, after '#### '."""
    found = NUMBER.findall(answer.rsplit("####", 1)[1]) if "####" in answer else []
    if not found:
        raise RunError("a maths answer has no '#### ' and number at its end, as GSM8K's answers do")
    return float(found[0].replace(",", ""))


def answered_number(answer: str, finished: bool = True) -> float | None:
    """The stated answer, or the last number of an answer the model finished. An answer cut off before
    stating one has none, since its last number is only part of the working."""
    found = STATED.findall(answer) or (NUMBER.findall(answer) if finished else [])
    return float(found[-1].replace(",", "")) if found else None


def is_correct(answer: str, gold: float, finished: bool = True) -> bool:
    value = answered_number(answer, finished)
    return value is not None and math.isclose(value, gold, rel_tol=0, abs_tol=1e-6)


def mcnemar(only_a: int, only_b: int) -> float:
    """The exact two-sided McNemar p-value: how likely a split this uneven of the prompts on which two
    measurements differ would be if neither were really better."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


class Measurement:
    """Each set is answered and saved, then its answers are judged and the labels saved apart, so a
    stopped measurement resumes from the last set saved, and a new judge re-judges saved answers."""

    def __init__(
        self,
        run: Run,
        edit: Edit | None,
        name: str,
        slots: int = DEFAULT_SLOTS,
        limit: int | None = None,
        progress: Progress = no_progress,
    ):
        self.run = run
        self.edit = edit
        self.name = name
        self.slots = slots
        self.limit = limit
        self.progress = progress
        self.folder = run.path("measurements", name)
        self.guard: Guard | None = None
        self.texts: dict[str, list[str]] | None = None
        # How many prompts each set holds in full, whatever the limit.
        self.sizes: dict[str, int] = {}

    def measure(self) -> dict[str, dict[str, Any]]:
        """Each set's results: what reading the answers gave, with the judge's labels for held-out sets."""
        run = self.run
        with run.locked():
            run.check_remote_code()
            _, quantized = run.gguf()
            all_ids = run.prompt_ids()
            self.sizes = {name: len(all_ids[name]) for name in SETS}
            ids = {name: all_ids[name][: self.limit] for name in SETS}
            self.folder.mkdir(parents=True, exist_ok=True)
            scales, edit = self._adapters() if self.edit else ({}, None)
            answered, todo = {}, []
            for name in SETS:
                inputs = self._answer_inputs(name, ids[name], edit)
                reused = self._reused(name, inputs)
                if reused is None:
                    todo.append((name, inputs))
                else:
                    answered[name] = reused
            if todo:
                room = max(len(p) + run.answer_tokens[name] for name, _ in todo for p in ids[name])
                server = run.server(quantized, list(scales), context_per_slot=room + 8, parallel=self.slots)
                with server:
                    for name, inputs in todo:
                        prompts = self._texts(name)
                        run.check_prompts(server, name, prompts, ids[name])
                        answered[name] = self._answer(server, name, ids[name], prompts, scales, inputs)
            try:
                judged = {name: self._judged(name, answered[name]) for name in HELD_OUT}
            finally:
                if self.guard:
                    self.guard.close()
        return {name: answered[name] | judged.get(name, {}) for name in SETS}

    def _adapters(self) -> tuple[dict[Path, float], dict[str, Any]]:
        """The edit's own adapters, one per edit target, and the strength each is applied at."""
        assert self.edit is not None
        run = self.run
        files, inputs = edit_adapters(run, self.edit, self.folder / "adapters")
        strengths = self.edit.strengths(run.shape[3], run.target_names)
        scales = {files[name]: strength for name, strength in strengths.items() if strength}
        return scales, {"adapters": digest(inputs), **self.edit.record(), "strengths": strengths}

    def _answer_inputs(self, name: str, ids: list[list[int]], edit: dict[str, Any] | None) -> dict[str, Any]:
        """What a set's answers depend on. Everything but the edit and the slots must match for two
        measurements to be compared."""
        settings = self.run.settings
        paired: dict[str, Any] = {
            "model": self.run.inputs["quantized"],
            "kv_cache": settings.deployment.kv_cache,
            "ids": digest(ids),
            "answer_tokens": self.run.answer_tokens[name],
        }
        if name == "harmless_held_out":
            paired["kl_tokens"] = settings.scoring.kl_tokens
        return {"paired": paired, "edit": edit, "slots": self.slots}

    def _private(self, name: str) -> Path:
        return self.run.path("private", "measurements", self.name, name)

    def _write_private(self, name: str, content: dict[str, list[str]]) -> None:
        path = self._private(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".building")
        partial.write_text(json.dumps(content))
        sync(partial)
        partial.replace(path)

    def _reused(self, name: str, inputs: dict[str, Any]) -> dict[str, Any] | None:
        """A set's saved results, if they were built from these inputs and their answers are still
        there to judge."""
        path = self.folder / f"{name}.json"
        if not self.run.fresh(path, inputs):
            return None
        result = json.loads(path.read_text())
        answers = self._private(f"{name}.json")
        if not answers.exists() or digest(json.loads(answers.read_text())["answers"]) != result["answers"]:
            self.run.report(f"Answering the {LABELS[name]} again: their saved answers are missing or changed")
            return None
        return result

    def _answer(
        self,
        server: LlamaServer,
        name: str,
        ids: list[list[int]],
        prompts: list[str],
        scales: dict[Path, float],
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        settings = self.run.settings
        started = time.perf_counter()
        answers = server.generate(
            ids,
            self.run.answer_tokens[name],
            scales,
            self.progress(f"{SHORT_LABELS[name]}: answering", len(ids)),
        )
        texts = [a.text for a in answers]
        result: dict[str, Any] = {
            "finished": [a.finished for a in answers],
            "tokens": [a.tokens for a in answers],
            "answer_seconds": time.perf_counter() - started,
        }
        if name in HELD_OUT:
            result["keyword_refused"] = [is_refusal(text) for text in texts]
            result["prompts"] = digest(prompts)
        if name == "maths":
            spec = settings.prompts.maths
            gold = [gold_number(a) for a in load_prompts(spec.model_copy(update={"column": "answer"}))]
            gold = gold[: len(answers)] if self.limit else gold
            result["correct"] = [
                is_correct(a.text, g, a.finished) for a, g in zip(answers, gold, strict=True)
            ]
        if name == "harmless_held_out":
            done = self.progress(f"{SHORT_LABELS[name]}: first tokens", len(ids))
            top = server.top_logprobs(ids, settings.scoring.kl_tokens, scales, done)
            result["first_token"] = [[[t, p] for t, p in sorted(found.items())] for found in top]
        result |= {
            "seconds": time.perf_counter() - started,
            "slots": self.slots,
            "paired": digest(inputs["paired"]),
            "answers": digest(texts),
        }
        self._write_private(f"{name}.json", {"answers": texts})
        path = self.folder / f"{name}.json"
        partial = path.with_name(path.name + ".building")
        partial.write_text(json.dumps(result))
        self.run.commit(partial, path, inputs)
        return result

    def _texts(self, name: str) -> list[str]:
        """A set's prompts as the model is asked them, read from their datasets once, when first needed."""
        if self.texts is None:
            self.texts = self.run.prompt_texts()[0]
        return self.texts[name][: self.limit]

    def _judge_inputs(self, answered: dict[str, Any]) -> dict[str, Any]:
        judge = self.run.settings.measure
        return {
            "answers": answered["answers"],
            "judge": [judge.judge, judge.judge_revision],
            "counting": COUNTING,
        }

    def _judged(self, name: str, answered: dict[str, Any]) -> dict[str, Any]:
        """The judge's labels for a held-out set's saved answers."""
        settings = self.run.settings
        inputs = self._judge_inputs(answered)
        path = self.folder / f"{name}.judged.json"
        if self.run.fresh(path, inputs):
            return json.loads(path.read_text())
        answers = json.loads(self._private(f"{name}.json").read_text())["answers"]
        prompts = self._texts(name)
        if digest(prompts) != answered["prompts"]:
            raise RunError(f"the {LABELS[name]} have changed since they were answered; run candor run first")
        if self.guard is None:
            self.guard = Guard(settings.measure.judge, settings.measure.judge_revision)
        started = time.perf_counter()
        done = self.progress(f"{SHORT_LABELS[name]}: judging", len(answers))
        replies = self.guard.replies(prompts, answers, done=done)
        verdicts = [read_verdict(reply) for reply in replies]
        result = {
            "judge_refused": [judged_refusal(a, v) for a, v in zip(answers, verdicts, strict=True)],
            "judge_safety": [v.safety for v in verdicts],
            "judge": digest(inputs["judge"]),
            "judge_seconds": time.perf_counter() - started,
        }
        self._write_private(f"{name}.judged.json", {"judge_replies": replies})
        partial = path.with_name(path.name + ".building")
        partial.write_text(json.dumps(result))
        self.run.commit(partial, path, inputs)
        return result


# What a measurement counts, each as a yes, no or unknown per prompt.
# Harmful and Harmless are the held-out sets; labels stay short so tables fit 80 columns.
COUNTS: list[tuple[str, str, Callable[[dict[str, Any]], list[bool | None]]]] = [
    ("Harmful refused, judge", "harmful_held_out", lambda r: r["judge_refused"]),
    ("Harmful refused, keywords", "harmful_held_out", lambda r: r["keyword_refused"]),
    (
        "Harmful answered unsafely",
        "harmful_held_out",
        lambda r: [None if safety is None else safety == "Unsafe" for safety in r["judge_safety"]],
    ),
    ("Harmful cut off", "harmful_held_out", lambda r: [not f for f in r["finished"]]),
    ("Harmless refused, judge", "harmless_held_out", lambda r: r["judge_refused"]),
    ("Harmless refused, keywords", "harmless_held_out", lambda r: r["keyword_refused"]),
    ("Maths correct", "maths", lambda r: r["correct"]),
    ("Maths cut off", "maths", lambda r: [not f for f in r["finished"]]),
]


def _count(values: list[bool | None]) -> str:
    known = [v for v in values if v is not None]
    unknown = len(values) - len(known)
    return f"{sum(known)}/{len(known)}" + (f" ({unknown} unlabelled)" if unknown else "")


def summary(results: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    """What one measurement found, as rows of a table."""
    rows = [(label, _count(values(results[name]))) for label, name, values in COUNTS]
    for name in HELD_OUT:
        result = results[name]
        pairs = zip(result["judge_refused"], result["keyword_refused"], strict=True)
        disagree = [judged != counted for judged, counted in pairs if judged is not None]
        rows.append((f"{SHORT_LABELS[name]}, keywords disagree", _count(disagree)))
    for name in SETS:
        result = results[name]
        rate = sum(result["tokens"]) / result["answer_seconds"] if result["answer_seconds"] else 0
        rows.append((f"{SHORT_LABELS[name]}, time", f"{seconds(result):.0f} s, {rate:.0f} tokens/s"))
    return rows


def seconds(result: dict[str, Any]) -> float:
    """How long a set took to answer and judge."""
    return result["seconds"] + result.get("judge_seconds", 0.0)


def load(run: Run, name: str) -> dict[str, dict[str, Any]]:
    """A measurement's results as candor measure returns them."""
    folder = run.path("measurements", name)
    files = [f"{s}.json" for s in SETS] + [f"{s}.judged.json" for s in HELD_OUT]
    if missing := [file for file in files if not (folder / file).exists()]:
        raise RunError(
            f"the measurement {name} is missing {', '.join(missing)}; run candor measure to finish it"
        )
    results = {s: json.loads((folder / f"{s}.json").read_text()) for s in SETS}
    for s in HELD_OUT:
        results[s] |= json.loads((folder / f"{s}.judged.json").read_text())
    return results


def compare(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> list[tuple[str, ...]]:
    """Two measurements of the same prompts, prompt by prompt: each count for both, the change, how
    many prompts went each way, and the McNemar p-value of that split; then the KL divergence of b's
    first tokens from a's."""
    for name in SETS:
        if a[name]["paired"] != b[name]["paired"]:
            raise RunError(
                f"the two measurements of the {LABELS[name]} differ in more than the edit and slots: the "
                "model file, the prompts or a setting changed between them"
            )
        if name in HELD_OUT and a[name]["judge"] != b[name]["judge"]:
            raise RunError(f"the {LABELS[name]} were judged by different judges")
    rows: list[tuple[str, ...]] = []
    for label, name, values in COUNTS:
        pairs = [
            (bool(x), bool(y))
            for x, y in zip(values(a[name]), values(b[name]), strict=True)
            if x is not None and y is not None
        ]
        a_yes, b_yes = sum(x for x, _ in pairs), sum(y for _, y in pairs)
        only_a = sum(x and not y for x, y in pairs)
        only_b = sum(y and not x for x, y in pairs)
        rows.append(
            (
                label,
                f"{a_yes}/{len(pairs)}",
                f"{b_yes}/{len(pairs)}",
                f"{b_yes - a_yes:+d}",
                f"{only_a} / {only_b}",
                f"{mcnemar(only_a, only_b):.3g}",
            )
        )
    first_a, first_b = a["harmless_held_out"]["first_token"], b["harmless_held_out"]["first_token"]
    divergences = [
        kl.from_top({int(t): p for t, p in x}, {int(t): p for t, p in y})
        for x, y in zip(first_a, first_b, strict=True)
    ]
    mean = sum(divergences) / len(divergences)
    rows.append(("Harmless KL from A", "", f"{mean:.4f}", "", "", ""))
    return rows
