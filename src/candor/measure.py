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
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import export, heretic, kl
from .edit import FULL_RANK
from .engine import Answer, LlamaServer, Thinking
from .files import write_atomic
from .judge import Guard, is_refusal, judged_refusal, read_verdict
from .kernel import Kernel
from .run import (
    ANSWERING,
    HELD_OUT,
    STAGES,
    Run,
    RunError,
    answering,
    digest,
    load_prompts,
    read_record,
    record_of,
    seeds,
)
from .targets import Writer

SETS = ("harmful_held_out", "harmless_held_out", "maths")
LABELS = {
    "harmful_held_out": "held-out harmful prompts",
    "harmless_held_out": "held-out harmless prompts",
    "maths": "maths questions",
}
# Short enough for progress bars to fit an 80-column terminal.
SHORT_LABELS = {"harmful_held_out": "Harmful", "harmless_held_out": "Harmless", "maths": "Maths"}
# How the judge's replies become labels. Bump it in any commit that changes that, so saved labels are
# judged again (the answers aren't).
COUNTING = 2
# The normal quantile a two-sided 95% interval reaches.
Z95 = statistics.NormalDist().inv_cdf(0.975)
# "Answer: 1,234.50" and the like, once dollar signs and boxes are left out; failing that, an answer's
# last number is taken as its answer.
STATED = re.compile(r"answer\W*?(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Starts a progress bar with a description and a total, and returns what to call as each item is done.
Progress = Callable[[str, int], Callable[[], None]]


def no_progress(_description: str, _total: int) -> Callable[[], None]:
    return lambda: None


@dataclass(frozen=True)
class Edit:
    """A finished edit: the refusal direction every layer uses (None: each layer its own), read where a
    thinking model's answer starts rather than at the end of the prompt when answering, and how strongly
    each component is edited at each layer."""

    name: str
    direction_layer: float | None
    kernels: dict[str, Kernel]
    answering: bool = False

    def strength(self, writer: Writer) -> float:
        kernel = self.kernels.get(writer.component)
        return kernel.at(writer.layer) if kernel else 0.0

    def strengths(self, writers: list[Writer], names: dict[str, str]) -> dict[str, float]:
        """Each edit target's strength, by its GGUF tensor name."""
        return {names[w.name]: self.strength(w) for w in writers}

    def record(self) -> dict[str, Any]:
        kernels = {component: asdict(kernel) for component, kernel in sorted(self.kernels.items())}
        # Only when set, so the records of edits read at the end of the prompt are as they were.
        position = {"answering": True} if self.answering else {}
        return {"direction_layer": self.direction_layer, "kernels": kernels, **position}


def heretic_trial(journal_path: Path, number: int, row_norm: str, think_close: str | None = None) -> Edit:
    """A trial from a heretic study, to be rebuilt with this run's own directions. A trial that read its
    directions after a response prefix that closes the thinking as this run's think_close does is rebuilt
    with the directions read where the answer starts. candor reads every layer's output before the
    model's final norm, and heretic its last one after it, so a per-layer trial's last layer is rebuilt
    with a direction heretic didn't read."""
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
    if (
        settings.get("row_normalization") == "full"
        and settings.get("full_normalization_lora_rank", 3) != FULL_RANK
    ):
        rank = settings["full_normalization_lora_rank"]
        problems.append(f'it built "full" edits at rank {rank}, and candor at {FULL_RANK}')
    if settings.get("winsorization_quantile", 1.0) < 1:
        problems.append("it winsorised the residuals, which candor doesn't")
    if settings.get("orthogonalize_direction") is not True:
        problems.append("it doesn't record orthogonalising its directions, as candor does")
    prefix = settings.get("response_prefix")
    if prefix and prefix != think_close:
        problems.append(
            f"it read its directions after a response prefix, {prefix!r}, that isn't the run's "
            "deployment.think_close; export heretic's own save of the trial with candor export --outside "
            "instead"
        )
    if problems:
        raise RunError(f"trial {number} can't be rebuilt like for like: " + "; ".join(problems))
    return Edit(f"heretic-{number}", trial.direction_layer, trial.kernels, answering=bool(prefix))


def edit_adapters(run: Run, edit: Edit, folder: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    """An edit's own adapters, one per edit target along the direction it reads, written to folder
    unless already there, by GGUF tensor name; and what they are built from."""
    directions = run.edit_directions(edit.name, edit.direction_layer, edit.answering)
    inputs = {**run.adapter_inputs(edit.answering), "direction_layer": edit.direction_layer}
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
    # "Answer: \boxed{18}" states 18, and "-$5" is -5.
    answer = answer.replace("\\boxed{", "").replace("$", "")
    found = STATED.findall(answer) or (NUMBER.findall(answer) if finished else [])
    return float(found[-1].replace(",", "")) if found else None


def is_correct(answer: str, gold: float, finished: bool = True) -> bool:
    value = answered_number(answer, finished)
    return value is not None and math.isclose(value, gold, rel_tol=0, abs_tol=1e-6)


def change_interval(only_a: int, only_b: int, pairs: int) -> tuple[float, float]:
    """A 95% interval for how many more of pairs paired prompts B says yes to than A, from how many only
    A and only B say yes to: Bonett and Price's adjusted Wald interval, which keeps a width when none
    differ."""
    a, b = (only_a + 1) / (pairs + 2), (only_b + 1) / (pairs + 2)
    half = Z95 * math.sqrt((a + b - (b - a) ** 2) / (pairs + 2))
    return max(b - a - half, -1) * pairs, min(b - a + half, 1) * pairs


def mean_interval(values: list[float], draws: int = 2000) -> tuple[float, float] | None:
    """A 95% interval for the mean of values over the prompts they were read on, within their range: a
    studentised bootstrap, drawn the same way every time, which holds better than plain percentiles
    where a few prompts carry much of the mean, as with KL. None for fewer than 3 values."""
    found = np.asarray(values)
    if len(found) < 3:
        return None
    mean, error = found.mean(), found.std(ddof=1) / math.sqrt(len(found))
    if error == 0:
        return float(mean), float(mean)
    drawn = found[np.random.default_rng(0).integers(0, len(found), (draws, len(found)))]
    errors = drawn.std(axis=1, ddof=1) / math.sqrt(len(found))
    # A draw of one value over and over has no spread to scale by.
    spread = errors > 0
    low, high = np.percentile((drawn.mean(axis=1)[spread] - mean) / errors[spread], [2.5, 97.5])
    return float(max(mean - high * error, found.min())), float(min(mean - low * error, found.max()))


def mcnemar(only_a: int, only_b: int) -> float:
    """The exact two-sided McNemar p-value: how likely a split this uneven of the prompts on which two
    measurements differ would be if neither were really better."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


class Measurement:
    """Each set's answers are saved as they come, then judged and the labels saved apart, so a stopped
    measurement resumes from the last answer saved, and a new judge re-judges saved answers."""

    def __init__(
        self,
        run: Run,
        edit: Edit | None,
        name: str,
        slots: int | None = None,
        limit: int | None = None,
        progress: Progress = no_progress,
        exported: bool = False,
        seed: int | None = None,
        outside: str | None = None,
    ):
        """With exported, the edit's exported GGUF is measured, rather than the run's model with the
        edit's adapters; with outside, the export of that name of a model edited elsewhere. slots
        replaces slots.measure. seed replaces the run's sampling seed, to see how much sampling alone
        moves the counts."""
        if exported and edit is None:
            raise ValueError("only an edit has an export to measure")
        if outside is not None and (edit is not None or exported):
            raise ValueError("a model edited elsewhere is measured on its own")
        self.run = run
        self.edit = edit
        self.exported = exported
        self.outside = outside
        self.name = name
        self.slots = run.settings.slots.measure if slots is None else slots
        self.limit = limit
        self.seed = run.settings.sampling.seed if seed is None else seed
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
            model, scales, edit = quantized, {}, None
            if self.outside is not None:
                model, made_from = export.outside_export(run, self.outside)
                edit = {"exported": digest(made_from)}
            elif self.exported:
                assert self.edit is not None
                model, made_from = export.exported(run, self.edit)
                edit = {"exported": digest(made_from)}
            # For a thinking model, KL is read after the unedited model's own thinking, which every
            # measurement compared with this one must share.
            kl_ids = run.kl_ids(all_ids, "harmless_held_out", self.limit)
            if not kl_ids:
                raise RunError(
                    f"the unedited model's thinking ended by itself on none of the first {self.limit} "
                    "held-out harmless prompts, so KL can't be read after it; measure more of them"
                )
            # Only once nothing before measuring can fail.
            self._keep_older()
            self.folder.mkdir(parents=True, exist_ok=True)
            if self.edit and not self.exported:
                scales, edit = self._adapters()
            answered, todo = {}, []
            for name in SETS:
                inputs = self._answer_inputs(name, ids[name], edit, kl_ids)
                reused = self._reused(name, inputs)
                if reused is None:
                    todo.append((name, inputs))
                else:
                    answered[name] = reused
            # A stopped measurement then leaves these sets missing, never another edit's answers or
            # labels beside its own.
            for name, _ in todo:
                run.discard(self.folder / f"{name}.json")
                run.discard(self.folder / f"{name}.judged.json")
            if todo:
                thinking_room = run.thinking_room()
                room = max(
                    len(p) + run.answer_tokens[name] + thinking_room for name, _ in todo for p in ids[name]
                )
                server = run.server(
                    model,
                    list(scales),
                    context_per_slot=room + 8,
                    parallel=self.slots,
                    slots_setting="--slots, or slots.measure in candor.toml",
                )
                with server:
                    thinking = run.thinking(server)
                    for name, inputs in todo:
                        prompts = self._texts(name)
                        run.check_prompts(server, name, prompts, ids[name])
                        firsts = {}
                        if name == "harmless_held_out":
                            firsts["first_token"] = kl_ids
                            if thinking:
                                # Also where the thinking starts, and where the answer starts after an
                                # empty thinking block, to see which of the three tells edits apart.
                                firsts["first_token_thinking"] = ids[name]
                                firsts["first_token_empty"] = all_ids[name + ANSWERING][: self.limit]
                        answered[name] = self._answer(
                            server, name, ids[name], firsts, prompts, scales, thinking, inputs
                        )
            try:
                judged = {name: self._judged(name, answered[name]) for name in HELD_OUT}
            finally:
                if self.guard:
                    self.guard.close()
        return {name: answered[name] | judged.get(name, {}) for name in SETS}

    def _keep_older(self) -> None:
        """Keep what older measuring code measured under this name, which no measurement made now can
        be compared with, under a name of its own rather than measure over it."""
        version = measured_by(self.folder)
        if version in (None, STAGES["measurements"]):
            return
        kept = f"{self.name}-measuring-{version}"
        folders = [(self.run.path(*where, self.name), self.run.path(*where, kept)) for where in PLACES]
        if any(to.exists() for _, to in folders):
            raise RunError(f"{self.name} was measured by older code and {kept} is taken; move one aside")
        for at, to in folders:
            if at.exists():
                at.rename(to)
        self.run.report(f"Keeping {self.name}, measured by older code, as {kept}")

    def _adapters(self) -> tuple[dict[Path, float], dict[str, Any]]:
        """The edit's own adapters, one per edit target, and the strength each is applied at; or, for an
        edit that isn't linear in strength, one adapter built at its strengths, applied at 1."""
        assert self.edit is not None
        run = self.run
        strengths = self.edit.strengths(run.shape[3], run.target_names)
        if not run.linear:
            path, inputs = run.edit_adapter(self.folder / "adapters", self.edit)
            return {path: 1.0}, {"adapters": digest(inputs), **self.edit.record(), "strengths": strengths}
        files, inputs = edit_adapters(run, self.edit, self.folder / "adapters")
        scales = {files[name]: strength for name, strength in strengths.items() if strength}
        return scales, {"adapters": digest(inputs), **self.edit.record(), "strengths": strengths}

    def _answer_inputs(
        self, name: str, ids: list[list[int]], edit: dict[str, Any] | None, kl_ids: list[list[int]]
    ) -> dict[str, Any]:
        """What a set's answers depend on. Everything but the edit and the slots must match for two
        measurements to be compared. An export counts as its edit: it is the run's model, quantised
        the same way, with the edit written in."""
        run = self.run
        deployment = run.settings.deployment
        paired: dict[str, Any] = {
            "model": run.inputs["quantized"],
            "kv_cache": deployment.kv_cache,
            "ids": digest(ids),
            "answer_tokens": run.answer_tokens[name],
            **answering(run.settings),
            # Measurements by older code drew or read their answers otherwise.
            "measuring": STAGES["measurements"],
        }
        if name == "harmless_held_out":
            paired["kl_tokens"] = run.settings.scoring.kl_tokens
            if deployment.think_close:
                paired["kl_ids"] = digest(kl_ids)
        inputs = {"paired": paired, "edit": edit, "slots": self.slots}
        # The seed decides which random draws sampled answers meet. Two measurements with different
        # seeds are still compared, prompt by prompt, to see how much sampling alone moves the counts.
        if "sampling" in paired:
            inputs["seed"] = self.seed
        return inputs

    def _private(self, name: str) -> Path:
        return self.run.path("private", "measurements", self.name, name)

    def _write_private(self, name: str, content: dict[str, list[str]]) -> None:
        path = self._private(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, json.dumps(content))

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
        firsts: dict[str, list[list[int]]],
        prompts: list[str],
        scales: dict[Path, float],
        thinking: Thinking | None,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """A set's answers and what reading them gives, committed once every prompt is answered. What
        needs no answers is done first, so failing there loses none; answers are saved as they come,
        so a measurement stopped part way answers only the prompts left."""
        run = self.run
        settings = run.settings
        started = time.perf_counter()
        if name == "maths":
            spec = settings.prompts.maths
            gold = [gold_number(a) for a in load_prompts(spec.model_copy(update={"column": "answer"}))]
            gold = gold[: len(ids)] if self.limit else gold
        read: dict[str, Any] = {}
        for key, prompts_read in firsts.items():
            done = self.progress(f"{SHORT_LABELS[name]}: first tokens", len(prompts_read))
            top = server.top_logprobs(prompts_read, settings.scoring.kl_tokens, scales, done)
            read[key] = [[[t, p] for t, p in sorted(found.items())] for found in top]
        answers, before, answering = self._answered(server, name, ids, scales, thinking, inputs)
        # Only what follows a thinking model's thinking is its answer, so only that is judged.
        texts = [a.text for a in answers]
        result: dict[str, Any] = {
            "finished": [a.finished for a in answers],
            "tokens": [a.tokens for a in answers],
            "answer_seconds": before + answering,
            "seed": self.seed,
            "machine": run.machine(),
        }
        if thinking:
            result["thinking"] = [a.thinking for a in answers]
            result["forced"] = [a.forced for a in answers]
        if name in HELD_OUT:
            result["keyword_refused"] = [is_refusal(text) for text in texts]
            result["prompts"] = digest(prompts)
        if name == "maths":
            result["correct"] = [
                is_correct(a.text, g, a.finished) for a, g in zip(answers, gold, strict=True)
            ]
        result |= read | {
            "seconds": before + time.perf_counter() - started,
            "slots": self.slots,
            "paired": digest(inputs["paired"]),
            "answers": digest(texts),
        }
        self._write_private(f"{name}.json", {"answers": texts})
        path = self.folder / f"{name}.json"
        partial = path.with_name(path.name + ".building")
        partial.write_text(json.dumps(result))
        self.run.commit(partial, path, inputs)
        self._private(f"{name}.answering.jsonl").unlink(missing_ok=True)
        return result

    def _answered(
        self,
        server: LlamaServer,
        name: str,
        ids: list[list[int]],
        scales: dict[Path, float],
        thinking: Thinking | None,
        inputs: dict[str, Any],
    ) -> tuple[list[Answer], float, float]:
        """Every prompt's answer, the seconds earlier sessions spent answering it and the seconds this
        one did. Each answer is saved as it comes, to the private folder, with the set's record; those a
        measurement stopped part way saved from the same inputs are kept, and only the rest are asked
        for, each with its own seed, as it would have been."""
        run = self.run
        journal = self._private(f"{name}.answering.jsonl")
        record = {"record": run.record_for(self.folder / f"{name}.json", inputs)}
        saved = _journal(journal)
        kept: dict[int, dict[str, Any]] = {e["i"]: e for e in saved[1:]} if saved[:1] == [record] else {}
        before = max((e["at"] for e in kept.values()), default=0.0)
        todo = [i for i in range(len(ids)) if i not in kept]
        if kept:
            run.report(f"Answering the {len(todo)} {LABELS[name]} a stopped measurement left of {len(ids)}")
        prompt_seeds = seeds(self.seed, len(ids))
        journal.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(journal, "".join(json.dumps(line) + "\n" for line in [record, *kept.values()]))
        with journal.open("a") as out:
            started = time.perf_counter()

            def save(place: int, answer: Answer) -> None:
                entry = {
                    "i": todo[place],
                    "answer": asdict(answer),
                    "at": before + time.perf_counter() - started,
                }
                out.write(json.dumps(entry) + "\n")
                out.flush()

            new = server.generate(
                [ids[i] for i in todo],
                run.answer_tokens[name],
                scales,
                self.progress(f"{SHORT_LABELS[name]}: answering", len(todo)),
                run.sampling(),
                [prompt_seeds[i] for i in todo],
                thinking,
                each=save,
            )
            answering = time.perf_counter() - started
        found = {i: Answer(**entry["answer"]) for i, entry in kept.items()} | dict(
            zip(todo, new, strict=True)
        )
        return [found[i] for i in range(len(ids))], before, answering

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
            "judge_machine": self.guard.machine,
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
# What a thinking model's measurement also counts: answers whose thinking reached its budget, so the
# model was made to stop thinking and answer.
THINKING_COUNTS: list[tuple[str, str, Callable[[dict[str, Any]], list[bool | None]]]] = [
    (f"{SHORT_LABELS[name]}, thinking cut", name, lambda r: r["forced"]) for name in SETS
]


def _counts(
    *measurements: dict[str, dict[str, Any]],
) -> list[tuple[str, str, Callable[[dict[str, Any]], Any]]]:
    """What every one of the measurements counts."""
    thinking = all("forced" in results[name] for results in measurements for name in SETS)
    return COUNTS + (THINKING_COUNTS if thinking else [])


def _count(values: list[bool | None]) -> str:
    known = [v for v in values if v is not None]
    unknown = len(values) - len(known)
    return f"{sum(known)}/{len(known)}" + (f" ({unknown} unlabelled)" if unknown else "")


def summary(results: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    """What one measurement found, as rows of a table."""
    rows = [(label, _count(values(results[name]))) for label, name, values in _counts(results)]
    for name in HELD_OUT:
        result = results[name]
        pairs = zip(result["judge_refused"], result["keyword_refused"], strict=True)
        disagree = [judged != counted for judged, counted in pairs if judged is not None]
        rows.append((f"{SHORT_LABELS[name]}, keywords disagree", _count(disagree)))
    for name in SETS:
        result = results[name]
        if "thinking" in result:
            thought = result["thinking"]
            middle = statistics.median(thought)
            rows.append((f"{SHORT_LABELS[name]}, thinking tokens", f"median {middle:g}, most {max(thought)}"))
    for name in SETS:
        result = results[name]
        written = sum(result["tokens"]) + sum(result.get("thinking", []))
        rate = written / result["answer_seconds"] if result["answer_seconds"] else 0
        rows.append((f"{SHORT_LABELS[name]}, time", f"{seconds(result):.0f} s, {rate:.0f} tokens/s"))
    return rows


def _kl(a: list[list[list[float]]], b: list[list[list[float]]]) -> list[float]:
    """The KL divergence of b's first tokens from a's at each prompt, as a measurement saves them."""
    return [
        kl.from_top({int(t): p for t, p in x}, {int(t): p for t, p in y}) for x, y in zip(a, b, strict=True)
    ]


def _journal(path: Path) -> list[dict[str, Any]]:
    """The lines a set's answers were saved in as they came, the first its record, leaving out a last one
    a stop cut short."""
    lines = path.read_text().splitlines() if path.exists() else []
    found = []
    for line in lines:
        try:
            found.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return found


# Where a measurement's numbers and its answers are kept, by its name.
PLACES = (("measurements",), ("private", "measurements"))


def measured_by(folder: Path) -> int | None:
    """The version of the measuring code that made what a measurement's folder holds, if anything."""
    for record in sorted(folder.glob("*.inputs.json")):
        if (version := (read_record(record) or {}).get("code", {}).get("measurements")) is not None:
            return version
    return None


def seconds(result: dict[str, Any]) -> float:
    """How long a set took to answer and judge."""
    return result["seconds"] + result.get("judge_seconds", 0.0)


# What a finished measurement holds: each set's results and the held-out sets' labels.
FINISHED = [f"{s}.json" for s in SETS] + [f"{s}.judged.json" for s in HELD_OUT]


def load(run: Run, name: str) -> dict[str, dict[str, Any]]:
    """A measurement's results as candor measure returns them."""
    folder = run.path("measurements", name)
    if missing := [file for file in FINISHED if not (folder / file).exists()]:
        raise RunError(
            f"the measurement {name} is missing {', '.join(missing)}; run candor measure to finish it"
        )
    results = {s: json.loads((folder / f"{s}.json").read_text()) for s in SETS}
    for s in HELD_OUT:
        judged = folder / f"{s}.judged.json"
        record = read_record(record_of(judged)) or {}
        if record.get("answers") != results[s]["answers"] or record.get("counting") != COUNTING:
            raise RunError(
                f"the measurement {name}'s labels for the {LABELS[s]} aren't of its answers as candor "
                "counts them now; run candor measure to finish it"
            )
        results[s] |= json.loads(judged.read_text())
    return results


def compare(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> list[tuple[str, ...]]:
    """Two measurements of the same prompts, prompt by prompt: each count for both, the change, how
    many prompts went each way, the McNemar p-value of that split and a 95% interval for the change;
    then the KL divergence of b's first tokens from a's, with a 95% interval for it, unless they were
    read on different backends; and the seeds, slots and machines where they differ."""
    for name in SETS:
        if a[name]["paired"] != b[name]["paired"]:
            raise RunError(
                f"the two measurements of the {LABELS[name]} differ in more than the edit, seed, slots and "
                "machine: the model file, the prompts, a setting or candor's measuring changed between them"
            )
        if name in HELD_OUT and a[name]["judge"] != b[name]["judge"]:
            raise RunError(f"the {LABELS[name]} were judged by different judges")
    rows: list[tuple[str, ...]] = []
    for label, name, values in _counts(a, b):
        pairs = [
            (bool(x), bool(y))
            for x, y in zip(values(a[name]), values(b[name]), strict=True)
            if x is not None and y is not None
        ]
        a_yes, b_yes = sum(x for x, _ in pairs), sum(y for _, y in pairs)
        only_a = sum(x and not y for x, y in pairs)
        only_b = sum(y and not x for x, y in pairs)
        low, high = change_interval(only_a, only_b, len(pairs))
        rows.append(
            (
                label,
                f"{a_yes}/{len(pairs)}",
                f"{b_yes}/{len(pairs)}",
                f"{b_yes - a_yes:+d}",
                f"{round(low):+d}, {round(high):+d}",
                f"{only_a} / {only_b}",
                f"{mcnemar(only_a, only_b):.2g}",
            )
        )
    harmless_a, harmless_b = a["harmless_held_out"], b["harmless_held_out"]
    places = [("first_token", "Harmless KL from A")]
    places += [
        (key, f"Harmless KL from A, {where}")
        for key, where in (
            ("first_token_empty", "empty thinking"),
            ("first_token_thinking", "thinking start"),
        )
        if key in harmless_a and key in harmless_b
    ]
    # KL read on another backend is a few percent off whatever the edit does (findings). Sets answered
    # again apart can each have been read on a machine of their own.
    platforms = {m["platform"] for m in (harmless_a.get("machine"), harmless_b.get("machine")) if m}
    apart = len(platforms) > 1
    for key, label in places:
        if apart:
            rows.append((label, "", "other backend"))
            continue
        divergences = _kl(harmless_a[key], harmless_b[key])
        interval = mean_interval(divergences)
        shown = f"{interval[0]:.3f}, {interval[1]:.3f}" if interval else ""
        rows.append((label, "", f"{statistics.fmean(divergences):.4f}", "", shown))
    rows = [(*row, *[""] * (7 - len(row))) for row in rows]
    for label, key in (("Sampling seed", "seed"), ("Slots", "slots")):
        if a["maths"].get(key) != b["maths"].get(key):
            rows.append((label, str(a["maths"].get(key)), str(b["maths"].get(key)), "", "", "", ""))
    gpus = [
        "; ".join(
            dict.fromkeys(
                ", ".join(found["machine"]["gpus"]) or "no GPU" for found in m.values() if "machine" in found
            )
        )
        or "not recorded"
        for m in (a, b)
    ]
    if gpus[0] != gpus[1]:
        rows.append(("GPU", *gpus, "", "", "", ""))
    judges = [
        {key: j[key] for key in ("device", "dtype")}
        if (j := m["harmful_held_out"].get("judge_machine"))
        else None
        for m in (a, b)
    ]
    if judges[0] != judges[1]:
        rows.append(
            (
                "Judged on",
                *(f"{j['device']}, {j['dtype']}" if j else "not recorded" for j in judges),
                "",
                "",
                "",
                "",
            )
        )
    return rows
