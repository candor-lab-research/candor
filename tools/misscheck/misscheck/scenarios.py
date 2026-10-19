"""Landscapes for the phase-3 checks, built on the harness's (landscape.py, variants.py).

Scaled(landscape, stretch k, efficiency e) changes a landscape in two ways, both for attn and mlp and
for both kinds of edit:
  stretch k     an edit at strength s scores as the original's edit at s / k: the same landscape in
                another strength unit. The best edit needs k times the strength at the same KL, and the
                zone where the harness penalises KL for leaving the fitted support (0.8-1.5) moves to
                0.8k-1.5k.
  efficiency e  an edit removes e times the refusal (the refusal logit's drop is scaled by e), so it
                needs about 1/e times the strength and pays that strength's KL.

The harness penalises KL outside the strengths the real searches drew (x e^3 per range width), so the
fitted and random landscapes keep their best inside 0.8-1.5 when the range is widened, but on a kink at
0.8 or 1.5. A free landscape (name starting with f) drops that penalty for strength only: the fitted
model extrapolated (refusal removed linear in strength, damage a power of it).

A position scenario holds one landscape per place the refusal direction is read ("think start" and
"answer start"). The test prompts are the same at both, so they share each prompt's difficulty; an
edit's own noise depends on its position too, as its answers would.

    uv run python scenarios.py refs --jobs=10    # compute (or show cached) every reference and front
"""

import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from multiprocessing import Pool
from pathlib import Path

import torch

from landscape import CANDOR, LAST, ONE_LAYER, PER_LAYER, Batch, Landscape, Space
from optimum import Best, best_of, describe, one_layer_grid
from variants import key, landscapes, reference as harness_reference

HERE = Path(__file__).parent
CACHE = HERE / "cache"
CAP = 2.0  # true refusals out of 100 the reference allows
NARROW = (0.8, 1.5)
WIDE = (0.5, 3.0)
THINK, ANSWER = "think start", "answer start"


def space(strength: tuple[float, float]) -> Space:
    return replace(CANDOR, strength=strength)


@dataclass
class Scaled(Landscape):
    stretch: float = 1.0
    efficiency: float = 1.0
    free: bool = False

    def intrinsic(self, batch: Batch) -> Batch:
        batch = super().intrinsic(batch)
        if self.stretch == 1:
            return batch
        settings = {
            name: value / self.stretch if name.split(".")[1] == "strength" else value
            for name, value in batch.settings.items()
        }
        return Batch(batch.one_layer, batch.direction_layer, settings)

    def logits(self, batch: Batch):
        logit, log_kl = super().logits(batch)
        if self.efficiency != 1:
            base = self.shape["base_logit"]
            logit = base - self.efficiency * (base - logit)
        return logit, log_kl


def scaled(original: Landscape, name: str, stretch: float = 1.0, efficiency: float = 1.0, free: bool = False) -> Scaled:
    fields = {k: v for k, v in vars(original).items() if k != "name"}
    if free:
        fields["support"] = replace(fields["support"], strength=(0.01, 100.0))
    return Scaled(name=name, **fields, stretch=stretch, efficiency=efficiency, free=free)


@dataclass
class Scenario:
    """One landscape per position (a model that doesn't think has only one, under None)."""

    name: str
    positions: dict


def _harness() -> dict[str, Landscape]:
    found = landscapes()
    return {name: found[name] for name in ["base", *[f"rand{i}" for i in range(1, 7)]]}


def scenarios() -> dict[str, Scenario]:
    h = _harness()
    base = h["base"]
    out = {}
    # A: the harness's landscapes, whose best sits inside 0.8-1.5.
    for name, land in h.items():
        out[name] = Scenario(name, {None: scaled(land, name)})
    # A: the same landscapes in a stretched strength unit, so the best needs about 2.0-2.8.
    for name, k in (("base", 2.4), ("rand4", 1.8), ("rand6", 1.8)):
        out[f"{name}x{k:g}"] = Scenario(f"{name}x{k:g}", {None: scaled(h[name], f"{name}x{k:g}", stretch=k)})
    # B: where the direction is read. Each position is base, scaled.
    same = scaled(base, "base")
    out["a-same"] = Scenario("a-same", {THINK: same, ANSWER: same})
    out["b-answer"] = Scenario("b-answer", {THINK: scaled(base, "base-e0.8", efficiency=0.8), ANSWER: same})
    out["c-think"] = Scenario("c-think", {THINK: same, ANSWER: scaled(base, "base-e0.95", efficiency=0.95)})
    out["d-high"] = Scenario(
        "d-high", {THINK: scaled(base, "basex4.5", stretch=4.5), ANSWER: scaled(base, "basex2.4", stretch=2.4)}
    )
    # The same with no strength penalty.
    for name, land in h.items():
        out[f"f{name}"] = Scenario(f"f{name}", {None: scaled(land, f"f{name}", free=True)})
    for name, k in (("base", 2.6), ("rand4", 1.8), ("rand6", 1.8)):
        label = f"f{name}x{k:g}"
        out[label] = Scenario(label, {None: scaled(h[name], label, stretch=k, free=True)})
    same = scaled(base, "fbase", free=True)
    out["fa-same"] = Scenario("fa-same", {THINK: same, ANSWER: same})
    out["fb-answer"] = Scenario("fb-answer", {THINK: scaled(base, "fbase-e0.8", efficiency=0.8, free=True), ANSWER: same})
    out["fc-think"] = Scenario("fc-think", {THINK: same, ANSWER: scaled(base, "fbase-e0.95", efficiency=0.95, free=True)})
    out["fd-high"] = Scenario(
        "fd-high",
        {THINK: scaled(base, "fbasex4.5", stretch=4.5, free=True), ANSWER: scaled(base, "fbasex2.6", stretch=2.6, free=True)},
    )
    return out


def _hints(land: Scaled) -> tuple[list[dict], list[dict]]:
    """The harness's known best edits of the landscape this one scales, moved into its strength unit."""
    original = landscapes()[land.name.removeprefix("f" if land.free else "").split("x")[0].split("-e")[0]]
    found = harness_reference(original)
    factor = land.stretch / land.efficiency

    def moved(p: dict) -> dict:
        return {k: v * factor if k.split(".")[-1] == "strength" else v for k, v in p.items()}

    per = [moved(found["per layer"].params)]
    one = [moved(b.params) for b in found["one layer"].values() if b.params]
    return per, one


def _land_key(land: Scaled) -> str:
    """The harness's key, which leaves out the support; a free landscape is told apart by its flag
    (dropped when False, so the penalised landscapes keep their keys)."""
    if land.free:
        return f"{land.name}-{key(land)}"
    fields = {k: v for k, v in vars(land).items() if k != "free"}
    return f"{land.name}-{key(Landscape_like(fields))}"


def _one(task) -> tuple[str, str, float, dict]:
    land, strength, scope, layer = task
    torch.set_num_threads(1)
    per, one = _hints(land)
    hints = per if scope == PER_LAYER else [h for h in one if abs(h["direction_layer"] - layer) <= 1]
    found = best_of(land, space(strength), scope, layer=layer, max_refusals=CAP, hints=hints)
    return scope, layer, asdict(found)


def references(tasks_by: dict[str, tuple[Scaled, tuple]], jobs: int) -> dict[str, dict]:
    """Each landscape's best edit at true refusals <= CAP, per layer and at each whole direction layer
    of candor's range, over the given strength range. Cached by landscape and range."""
    out, todo = {}, []
    for label, (land, strength) in tasks_by.items():
        path = CACHE / f"ref-{_land_key(land)}-{strength[0]:g}-{strength[1]:g}.json"
        if path.exists():
            out[label] = json.loads(path.read_text())
            continue
        layers = one_layer_grid(CANDOR)
        todo += [(label, path, (land, strength, PER_LAYER, 0.0))]
        todo += [(label, path, (land, strength, ONE_LAYER, d)) for d in layers]
    if todo:
        started = time.time()
        with Pool(jobs) as pool:
            results = pool.map(_one, [t[2] for t in todo], chunksize=1)
        print(f"{len(todo)} descents in {time.time() - started:.0f} s", flush=True)
        grouped = {}
        for (label, path, _), (scope, layer, found) in zip(todo, results, strict=True):
            entry = grouped.setdefault(label, (path, {"per layer": None, "one layer": {}}))[1]
            if scope == PER_LAYER:
                entry["per layer"] = found
            else:
                entry["one layer"][str(layer)] = found
        for label, (path, entry) in grouped.items():
            path.write_text(json.dumps(entry, indent=1))
            out[label] = entry
    return out


def best_of_entry(entry: dict) -> Best:
    found = [Best(**entry["per layer"]), *[Best(**b) for b in entry["one layer"].values()]]
    return min(found, key=lambda b: b.kl)


class Landscape_like:
    """An object whose vars are the given fields, for the harness's key."""

    def __init__(self, fields: dict):
        self.__dict__.update(fields)


def zone(land: Scaled) -> tuple[float, float] | None:
    """The strengths inside the wide range where the landscape's KL isn't penalised. A descent over the
    wide range can stall at the penalty's kink, so the best is also sought over this zone alone."""
    low, high = max(WIDE[0], 0.8 * land.stretch), min(WIDE[1], 1.5 * land.stretch)
    return (low, high) if low < high and not land.free else None


def reference_tasks(names: list[str] | None = None) -> dict[str, tuple[Scaled, tuple]]:
    """Every (landscape, strength range) a reference is needed for, for these scenarios (or all)."""
    tasks = {}
    for scenario in scenarios().values():
        if names is not None and scenario.name not in names:
            continue
        for land in scenario.positions.values():
            tasks[f"{land.name} wide"] = (land, WIDE)
            if zone(land):
                tasks[f"{land.name} zone"] = (land, zone(land))
            if scenario.name == land.name and land.stretch != 1:
                tasks[f"{land.name} narrow"] = (land, NARROW)
    return tasks


def twin(land: Scaled) -> str | None:
    """A free landscape's penalised twin: its KL is never below the free one's, so its best bounds it."""
    return land.name.removeprefix("f") if land.free else None


def land_truth(land: Scaled, refs: dict) -> Best:
    """A landscape's best edit at true refusals <= CAP over the wide range (and its unpenalised zone,
    and for a free landscape its penalised twin's)."""
    names = [land.name, twin(land)] if land.free else [land.name]
    found = [best_of_entry(refs[f"{n} {label}"]) for n in names for label in ("wide", "zone") if f"{n} {label}" in refs]
    return min(found, key=lambda b: b.kl)


def truth(scenario: Scenario, strength: tuple[float, float], refs: dict) -> Best:
    """The scenario's best edit over every position, at true refusals <= CAP, in a strength range."""
    if strength == WIDE:
        return min((land_truth(land, refs) for land in scenario.positions.values()), key=lambda b: b.kl)
    return min((best_of_entry(refs[f"{land.name} narrow"]) for land in scenario.positions.values()), key=lambda b: b.kl)


def load_refs(names: list[str] | None = None) -> dict:
    return references(reference_tasks(names), jobs=1)


def main() -> None:
    if sys.argv[1:2] != ["refs"]:
        raise SystemExit(__doc__)
    jobs = int(next((a for a in sys.argv if a.startswith("--jobs=")), "--jobs=10").split("=")[1])
    refs = references(reference_tasks(), jobs)
    for label in sorted(refs):
        b = best_of_entry(refs[label])
        print(f"{label:22s} {describe(b)}")
    for name, curve in fronts(jobs).items():
        print(f"{name:22s} front: " + "  ".join(f"{r:g}:{kl:.4f}" for r, kl in curve.items()))



LEVELS = (0.25, 0.5, 1.0, 1.5, 3.0, 4.0, 6.0, 10.0)


def _front_one(task) -> tuple[str, float, float]:
    label, land, strength, scope, layer, level, hints = task
    torch.set_num_threads(1)
    found = best_of(land, space(strength), scope, layer=layer, max_refusals=level, hints=hints,
                    samples=20_000, steps=1500, seeds=2)
    return label, level, found.kl


def fronts(jobs: int = 1, names: list[str] | None = None) -> dict[str, dict[float, float]]:
    """Each wide landscape's best KL at true refusals <= each level (LEVELS, and CAP from its reference),
    per layer and at the two direction layers best at CAP, started from the CAP best too. Cached."""
    refs = load_refs(names)
    raw, todo = {}, []
    lands = {land.name: land for s in scenarios().values() for land in s.positions.values()}
    for name, land in lands.items():
        for label, strength in (("wide", WIDE), ("zone", zone(land))):
            if strength is None:
                continue
            path = CACHE / f"front-{_land_key(land)}{'' if label == 'wide' else '-zone'}.json"
            if path.exists():
                raw[(name, label)] = {float(r): kl for r, kl in json.loads(path.read_text()).items()}
                continue
            entry = refs[f"{name} {label}"]
            per, one = _hints(land)
            per = per + ([entry["per layer"]["params"]] if entry["per layer"]["params"] else [])
            layers = sorted(entry["one layer"].items(), key=lambda item: item[1]["kl"])[:2]
            for level in LEVELS:
                todo.append(((name, label, path), land, strength, PER_LAYER, 0.0, level, per))
                for layer, b in layers:
                    hints = [h for h in one if abs(h["direction_layer"] - float(layer)) <= 1]
                    hints += [b["params"]] if b["params"] else []
                    todo.append(((name, label, path), land, strength, ONE_LAYER, float(layer), level, hints))
    if todo:
        started = time.time()
        with Pool(jobs) as pool:
            results = pool.map(_front_one, todo, chunksize=1)
        print(f"{len(todo)} front descents in {time.time() - started:.0f} s", flush=True)
        found = {}
        for (name, label, path), level, kl in results:
            curve = found.setdefault((name, label, path), {})
            curve[level] = min(curve.get(level, math.inf), kl)
        for (name, label, path), curve in found.items():
            curve[CAP] = best_of_entry(refs[f"{name} {label}"]).kl
            path.write_text(json.dumps(dict(sorted(curve.items()))))
            raw[(name, label)] = curve
    out = {}
    for name, land in lands.items():
        curves = [c for (n, _), c in raw.items() if n in (name, twin(land))]
        curve = {r: min(c[r] for c in curves) for r in sorted(curves[0])}
        levels = list(curve)
        for low, high in zip(levels, levels[1:]):
            curve[high] = min(curve[high], curve[low])
        out[name] = curve
    return out


def scenario_front(scenario: Scenario, curves: dict) -> dict[float, float]:
    """The scenario's best KL at each level of true refusals, over its positions."""
    levels = sorted(next(iter(curves.values())))
    return {r: min(curves[land.name][r] for land in scenario.positions.values()) for r in levels}


if __name__ == "__main__":
    main()
