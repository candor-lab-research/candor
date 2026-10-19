"""The true best edit of a landscape: the lowest KL among edits whose true refusals are at most a
limit, found by broad random sampling and then gradient descent from the best samples.

A search is judged against this, so it is found with far more effort than any search spends: by
default 50,000 random edits per family and direction layer, then 3,000 steps of Adam from the 32
best, from three seeds.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from landscape import CANDOR, COMPONENTS, HERETIC, ONE_LAYER, PER_LAYER, SETTINGS, WIDE, Batch, Landscape, Space


@dataclass
class Best:
    params: dict
    refusals: float  # true expected refusals out of 100
    kl: float


def _batch(raw: dict[str, torch.Tensor], space: Space, scope: str, layer: torch.Tensor) -> Batch:
    """Settings from unconstrained numbers, squeezed into the space."""
    settings = {}
    for c in COMPONENTS:
        for s in SETTINGS:
            low, high = space.range(s)
            settings[f"{c}.{s}"] = low + (high - low) * torch.sigmoid(raw[f"{c}.{s}"])
    one = torch.full((len(layer),), scope == ONE_LAYER)
    return Batch(one, layer, settings)


def _params(batch: Batch, i: int, scope: str) -> dict:
    p = {name: float(value[i]) for name, value in batch.settings.items()}
    p["direction_scope"] = scope
    p["direction_layer"] = float(batch.direction_layer[i])
    return p


def best(
    landscape: Landscape,
    space: Space,
    scope: str,
    max_refusals: float = 1.0,
    layer: float = 0.0,
    samples: int = 50_000,
    starts: int = 32,
    steps: int = 3000,
    seed: int = 0,
    hints: list[dict] = (),
) -> Best:
    """The lowest-KL edit of one scope with true refusals <= max_refusals (out of 100), a one-layer
    edit reading its direction at layer. The refusal limit is a penalty that tightens as the descent
    goes on, aimed a hair inside the limit. Hints are edits whose settings start a descent too (moved
    into the space if outside it), for good edits known in advance."""
    generator = torch.Generator().manual_seed(seed)
    limit = math.log(max_refusals / (100 - max_refusals)) - 0.002
    names = [f"{c}.{s}" for c in COMPONENTS for s in SETTINGS]
    direction = torch.full((samples,), float(layer))

    def scored(raw):
        batch = _batch(raw, space, scope, direction[: len(next(iter(raw.values())))])
        logit, log_kl = landscape.logits(batch)
        return batch, logit, log_kl

    with torch.no_grad():
        raw = {n: torch.logit(torch.rand(samples, generator=generator).clamp(1e-4, 1 - 1e-4)) for n in names}
        _, logit, log_kl = scored(raw)
        top = torch.argsort(log_kl + 10 * torch.relu(logit - limit) ** 2)[:starts]
        raw = {n: v[top] for n, v in raw.items()}
        known = {}
        for n in names:
            low, high = space.range(n.split(".")[1])
            share = torch.tensor([(h[n] - low) / (high - low) for h in hints], dtype=torch.float64)
            known[n] = torch.logit(share.clamp(1e-6, 1 - 1e-6))
            raw[n] = torch.cat([raw[n], known[n]])
    raw = {n: v.clone().requires_grad_() for n, v in raw.items()}
    optimiser = torch.optim.Adam(raw.values(), lr=0.05)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps, eta_min=5e-4)
    for step in range(steps):
        optimiser.zero_grad()
        _, logit, log_kl = scored(raw)
        weight = 10 * 1000 ** (step / steps)
        (log_kl + weight * torch.relu(logit - limit) ** 2).sum().backward()
        optimiser.step()
        schedule.step()
    with torch.no_grad():
        # The hints as they were stay candidates too, in case every descent from them strayed.
        batch, logit, log_kl = scored({n: torch.cat([raw[n], known[n]]) for n in names})
        refusals = 100 * torch.sigmoid(logit)
        feasible = refusals <= max_refusals + 0.01
        if not feasible.any():
            return Best({}, float("nan"), float("inf"))
        kl = torch.where(feasible, torch.exp(log_kl), torch.tensor(float("inf")))
        i = int(torch.argmin(kl))
        return Best(_params(batch, i, scope), float(refusals[i]), float(kl[i]))


def best_of(landscape: Landscape, space: Space, scope: str, layer: float = 0.0, seeds: int = 3,
            max_refusals: float = 1.0, **kwargs) -> Best:
    """best() from several seeds, as one descent can stop in a basin a few percent short."""
    hints = kwargs.pop("hints", ())
    found = [
        best(landscape, space, scope, max_refusals, layer=layer, seed=s, hints=hints if s == 0 else (), **kwargs)
        for s in range(seeds)
    ]
    return min(found, key=lambda b: b.kl)


def one_layer_grid(space: Space, step: float = 1.0) -> list[float]:
    """The direction layers to fix in turn: whole layers, or for a float space a finer grid. A descent
    over the direction layer itself stalls between layers, so the layer is never descended on."""
    low, high = space.direction_layer
    if space.whole_layers:
        return [float(d) for d in range(math.ceil(low), math.floor(high) + 1)]
    grid = np.arange(math.ceil(low / step) * step, high + 1e-9, step)
    return sorted({low, high, *map(float, grid)})


def one_layer_bests(landscape: Landscape, space: Space, step: float = 1.0, **kwargs) -> dict[float, Best]:
    return {d: best_of(landscape, space, ONE_LAYER, layer=d, **kwargs) for d in one_layer_grid(space, step)}


def describe(b: Best) -> str:
    if not b.params:
        return "none feasible"
    p = b.params
    layer = f"layer {p['direction_layer']:.2f}" if p["direction_scope"] == ONE_LAYER else "per layer"
    kernels = "; ".join(
        f"{c}: s {p[f'{c}.strength']:.2f} peak {p[f'{c}.peak_layer']:.3f} edge {p[f'{c}.edge_strength']:.2f} "
        f"reach {p[f'{c}.reach']:.3f}"
        for c in COMPONENTS
    )
    return f"{layer}: {b.refusals:.2f} refused, KL {b.kl:.4f} ({kernels})"


# Trials with more true refusals than this are not near zero refusals and are not judged.
NEAR_ZERO = 2.0


def best_at(curve: dict[float, float], refusals) -> np.ndarray:
    """The landscape's best KL at this many true refusals (a number or an array), interpolated on its
    front in log-log; below the front's first point, that point's KL, which judges such edits more
    strictly."""
    r = np.maximum(np.asarray(refusals, dtype=float), min(curve))
    points = sorted(curve)
    return np.exp(np.interp(np.log(r), np.log(points), np.log([curve[p] for p in points])))


def gaps(curve: dict[float, float], refusals, kls) -> np.ndarray:
    """How far each edit's true KL is above the best at its true refusals; inf above NEAR_ZERO."""
    refusals, kls = np.asarray(refusals, dtype=float), np.asarray(kls, dtype=float)
    return np.where(refusals <= NEAR_ZERO, kls / best_at(curve, refusals) - 1, np.inf)


def region_volume(landscape: Landscape, space: Space, curve: dict[float, float], tols=(0.03, 0.10),
                  samples: int = 2_000_000, seed: int = 0) -> dict[float, dict[str, float]]:
    """The share of draws from the space, drawn as candor's random phase draws them (scope 50/50,
    settings uniform), that land within tol of the landscape's best trade-off near zero refusals,
    by scope. A random search of n trials then hits with probability 1 - (1 - share)^n."""
    rng = np.random.default_rng(seed)
    hits = {tol: {ONE_LAYER: 0, PER_LAYER: 0} for tol in tols}
    chunk = 100_000
    for start in range(0, samples, chunk):
        n = min(chunk, samples - start)
        one = rng.random(n) < 0.5
        low, high = space.direction_layer
        if space.whole_layers:
            layer = rng.integers(int(low), int(high) + 1, n).astype(float)
        else:
            layer = rng.uniform(low, high, n)
        settings = {}
        for c in COMPONENTS:
            for s in SETTINGS:
                a, b = space.range(s)
                settings[f"{c}.{s}"] = torch.from_numpy(rng.uniform(a, b, n))
        rate, kl = landscape.true_batch(Batch(torch.from_numpy(one), torch.from_numpy(layer), settings))
        gap = gaps(curve, 100 * rate, kl)
        for tol in tols:
            hits[tol][ONE_LAYER] += int(((gap <= tol) & one).sum())
            hits[tol][PER_LAYER] += int(((gap <= tol) & ~one).sum())
    return {tol: {scope: count / samples for scope, count in by_scope.items()} for tol, by_scope in hits.items()}
