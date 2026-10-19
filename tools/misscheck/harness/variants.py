"""The landscapes to test a search on, and each one's true best edit.

    base       the fit to rude-llama's two searches (fit.py). Best: a one-layer edit at layer 12.
    per-layer  (a) one-layer edits cost more KL, so the best is a per-layer edit and the one-layer
               family is a slightly worse decoy.
    outside    (b) the one-layer landscape sits 4 layers shallower, so its good direction layers are
               8 and 10 (0.30 and 0.37 of the depth) with peaks near 0.46-0.62: just outside candor's
               ranges (layers 11-24 = 0.4-0.9, peaks 0.6-1.0), where it can draw no good one-layer edit.
    needle     (c) the one-layer basin around base's best is narrower in every setting, and the
               other good direction layers fall away.
    rand1..6   base with its shape perturbed at random (which direction layers fit, which depths
               remove refusal and at what KL cost), the one-layer family's KL scaled by 0.8-1.25 and
               its region moved -2..+2 layers: models we have no data for. Their bests are found over
               whole direction layers in today's ranges only.

    uv run python variants.py            # compute (or show cached) best edits for every landscape
"""

import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from landscape import CANDOR, HERETIC, ONE_LAYER, PER_LAYER, WIDE, Landscape, load_shape
from optimum import Best, best_of, describe, one_layer_bests

HERE = Path(__file__).parent
FITTED = HERE / "fitted.json"
CACHE = HERE / "cache"
# The one-layer family's best in (a) is this much worse than the per-layer best.
DECOY = 1.10
# True refusals (out of 100) at which the best trade-off is found, for judging how close a search got
# to it near zero refusals.
FRONT = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
RANDOM = 6


def perturbed(fitted: Landscape, i: int) -> Landscape:
    """A random landscape near the fit: each layer-profile knot moved by noise, so the best direction
    layer, the depths that matter and the best family can all change."""
    g = torch.Generator().manual_seed(1000 + i)
    shape = {k: v.clone() for k, v in fitted.shape.items()}
    shape["align.by_layer"] = shape["align.by_layer"] + 0.8 * torch.randn(len(shape["align.by_layer"]), generator=g)
    shape["harm.by_layer"] = shape["harm.by_layer"] + 0.3 * torch.randn(len(shape["harm.by_layer"]), generator=g)
    for scope in ("one", "per"):
        for c in ("attn", "mlp"):
            for kind in ("refuse", "damage"):
                knots = shape[f"{kind}.{scope}.{c}"]
                shape[f"{kind}.{scope}.{c}"] = knots + 0.4 * torch.randn(len(knots), generator=g)
    one_layer_kl = math.exp(math.log(0.8) + (math.log(1.25) - math.log(0.8)) * float(torch.rand(1, generator=g)))
    shift = float(torch.randint(-2, 3, (1,), generator=g))
    return Landscape(f"rand{i}", shape, one_layer_kl=one_layer_kl, shift=shift)


def base() -> Landscape:
    return Landscape("base", load_shape(FITTED))


def landscapes() -> dict[str, Landscape]:
    fitted = base()
    reference_base = reference(fitted)
    one, per = reference_base["one layer best"], reference_base["per layer"]
    found = {
        "base": fitted,
        "per-layer": Landscape("per-layer", fitted.shape, one_layer_kl=DECOY * per.kl / one.kl),
        "outside": Landscape("outside", fitted.shape, shift=4.0),
        "needle": Landscape("needle", fitted.shape, needle=1.5, centre=one.params),
        **{f"rand{i}": perturbed(fitted, i) for i in range(1, RANDOM + 1)},
    }
    # base and needle again, with each edit's prompt noise smooth over the settings (length 0.1).
    found.update({f"{name}-smooth": replace(found[name], name=f"{name}-smooth", noise_length=0.1)
                  for name in ("base", "needle")})
    return found


def key(landscape: Landscape) -> str:
    settings = {
        k: v for k, v in vars(landscape).items()
        if k not in ("shape", "support") and not (k == "noise_length" and v == 0)
    }
    shape = {k: v.tolist() for k, v in landscape.shape.items()}
    text = json.dumps([settings, shape, asdict(landscape.support)], sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def reference(landscape: Landscape, wide: bool | None = None) -> dict:
    """The landscape's best edits with true refusals <= 1 out of 100: per layer, one layer at each
    direction layer on a half-layer grid over heretic's range (which holds candor's whole layers), and
    over a far wider space. "best" is the best a search with today's ranges could draw; "best anywhere"
    includes the wider space. Cached by the landscape's settings."""
    if landscape.noise_length:  # the truth is its noiseless twin's
        return reference(replace(landscape, name=landscape.name.removesuffix("-smooth"), noise_length=0.0), wide)
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{landscape.name}-{key(landscape)}.json"
    if path.exists():
        raw = json.loads(path.read_text())
        return _decode(raw)
    torch.set_num_threads(2)
    started = time.time()
    quick = landscape.name.startswith("rand")
    wide = not quick if wide is None else wide
    # A variant's best edits are base's best edits moved by the variant, so those start descents too.
    hints_per, hints_one = [], []
    if landscape.name != "base":
        fitted = reference(base())
        hints_per = [fitted["per layer"].params]
        hints_one = [landscape.from_intrinsic(b.params) for b in fitted["one layer"].values() if b.params]
    found = {
        "per layer": best_of(landscape, HERETIC, PER_LAYER, hints=hints_per),
        "one layer": one_layer_bests(landscape, CANDOR if quick else HERETIC, step=0.5, hints=hints_one),
    }
    if wide:
        known_one = hints_one + [b.params for b in found["one layer"].values() if b.params]
        found["wide per layer"] = best_of(landscape, WIDE, PER_LAYER, hints=[*hints_per, found["per layer"].params])
        found["wide one layer"] = one_layer_bests(landscape, WIDE, step=1.0, hints=known_one)
    found["seconds"] = time.time() - started
    path.write_text(json.dumps(_encode(found), indent=1))
    return _decode(json.loads(path.read_text()))


def front(landscape: Landscape, space=HERETIC) -> dict[float, float]:
    """The landscape's best KL at true refusals <= r, for each r in FRONT, over today's ranges (or a
    wider space). The one-layer side is searched at the three direction layers best at r = 1. Cached."""
    if landscape.noise_length:
        return front(replace(landscape, name=landscape.name.removesuffix("-smooth"), noise_length=0.0), space)
    path = CACHE / f"{landscape.name}-{key(landscape)}-front-{'wide' if space is WIDE else 'today'}.json"
    if path.exists():
        return {float(r): kl for r, kl in json.loads(path.read_text()).items()}
    found = reference(landscape)
    grid = found["one layer"] if space is HERETIC else found["wide one layer"]
    layers = sorted(grid, key=lambda d: grid[d].kl)[:3]
    hints_per = [found["per layer"].params]
    hints_one = [b.params for b in grid.values() if b.params]
    torch.set_num_threads(2)
    out = {}
    for r in FRONT:
        candidates = [best_of(landscape, space, PER_LAYER, max_refusals=r, hints=hints_per)]
        candidates += [best_of(landscape, space, ONE_LAYER, layer=d, max_refusals=r, hints=hints_one) for d in layers]
        out[r] = min(b.kl for b in candidates)
    # The best KL can only fall as more refusals are allowed; a descent that fell short is overruled.
    for low, high in zip(FRONT, FRONT[1:]):
        out[high] = min(out[high], out[low])
    path.write_text(json.dumps(out))
    return out


def _encode(found: dict) -> dict:
    out = {}
    for key, value in found.items():
        if isinstance(value, Best):
            out[key] = asdict(value)
        elif isinstance(value, dict):
            out[key] = {str(d): asdict(b) for d, b in value.items()}
        else:
            out[key] = value
    return out


def _decode(raw: dict) -> dict:
    found = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "kl" in value:
            found[key] = Best(**value)
        elif isinstance(value, dict):
            found[key] = {float(d): Best(**b) for d, b in value.items()}
        else:
            found[key] = value
    found["one layer best"] = min(found["one layer"].values(), key=lambda b: b.kl)
    found["whole-layer best"] = min(
        (b for d, b in found["one layer"].items() if float(d).is_integer()), key=lambda b: b.kl
    )
    found["best"] = min(found["per layer"], found["one layer best"], key=lambda b: b.kl)
    if "wide one layer" in found:
        found["wide one layer best"] = min(found["wide one layer"].values(), key=lambda b: b.kl)
        found["best anywhere"] = min(
            found["best"], found["wide per layer"], found["wide one layer best"], key=lambda b: b.kl
        )
    return found


def main() -> None:
    names = sys.argv[1:] or ["base", "per-layer", "outside", "needle"]
    all_landscapes = landscapes()
    for name in names:
        found = reference(all_landscapes[name])
        print(f"\n== {name} (found in {found['seconds']:.0f} s)")
        print("  per layer:         ", describe(found["per layer"]))
        print("  one layer, best:   ", describe(found["one layer best"]))
        print("  one layer, whole:  ", describe(found["whole-layer best"]))
        print("  best anywhere:     ", describe(found["best anywhere"]))
        row = "  ".join(f"{d:g}:{b.kl:.3f}" for d, b in found["one layer"].items() if b.params and b.kl < 1)
        print("  one-layer best KL by direction layer:", row)
        print("  best KL at true refusals <= r:", "  ".join(f"{r:g}:{kl:.4f}" for r, kl in front(all_landscapes[name]).items()))


if __name__ == "__main__":
    main()
