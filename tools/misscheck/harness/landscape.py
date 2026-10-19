"""Synthetic score landscapes over candor's search parameters, for measuring how often a search design
misses the best edit.

A landscape gives an edit's true refusal rate and KL divergence from a small mechanistic model. Each
layer is edited with the strength candor's kernel gives it. That strength removes refusal in proportion
to how much the layer and component matter and, for a one-layer edit, how well the direction read at
the direction layer fits the layer being edited. It does damage in proportion to the strength raised
to a power, and KL grows as a power of the summed damage. The model's shape is fitted to candor's and heretic's searches on rude-llama
(fit.py), and the variants change that fitted shape (variants.py).

A Scorer adds the noise of a fixed set of test prompts. Scoring is greedy in candor, so the same edit
always gets the same score; here too. Each prompt has a fixed difficulty, so nearby edits refuse
mostly the same prompts, and the count over 100 prompts is binomial-like across prompt sets.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

LAYERS = 28
LAST = LAYERS - 1
COMPONENTS = ("attn", "mlp")
SETTINGS = ("strength", "peak_layer", "edge_strength", "reach")
ONE_LAYER, PER_LAYER = "one layer", "per layer"
KNOTS = 8


def interpolate(knots: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A piecewise-linear curve over layers 0..LAST through evenly spaced knots, at x (any shape)."""
    step = LAST / (len(knots) - 1)
    position = (x / step).clamp(0, len(knots) - 1 - 1e-9)
    low = position.floor().long()
    fraction = position - low
    return knots[low] * (1 - fraction) + knots[low + 1] * fraction


def kernel_weights(strength, peak, edge, reach) -> torch.Tensor:
    """candor's kernel (kernel.Kernel.at) for a batch of edits: (N, LAYERS) strengths."""
    layers = torch.arange(LAYERS, dtype=torch.float64)
    distance = (layers - peak[:, None] * LAST).abs()
    width = reach[:, None] * LAST
    peak_strength = strength[:, None]
    weights = peak_strength + distance / width * (edge[:, None] * peak_strength - peak_strength)
    return torch.where(distance > width, torch.zeros_like(weights), weights)


@dataclass
class Batch:
    """Edits as tensors: one_layer (bool), direction_layer (float, ignored per layer), and per component
    each of SETTINGS as fractions, as candor's search draws them."""

    one_layer: torch.Tensor
    direction_layer: torch.Tensor
    settings: dict[str, torch.Tensor]

    @classmethod
    def from_params(cls, params: list[dict]) -> "Batch":
        """From parameter dicts as candor's search names them (search.suggest)."""
        one = torch.tensor([p["direction_scope"] == ONE_LAYER for p in params])
        layer = torch.tensor([float(p.get("direction_layer") or 0.0) for p in params])
        settings = {
            f"{c}.{s}": torch.tensor([float(p[f"{c}.{s}"]) for p in params]) for c in COMPONENTS for s in SETTINGS
        }
        return cls(one, layer, settings)

    def __len__(self) -> int:
        return len(self.one_layer)


def _softplus(x):
    return torch.nn.functional.softplus(x)


def default_shape() -> dict[str, torch.Tensor]:
    """A starting shape for fitting. Layer profiles have KNOTS knots; what depends on the direction layer
    has one knot per layer, as each layer's direction is its own."""
    shape = {
        "base_logit": torch.tensor(math.log(99)),
        "power": torch.tensor(0.7),
        "outer.one": torch.tensor(0.0),
        "outer.per": torch.tensor(0.0),
        "kl_floor": torch.tensor(-9.0),
        "align.up": torch.tensor(1.5),
        "align.down": torch.tensor(1.5),
        "align.floor": torch.tensor(-1.0),
        "align.by_layer": torch.zeros(LAYERS),
        "harm.up": torch.tensor(1.5),
        "harm.down": torch.tensor(1.5),
        "harm.floor": torch.tensor(0.0),
        "harm.by_layer": torch.zeros(LAYERS),
    }
    for scope in ("one", "per"):
        for c in COMPONENTS:
            shape[f"refuse.{scope}.{c}"] = torch.full((KNOTS,), -1.0)
            shape[f"damage.{scope}.{c}"] = torch.full((KNOTS,), -5.0)
    return shape


def _fit_to_direction(shape, prefix: str, offset: torch.Tensor, direction_layer: torch.Tensor) -> torch.Tensor:
    """How much a direction read at the direction layer acts on each layer, for a one-layer edit: a
    per-direction-layer scale times a bump around the direction layer over a floor. (N, LAYERS)"""
    width = torch.where(offset > 0, _softplus(shape[f"{prefix}.up"]), _softplus(shape[f"{prefix}.down"])) * 4
    floor = torch.sigmoid(shape[f"{prefix}.floor"])
    knots = shape[f"{prefix}.by_layer"]
    # A direction read between two layers blends theirs, and is taken to act as the same blend of
    # their effects, so it does about as well as its two layers and no better.
    scale = interpolate(torch.sigmoid(knots) if prefix == "align" else torch.exp(knots), direction_layer)[:, None]
    bump = floor + (1 - floor) * torch.exp(-((offset / width) ** 2))
    return scale * bump


def evaluate(shape: dict[str, torch.Tensor], batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
    """The refusal logit and log KL of each edit in the batch."""
    layers = torch.arange(LAYERS, dtype=torch.float64)
    power = _softplus(shape["power"]) + 1
    offset = layers[None, :] - batch.direction_layer[:, None]
    align = _fit_to_direction(shape, "align", offset, batch.direction_layer)
    harm = _fit_to_direction(shape, "harm", offset, batch.direction_layer)
    one = batch.one_layer[:, None]
    removed = torch.zeros(len(batch))
    damage = torch.zeros(len(batch))
    for c in COMPONENTS:
        s = batch.settings
        weights = kernel_weights(s[f"{c}.strength"], s[f"{c}.peak_layer"], s[f"{c}.edge_strength"], s[f"{c}.reach"])
        refuse_one = _softplus(interpolate(shape[f"refuse.one.{c}"], layers))[None, :] * align
        refuse_per = _softplus(interpolate(shape[f"refuse.per.{c}"], layers))[None, :]
        damage_one = torch.exp(interpolate(shape[f"damage.one.{c}"], layers))[None, :] * harm
        damage_per = torch.exp(interpolate(shape[f"damage.per.{c}"], layers))[None, :]
        removed = removed + (weights * torch.where(one, refuse_one, refuse_per)).sum(1)
        damage = damage + (weights.clamp(min=0) ** power * torch.where(one, damage_one, damage_per)).sum(1)
    # KL grows as a power of the summed damage, steeper or gentler than additive, per scope.
    outer = torch.where(batch.one_layer, torch.exp(shape["outer.one"]), torch.exp(shape["outer.per"]))
    log_kl = torch.log(torch.exp(outer * torch.log(damage + 1e-12)) + torch.exp(shape["kl_floor"]))
    return shape["base_logit"] - removed, log_kl


def save_shape(shape: dict[str, torch.Tensor], path: Path) -> None:
    path.write_text(json.dumps({k: v.detach().tolist() for k, v in shape.items()}, indent=1))


def load_shape(path: Path) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v) for k, v in json.loads(path.read_text()).items()}


@dataclass(frozen=True)
class Space:
    """Ranges an edit's settings are drawn from, as fractions as candor draws them, and the direction
    layer as a layer index."""

    strength: tuple[float, float]
    peak_layer: tuple[float, float]
    edge_strength: tuple[float, float]
    reach: tuple[float, float]
    direction_layer: tuple[float, float]
    whole_layers: bool

    def range(self, setting: str) -> tuple[float, float]:
        return getattr(self, setting)


# candor's search today (config.Search defaults on a 28-layer model).
CANDOR = Space((0.8, 1.5), (0.6, 1.0), (0.0, 1.0), (1 / LAST, 0.6), (11, 24), True)
# heretic's: the same with a float direction layer, 0.4-0.9 of the depth.
HERETIC = Space((0.8, 1.5), (0.6, 1.0), (0.0, 1.0), (1 / LAST, 0.6), (0.4 * LAST, 0.9 * LAST), False)
# Far wider than either.
WIDE = Space((0.4, 2.2), (0.15, 1.0), (0.0, 1.0), (1 / LAST, 1.0), (2, 26), False)


def outside(batch: Batch, space: Space) -> torch.Tensor:
    """How far each edit lies outside the space, summed over its settings in widths of their ranges."""
    total = torch.zeros(len(batch))
    for name, value in batch.settings.items():
        low, high = space.range(name.split(".")[1])
        total = total + (torch.relu(low - value) + torch.relu(value - high)) / (high - low)
    low, high = space.direction_layer
    layer = (torch.relu(low - batch.direction_layer) + torch.relu(batch.direction_layer - high)) / (high - low)
    return total + torch.where(batch.one_layer, layer, torch.zeros_like(layer))


@dataclass
class Landscape:
    """A score landscape: true refusal rates and KL for any edit, and how long scoring one takes.

    The fitted shape is only evidence inside the ranges the two real searches drew from (support), so
    by default an edit outside them is scored as worse: its KL is multiplied by e^outside_cost for
    each range-width it lies outside. The variants change one-layer edits only: one_layer_kl scales
    their KL, shift scores them as the fitted landscape's edits that many layers deeper (so the best
    one-layer region sits that many layers shallower), and needle stretches their distance from centre
    (the fitted best one-layer edit), shrinking its basin by that factor in every setting.
    """

    name: str
    shape: dict[str, torch.Tensor]
    support: Space = HERETIC
    outside_cost: float = 3.0
    one_layer_kl: float = 1.0
    shift: float = 0.0
    needle: float = 1.0
    centre: dict | None = None
    # Seconds to score an edit on the user's Mac: generating and judging each harmful answer, where a
    # refusal is shorter than a compliance, plus the KL pass. Fitted to candor's 200 trials
    # (seconds = 74.1 - 0.286 * refusals out of 100); the KL pass's share is assumed.
    seconds_complied: float = 0.691
    seconds_refused: float = 0.405
    seconds_kl: float = 5.0
    # 0: each edit's own share of each prompt's noise is independent of every other edit's. Above 0:
    # it varies smoothly over the settings with this length scale (in widths of heretic's ranges), so
    # nearby edits refuse nearly the same prompts, as greedy answers of nearly equal weights would.
    noise_length: float = 0.0

    def intrinsic(self, batch: Batch) -> Batch:
        """The edit of the fitted landscape that this one scores each edit as."""
        if self.shift == 0 and self.needle == 1:
            return batch
        one = batch.one_layer
        layer = torch.where(one, batch.direction_layer + self.shift, batch.direction_layer)
        settings = {
            name: torch.where(one, value + self.shift / LAST, value) if name.endswith("peak_layer") else value
            for name, value in batch.settings.items()
        }
        if self.needle != 1:
            c = self.centre
            layer = torch.where(one, c["direction_layer"] + (layer - c["direction_layer"]) * self.needle, layer)
            settings = {
                name: torch.where(one, c[name] + (value - c[name]) * self.needle, value)
                for name, value in settings.items()
            }
        return Batch(one, layer, settings)

    def from_intrinsic(self, params: dict) -> dict:
        """The edit this landscape scores as the fitted landscape's edit params (intrinsic's inverse)."""
        if params["direction_scope"] != ONE_LAYER:
            return dict(params)
        out = dict(params)
        if self.needle != 1:
            c = self.centre
            out = {k: c[k] + (v - c[k]) / self.needle if k in c and k != "direction_scope" else v for k, v in out.items()}
        out["direction_layer"] -= self.shift
        for k in out:
            if k.endswith("peak_layer"):
                out[k] -= self.shift / LAST
        return out

    def logits(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Each edit's true refusal logit and log KL; differentiable in the batch's settings."""
        fitted = self.intrinsic(batch)
        logit, log_kl = evaluate(self.shape, fitted)
        log_kl = log_kl + self.outside_cost * outside(fitted, self.support)
        return logit, log_kl + torch.where(batch.one_layer, math.log(self.one_layer_kl), 0.0)

    @torch.no_grad()
    def true_batch(self, batch: Batch) -> tuple[np.ndarray, np.ndarray]:
        """Each edit's true refusal rate (0-1) and KL divergence."""
        logit, log_kl = self.logits(batch)
        return torch.sigmoid(logit).numpy(), torch.exp(log_kl).numpy()

    def true(self, params: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        return self.true_batch(Batch.from_params(params))

    def seconds(self, refusals: int, prompts: int, kl_pass: bool = True) -> float:
        answered = prompts - refusals
        return answered * self.seconds_complied + refusals * self.seconds_refused + (self.seconds_kl if kl_pass else 0)

    def scorer(self, seed: int, prompts: int = 100, prompt_correlation: float = 0.5, kl_noise: float = 0.02) -> "Scorer":
        return Scorer(self, seed, prompts, prompt_correlation, kl_noise)


@dataclass
class Score:
    refusals: int
    prompts: int
    kl: float
    seconds: float


@dataclass
class Scorer:
    """Scores edits on one fixed set of test prompts, as candor's search does: deterministic per edit,
    and nearby edits tend to refuse the same prompts.

    Each prompt i has a fixed difficulty z_i; an edit with true refusal rate p refuses it when
    Phi(sqrt(rho) z_i + sqrt(1 - rho) e_i) < p, with e_i fixed for that edit and prompt. Over prompt
    sets the count is binomial-like. Measured KL is the true KL times exp(kl_noise * e), e fixed per edit.
    """

    landscape: Landscape
    seed: int
    prompts: int
    prompt_correlation: float
    kl_noise: float
    difficulty: np.ndarray = field(init=False)

    def __post_init__(self):
        self.difficulty = np.random.default_rng([self.seed, 1]).standard_normal(self.prompts)
        self._features = {}

    def _smooth_noise(self, params: dict) -> np.ndarray:
        """Each prompt's edit noise as a smooth random function of the settings (random Fourier
        features of a squared-exponential Gaussian process with unit variance), one per scope."""
        scope = params["direction_scope"]
        if scope not in self._features:
            rng = np.random.default_rng([self.seed, 4, scope == ONE_LAYER])
            dims, features = 1 + len(COMPONENTS) * len(SETTINGS), 128
            self._features[scope] = (
                rng.standard_normal((dims, features)) / self.landscape.noise_length,
                rng.uniform(0, 2 * math.pi, features),
                rng.standard_normal((features, self.prompts)) * math.sqrt(2 / features),
            )
        w, b, a = self._features[scope]
        support = self.landscape.support
        low, high = support.direction_layer
        x = [(float(params.get("direction_layer") or 0.0) - low) / (high - low)]
        for c in COMPONENTS:
            for s in SETTINGS:
                low, high = support.range(s)
                x.append((params[f"{c}.{s}"] - low) / (high - low))
        return np.cos(np.array(x) @ w + b) @ a

    def _edit_noise(self, params: dict) -> np.random.Generator:
        key = json.dumps({k: round(v, 9) if isinstance(v, float) else v for k, v in sorted(params.items())})
        digest = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")
        return np.random.default_rng([self.seed, 2, digest])

    def unedited(self) -> int:
        """The unedited model's refusals on the full set, which candor's objectives divide by."""
        rate = float(torch.sigmoid(self.landscape.shape["base_logit"]))
        return int((self._uniform(np.random.default_rng([self.seed, 3])) < rate).sum())

    def _uniform(self, noise: np.random.Generator, own: np.ndarray | None = None) -> np.ndarray:
        rho = self.prompt_correlation
        independent = noise.standard_normal(self.prompts)
        own = independent if own is None else own
        latent = math.sqrt(rho) * self.difficulty + math.sqrt(1 - rho) * own
        return 0.5 * (1 + _erf(latent / math.sqrt(2)))

    def score(self, params: dict, prompts: int | None = None, kl_pass: bool = True) -> Score:
        """Score one edit on the first `prompts` test prompts (all of them by default), with or without
        the KL pass. A strategy that prunes can score a few prompts first and, if it goes on, all of them:
        the first few refuse the same way both times, and it should charge only the difference in
        seconds for the second call."""
        prompts = self.prompts if prompts is None else prompts
        rate, kl = self.landscape.true([params])
        noise = self._edit_noise(params)
        own = self._smooth_noise(params) if self.landscape.noise_length else None
        refusals = int((self._uniform(noise, own)[:prompts] < rate[0]).sum())
        measured_kl = float(kl[0] * math.exp(self.kl_noise * noise.standard_normal())) if kl_pass else float("nan")
        return Score(refusals, prompts, measured_kl, self.landscape.seconds(refusals, prompts, kl_pass))


def _erf(x: np.ndarray) -> np.ndarray:
    return torch.special.erf(torch.from_numpy(x)).numpy()
