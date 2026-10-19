"""Budget hits on the phase-3 scenario landscapes: how often an edit's thinking runs to the thinking
budget, so the model is made to stop and answer (candor's Score.forced).

An edit's chance of reaching the budget on a prompt, q, follows K2-Horizon-7B's search (k2/fit.py):

    logit q = A + B log(KL / KL*) + C p + S u(edit)

KL and p are the edit's true KL and refusal rate, KL* the scenario's best KL at true refusals <= 2 with
no budget limit, and u a smooth random function of the edit's settings with unit variance (random
Fourier features of a squared-exponential kernel, length LENGTH in widths of the settings' ranges), one
for each kind of edit at each position, fixed by the scenario. B, C, S and LENGTH are K2's; A is set so
the simulated searches' first 137 trials match K2's budget hits (budget-calib.txt).

Two stress variants add bumps of budget hits beside good edits, each bump's own best edit kept at
STRESS_AT and about half of its near edits made to fail the limit (see _bump):
  stress  one bump beside the scenario's best edit within both caps; its near edits are those of the
          best's kind, position and direction layer within 10% of its KL at true refusals <= 2. Few of
          the edits a search picks are among them (they mostly truly refuse 3-6), so it is a weak stress.
  cliffs  a bump beside each basin (position, kind, direction layer) whose best edit is within 10% of
          the scenario's best, its near edits those within 10% of the best's KL at true refusals <= 6:
          the edits a search can pick.

A run's prompts each have a fixed propensity w_i (from the run's seed, apart from their refusal
difficulty); an edit reaches the budget on prompt i when Phi(sqrt(RHO) w_i + sqrt(1 - RHO) e_i) < q, e_i
fixed for that edit and prompt and drawn apart from its refusal and KL noise, so budget hits leave its
refusals and KL as they were.

    PYTHONPATH=candor-8dfc0f5/src:. uv run python budget.py truth --scenarios a-same --stress a-same \
        --cliffs a-same --jobs 10   # truths and bumps, cached in cache/ (budget_run.sh does every scenario)
"""

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from landscape import COMPONENTS, ONE_LAYER, PER_LAYER, SETTINGS, Batch, Landscape
from optimum import _batch, _params
from scenarios import CAP, WIDE, Scenario, load_refs, scenarios, space, truth, twin

HERE = Path(__file__).parent
CACHE = HERE / "cache"

# Fitted to K2's 146 trials (k2/fit.py): a binomial model with a normal random effect per edit, whose
# spread is 1.76 (1.32-1.41 with kind and position in the fit). S and A are then set so the simulated
# searches' first 137 trials match K2's (budget-calib.txt): at S 1.76 they had too many trials with no
# budget hits and too few near misses (9-14).
A, B, C, S = -5.118, 1.582, -3.025, 1.3
INTERCEPT = -4.4
# Nearby K2 edits share their budget hits beyond what KL and refusals give at about 0.25 range-widths
# apart (correlation +0.4) and not beyond 0.35 (k2/fit.py).
LENGTH = 0.2
RHO = 0.5
FEATURES = 256
LIMIT = 8.0  # expected budget hits of 100 an edit may have to count in the truth
STRESS_AT = 0.06  # a bump's own best edit's share of budget hits
CLIFF_REFUSALS = 6.0  # cliffs' near edits truly refuse at most this many of 100
NAMES = [f"{c}.{s}" for c in COMPONENTS for s in SETTINGS]
LAYER_RANGE = (11.0, 24.0)
RANGES = {"strength": WIDE, "peak_layer": (0.6, 1.0), "edge_strength": (0.0, 1.0), "reach": (1 / 27, 0.6)}


def coords(batch: Batch) -> torch.Tensor:
    """Each edit's settings as shares of their ranges (strength over 0.5-3.0), its direction layer first
    (0 for an edit per layer). (N, 9)"""
    low, high = LAYER_RANGE
    layer = torch.where(
        batch.one_layer, (batch.direction_layer - low) / (high - low), torch.zeros(len(batch))
    )
    columns = [layer]
    for name in NAMES:
        low, high = RANGES[name.split(".")[1]]
        columns.append((batch.settings[name] - low) / (high - low))
    return torch.stack(columns, 1)


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.blake2b(json.dumps(parts).encode(), digest_size=8).digest(), "little")


@dataclass
class Budget:
    """A scenario's budget landscape: A, B, C, S as fitted, the smooth field u per (one layer, position),
    and any bumps (stress, cliffs)."""

    best_kl: float
    fields: dict
    intercept: float = INTERCEPT
    bumps: tuple = ()

    def logit(
        self, batch: Batch, position, logit_refusal: torch.Tensor, log_kl: torch.Tensor
    ) -> torch.Tensor:
        x = coords(batch)
        u = torch.zeros(len(batch))
        for one in (True, False):
            w, b, a = self.fields[(one, position)]
            u = torch.where(batch.one_layer == one, torch.cos(x @ w + b) @ a, u)
        out = (
            self.intercept + B * (log_kl - math.log(self.best_kl)) + C * torch.sigmoid(logit_refusal) + S * u
        )
        for bump in self.bumps:
            if bump["position"] == position:
                distance = ((x - torch.tensor(bump["centre"])) ** 2).sum(1)
                near = bump["height"] * torch.exp(-distance / (2 * bump["width"] ** 2))
                out = out + torch.where(batch.one_layer == bump["one"], near, torch.zeros_like(near))
        return out

    def hits(self, land: Landscape, position, params: list[dict]) -> np.ndarray:
        """Expected budget hits of 100 for each edit."""
        batch = Batch.from_params(params)
        with torch.no_grad():
            logit, log_kl = land.logits(batch)
            return 100 * torch.sigmoid(self.logit(batch, position, logit, log_kl)).numpy()


def fields(name: str, positions) -> dict:
    """The scenario's smooth fields, one per kind of edit and position."""
    out = {}
    for position in positions:
        for one in (True, False):
            rng = np.random.default_rng(_seed("budget field", name, position, one))
            out[(one, position)] = (
                torch.from_numpy(rng.standard_normal((1 + len(NAMES), FEATURES)) / LENGTH),
                torch.from_numpy(rng.uniform(0, 2 * math.pi, FEATURES)),
                torch.from_numpy(rng.standard_normal(FEATURES) * math.sqrt(2 / FEATURES)),
            )
    return out


class Draws:
    """One run's budget hits: each prompt's propensity, and each edit's own noise per prompt, drawn from
    generators of their own so the refusal and KL draws are untouched."""

    def __init__(self, seed: int, prompts: int = 100):
        self.seed = seed
        self.propensity = np.random.default_rng([seed, 6]).standard_normal(prompts)

    def forced(self, edit: dict, q: float) -> np.ndarray:
        key = json.dumps({k: round(v, 9) if isinstance(v, float) else v for k, v in sorted(edit.items())})
        digest = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")
        own = np.random.default_rng([self.seed, 5, digest]).standard_normal(len(self.propensity))
        latent = math.sqrt(RHO) * self.propensity + math.sqrt(1 - RHO) * own
        return 0.5 * (1 + torch.special.erf(torch.from_numpy(latent / math.sqrt(2))).numpy()) < q


# ---- the truth: the least KL at true refusals <= 2 and expected budget hits <= LIMIT


@dataclass
class Truth:
    kl: float
    refusals: float
    hits: float
    position: str | None
    params: dict
    unlimited_kl: float  # the best KL with no budget limit


def _candidates(scenario: Scenario, refs: dict) -> list[tuple[float, str | None, str, float, list[dict]]]:
    """Every (position, kind, direction layer) with its best KL at true refusals <= 2 and no budget limit,
    lowest first, and the edits that reached it (the references' descents), as starts."""
    found = {}
    for position, land in scenario.positions.items():
        names = [land.name, twin(land)] if land.free else [land.name]
        for name in names:
            for label in ("wide", "zone"):
                entry = refs.get(f"{name} {label}")
                if entry is None:
                    continue
                bests = [(PER_LAYER, 0.0, entry["per layer"])]
                bests += [(ONE_LAYER, float(layer), b) for layer, b in entry["one layer"].items()]
                for scope, layer, b in bests:
                    if not b["params"]:
                        continue
                    kl, hints = found.get((position, scope, layer), (math.inf, []))
                    found[(position, scope, layer)] = (min(kl, b["kl"]), [*hints, b["params"]])
    return sorted((kl, p, s, layer, hints) for (p, s, layer), (kl, hints) in found.items())


def best_within(
    land: Landscape,
    budget: Budget,
    position,
    scope: str,
    layer: float,
    hints: list[dict],
    samples: int = 50_000,
    starts: int = 32,
    steps: int = 3000,
    seed: int = 0,
) -> tuple[float, dict]:
    """optimum.best with the budget limit as a second penalty: the least KL of one kind (at one direction
    layer) with true refusals <= CAP and expected budget hits <= LIMIT, over strength 0.5-3.0."""
    region = space(WIDE)
    generator = torch.Generator().manual_seed(seed)
    limit_r = math.log(CAP / (100 - CAP)) - 0.002
    limit_q = math.log(LIMIT / (100 - LIMIT)) - 0.002
    direction = torch.full((samples,), float(layer))

    def scored(raw):
        batch = _batch(raw, region, scope, direction[: len(next(iter(raw.values())))])
        logit, log_kl = land.logits(batch)
        return batch, logit, log_kl, budget.logit(batch, position, logit, log_kl)

    def penalty(logit, lq):
        return torch.relu(logit - limit_r) ** 2 + torch.relu(lq - limit_q) ** 2

    with torch.no_grad():
        raw = {n: torch.logit(torch.rand(samples, generator=generator).clamp(1e-4, 1 - 1e-4)) for n in NAMES}
        _, logit, log_kl, lq = scored(raw)
        top = torch.argsort(log_kl + 10 * penalty(logit, lq))[:starts]
        raw = {n: v[top] for n, v in raw.items()}
        known = {}
        for n in NAMES:
            low, high = region.range(n.split(".")[1])
            share = torch.tensor([(h[n] - low) / (high - low) for h in hints], dtype=torch.float64)
            known[n] = torch.logit(share.clamp(1e-6, 1 - 1e-6))
            raw[n] = torch.cat([raw[n], known[n]])
    raw = {n: v.clone().requires_grad_() for n, v in raw.items()}
    optimiser = torch.optim.Adam(raw.values(), lr=0.05)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps, eta_min=5e-4)
    for step in range(steps):
        optimiser.zero_grad()
        _, logit, log_kl, lq = scored(raw)
        (log_kl + 10 * 1000 ** (step / steps) * penalty(logit, lq)).sum().backward()
        optimiser.step()
        schedule.step()
    with torch.no_grad():
        batch, logit, log_kl, lq = scored({n: torch.cat([raw[n], known[n]]) for n in NAMES})
        feasible = (100 * torch.sigmoid(logit) <= CAP + 0.01) & (100 * torch.sigmoid(lq) <= LIMIT + 0.01)
        if not feasible.any():
            return math.inf, {}
        kl = torch.where(feasible, torch.exp(log_kl), torch.tensor(math.inf))
        i = int(torch.argmin(kl))
        return float(kl[i]), _params(batch, i, scope)


def _descend(task) -> tuple[float, dict]:
    land, budget, position, scope, layer, hints = task
    torch.set_num_threads(1)
    found = [
        best_within(land, budget, position, scope, layer, hints if s == 0 else [], seed=s) for s in range(3)
    ]
    return min(found, key=lambda f: f[0])


def find_truth(scenario: Scenario, budget: Budget, refs: dict, pool=None, known: tuple = ()) -> Truth:
    """Branch and bound over (position, kind, direction layer), lowest unlimited KL first: a candidate
    whose best edit is within the budget limit is done; otherwise descend with both limits. Stops once
    the next candidate's unlimited best can't beat the best found. Known edits (Truths found before)
    start it if they are within both limits here."""
    best = Truth(math.inf, math.nan, math.nan, None, {}, math.inf)
    candidates = _candidates(scenario, refs)
    best.unlimited_kl = candidates[0][0]
    for edit in known:
        land = scenario.positions[edit.position]
        rate, kl = land.true([edit.params])
        hits = budget.hits(land, edit.position, [edit.params])[0]
        if 100 * rate[0] <= CAP + 0.01 and hits <= LIMIT and kl[0] < best.kl:
            best = Truth(
                float(kl[0]), float(100 * rate[0]), float(hits), edit.position, edit.params, best.unlimited_kl
            )
    todo = []
    for lower, position, scope, layer, hints in candidates:
        if lower >= best.kl:
            break
        land = scenario.positions[position]
        params = [{**h, "direction_scope": scope, "direction_layer": layer} for h in hints]
        rate, kl = land.true(params)
        hits = budget.hits(land, position, params)
        ok = (100 * rate <= CAP + 0.01) & (hits <= LIMIT)
        if ok.any():
            i = int(np.argmin(np.where(ok, kl, np.inf)))
            if kl[i] < best.kl:
                best = Truth(
                    float(kl[i]), float(100 * rate[i]), float(hits[i]), position, params[i], best.unlimited_kl
                )
        if not ok.any() or kl[ok].min() > lower * 1.0001:
            todo.append((lower, (land, budget, position, scope, layer, params)))
    # Descents for the candidates whose own best fails the limit, in batches, bounded as they come in.
    while todo:
        todo = [t for t in todo if t[0] < best.kl]
        batch, todo = todo[:12], todo[12:]
        results = (pool.map if pool else map)(_descend, [t[1] for t in batch])
        for (_, task), (kl, params) in zip(batch, results, strict=True):
            if kl < best.kl:
                land, _, position, scope, layer, _ = task
                params = {**params, "direction_scope": scope, "direction_layer": layer}
                rate, _ = land.true([params])
                hits = budget.hits(land, position, [params])
                best = Truth(kl, float(100 * rate[0]), float(hits[0]), position, params, best.unlimited_kl)
    return best


# ---- the stress bumps


def _near(
    land: Landscape, params: dict, kl: float, refusals: float, count: int = 4000, seed: int = 0
) -> list[dict]:
    """Edits of the edit's kind, position and direction layer with true KL <= kl and true refusals <=
    refusals: the edit moved by normal steps of 0.02-0.2 range-widths in its eight settings."""
    rng = np.random.default_rng(seed)
    x = coords(Batch.from_params([params]))[0].numpy()
    found = []
    for scale in (0.02, 0.05, 0.1, 0.2):
        moved = x[None, 1:] + scale * rng.standard_normal((count, len(NAMES)))
        drawn = []
        for row in moved:
            p = dict(params)
            for name, share in zip(NAMES, row, strict=True):
                low, high = RANGES[name.split(".")[1]]
                p[name] = float(low + np.clip(share, 0, 1) * (high - low))
            drawn.append(p)
        rate, kls = land.true(drawn)
        found += [p for p, r, k in zip(drawn, rate, kls, strict=True) if 100 * r <= refusals and k <= kl]
    return found


def _bump(
    land: Landscape,
    budget: Budget,
    position,
    params: dict,
    near: list[dict],
    side_seed: int,
    share: float = 0.5,
) -> tuple[Budget, dict]:
    """The budget landscape with one more bump beside an edit: centred off it towards its near edits on a
    random side of it, as wide as they spread, its height set so the edit has STRESS_AT of budget hits
    (or a little more than it had), and its distance chosen so about `share` of the near edits fail."""
    x0 = coords(Batch.from_params([params]))[0]
    xs = coords(Batch.from_params(near))
    offsets = xs - x0
    side = torch.from_numpy(np.random.default_rng(side_seed).standard_normal(xs.shape[1]))
    side[0] = 0.0
    direction = offsets[(offsets @ side) > 0].mean(0)
    direction = direction / direction.norm()
    width = float(offsets.pow(2).sum(1).mean().sqrt())
    before = budget.hits(land, position, [params])[0] / 100
    lift = max(math.log(STRESS_AT / (1 - STRESS_AT)) - math.log(before / (1 - before)), 0.05)
    one = params["direction_scope"] == ONE_LAYER
    failed_before = float((budget.hits(land, position, near) > LIMIT).mean())
    tried = []
    for k in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
        bump = {
            "position": position,
            "one": one,
            "centre": (x0 + k * width * direction).tolist(),
            "width": width,
            "height": lift * math.exp(k * k / 2),
        }
        stressed = Budget(budget.best_kl, budget.fields, budget.intercept, (*budget.bumps, bump))
        failed = float((stressed.hits(land, position, near) > LIMIT).mean())
        tried.append((abs(failed - share), k, stressed, failed, bump))
    _, k, stressed, failed, bump = min(tried, key=lambda t: t[0])
    return stressed, {
        "near": len(near),
        "k": k,
        "width": width,
        "height": bump["height"],
        "failed": failed,
        "failed_before": failed_before,
    }


def stress(scenario: Scenario, base: Budget, truth_: Truth) -> tuple[Budget, dict]:
    """The base budget landscape with one bump beside the best edit (see the module's docstring)."""
    land = scenario.positions[truth_.position]
    near = _near(land, truth_.params, 1.10 * truth_.kl, CAP)
    if len(near) < 50:
        raise SystemExit(f"{scenario.name}: only {len(near)} edits near the best")
    return _bump(land, base, truth_.position, truth_.params, near, _seed("stress side", scenario.name))


def cliffs(scenario: Scenario, base: Budget, truth_: Truth, refs: dict) -> tuple[Budget, list[dict]]:
    """The base budget landscape with a bump beside each basin whose best is within 10% of the
    scenario's best (see the module's docstring), lowest KL first."""
    budget, made = base, []
    for lower, position, scope, layer, hints in _candidates(scenario, refs):
        if lower > 1.10 * truth_.kl:
            break
        land = scenario.positions[position]
        params = [{**h, "direction_scope": scope, "direction_layer": layer} for h in hints]
        rate, kl = land.true(params)
        ok = 100 * rate <= CAP + 0.01
        if not ok.any():
            continue
        best = params[int(np.argmin(np.where(ok, kl, np.inf)))]
        near = _near(land, best, 1.10 * truth_.kl, CLIFF_REFUSALS)
        if len(near) < 50:
            continue
        budget, how = _bump(
            land, budget, position, best, near, _seed("cliff side", scenario.name, position, scope, layer)
        )
        made.append({"position": position, "scope": scope, "layer": layer, **how})
    return budget, made


# ---- each scenario's budget landscape and truth, cached


def budget_for(name: str, variant: str, intercept: float = INTERCEPT) -> Budget:
    scenario = scenarios()[name]
    refs = load_refs()
    base = Budget(truth(scenario, WIDE, refs).kl, fields(name, list(scenario.positions)), intercept)
    if variant == "base":
        return base
    path = CACHE / f"budget-{variant}-{name}-{intercept:g}.json"
    if not path.exists():
        raise SystemExit(f"no {variant} bumps for {name}: run budget.py truth first")
    saved = json.loads(path.read_text())
    bumps = [saved["bump"]] if "bump" in saved else saved["bumps"]
    return Budget(base.best_kl, base.fields, intercept, tuple(bumps))


def truth_for(name: str, variant: str, intercept: float = INTERCEPT) -> Truth:
    path = CACHE / f"budget-truth-{variant}-{name}-{intercept:g}.json"
    if not path.exists():
        raise SystemExit(f"no truth for {name} ({variant}): run budget.py truth first")
    return Truth(**json.loads(path.read_text()))


def _save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=1, default=float))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["truth"])
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--stress", default="", help="scenarios to build a stress bump for")
    parser.add_argument("--cliffs", default="", help="scenarios to build cliffs for")
    parser.add_argument("--intercept", type=float, default=INTERCEPT)
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()
    refs = load_refs()
    names = [n for n in args.scenarios.split(",") if n]
    with Pool(args.jobs) as pool:
        others = [n for n in f"{args.stress},{args.cliffs}".split(",") if n and n not in names]
        for name in names + list(dict.fromkeys(others)):
            scenario = scenarios()[name]
            path = CACHE / f"budget-truth-base-{name}-{args.intercept:g}.json"
            if path.exists():
                found = Truth(**json.loads(path.read_text()))
            else:
                found = find_truth(scenario, budget_for(name, "base", args.intercept), refs, pool)
                _save(path, found.__dict__)
            print(
                f"{name:10s} base   best KL {found.kl:.4f} (no limit {found.unlimited_kl:.4f}, "
                f"{found.kl / found.unlimited_kl - 1:+.1%}), refusals {found.refusals:.2f}, "
                f"hits {found.hits:.2f}",
                flush=True,
            )
        for name in [n for n in args.stress.split(",") if n]:
            scenario = scenarios()[name]
            path = CACHE / f"budget-stress-{name}-{args.intercept:g}.json"
            base_truth = truth_for(name, "base", args.intercept)
            if not path.exists():
                stressed, how = stress(scenario, budget_for(name, "base", args.intercept), base_truth)
                _save(path, {"bump": stressed.bumps[-1], **how})
            saved = json.loads(path.read_text())
            truth_path = CACHE / f"budget-truth-stress-{name}-{args.intercept:g}.json"
            if not truth_path.exists():
                found = find_truth(
                    scenario, budget_for(name, "stress", args.intercept), refs, pool, (base_truth,)
                )
                _save(truth_path, found.__dict__)
            found = truth_for(name, "stress", args.intercept)
            print(
                f"{name:10s} stress best KL {found.kl:.4f} (base {base_truth.kl:.4f}), "
                f"hits {found.hits:.2f}; "
                f"{saved['near']} near edits, {saved['failed_before']:.0%} fail the limit before the bump, "
                f"{saved['failed']:.0%} with it (k {saved['k']}, width {saved['width']:.3f}, "
                f"height {saved['height']:.1f})",
                flush=True,
            )
        for name in [n for n in args.cliffs.split(",") if n]:
            scenario = scenarios()[name]
            path = CACHE / f"budget-cliffs-{name}-{args.intercept:g}.json"
            base_truth = truth_for(name, "base", args.intercept)
            if not path.exists():
                stressed, made = cliffs(scenario, budget_for(name, "base", args.intercept), base_truth, refs)
                _save(path, {"bumps": list(stressed.bumps), "made": made})
            saved = json.loads(path.read_text())
            truth_path = CACHE / f"budget-truth-cliffs-{name}-{args.intercept:g}.json"
            if not truth_path.exists():
                found = find_truth(
                    scenario, budget_for(name, "cliffs", args.intercept), refs, pool, (base_truth,)
                )
                _save(truth_path, found.__dict__)
            found = truth_for(name, "cliffs", args.intercept)
            made = saved["made"]
            print(
                f"{name:10s} cliffs best KL {found.kl:.4f} (base {base_truth.kl:.4f}), "
                f"hits {found.hits:.2f}; "
                f"{len(made)} basins, near edits failing the limit "
                + ", ".join(f"{m['failed_before']:.0%}->{m['failed']:.0%}" for m in made),
                flush=True,
            )


if __name__ == "__main__":
    main()
