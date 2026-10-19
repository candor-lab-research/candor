"""Fit a landscape's shape to candor's and heretic's searches on rude-llama.

candor's 200 trials are on candor's scale (judge refusals, llama.cpp KL). heretic's are on its own
scale (keyword refusals, which over-count, and its own KL), so they are tied to candor's by a fitted
monotone map: keyword rate = f + (1 - f) sigmoid(u0 + u1 logit p), log KL_h = k0 + k1 log KL, with
u0 and k0 free to differ between one-layer and per-layer edits (heretic's own directions and prompts
need not shift both alike). Trial 170 scored by candor's own scorer anchors the one-layer map. heretic's one-layer trials reading layers above 17.5 are
left out: candor's own trials there show keywords missing most of the judge's refusals.

    uv run python fit.py              # fit, save fitted.json, print the checks
    uv run python fit.py --folds 5    # also cross-validate on candor's trials
"""

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from landscape import COMPONENTS, ONE_LAYER, PER_LAYER, SETTINGS, Batch, default_shape, evaluate, save_shape

HERE = Path(__file__).parent
DATA = HERE.parent
# Trial 170 of heretic's search, as candor's scorer scored it (CONTEXT.md).
TRIAL_170 = {
    "attn.strength": 1.327, "attn.peak_layer": 0.611, "attn.edge_strength": 0.79, "attn.reach": 0.50,
    "mlp.strength": 1.26, "mlp.peak_layer": 0.615, "mlp.edge_strength": 0.45, "mlp.reach": 0.474,
}
ANCHORS = [(12.039, 0, 0.0834), (12.0, 0, 0.0818)]
ANCHOR_WEIGHT = 10.0
LATE_KEYWORDS = 17.5


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path)))


def params_of(row: dict) -> dict:
    p = {f"{c}.{s}": float(row[f"{c}.{s}"]) for c in COMPONENTS for s in SETTINGS}
    p["direction_scope"] = ONE_LAYER if row["scope"] == "one_layer" else PER_LAYER
    p["direction_layer"] = float(row["direction_layer"]) if row["direction_layer"] else 0.0
    return p


def candor_data(with_anchors: bool = True) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    rows = read(DATA / "candor_search_rude.csv")
    params = [params_of(r) for r in rows]
    refusals = [int(r["refusals_judge"]) for r in rows]
    kls = [float(r["kl"]) for r in rows]
    weights = [1.0] * len(rows)
    if with_anchors:
        for layer, refused, kl in ANCHORS:
            params.append({**TRIAL_170, "direction_scope": ONE_LAYER, "direction_layer": layer})
            refusals.append(refused)
            kls.append(kl)
            weights.append(ANCHOR_WEIGHT)
    return params, np.array(refusals), np.array(kls), np.array(weights)


def heretic_data() -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows = [
        r for r in read(DATA / "heretic_search_rude.csv")
        if r["scope"] == "per_layer" or float(r["direction_layer"]) <= LATE_KEYWORDS
    ]
    return [params_of(r) for r in rows], np.array([int(r["refusals_keywords"]) for r in rows]), np.array(
        [float(r["kl_heretic_scale"]) for r in rows]
    )


def beta_binomial(k, n, p, concentration):
    n = torch.full_like(k, n)
    p = p.clamp(1e-6, 1 - 1e-6)
    a, b = p * concentration, (1 - p) * concentration
    return (
        torch.lgamma(n + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(n - k + 1.0)
        + torch.lgamma(k + a) + torch.lgamma(n - k + b) - torch.lgamma(n + a + b)
        - torch.lgamma(a) - torch.lgamma(b) + torch.lgamma(a + b)
    )


def mapping_start() -> dict[str, torch.Tensor]:
    return {
        "keyword_floor": torch.tensor(-3.5),
        "keyword_shift": torch.tensor(0.0),
        "keyword_slope": torch.tensor(1.0),
        "kl_shift": torch.tensor(-1.0),
        "kl_shift_one": torch.tensor(0.0),
        "keyword_shift_one": torch.tensor(0.0),
        "kl_slope": torch.tensor(1.0),
        "log_sigma_candor": torch.tensor(-1.0),
        "log_sigma_heretic": torch.tensor(-1.0),
        "log_conc_candor": torch.tensor(3.0),
        "log_conc_heretic": torch.tensor(3.0),
    }


def smoothness(shape) -> torch.Tensor:
    """Curvature of the fitted curves. What depends on the direction layer is let bend more: one
    layer's direction can differ sharply from the next one's (candor's layer 11 does little)."""
    total = torch.tensor(0.0)
    for name, value in shape.items():
        if value.dim() == 1:
            weight = 0.05 if name.endswith("by_layer") else 1.0
            total = total + weight * (value[2:] - 2 * value[1:-1] + value[:-2]).pow(2).sum()
    return total


def fit(candor, heretic, steps: int = 3000, seed: int = 0, smooth: float = 1.0, verbose: bool = False):
    torch.manual_seed(seed)
    c_params, c_ref, c_kl, c_w = candor
    h_params, h_ref, h_kl = heretic
    c_batch, h_batch = Batch.from_params(c_params), Batch.from_params(h_params)
    c_ref, c_kl, c_w = torch.tensor(c_ref, dtype=torch.float64), torch.tensor(c_kl), torch.tensor(c_w)
    h_ref, h_kl = torch.tensor(h_ref, dtype=torch.float64), torch.tensor(h_kl)
    shape = default_shape()
    for name, value in shape.items():
        shape[name] = (value + 0.3 * torch.randn_like(value)).requires_grad_()
    extra = {k: v.clone().requires_grad_() for k, v in mapping_start().items()}
    optimiser = torch.optim.Adam([*shape.values(), *extra.values()], lr=0.05)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)
    for step in range(steps):
        optimiser.zero_grad()
        loss = -log_likelihood(shape, extra, c_batch, c_ref, c_kl, c_w, h_batch, h_ref, h_kl) + smooth * smoothness(shape)
        loss.backward()
        optimiser.step()
        schedule.step()
        if verbose and step % 500 == 0:
            print(step, round(loss.item(), 2))
    return {k: v.detach() for k, v in shape.items()}, {k: v.detach() for k, v in extra.items()}, loss.item()


def log_likelihood(shape, extra, c_batch, c_ref, c_kl, c_w, h_batch, h_ref, h_kl):
    logit, log_kl = evaluate(shape, c_batch)
    sigma_c = torch.exp(extra["log_sigma_candor"])
    total = (c_w * beta_binomial(c_ref, 100.0, torch.sigmoid(logit), torch.exp(extra["log_conc_candor"]))).sum()
    total = total + (c_w * torch.distributions.Normal(log_kl, sigma_c).log_prob(torch.log(c_kl))).sum()
    logit_h, log_kl_h = evaluate(shape, h_batch)
    floor = torch.sigmoid(extra["keyword_floor"])
    one = h_batch.one_layer.double()
    shift = extra["keyword_shift"] + one * extra["keyword_shift_one"]
    keyword_rate = floor + (1 - floor) * torch.sigmoid(shift + extra["keyword_slope"] * logit_h)
    total = total + beta_binomial(h_ref, 100.0, keyword_rate, torch.exp(extra["log_conc_heretic"])).sum()
    mapped = extra["kl_shift"] + one * extra["kl_shift_one"] + extra["kl_slope"] * log_kl_h
    total = total + torch.distributions.Normal(mapped, torch.exp(extra["log_sigma_heretic"])).log_prob(torch.log(h_kl)).sum()
    return total


def best_of(candor, heretic, starts: int, steps: int):
    fits = [fit(candor, heretic, steps=steps, seed=s) for s in range(starts)]
    return min(fits, key=lambda f: f[2])


def correlation(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def predict(shape, params):
    logit, log_kl = evaluate(shape, Batch.from_params(params))
    return 100 * torch.sigmoid(logit).numpy(), torch.exp(log_kl).numpy()


def report(shape, extra, label: str) -> None:
    params, refusals, kls, _ = candor_data(with_anchors=False)
    predicted_ref, predicted_kl = predict(shape, params)
    one = np.array([p["direction_scope"] == ONE_LAYER for p in params])
    print(f"\n== {label}")
    for name, mask in (("all", np.ones_like(one)), ("one layer", one), ("per layer", ~one)):
        mask = mask.astype(bool)
        print(
            f"candor {name:9s} n={mask.sum():3d}: refusals r={correlation(predicted_ref[mask], refusals[mask]):.3f} "
            f"MAE={np.abs(predicted_ref[mask] - refusals[mask]).mean():5.1f}; log KL r="
            f"{correlation(np.log(predicted_kl[mask]), np.log(kls[mask])):.3f} median |error|="
            f"{np.median(np.abs(predicted_kl[mask] / kls[mask] - 1)) * 100:4.1f}%"
        )
    for layer, refused, kl in ANCHORS:
        r, k = predict(shape, [{**TRIAL_170, "direction_scope": ONE_LAYER, "direction_layer": layer}])
        print(f"Trial 170 at layer {layer}: landscape {r[0]:.1f} refused, KL {k[0]:.4f}; scored {refused}, {kl}")
    print("mapping:", {k: round(float(v), 3) for k, v in extra.items()})


def cross_validate(folds: int, starts: int, steps: int) -> None:
    params, refusals, kls, weights = candor_data(with_anchors=False)
    heretic = heretic_data()
    order = np.random.default_rng(0).permutation(len(params))
    predicted_ref, predicted_kl = np.zeros(len(params)), np.zeros(len(params))
    for fold in range(folds):
        test = order[fold::folds]
        train = np.setdiff1d(order, test)
        anchors = candor_data(with_anchors=True)
        train_set = (
            [params[i] for i in train] + anchors[0][-len(ANCHORS):],
            np.concatenate([refusals[train], anchors[1][-len(ANCHORS):]]),
            np.concatenate([kls[train], anchors[2][-len(ANCHORS):]]),
            np.concatenate([weights[train], anchors[3][-len(ANCHORS):]]),
        )
        shape, _, _ = best_of(train_set, heretic, starts, steps)
        r, k = predict(shape, [params[i] for i in test])
        predicted_ref[test], predicted_kl[test] = r, k
    one = np.array([p["direction_scope"] == ONE_LAYER for p in params])
    print(f"\n== {folds}-fold cross-validation on candor's trials (out of sample)")
    for name, mask in (("all", np.ones_like(one)), ("one layer", one), ("per layer", ~one)):
        mask = mask.astype(bool)
        print(
            f"candor {name:9s}: refusals r={correlation(predicted_ref[mask], refusals[mask]):.3f} "
            f"MAE={np.abs(predicted_ref[mask] - refusals[mask]).mean():5.1f}; log KL r="
            f"{correlation(np.log(predicted_kl[mask]), np.log(kls[mask])):.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--folds", type=int, default=0)
    parser.add_argument("--out", default=str(HERE / "fitted.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.time()
    shape, extra, loss = best_of(candor_data(), heretic_data(), args.starts, args.steps)
    save_shape(shape, Path(args.out))
    save_shape(extra, Path(args.out).with_name("fitted_mapping.json"))
    report(shape, extra, f"fit with anchors (loss {loss:.1f})")
    free, free_extra, free_loss = best_of(candor_data(with_anchors=False), heretic_data(), args.starts, args.steps)
    report(free, free_extra, f"fit WITHOUT the Trial 170 anchors (loss {free_loss:.1f}): does the data alone predict it?")
    if args.folds:
        cross_validate(args.folds, max(1, args.starts // 2), args.steps)
    print(f"\n{time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
