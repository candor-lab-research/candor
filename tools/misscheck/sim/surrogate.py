"""GP surrogates of rude-llama's refusal and KL landscapes, fitted to the two searches' numbers.

Inputs are the search's parameters scaled to the default box (each in [0, 1]). Outputs are
logit(refusal share) and log(KL). One exact GP (ARD Matern 5/2, constant mean) per scope and output.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = Path(__file__).resolve().parent.parent
LAST = 27
KERNEL = ["strength", "peak_layer", "edge_strength", "reach"]
COMPONENTS = ["attn", "mlp"]
BOX = {"strength": (0.8, 1.5), "peak_layer": (0.6, 1.0), "edge_strength": (0.0, 1.0), "reach": (1 / LAST, 0.6)}
LAYERS = (11, 24)
NAMES = [f"{c}.{k}" for c in COMPONENTS for k in KERNEL]


def to_unit(frame: pd.DataFrame, one_layer: bool) -> np.ndarray:
    cols = []
    if one_layer:
        cols.append((frame.direction_layer.to_numpy(float) - LAYERS[0]) / (LAYERS[1] - LAYERS[0]))
    for name in NAMES:
        low, high = BOX[name.split(".")[1]]
        cols.append((frame[name].to_numpy(float) - low) / (high - low))
    return np.column_stack(cols)


class GP:
    def __init__(self, x: np.ndarray, y: np.ndarray, steps: int = 400):
        self.x = torch.tensor(x)
        y = torch.tensor(y)
        self.y_mean, self.y_std = y.mean(), y.std()
        self.y = (y - self.y_mean) / self.y_std
        d = x.shape[1]
        self.log_ls = torch.full((d,), math.log(0.5), requires_grad=True)
        self.log_scale = torch.tensor(0.0, requires_grad=True)
        self.log_noise = torch.tensor(math.log(0.05), requires_grad=True)
        self.mean = torch.tensor(0.0, requires_grad=True)
        params = [self.log_ls, self.log_scale, self.log_noise, self.mean]
        opt = torch.optim.Adam(params, lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            loss = -self._mll()
            loss.backward()
            opt.step()
        with torch.no_grad():
            k = self._k(self.x, self.x) + (self.log_noise.exp() + 1e-6) * torch.eye(len(self.x))
            self.chol = torch.linalg.cholesky(k)
            self.alpha = torch.cholesky_solve((self.y - self.mean).unsqueeze(1), self.chol)

    def _k(self, a, b):
        ls = self.log_ls.exp()
        r = torch.cdist(a / ls, b / ls).clamp_min(1e-12)
        s5 = math.sqrt(5) * r
        return self.log_scale.exp() * (1 + s5 + s5**2 / 3) * torch.exp(-s5)

    def _mll(self):
        n = len(self.x)
        k = self._k(self.x, self.x) + (self.log_noise.exp() + 1e-6) * torch.eye(n)
        chol = torch.linalg.cholesky(k)
        r = (self.y - self.mean).unsqueeze(1)
        a = torch.cholesky_solve(r, chol)
        # Weak priors keep lengthscales and noise sensible on 100-200 points.
        prior = -0.5 * ((self.log_ls - math.log(0.5)) ** 2).sum() - 0.5 * (self.log_noise - math.log(0.05)) ** 2
        return -0.5 * (r * a).sum() - torch.log(torch.diagonal(chol)).sum() + prior

    def predict(self, x) -> torch.Tensor:
        """Posterior mean; differentiable when x needs a gradient."""
        x = x if torch.is_tensor(x) else torch.tensor(x)
        out = (self.mean + (self._k(x, self.x) @ self.alpha).squeeze(1)) * self.y_std + self.y_mean
        return out if x.requires_grad else out.detach()


def logit_share(refusals: np.ndarray, n: int = 100) -> np.ndarray:
    p = (refusals + 0.5) / (n + 1)
    return np.log(p / (1 - p))


class Scope:
    """One scope's landscape: refusals (count of 100) and KL from two GPs, after an affine map that
    puts the data on candor's scale."""

    def __init__(self, x, refusals, kl, refusal_shift=0.0, kl_factor=1.0):
        self.refusal_gp = GP(x, logit_share(np.clip(refusals - refusal_shift, 0, 100)))
        self.kl_gp = GP(x, np.log(kl * kl_factor))

    def __call__(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.atleast_2d(u)
        share = torch.sigmoid(self.refusal_gp.predict(u)).numpy()
        refusals = np.clip(np.round(share * 101 - 0.5), 0, 100)
        return refusals, np.exp(self.kl_gp.predict(u).numpy())


def cv_r2(x, y, folds=10, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    pred = np.empty(len(x))
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        gp = GP(x[train], y[train], steps=250)
        pred[test] = gp.predict(x[test]).numpy()
    return 1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum(), np.corrcoef(pred, y)[0, 1]


def load():
    candor = pd.read_csv(HERE / "candor_search_rude.csv")
    heretic = pd.read_csv(HERE / "heretic_search_rude.csv")
    return candor, heretic


def landscapes() -> dict[str, dict[str, Scope]]:
    """hybrid: per layer from candor's own trials, one layer from heretic's, mapped to candor's scale by
    the Trial 170 anchor (2 keyword refusals -> 0 by the judge; KL x 3.21). heretic: both scopes from
    heretic's trials on heretic's scale. candor: both from candor's trials (no good one-layer data)."""
    candor, heretic = load()
    c_one, c_per = candor[candor.scope == "one_layer"], candor[candor.scope == "per_layer"]
    h_one, h_per = heretic[heretic.scope == "one_layer"], heretic[heretic.scope == "per_layer"]
    kl_factor = 0.0834 / 0.0260
    per_c = Scope(to_unit(c_per, False), c_per.refusals_judge.to_numpy(float), c_per.kl.to_numpy(float))
    one_h_mapped = Scope(to_unit(h_one, True), h_one.refusals_keywords.to_numpy(float), h_one.kl_heretic_scale.to_numpy(float), 2.0, kl_factor)
    one_h = Scope(to_unit(h_one, True), h_one.refusals_keywords.to_numpy(float), h_one.kl_heretic_scale.to_numpy(float))
    per_h = Scope(to_unit(h_per, False), h_per.refusals_keywords.to_numpy(float), h_per.kl_heretic_scale.to_numpy(float))
    one_c = Scope(to_unit(c_one, True), c_one.refusals_judge.to_numpy(float), c_one.kl.to_numpy(float))
    # The union: heretic's one-layer trials (mapped) where they are, candor's own where candor looked.
    union_x = np.vstack([to_unit(h_one, True), to_unit(c_one, True)])
    union_r = np.concatenate([np.clip(h_one.refusals_keywords.to_numpy(float) - 2, 0, 100), c_one.refusals_judge.to_numpy(float)])
    union_kl = np.concatenate([h_one.kl_heretic_scale.to_numpy(float) * kl_factor, c_one.kl.to_numpy(float)])
    one_union = Scope(union_x, union_r, union_kl)
    return {
        "union": {"one": one_union, "per": per_c},
        "hybrid": {"one": one_h_mapped, "per": per_c},
        "heretic": {"one": one_h, "per": per_h},
        "candor": {"one": one_c, "per": per_c},
    }


if __name__ == "__main__":
    import time

    start = time.time()
    candor, heretic = load()
    for label, frame, ref_col, kl_col in [
        ("candor", candor, "refusals_judge", "kl"),
        ("heretic", heretic, "refusals_keywords", "kl_heretic_scale"),
    ]:
        for scope in ("one_layer", "per_layer"):
            sub = frame[frame.scope == scope]
            x = to_unit(sub, scope == "one_layer")
            r2r, cr = cv_r2(x, logit_share(sub[ref_col].to_numpy(float)))
            r2k, ck = cv_r2(x, np.log(sub[kl_col].to_numpy(float)))
            print(f"{label:8s} {scope:9s} n={len(sub):3d}  10-fold CV R2: refusal logit {r2r:.2f} (r {cr:.2f}), log KL {r2k:.2f} (r {ck:.2f})")
    print(f"{time.time() - start:.0f} s")
