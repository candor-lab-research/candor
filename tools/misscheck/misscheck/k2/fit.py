"""How K2-Horizon-7B's budget hits depend on its edits, from k2/trials.csv (numbers only).

A binomial model of each trial's budget hits on the prompts it judged, with a normal random effect per
edit on the logit (40-point Gauss-Hermite quadrature):

    logit q = a + b log(KL / KL*) + c refusals/judged [+ kind and position terms] + s z

KL* is the least KL of K2's trials scored in full within 2 refusals. Then, from the kl+refusals fit, each
trial's posterior mean z, and how alike z is for pairs of edits of the same kind and position by how
far apart their settings are (shares of K2's ranges), for the smooth field's length in budget.py.

    uv run python k2/fit.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_default_dtype(torch.float64)
trials = pd.read_csv(Path(__file__).parent / "trials.csv")
best = trials[(trials.judged == 100) & (trials.refusals <= 2)].kl.min()
x_kl = torch.tensor(np.log(trials.kl / best).values)
x_ref = torch.tensor((trials.refusals / trials.judged).values)
one = torch.tensor(trials.one.values, dtype=torch.float64)
answer = torch.tensor(trials.answer.values, dtype=torch.float64)
n = torch.tensor(trials.judged.values, dtype=torch.float64)
k = torch.tensor(trials.forced.values, dtype=torch.float64)
nodes, weights = np.polynomial.hermite_e.hermegauss(40)
nodes, log_weights = torch.tensor(nodes), torch.log(torch.tensor(weights / weights.sum()))


def fit(columns: list[torch.Tensor]) -> tuple[float, list[float], float]:
    p = torch.zeros(len(columns) + 2, requires_grad=True)
    optimiser = torch.optim.LBFGS([p], max_iter=500, line_search_fn="strong_wolfe")

    def loss():
        optimiser.zero_grad()
        eta = p[0] + sum(p[i + 1] * c for i, c in enumerate(columns)) + torch.zeros(len(n))
        logit = eta[:, None] + torch.nn.functional.softplus(p[-1]) * nodes[None, :]
        ll = k[:, None] * torch.nn.functional.logsigmoid(logit) + (n - k)[
            :, None
        ] * torch.nn.functional.logsigmoid(-logit)
        out = -torch.logsumexp(ll + log_weights[None, :], 1).sum()
        out.backward()
        return out

    optimiser.step(loss)
    return float(loss().detach()), p[:-1].tolist(), float(torch.nn.functional.softplus(p[-1].detach()))


print(f"KL* {best:.4f}; {len(trials)} trials")
models = {
    "kl": [x_kl],
    "kl+refusals": [x_kl, x_ref],
    "kl+refusals+kind+position": [x_kl, x_ref, one, answer],
    "kl+refusals+kind*position": [x_kl, x_ref, one, answer, one * answer],
    "refusals": [x_ref],
}
fits = {}
for name, columns in models.items():
    nll, coef, spread = fit(columns)
    fits[name] = (coef, spread)
    print(
        f"{name:26s} AIC {2 * nll + 2 * (len(coef) + 1):8.2f}  "
        f"coefficients {np.round(coef, 3)}  s {spread:.3f}"
    )

(a, b, c), s = fits["kl+refusals"]
eta = a + b * x_kl.numpy() + c * x_ref.numpy()
grid, w = np.polynomial.hermite_e.hermegauss(60)
w = w / w.sum()
z = []
for e, hits, judged in zip(eta, trials.forced, trials.judged, strict=True):
    logit = e + s * grid
    ll = hits * -np.logaddexp(0, -logit) + (judged - hits) * -np.logaddexp(0, logit)
    post = w * np.exp(ll - ll.max())
    z.append((post * grid).sum() / post.sum())
z = np.array(z)
ranges = {
    f"{part}.{setting}": span
    for part in ("attn", "mlp")
    for setting, span in (
        ("strength", (0.8, 5.0)),
        ("peak_layer", (0.6, 1.0)),
        ("edge_strength", (0.0, 1.0)),
        ("reach", (0.01, 0.6)),
    )
}
x = np.stack([(trials[name] - low) / (high - low) for name, (low, high) in ranges.items()], 1)
layer = ((trials.direction_layer.fillna(0) - 14) / 18).values
pairs = []
for i in range(len(trials)):
    for j in range(i):
        if trials.one[i] == trials.one[j] and trials.answer[i] == trials.answer[j]:
            far = np.sqrt(((x[i] - x[j]) ** 2).sum() + trials.one[i] * (layer[i] - layer[j]) ** 2)
            pairs.append((far, z[i], z[j]))
pairs = np.array(pairs)
for low, high in [(0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 3)]:
    m = (pairs[:, 0] >= low) & (pairs[:, 0] < high)
    print(
        f"pairs {low}-{high} apart: {m.sum():5d}, "
        f"correlation of z {np.corrcoef(pairs[m, 1], pairs[m, 2])[0, 1]:+.2f}"
    )
