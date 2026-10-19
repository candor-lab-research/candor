"""The best each landscape allows: least KL with refusals <= cap, per scope, by dense random search
plus a (1+1) evolution strategy from the best few points. Cached to optima.json."""

import json
import time
from pathlib import Path

import numpy as np

from surrogate import landscapes

CAPS = (0, 2, 5)
HERE = Path(__file__).resolve().parent


def snap(u: np.ndarray, one: bool) -> np.ndarray:
    u = np.clip(u, 0, 1)
    if one:
        u[:, 0] = np.round(u[:, 0] * 13) / 13
    return u


def best_under(scope, one: bool, cap: int, rng, n=200_000) -> tuple[float, list[float]]:
    d = 9 if one else 8
    u = snap(rng.random((n, d)), one)
    refusals, kl = scope(u)
    kl = np.where(refusals <= cap, kl, np.inf)
    starts = u[np.argsort(kl)[:16]]
    best_kl, best_u = np.inf, None
    for start in starts:
        x, fx = start.copy(), kl[np.argsort(kl)[0]] if False else None
        r, k = scope(x[None])
        fx = k[0] if r[0] <= cap else np.inf
        step = 0.05
        for _ in range(60):
            cand = snap(x + rng.normal(0, step, (32, d)), one)
            r, k = scope(cand)
            k = np.where(r <= cap, k, np.inf)
            i = np.argmin(k)
            if k[i] < fx:
                x, fx = cand[i], k[i]
            else:
                step *= 0.85
        if fx < best_kl:
            best_kl, best_u = fx, x
    return float(best_kl), [float(v) for v in best_u] if best_u is not None else []


def main():
    start = time.time()
    rng = np.random.default_rng(0)
    out = {}
    for name, scopes in landscapes().items():
        out[name] = {}
        for scope_name, scope in scopes.items():
            for cap in CAPS:
                kl, u = best_under(scope, scope_name == "one", cap, rng)
                out[name][f"{scope_name}@{cap}"] = {"kl": kl, "u": u}
                print(name, scope_name, cap, round(kl, 4), np.round(u, 2).tolist(), flush=True)
    (HERE / "optima.json").write_text(json.dumps(out, indent=1))
    print(f"{time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
