"""How big each landscape's good regions are: the share of uniform random draws (the startup phase)
that land at or below a refusal count, and their best KL."""
import numpy as np
from surrogate import landscapes
from optimum import snap

rng = np.random.default_rng(1)
for name, scopes in landscapes().items():
    for s, scope in scopes.items():
        one = s == "one"
        u = snap(rng.random((400_000, 9 if one else 8)), one)
        r, k = scope(u)
        row = []
        for cap in (2, 5, 10, 20, 35):
            m = r <= cap
            row.append(f"<= {cap}: {m.mean():.4%}")
        print(f"{name:8s} {s}: " + ", ".join(row))
