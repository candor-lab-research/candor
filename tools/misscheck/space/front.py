"""Model-based guess at the one-layer family's front in candor's KL scale: heretic's refusal model x candor's KL model."""
import sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from load import candor, heretic
from profiles import profiles
from surrogate import Ridge, design

rng = np.random.default_rng(1)
c, h = candor(), heretic()
h1, c1 = h[h.scope == "one_layer"], c[c.scope == "one_layer"]
refusal = Ridge(design(h1, True), np.log((h1.ref.values + .5) / (100 - h1.ref.values + .5)))
kl = Ridge(design(c1, True), np.log(c1.kl.values))
Pg = profiles(h1[h1.ref <= 6])


def draw(n, peak=(.6, 1), reach=(1 / 27, .6), strength=(.8, 1.5)):
    d = {"direction_layer": rng.integers(11, 25, n).astype(float)}
    for comp in ("attn", "mlp"):
        d[f"{comp}.strength"] = rng.uniform(*strength, n)
        d[f"{comp}.peak_layer"] = rng.uniform(*peak, n)
        d[f"{comp}.edge_strength"] = rng.uniform(0, 1, n)
        d[f"{comp}.reach"] = rng.uniform(*reach, n)
    return pd.DataFrame(d)


def evaluate(df):
    X = design(df, True)
    df["ref_kw"] = 100 / (1 + np.exp(-refusal(X)))
    df["kl_candor"] = np.exp(kl(X))
    P = profiles(df)
    df["dgood"] = np.sqrt(((P[:, None, :] - Pg[None]) ** 2).mean(-1)).min(1)
    return df


if __name__ == "__main__":
    t0 = time.time()
    for name, kw in (("current ranges", {}), ("peak .3-1, reach to 1", {"peak": (.3, 1), "reach": (1 / 27, 1)})):
        df = pd.concat([evaluate(draw(20_000, **kw)) for _ in range(10)])
        for support in (0.2, 0.3):
            s = df[df.dgood <= support]
            rows = [f"{name}, within {support} of a known good shape (n={len(s)}):"]
            for cap in (3, 5, 8):
                ok = s[s.ref_kw <= cap]
                rows.append(f"pred kw<={cap}: min pred candor KL {ok.kl_candor.min():.4f}" if len(ok) else f"pred kw<={cap}: none")
            print("  ".join(rows))
    print(f"took {time.time() - t0:.0f}s")
