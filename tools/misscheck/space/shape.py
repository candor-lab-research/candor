"""How box-like the good region is in each parametrisation: shuffle each parameter independently among
good draws and see how many shuffled draws are still good (1 = an axis-aligned box, low = a thin diagonal)."""
import sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from load import candor, heretic
from profiles import profiles
from volume import symmetric, three_knots, nearest

rng = np.random.default_rng(2)
N = 400_000


def params_symmetric(peak, reach):
    return {"s": rng.uniform(.8, 1.5, N), "p": rng.uniform(*peak, N), "e": rng.uniform(0, 1, N), "r": rng.uniform(*reach, N)}


def params_knots():
    return {"s": rng.uniform(.8, 1.5, N), "p": rng.uniform(.3, 1, N), "a": rng.uniform(0, 1, N), "e1": rng.uniform(0, 1, N), "e2": rng.uniform(0, 1, N)}


SPACES = {
    "current": (lambda: params_symmetric((.6, 1), (1 / 27, .6)), lambda q: symmetric(q["s"], q["p"], q["e"], q["r"])),
    "reach to 1.0": (lambda: params_symmetric((.6, 1), (1 / 27, 1)), lambda q: symmetric(q["s"], q["p"], q["e"], q["r"])),
    "three knots": (params_knots, lambda q: three_knots(q["s"], q["p"], q["a"], q["e1"], q["e2"])),
}

if __name__ == "__main__":
    t0 = time.time()
    h = heretic()
    centres = profiles(h[(h.scope == "one_layer") & (h.ref <= 6)])
    for name, (make, build) in SPACES.items():
        qa, qm = make(), make()
        d = nearest(np.concatenate([build(qa), build(qm)], 1), centres)
        good = d <= 0.25
        ga = {k: v[good] for k, v in qa.items()}
        gm = {k: v[good] for k, v in qm.items()}
        n = good.sum()
        shuffled_a = {k: rng.permutation(v) for k, v in ga.items()}
        shuffled_m = {k: rng.permutation(v) for k, v in gm.items()}
        ds = nearest(np.concatenate([build(shuffled_a), build(shuffled_m)], 1), centres)
        spread = {f"{c}.{k}": float(np.subtract(*np.quantile(v, [.9, .1]))) for c, g in (("attn", ga), ("mlp", gm)) for k, v in g.items()}
        corr = {c: float(np.corrcoef(g["p"], g["r"] if "r" in g else g["a"])[0, 1]) for c, g in (("attn", ga), ("mlp", gm))}
        print(f"{name}: P(good)={n / N:.4f}; shuffled still good {np.mean(ds <= .25):.2f}; "
              f"peak-vs-{'reach' if 'r' in ga else 'a'} corr attn {corr['attn']:.2f} mlp {corr['mlp']:.2f}")
        print("   10-90% spread of good draws:", {k: round(v, 2) for k, v in spread.items()})
    print(f"took {time.time() - t0:.0f}s")
