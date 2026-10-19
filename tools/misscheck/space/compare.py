"""Compare kernel parametrisations by how much of their uniform prior lands near known good shapes
(volume), and how box-like that good set is (shuffle test, largest within-component correlation)."""
import sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from load import candor, heretic
from profiles import profiles
from volume import symmetric, asymmetric, three_knots, nearest

rng = np.random.default_rng(3)
N = 300_000
U = lambda lo, hi: rng.uniform(lo, hi, N)


def sym(peak=(.6, 1), reach=(1 / 27, .6), s=(.8, 1.5)):
    return lambda: {"s": U(*s), "p": U(*peak), "e": U(0, 1), "r": U(*reach)}, lambda q: symmetric(q["s"], q["p"], q["e"], q["r"])


def start_peak(start=(-.3, .9), peak=(.6, 1), s=(.8, 1.5)):
    """Same kernel, drawn as where the band starts (f) and peaks (p); it reaches as far past the peak."""
    return (lambda: {"s": U(*s), "f": U(*start), "p": U(*peak), "e": U(0, 1)},
            lambda q: symmetric(q["s"], q["p"], q["e"], np.maximum(q["p"] - q["f"], 1 / 27)))


def asym(peak=(.3, 1)):
    return (lambda: {"s": U(.8, 1.5), "p": U(*peak), "e": U(0, 1), "l": U(1 / 27, 1), "rr": U(1 / 27, 1)},
            lambda q: asymmetric(q["s"], q["p"], q["e"], q["l"], q["rr"]))


def knots(peak=(.3, 1)):
    return (lambda: {"s": U(.8, 1.5), "p": U(*peak), "a": U(0, 1), "e1": U(0, 1), "e2": U(0, 1)},
            lambda q: three_knots(q["s"], q["p"], q["a"], q["e1"], q["e2"]))


SPACES = {
    "current: peak .6-1, reach 1/27-.6": sym(),
    "current, reach 1/27-1.0": sym(reach=(1 / 27, 1)),
    "current, peak .4-1, reach 1/27-1.0": sym(peak=(.4, 1), reach=(1 / 27, 1)),
    "start+peak: start -.3-.9, peak .6-1": start_peak(),
    "start+peak: start -.3-.9, peak .4-1": start_peak(peak=(.4, 1)),
    "asymmetric: peak .3-1, left/right reach 1/27-1": asym(),
    "three knots: peak .3-1": knots(),
    "three knots: peak .5-1": knots(peak=(.5, 1)),
}


def max_corr(q):
    keys = list(q)
    m = np.corrcoef(np.stack([q[k] for k in keys]))
    np.fill_diagonal(m, 0)
    i, j = np.unravel_index(np.abs(m).argmax(), m.shape)
    return f"{keys[i]}~{keys[j]} {m[i, j]:+.2f}"


if __name__ == "__main__":
    t0 = time.time()
    c, h = candor(), heretic()
    targets = {"one-layer good": profiles(h[(h.scope == "one_layer") & (h.ref <= 6)]),
               "per-layer good": profiles(c[(c.scope == "per_layer") & (c.ref <= 3)])}
    for name, (make, build) in SPACES.items():
        qa, qm = make(), make()
        draws = np.concatenate([build(qa), build(qm)], 1)
        cols = []
        for tname, centres in targets.items():
            d = nearest(draws, centres)
            good = d <= .25
            sa = {k: rng.permutation(v[good]) for k, v in qa.items()}
            sm = {k: rng.permutation(v[good]) for k, v in qm.items()}
            kept = np.mean(nearest(np.concatenate([build(sa), build(sm)], 1), centres) <= .25)
            cols.append(f"{tname}: P(d<=.2) {np.mean(d <= .2):.4f}, P(d<=.25) {good.mean():.4f}, shuffle-kept {kept:.2f}, "
                        f"max corr (attn) {max_corr({k: v[good] for k, v in qa.items()})}")
        print(name + "\n    " + "\n    ".join(cols))
    print(f"took {time.time() - t0:.0f}s")
