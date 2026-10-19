"""How much of each parametrisation's uniform prior lands on profiles close to known good recipes."""
import sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from load import candor, heretic
from profiles import LAYERS, LAST, profiles

rng = np.random.default_rng(0)
N = 200_000
X = np.arange(LAYERS) / LAST  # layer positions as fractions of depth


def u(lo, hi, n=N):
    return rng.uniform(lo, hi, n)


def symmetric(s, p, e, r):
    """The current kernel: s at p, linear to e*s at distance r, zero beyond."""
    d = np.abs(X - p[:, None])
    out = s[:, None] * (1 + d / r[:, None] * (e[:, None] - 1))
    return np.where(d > r[:, None], 0.0, out)


def asymmetric(s, p, e, left, right):
    """Like the current kernel, but reaching left and right separately."""
    d = X - p[:, None]
    r = np.where(d < 0, left[:, None], right[:, None])
    out = s[:, None] * (1 + np.abs(d) / r * (e[:, None] - 1))
    return np.where(np.abs(d) > r, 0.0, out)


def three_knots(s, p, a, e1, e2):
    """Zero before the first layer f = a*p; e1*s at f rising linearly to s at p; then linearly to e2*s at the last layer."""
    f = a * p
    rise = s[:, None] * (e1[:, None] + (1 - e1[:, None]) * (X - f[:, None]) / np.maximum(p - f, 1e-9)[:, None])
    fall = s[:, None] * (1 + (e2[:, None] - 1) * (X - p[:, None]) / np.maximum(1 - p, 1e-9)[:, None])
    out = np.where(X < p[:, None], rise, fall)
    return np.where(X < f[:, None] - 1e-9, 0.0, out)


def start_end(s, a, b, e, q):
    """A band from a to b (fractions of depth, b>=a), peak at a+q*(b-a), linear down to e*s at both band ends."""
    p = a + q * (b - a)
    d = X - p[:, None]
    half = np.where(d < 0, (p - a)[:, None], (b - p)[:, None])
    out = s[:, None] * (1 + np.abs(d) / np.maximum(half, 1e-9) * (e[:, None] - 1))
    inside = (X >= a[:, None] - 1e-9) & (X <= b[:, None] + 1e-9)
    return np.where(inside, out, 0.0)


SPACES = {
    "current (s .8-1.5, peak .6-1, edge 0-1, reach 1/27-.6)":
        lambda: symmetric(u(.8, 1.5), u(.6, 1), u(0, 1), u(1 / 27, .6)),
    "current, reach to 1.0":
        lambda: symmetric(u(.8, 1.5), u(.6, 1), u(0, 1), u(1 / 27, 1.0)),
    "current, peak .3-1, reach to 1.0":
        lambda: symmetric(u(.8, 1.5), u(.3, 1), u(0, 1), u(1 / 27, 1.0)),
    "current, peak .3-1, reach to 1.0, s .5-2":
        lambda: symmetric(u(.5, 2.0), u(.3, 1), u(0, 1), u(1 / 27, 1.0)),
    "asymmetric reach (peak .6-1, left/right 1/27-1)":
        lambda: asymmetric(u(.8, 1.5), u(.6, 1), u(0, 1), u(1 / 27, 1), u(1 / 27, 1)),
    "asymmetric reach (peak .3-1, left/right 1/27-1)":
        lambda: asymmetric(u(.8, 1.5), u(.3, 1), u(0, 1), u(1 / 27, 1), u(1 / 27, 1)),
    "start/end band (a 0-1, b=a+w, w 1/27-1 clipped, peak q 0-1)":
        lambda: (lambda a: start_end(u(.8, 1.5), a, np.minimum(1.0, a + u(1 / 27, 1.0)), u(0, 1), u(0, 1)))(u(0, 1)),
    "start/end band, end = last layer with prob ~1/2 (w 1/27-2)":
        lambda: (lambda a: start_end(u(.8, 1.5), a, np.minimum(1.0, a + u(1 / 27, 2.0)), u(0, 1), u(0, 1)))(u(0, .95)),
    "three knots (peak .3-1, first=a*peak, e1, e_last)":
        lambda: three_knots(u(.8, 1.5), u(.3, 1), u(0, 1), u(0, 1), u(0, 1)),
}


def nearest(draws, centres, chunk=20_000):
    out = np.empty(len(draws))
    for i in range(0, len(draws), chunk):
        block = draws[i:i + chunk]
        out[i:i + chunk] = np.sqrt(((block[:, None, :] - centres[None]) ** 2).mean(-1)).min(1)
    return out


if __name__ == "__main__":
    t0 = time.time()
    c, h = candor(), heretic()
    targets = {
        "heretic one-layer <=6 kw (14)": profiles(h[(h.scope == "one_layer") & (h.ref <= 6)]),
        "candor per-layer <=3 judge (16)": profiles(c[(c.scope == "per_layer") & (c.ref <= 3)]),
        "trial 170 alone": profiles(h[h.trial == 170]),
    }
    for name, make in SPACES.items():
        draws = np.concatenate([make(), make()], axis=1)  # attn, mlp drawn independently
        line = [name]
        for tname, centres in targets.items():
            d = nearest(draws, centres)
            line.append(f"{tname}: P(d<=.2)={np.mean(d <= .2):.4f} P(d<=.3)={np.mean(d <= .3):.4f}")
        print("\n  ".join(line))
    print(f"took {time.time() - t0:.0f}s")
