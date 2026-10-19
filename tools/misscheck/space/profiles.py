import numpy as np
LAYERS, LAST = 28, 27


def profile(s, p, e, r):
    """Strength per layer for the current kernel: s at layer p*LAST, falling linearly to e*s at r*LAST."""
    s, p, e, r = (np.asarray(v, float)[..., None] for v in (s, p, e, r))
    d = np.abs(np.arange(LAYERS) - p * LAST)
    reach = r * LAST
    out = s + d / np.maximum(reach, 1e-12) * (e * s - s)
    return np.where(d > reach, 0.0, out)


def profiles(df):
    """attn and mlp profiles side by side, shape (n, 56)."""
    parts = [profile(df[f"{c}.strength"], df[f"{c}.peak_layer"], df[f"{c}.edge_strength"], df[f"{c}.reach"])
             for c in ("attn", "mlp")]
    return np.concatenate(parts, axis=-1)
