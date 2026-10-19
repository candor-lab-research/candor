"""KL divergence between first-token distributions."""

import math


def from_top(base: dict[int, float], edited: dict[int, float]) -> float:
    """KL(base || edited) from the top tokens a server lists, as {token id: log probability}.

    The base model's listed tokens are compared one by one and every other token is merged into one
    bucket, which converges on the exact value as more tokens are listed. A listed base token that
    edited doesn't list gets edited's smallest listed probability.
    """
    floor = min(edited.values())
    kl = base_total = edited_total = 0.0
    for token, logp in base.items():
        logq = edited.get(token, floor)
        p = math.exp(logp)
        kl += p * (logp - logq)
        base_total += p
        edited_total += math.exp(logq)
    base_rest = max(1.0 - base_total, 0.0)
    if base_rest > 0:
        kl += base_rest * math.log(base_rest / max(1.0 - edited_total, 1e-12))
    return kl


def mean_from_top(base: list[dict[int, float]], edited: list[dict[int, float]]) -> float:
    return sum(map(from_top, base, edited)) / len(base)
