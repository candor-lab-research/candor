"""Second replay: the cap-curtailment family with the knobs that set its cost.

Rule 'stop when refusals > cap and >= G fully scored trials have fewer refusals than seen so far'
(G = 25 leaves TPE's proposals unchanged; G = 1 is nearly plain curtailment at the cap).
Knobs: KL measured first or only for trials that finish, judge check every 8 or 16 answers, and
how many answers are still being generated when a trial stops.
"""

import sys

import numpy as np
from math import comb

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import replay as rp  # noqa: E402

N = 100


def tail(k, total, good, draws):
    """P(at least k of the draws are good) without replacement."""
    return sum(comb(good, i) * comb(total - good, draws - i) for i in range(k, min(good, draws) + 1)) / comb(total, draws)


def run(refused, order, cost, cap, g, chunk, lookahead, kl_last, alpha=None):
    total, judged, fully, fully_cost = 0.0, 0, 0, 0.0
    done_r = []
    for t in range(len(rp.R)):
        seq = refused[t, order[t]]
        cum = np.concatenate([[0], np.cumsum(seq)])
        stop = None
        for n in range(chunk, N, chunk):
            k = cum[n]
            if k > cap and sum(r < k for r in done_r) >= g:
                stop = n
                break
            if alpha is not None and k > 0 and tail(k, N, cap, n) < alpha:
                stop = n
                break
        if stop is None:
            c = cost["fixed"] + cost["judge"] * N + cost["refused"] * seq.sum() + cost["answered"] * (N - seq.sum())
            total += c
            fully_cost += c
            judged += N
            fully += 1
            done_r.append(rp.R[t])
        else:
            gen = seq[: min(N, stop + lookahead)]
            kl = 0.0 if kl_last else cost["fixed"] - 0.5
            total += 0.5 + kl + cost["judge"] * stop + cost["refused"] * gen.sum() + cost["answered"] * (len(gen) - gen.sum())
            judged += stop
    return total / len(rp.R), judged / len(rp.R), fully, fully_cost / max(total, 1e-9)


def main():
    reps = 20
    for model in sys.argv[1:] or ["3B"]:
        cost = rp.costs(model)
        for om, sigma, ordering in [("random", 0.0, "fixed"), ("nested", 1.0, "hardest first")]:
            print(f"\n=== {model}; refusals {om}{'' if om == 'random' else ' sigma 1'}, order {ordering}")
            base = None
            configs = [
                ("full scoring", dict(cap=10**9, g=1, chunk=16, lookahead=12, kl_last=False)),
                ("G=25 (sampler unchanged), KL first, chunk 16", dict(cap=5, g=25, chunk=16, lookahead=12, kl_last=False)),
                ("G=25, KL last, chunk 8", dict(cap=5, g=25, chunk=8, lookahead=8, kl_last=True)),
                ("G=10, KL last, chunk 8", dict(cap=5, g=10, chunk=8, lookahead=8, kl_last=True)),
                ("G=1, KL first, chunk 16", dict(cap=5, g=1, chunk=16, lookahead=12, kl_last=False)),
                ("G=1, KL last, chunk 16", dict(cap=5, g=1, chunk=16, lookahead=12, kl_last=True)),
                ("G=1, KL last, chunk 8", dict(cap=5, g=1, chunk=8, lookahead=8, kl_last=True)),
                ("G=1, KL last, chunk 8, no wasted answers", dict(cap=5, g=1, chunk=8, lookahead=0, kl_last=True)),
                ("G=1, KL last, chunk 8, cap 0", dict(cap=0, g=1, chunk=8, lookahead=8, kl_last=True)),
                ("G=1, KL last, chunk 8, cap 10", dict(cap=10, g=1, chunk=8, lookahead=8, kl_last=True)),
            ]
            if om == "random":
                configs.append(("G=1 + statistical stop at 1% (random order only)", dict(cap=5, g=1, chunk=8, lookahead=8, kl_last=True, alpha=0.01)))
            for label, kw in configs:
                out = []
                for rep in range(reps):
                    rng = np.random.default_rng(rep)
                    refused = rp.outcomes(om, sigma, rng)
                    order = rp.orders(refused, ordering, rng)
                    out.append(run(refused, order, cost, **kw))
                t, j, f, share = map(np.mean, zip(*out))
                base = base or t
                print(f"{label:52s} {t:7.1f} s/trial  x{base / t:.2f} trials  judged {j:5.1f}  fully scored {f:5.1f}  "
                      f"time on fully scored {100 * share:3.0f}%")


if __name__ == "__main__":
    main()
