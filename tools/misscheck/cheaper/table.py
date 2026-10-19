import sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import replay as rp
import replay2 as r2

configs = [
    ("today: full scoring", dict(cap=10**9, g=1, chunk=16, lookahead=12, kl_last=False)),
    ("P1 exact, sampler unchanged (G=25), chunk 16", dict(cap=5, g=25, chunk=16, lookahead=12, kl_last=True)),
    ("P1+P2 curtail at cap (G=1), chunk 16", dict(cap=5, g=1, chunk=16, lookahead=12, kl_last=True)),
    ("P1+P2 + cancel in-flight answers", dict(cap=5, g=1, chunk=16, lookahead=0, kl_last=True)),
    ("P1+P2 chunk 8 (judge batch 8)", dict(cap=5, g=1, chunk=8, lookahead=8, kl_last=True)),
]
for model in ["3B", "9B thinking 5x", "9B thinking 10x"]:
    cost = rp.costs(model)
    for om, sigma, ordering in [("random", 0.0, "fixed"), ("nested", 1.0, "hardest first")]:
        print(f"\n{model}; order {ordering}")
        base = None
        for label, kw in configs:
            out = []
            for rep in range(20):
                rng = np.random.default_rng(rep)
                refused = rp.outcomes(om, sigma, rng)
                order = rp.orders(refused, ordering, rng)
                out.append(r2.run(refused, order, cost, **kw))
            t = np.mean([o[0] for o in out])
            base = base or t
            print(f"  {label:48s} {t:7.1f} s/trial  200 trials {200 * t / 3600:5.1f} h  x{base / t:.2f} trials per wall-clock")
