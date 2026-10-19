import sys, numpy as np, math
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import budget as b
rng = np.random.default_rng(1)
rs = {"one layer": [], "per layer": []}; best = {"one layer": math.inf, "per layer": math.inf}
for _ in range(20000):
    p = {"direction_scope": rng.choice(["one layer", "per layer"]), "direction_layer": int(rng.integers(11, 25))}
    for c in ("attn", "mlp"):
        p[f"{c}.strength"] = rng.uniform(0.8, 1.5); p[f"{c}.peak_layer"] = rng.uniform(0.6, 1.0)
        p[f"{c}.edge_strength"] = rng.uniform(0, 1); p[f"{c}.reach"] = rng.uniform(1/27, 0.6)
    r, kl = b.noiseless(p); rs[p["direction_scope"]].append(r)
    if r <= b.CAP: best[p["direction_scope"]] = min(best[p["direction_scope"]], kl)
for s in rs:
    a = np.array(rs[s]); print(s, "median R", np.median(a), "P(R<=5) %.4f" % (a <= 5).mean(), "P(R<=20) %.3f" % (a <= 20).mean(), "best feasible KL (random 10k) %.4f" % best[s])
print("histogram all", np.histogram(np.concatenate([rs[s] for s in rs]), bins=[0,1,3,6,11,21,36,51,76,91,101])[0].tolist())
