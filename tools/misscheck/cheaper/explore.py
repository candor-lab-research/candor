import csv, statistics as st, numpy as np
from pathlib import Path
D = str(Path(__file__).resolve().parent.parent) + "/"
rows = list(csv.DictReader(open(D + "candor_search_rude.csv")))
r = np.array([int(x["refusals_judge"]) for x in rows]); kw = np.array([int(x["refusals_keywords"]) for x in rows])
kl = np.array([float(x["kl"]) for x in rows]); s = np.array([float(x["seconds"]) for x in rows])
scope = np.array([x["scope"] for x in rows])
print("n", len(rows), "seconds mean %.1f median %.1f min %.1f max %.1f" % (s.mean(), np.median(s), s.min(), s.max()))
A = np.vstack([np.ones_like(r), 100 - r]).T
coef, res, *_ = np.linalg.lstsq(A, s, rcond=None)
pred = A @ coef
print("seconds ~ %.2f + %.3f * (100 - refusals); R2 %.3f" % (coef[0], coef[1], 1 - ((s - pred) ** 2).sum() / ((s - s.mean()) ** 2).sum()))
A2 = np.vstack([np.ones_like(r), 100 - kw]).T
coef2, *_ = np.linalg.lstsq(A2, s, rcond=None); pred2 = A2 @ coef2
print("seconds ~ %.2f + %.3f * (100 - kw); R2 %.3f" % (coef2[0], coef2[1], 1 - ((s - pred2) ** 2).sum() / ((s - s.mean()) ** 2).sum()))
print("first trial seconds (warmup?)", s[:3])
for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 90), (90, 101)]:
    m = (r >= lo) & (r < hi)
    print(f"refusals {lo}-{hi-1}: n={m.sum()} mean s={s[m].mean():.1f}" if m.any() else f"{lo}-{hi}: none")
print("refusal histogram:", np.histogram(r, bins=[0,1,2,3,5,10,20,35,50,75,90,95,100,101])[0].tolist())
print("trials with refusals <= 5:", (r <= 5).sum(), " <=10:", (r <= 10).sum(), " <=20:", (r<=20).sum(), " >=50:", (r>=50).sum())
print("KL quantiles", np.quantile(kl, [0, .1, .25, .5, .75, .9, 1]).round(4).tolist())
# correlation between KL and refusals
print("spearman-ish corr(kl, refusals) %.2f" % np.corrcoef(np.argsort(np.argsort(kl)), np.argsort(np.argsort(r)))[0, 1])
# by scope
for sc in ["one_layer", "per_layer"]:
    m = scope == sc
    print(sc, "n", m.sum(), "min ref", r[m].min(), "median ref", np.median(r[m]), "mean s %.1f" % s[m].mean())
