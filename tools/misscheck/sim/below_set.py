"""Which trials formed TPE's 'below' (good) set in candor's real run, at several points."""
import numpy as np
import pandas as pd
from optuna.study._multi_objective import _fast_non_domination_rank
from optuna.samplers._tpe.sampler import default_gamma

c = pd.read_csv("candor_search_rude.csv")
share = c.refusals_judge / 99
obj1 = np.where(c.kl >= 0.01, c.kl, share * 0.01)
vals = np.column_stack([obj1, share])
for n in (60, 100, 140, 199):
    v = vals[:n]
    k = default_gamma(n)
    ranks = _fast_non_domination_rank(v.copy(), n_below=k)
    order = np.argsort(ranks, kind="stable")[:k]  # approx: tie-break by HSSP ignored
    sub = c.iloc[order]
    print(f"after {n} trials: below set of {k}; rank<= {ranks[order].max()}")
    print(sub[["trial", "scope", "direction_layer", "mlp.peak_layer", "mlp.reach", "attn.peak_layer", "attn.reach", "refusals_judge", "kl"]].round(3).to_string(index=False))
