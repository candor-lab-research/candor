"""Tables from budget_sim.py's runs: does search.max_budget_hits = 8 make the search miss the best edit
more often, and how many prompts does it save?

The truth is the least true KL among edits with true refusals <= 2 and expected budget hits <= 8 (budget.py),
lowered to any simulated trial within both that beat it. A run's pick is candor's: among its trials scored
in full with at most 2 refusals and at most 8 budget hits measured, the least measured KL (for the off arm
this filter is applied at the end; the search itself never saw the limit). It hits when its true KL is
within 10% (3%) of the truth's; a run with no such trial misses.

Columns: hit rate with its Wilson 95% interval; the paired difference against off (same scenario and seed)
with a paired-bootstrap 95% interval and McNemar's exact p; none = runs with no pick; heavy = picks whose
expected budget hits are over 8 (measured within the limit by luck); judged = harmful prompts judged per
trial; forced = answers per trial that ran to the budget; K2 h = the hours these trials would take on
K2-Horizon-7B's pod, from K2's seconds per trial fitted on prompts judged, budget hits and refusals.

    PYTHONPATH=candor-cb0fa99/src:. uv run python budget_report.py > budget-tables.txt
"""

import json
import math

import numpy as np
import pandas as pd

import budget
from budget_sim import HERE, LIMIT, PROMPTS, RUNS, load
from report import wilson

LENGTHS = (300, 400, 600)
ARMS = ("off", "on", "switch")
GROUPS = {
    "positions, penalised (D3grow)": (["a-same", "b-answer", "c-think", "d-high"], "D3grow"),
    "positions, free (D3grow)": (["fa-same", "fb-answer", "fc-think", "fd-high"], "D3grow"),
    "positions, penalised (D3, wide)": (["a-same", "b-answer", "c-think", "d-high"], "D3"),
    "positions, free (D3, wide)": (["fa-same", "fb-answer", "fc-think", "fd-high"], "D3"),
    "strength, all (grow)": (
        [
            *["base", "rand1", "rand2", "rand3", "rand4", "rand5", "rand6"],
            *["basex2.4", "rand4x1.8", "rand6x1.8"],
            *["fbase", "frand1", "frand2", "frand3", "frand4", "frand5", "frand6"],
            *["fbasex2.6", "frand4x1.8", "frand6x1.8"],
        ],
        "grow",
    ),
}
BOOTSTRAP = 4000


def seconds_model() -> np.ndarray:
    """K2's seconds per trial against prompts judged, budget hits and refusals (least squares)."""
    k2 = pd.read_csv(HERE / "k2" / "trials.csv")
    x = np.stack([np.ones(len(k2)), k2.judged, k2.forced, k2.refusals], 1)
    return np.linalg.lstsq(x, k2.seconds.values, rcond=None)[0]


def best_kl(name: str, variant: str) -> tuple[float, float]:
    """The truth's KL, and the same lowered to any simulated trial of the scenario within both limits."""
    kl = budget.truth_for(name, variant).kl
    lowest = kl
    folder = RUNS / variant / name
    for path in folder.glob("*/*/*.npz"):
        t = np.load(path)["trials"]
        ok = (t[:, 5] <= 2) & (t[:, 11] <= LIMIT)
        if ok.any():
            lowest = min(lowest, float(t[ok, 6].min()))
    return kl, lowest


def judge(trials: np.ndarray, length: int, truth_kl: float, seconds: np.ndarray) -> dict:
    t = trials[:length]
    judged, refusals, kl, true_kl, forced, true_hits = t[:, 2], t[:, 3], t[:, 4], t[:, 6], t[:, 10], t[:, 11]
    hours = (
        float((seconds[0] + seconds[1] * judged + seconds[2] * forced + seconds[3] * refusals).sum()) / 3600
    )
    out = {"judged": float(judged.mean()), "forced": float(forced.mean()), "hours": hours}
    eligible = (judged == PROMPTS) & (refusals <= 2) & (forced <= LIMIT)
    if not eligible.any():
        return {**out, "pick": False, "gap": math.inf, "heavy": False}
    i = np.flatnonzero(eligible)[np.lexsort((refusals[eligible], kl[eligible]))[0]]
    return {**out, "pick": True, "gap": true_kl[i] / truth_kl - 1, "heavy": bool(true_hits[i] > LIMIT)}


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p from the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(hits: np.ndarray, base: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """The difference in hit rate, its paired-bootstrap 95% interval, and McNemar's p."""
    diff = hits.astype(float) - base.astype(float)
    draws = rng.integers(0, len(diff), (BOOTSTRAP, len(diff)))
    means = diff[draws].mean(1)
    b, c = int(((hits == 1) & (base == 0)).sum()), int(((hits == 0) & (base == 1)).sum())
    return (
        float(diff.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
        mcnemar(b, c),
    )


def runs_of(names: list[str], variant: str, design: str) -> dict:
    """Each arm's runs, paired: only seeds every arm has, for every scenario."""
    out = {arm: [] for arm in ARMS}
    for name in names:
        found = {arm: load(name, variant, design, arm) for arm in ARMS}
        seeds = sorted(set.intersection(*(set(f) for f in found.values())))
        for arm in ARMS:
            out[arm] += [(name, seed, found[arm][seed]) for seed in seeds]
    return out


def pct(x: float) -> str:
    return f"{x:5.1%}"


def table(variant: str, seconds: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    truths = {}
    for names, _ in GROUPS.values():
        for name in names:
            if name not in truths:
                truths[name] = best_kl(name, variant)
    lowered = {n: (kl, low) for n, (kl, low) in truths.items() if low < kl * (1 - 1e-6)}
    for name, (kl, low) in lowered.items():
        print(f"  {name}: the truth's KL lowered to a simulated trial's, {kl:.4f} -> {low:.4f}")
    print(
        f"{'group':32s} {'arm':6s} {'trials':>6s} {'runs':>5s}  {'within 10% (95% CI)':>21s}  "
        f"{'vs off (95% CI), McNemar p':>30s}  {'within 3%':>9s} {'vs off':>7s}  {'none':>5s} {'heavy':>5s} "
        f"{'judged':>6s} {'forced':>6s} {'K2 h':>5s}"
    )
    for label, (names, design) in GROUPS.items():
        arms = runs_of(names, variant, design)
        if not arms["off"]:
            continue
        for length in LENGTHS:
            judged = {
                arm: [judge(t, length, truths[name][1], seconds) for name, _, t in runs]
                for arm, runs in arms.items()
            }
            hit10 = {arm: np.array([j["gap"] <= 0.10 for j in js]) for arm, js in judged.items()}
            hit3 = {arm: np.array([j["gap"] <= 0.03 for j in js]) for arm, js in judged.items()}
            for arm in ARMS:
                js, n = judged[arm], len(judged[arm])
                low, high = wilson(int(hit10[arm].sum()), n)
                rate = f"{pct(hit10[arm].mean())} ({pct(low)}-{pct(high)})"
                if arm == "off":
                    versus = versus3 = ""
                else:
                    d, lo, hi, p = paired(hit10[arm], hit10["off"], rng)
                    versus = f"{d:+6.1%} ({lo:+6.1%} to {hi:+6.1%}) p {p:.2f}"
                    versus3 = f"{hit3[arm].mean() - hit3['off'].mean():+6.1%}"
                print(
                    f"{label:32s} {arm:6s} {length:6d} {n:5d}  {rate:>21s}  {versus:>30s}  "
                    f"{pct(hit3[arm].mean()):>9s} {versus3:>7s}  "
                    f"{pct(np.mean([not j['pick'] for j in js])):>5s} "
                    f"{pct(np.mean([j['heavy'] for j in js])):>5s} {np.mean([j['judged'] for j in js]):6.1f} "
                    f"{np.mean([j['forced'] for j in js]):6.2f} {np.mean([j['hours'] for j in js]):5.0f}"
                )
        print()


def by_scenario(variant: str, length: int) -> None:
    """Each scenario's hit rate within 10% per arm at one length, the best edit's expected budget hits,
    and near over 8: of the off arm's would-be hits (scored in full, at most 2 refusals measured, true KL
    within 10% of the truth's), the share measured over 8 budget hits, which the limit turns away."""
    print(
        f"{'scenario':10s} {'design':6s} {'runs':>5s}  "
        + "  ".join(f"{arm:>6s}" for arm in ARMS)
        + f"  {'on-off':>7s} {'switch-off':>10s}  {'best hits':>9s} {'near over 8':>11s}"
    )
    rows = []
    for names, design in GROUPS.values():
        for name in names:
            arms = runs_of([name], variant, design)
            if not arms["off"]:
                continue
            _, kl = best_kl(name, variant)
            hits = {
                arm: np.array([judge(t, length, kl, np.zeros(4))["gap"] <= 0.10 for _, _, t in runs])
                for arm, runs in arms.items()
            }
            off = np.concatenate([t[:length] for _, _, t in arms["off"]])
            near = (off[:, 2] == PROMPTS) & (off[:, 3] <= 2) & (off[:, 6] <= 1.10 * kl)
            over = float((off[near, 10] > LIMIT).mean()) if near.any() else math.nan
            truth = budget.truth_for(name, variant)
            on, switch = hits["on"].mean() - hits["off"].mean(), hits["switch"].mean() - hits["off"].mean()
            rows.append((on, switch, over))
            print(
                f"{name:10s} {design:6s} {len(hits['off']):5d}  "
                + "  ".join(f"{hits[a].mean():6.0%}" for a in ARMS)
                + f"  {on:+7.1%} {switch:+10.1%}  {truth.hits:9.1f} {over:11.1%}"
            )
    a = np.array(rows)
    ok = ~np.isnan(a[:, 2])
    high = ok & (a[:, 2] > 0.05)
    print(
        f"Correlation over scenarios with near over 8: on-off {np.corrcoef(a[ok, 0], a[ok, 2])[0, 1]:+.2f}, "
        f"switch-off {np.corrcoef(a[ok, 1], a[ok, 2])[0, 1]:+.2f}. "
        f"Where near over 8 is above 5% ({high.sum()} of "
        f"{ok.sum()}): on-off {a[high, 0].mean():+.1%}, switch-off {a[high, 1].mean():+.1%}; elsewhere "
        f"{a[ok & ~high, 0].mean():+.1%}, {a[ok & ~high, 1].mean():+.1%}."
    )
    print()


def chi2_p(q: float, df: int) -> float:
    """Upper tail of a chi-square (Wilson-Hilferty's normal approximation)."""
    z = ((q / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def pooled(variant: str) -> None:
    """Each arm against off over every group, and over the thinking groups alone; and whether the
    scenarios differ in it by more than chance (Cochran's Q over scenario and design at 600 trials)."""
    rng = np.random.default_rng(1)
    pools = {"every group": list(GROUPS), "thinking groups": [g for g in GROUPS if g.startswith("positions")]}
    cells = {}
    for label, (names, design) in GROUPS.items():
        for name in names:
            arms = runs_of([name], variant, design)
            if arms["off"]:
                _, kl = best_kl(name, variant)
                cells[(label, name)] = {
                    length: {
                        arm: np.array([judge(t, length, kl, np.zeros(4))["gap"] <= 0.10 for _, _, t in runs])
                        for arm, runs in arms.items()
                    }
                    for length in LENGTHS
                }
    print(
        f"{'pool':16s} {'arm':6s} {'trials':>6s} {'runs':>5s}  {'off':>6s} {'arm':>6s}  "
        f"{'vs off (95% CI), McNemar p':>30s}"
    )
    for pool, groups in pools.items():
        chosen = [c for (label, _), c in cells.items() if label in groups]
        for arm in ARMS[1:]:
            for length in LENGTHS:
                hits = np.concatenate([c[length][arm] for c in chosen])
                base = np.concatenate([c[length]["off"] for c in chosen])
                d, lo, hi, p = paired(hits, base, rng)
                print(
                    f"{pool:16s} {arm:6s} {length:6d} {len(hits):5d}  {base.mean():6.1%} {hits.mean():6.1%}  "
                    f"{d:+6.1%} ({lo:+6.1%} to {hi:+6.1%}) p {p:.2f}"
                )
    for arm in ARMS[1:]:
        diffs, weights = [], []
        for c in cells.values():
            hits, base = c[600][arm], c[600]["off"]
            n = len(hits)
            b, cc = int(((hits == 1) & (base == 0)).sum()), int(((hits == 0) & (base == 1)).sum())
            var = (b + cc - (b - cc) ** 2 / n) / n**2
            if var > 0:
                diffs.append((b - cc) / n)
                weights.append(1 / var)
        diffs, weights = np.array(diffs), np.array(weights)
        mean = (weights * diffs).sum() / weights.sum()
        q = float((weights * (diffs - mean) ** 2).sum())
        print(
            f"{arm} - off at 600 trials over {len(diffs)} scenario-designs: "
            f"Cochran's Q {q:.1f} on {len(diffs) - 1} df, "
            f"p {chi2_p(q, len(diffs) - 1):.2f} (spread of the per-scenario differences beyond chance), "
            f"sd of the differences {diffs.std():.1%}"
        )
    print()


def equal_time(variant: str, seconds: np.ndarray) -> None:
    """Each arm at equal K2 pod time rather than equal trials: a run is judged at the trials it finishes
    within the off arm's mean K2 hours at 300 and 400 trials (at most 600)."""
    rng = np.random.default_rng(2)
    print(
        f"{'group':32s} {'hours':>5s} {'arm':6s} {'trials':>6s}  {'within 10%':>10s}  "
        f"{'vs off (95% CI), McNemar p':>30s}"
    )
    for label, (names, design) in GROUPS.items():
        arms = runs_of(names, variant, design)
        if not arms["off"]:
            continue
        kls = {name: best_kl(name, variant)[1] for name in names}

        def hours(t):
            return (
                np.cumsum(seconds[0] + seconds[1] * t[:, 2] + seconds[2] * t[:, 10] + seconds[3] * t[:, 3])
                / 3600
            )

        for length in LENGTHS[:2]:
            budget_hours = float(np.mean([hours(t)[length - 1] for _, _, t in arms["off"]]))
            hits, counts = {}, {}
            for arm, runs in arms.items():
                reached = [int(np.searchsorted(hours(t), budget_hours, side="right")) for _, _, t in runs]
                counts[arm] = float(np.mean(reached))
                hits[arm] = np.array(
                    [
                        judge(t, n, kls[name], seconds)["gap"] <= 0.10
                        for (name, _, t), n in zip(runs, reached, strict=True)
                    ]
                )
            for arm in ARMS:
                versus = ""
                if arm != "off":
                    d, lo, hi, p = paired(hits[arm], hits["off"], rng)
                    versus = f"{d:+6.1%} ({lo:+6.1%} to {hi:+6.1%}) p {p:.2f}"
                print(
                    f"{label:32s} {budget_hours:5.0f} {arm:6s} {counts[arm]:6.0f}  "
                    f"{hits[arm].mean():10.1%}  {versus:>30s}"
                )
    print()


def main() -> None:
    seconds = seconds_model()
    print(
        "K2's seconds per trial = {:.0f} + {:.2f} x prompts judged + {:.1f} x budget hits {:+.2f} x refusals "
        "(least squares on its 146 trials)".format(*seconds)
    )
    print(
        f"Budget model: logit q = {budget.INTERCEPT:g} + {budget.B:g} log(KL/KL*) {budget.C:+g} p "
        f"+ {budget.S:g} u, length {budget.LENGTH:g}, prompt correlation {budget.RHO:g}"
    )
    for variant in ("base", "stress", "cliffs"):
        if not (RUNS / variant).exists():
            continue
        print(f"\n== {variant} budget landscape\n")
        if variant != "base":
            failed = []
            for path in sorted(budget.CACHE.glob(f"budget-{variant}-*-{budget.INTERCEPT:g}.json")):
                saved = json.loads(path.read_text())
                failed += [(m["failed_before"], m["failed"]) for m in saved.get("made", [saved])]
            before, after = np.array(failed).T
            near = (
                "of the best's kind, position and direction layer within 10% of its KL at true refusals <= 2"
                if variant == "stress"
                else "of each basin whose best is within 10% of the scenario's best, within 10% of that KL "
                "at true "
                f"refusals <= {budget.CLIFF_REFUSALS:g}"
            )
            print(
                f"{len(failed)} bumps. Edits {near} that fail the limit: "
                f"median {np.median(before):.0%} without the "
                f"bumps, {np.median(after):.0%} with them (range {after.min():.0%}-{after.max():.0%}); "
                "each bump's "
                f"own best at {100 * budget.STRESS_AT:g} expected budget hits or a little over what it had. "
                "See "
                "near over 8 below for how many of the search's would-be hits fail.\n"
            )
        table(variant, seconds)
        print(f"Pooled, within 10% ({variant}):")
        pooled(variant)
        print(
            f"At equal K2 pod time, within 10% ({variant}): "
            "the trials each arm finishes in the off arm's hours at "
            "300 and 400 trials"
        )
        equal_time(variant, seconds)
        print(f"Per scenario, within 10% at 600 trials ({variant}):")
        by_scenario(variant, 600)


if __name__ == "__main__":
    main()
