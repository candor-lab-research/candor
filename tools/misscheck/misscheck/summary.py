"""The final tables: each question's designs on groups of scenarios, pooled, judged as report.py judges.

    uv run python summary.py
"""

from report import BUDGETS, best_position, cell, ci, envelope, load
from scenarios import WIDE, fronts, load_refs, scenario_front, scenarios, truth

RANGE = {
    "penalised, best inside 0.8-1.5": ["base", "rand1", "rand2", "rand3", "rand4", "rand5", "rand6"],
    "free, best inside (or narrow within 4%)": ["fbase", "frand1", "frand3", "frand4", "frand6"],
    "free, best at attn 2.0-2.3 (narrow +27-35%)": ["frand2", "frand5"],
    "penalised, best needs 2.0-2.5": ["basex2.4", "rand4x1.8", "rand6x1.8"],
    "free, best needs 2.0-2.6": ["fbasex2.6", "frand4x1.8", "frand6x1.8"],
}
RANGE_DESIGNS = ["narrow", "wide", "widelog", "grow"]
POSITION = {name: [name] for name in ["a-same", "b-answer", "c-think", "d-high", "fa-same", "fb-answer", "fc-think", "fd-high"]}
POSITION |= {"penalised a-d pooled": ["a-same", "b-answer", "c-think", "d-high"],
             "free a-d pooled": ["fa-same", "fb-answer", "fc-think", "fd-high"]}
POSITION_DESIGNS = ["D1", "D2", "D3", "D3x", "D4", "D1log", "D3log", "D3grow"]


JUDGING = {}


def judging(name: str, refs, curves) -> tuple:
    """The scenario's best KL and front, lowered to any simulated trial that beat them, and the best
    edit's position."""
    if name not in JUDGING:
        scenario = scenarios()[name]
        best = truth(scenario, WIDE, refs)
        curve, kl = envelope(name, scenario_front(scenario, curves), best.kl)
        JUDGING[name] = (curve, kl, best_position(scenario, best))
    return JUDGING[name]


def judged_cells(names: list[str], design: str, refs, curves) -> dict | None:
    totals = {b: {"n": 0, "hit10": 0, "hit3": 0, "front": 0, "found": 0, "pos": 0, "judged": 0.0} for b in BUDGETS}
    for name in names:
        runs = load(name, design)
        if not runs:
            return None
        curve, kl, position = judging(name, refs, curves)
        for budget in BUDGETS:
            c = cell(runs, budget, kl, position, curve)
            t = totals[budget]
            for k in ("n", "hit10", "hit3", "front", "found", "pos"):
                t[k] += c[k]
            t["judged"] += c["judged"] * c["n"]
    return totals


def table(groups: dict, designs: list[str], positions: bool) -> None:
    refs, curves = load_refs(), fronts()
    head = "  ".join(f"{'within 10% at ' + str(b):>19s}" for b in BUDGETS)
    print(f"{'scenarios':44s} {'design':7s} {'runs':>4s}  {head}  {'within 3% at 300/400/600':>24s}  "
          f"{'found@600':>9s} {'judged':>6s}" + ("  right position@600" if positions else ""))
    for label, names in groups.items():
        for design in designs:
            totals = judged_cells(names, design, refs, curves)
            if totals is None:
                continue
            n = totals[BUDGETS[0]]["n"]
            tens = "  ".join(f"{ci(totals[b]['hit10'], n):>19s}" for b in BUDGETS)
            threes = " / ".join(f"{totals[b]['hit3'] / n:4.0%}" for b in BUDGETS)
            last = totals[BUDGETS[-1]]
            line = (f"{label:44s} {design:7s} {n:4d}  {tens}  {threes:>24s}  {last['found'] / n:9.0%} "
                    f"{last['judged'] / n:6.1f}")
            if positions:
                line += f"  {last['pos'] / n:19.0%}"
            print(line, flush=True)
        print()


if __name__ == "__main__":
    print("== Question A: strength range (judged = prompts judged per trial over 600 trials)\n")
    table(RANGE, RANGE_DESIGNS, False)
    print("== Question B: where the direction is read, strength 0.5-3.0\n")
    table(POSITION, POSITION_DESIGNS, True)
