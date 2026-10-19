"""Full trial logs (family, restart, refusals, KL in order) for the allocation and agreement analyses."""

import json
import sys
import time

import designs


def traced(land, budget, seed, restarts, sampler):
    rows = []
    per = budget // (2 * restarts)
    for i, scope in enumerate(("one", "per")):
        for j in range(restarts):
            s = seed * 100 + i * 10 + j
            if sampler == "gp":
                log = designs.gp_family(land, scope, per, s, startup=min(20, per // 3))
            elif sampler == "gp_wide":
                log = designs.gp_family(land, scope, per, s, startup=min(20, per // 3), narrow=designs.WIDE)
            else:
                log = designs.family(land, scope, per, s, True, 0.3, sobol=True)
            rows += [[scope, j, r, k] for _, r, k, _ in log]
    return rows


def run(job):
    landscape, budget, seed, restarts, sampler = job
    land = designs.Shifted(designs.Landscape(designs.LANDS["union"])) if landscape == "shifted" else designs.Landscape(designs.LANDS[landscape])
    start = time.time()
    rows = traced(land, budget, seed, restarts, sampler)
    return {"landscape": landscape, "budget": budget, "seed": seed, "restarts": restarts, "sampler": sampler,
            "trials": rows, "seconds": time.time() - start}


if __name__ == "__main__":
    import multiprocessing as mp

    lands = sys.argv[1].split(",")
    budgets = [int(b) for b in sys.argv[2].split(",")]
    restarts = [int(r) for r in sys.argv[3].split(",")]
    first = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    seeds = range(first, first + int(sys.argv[4]))
    out = sys.argv[5]
    sampler = sys.argv[6]
    jobs = [(l, b, s, r, sampler) for l in lands for b in budgets for r in restarts for s in seeds]
    start = time.time()
    with mp.get_context("spawn").Pool(2, initializer=designs.init) as pool, open(out, "a") as f:
        for result in pool.imap_unordered(run, jobs):
            f.write(json.dumps(result) + "\n")
            f.flush()
    print(f"{len(jobs)} runs in {time.time() - start:.0f} s")
