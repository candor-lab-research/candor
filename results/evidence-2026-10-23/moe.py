"""The gap for the models gaps.py leaves out: mixture-of-experts models of any size, and dense models
over 36B. The same candidates, filters and counts as gaps.py otherwise, plus each model's experts
(blank where its config names them otherwise).

    python3 -I results/evidence-2026-10-23/moe.py > results/evidence-2026-10-23/moe.tsv
"""

import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gaps

gaps.MAX_PARAMS = math.inf


def main():
    expand = ["downloads", "createdAt", "safetensors", "tags", "config"]
    pool = {}
    for sort in ("downloads", "trendingScore"):
        for model in gaps.get(
            {"pipeline_tag": "text-generation", "sort": sort, "limit": 1000, "expand[]": expand}
        ):
            pool[model["id"]] = model
    candidates = [m for m in pool.values() if gaps.editable(m)]
    with ThreadPoolExecutor(8) as threads:
        found = list(threads.map(gaps.experts, candidates))
        kept = [(m, e) for m, e in zip(candidates, found, strict=True) if e[0] or (gaps.size(m) or 0) > 36e9]
        print(f"{len(pool)} models, {len(candidates)} editable, {len(kept)} MoE or over 36B", file=sys.stderr)
        rows = list(threads.map(lambda pair: gaps.gap(pair[0]), kept))
    for row, (_, (total, active)) in zip(rows, kept, strict=True):
        row["experts"], row["active_experts"] = total or "", active or ""
    rows.sort(key=lambda r: r["gap"], reverse=True)
    print("\t".join(rows[0]))
    for row in rows:
        print("\t".join(str(v) for v in row.values()))


if __name__ == "__main__":
    main()
