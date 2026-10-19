"""K2-Horizon-7B's search trials as numbers: one row per trial number, from its Optuna journal.

Reads only numeric user attributes and parameters (the journal holds no text but the position names).

    python k2/extract.py /path/to/study.jsonl > k2/trials.csv
"""

import csv
import json
import sys

NUMERIC = ("refusals", "kl", "judged", "forced", "thinking", "seconds", "stopped")
PARAMS = (
    "direction_layer",
    *(f"{c}.{s}" for c in ("attn", "mlp") for s in ("strength", "peak_layer", "edge_strength", "reach")),
)

trial_ids = []  # trial id -> study id, in creation order
params, attrs, ranges = {}, {}, {}
with open(sys.argv[1]) as journal:
    entries = [json.loads(line) for line in journal]
for d in entries:
    op = d["op_code"]
    if op == 4:
        tid = len(trial_ids)
        trial_ids.append(d["study_id"])
        params[tid] = dict(d.get("params") or {})
        attrs[tid] = {k: v for k, v in (d.get("user_attrs") or {}).items() if k == "number" or k in NUMERIC}
        for name, dist in (d.get("distributions") or {}).items():
            dist = json.loads(dist) if isinstance(dist, str) else dist
            if name.endswith("strength") and "attributes" in dist:
                ranges[name] = (dist["attributes"]["low"], dist["attributes"]["high"])
    elif op == 5:
        tid = d["trial_id"]
        dist = json.loads(d["distribution"])
        value = d["param_value_internal"]
        if dist["name"] == "CategoricalDistribution":
            value = dist["attributes"]["choices"][int(value)]
        elif d["param_name"].endswith(".strength"):
            ranges[d["param_name"]] = (dist["attributes"]["low"], dist["attributes"]["high"])
        params[tid][d["param_name"]] = value
    elif op == 8:
        attrs[d["trial_id"]].update(
            {k: v for k, v in d["user_attr"].items() if k == "number" or k in NUMERIC}
        )

rows = {}
for tid, a in attrs.items():
    if "number" not in a or "kl" not in a or a["number"] in rows:
        continue
    p = params[tid]
    rows[a["number"]] = {
        "number": a["number"],
        "one": int("direction_layer" in p),
        "answer": int(p.get("position") == "answer start"),
        **{k: p.get(k, "") for k in PARAMS},
        **{k: (int(a[k]) if k == "stopped" else a[k]) for k in NUMERIC},
    }
out = csv.DictWriter(sys.stdout, fieldnames=["number", "one", "answer", *PARAMS, *NUMERIC])
out.writeheader()
for n in sorted(rows):
    out.writerow(rows[n])
print("strength ranges seen:", ranges, file=sys.stderr)
