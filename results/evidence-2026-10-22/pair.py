"""Each trial's harmful answers at 256 tokens (step 5) against 1,024 (step 6), prompt by prompt.
Numbers only.

    uv run python results/evidence-2026-10-22/pair.py"""

import json
from collections import Counter
from pathlib import Path

from candor.measure import mcnemar

FOLDER = "runs/qwen27b-{}/measurements/trial-{}-exported-slots-32/harmful_held_out.{}json"


def read(run: str, trial: int, labels: str = "") -> dict:
    return json.loads(Path(FOLDER.format(run, trial, labels)).read_text())


for trial in (251, 532):
    short, long = read("medium", trial), read("long", trial)
    same = sum(a == b for a, b in zip(short["thinking"], long["thinking"], strict=True))
    print(f"{trial}: thinking lengths equal on {same} of 288; finished {sum(short['finished'])} at 256,")
    print(f"  {sum(long['finished'])} at 1,024")
    for labels in ("judged.", "second_judge."):
        a = [s == "Unsafe" for s in read("medium", trial, labels)["judge_safety"]]
        safety = read("long", trial, labels)["judge_safety"]
        b = [s == "Unsafe" for s in safety]
        only_a = sum(x and not y for x, y in zip(a, b, strict=True))
        only_b = sum(y and not x for x, y in zip(a, b, strict=True))
        not_unsafe = [i for i, unsafe in enumerate(b) if not unsafe]
        cut = sum(not long["finished"][i] for i in not_unsafe)
        print(f"  {labels[:-1]}: unsafe {sum(a)} at 256, {sum(b)} at 1,024; only 256 {only_a}, only 1,024")
        print(f"    {only_b}, p {mcnemar(only_a, only_b):.2g}; labels at 1,024 {dict(Counter(safety))};")
        print(f"    not unsafe at 1,024 and cut there: {cut} of {len(not_unsafe)}")
