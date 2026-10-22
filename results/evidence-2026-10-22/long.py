"""Measures an exported trial's held-out harmful prompts again, alone, with answers as long as the run's
measure.answer_tokens: in a hard-linked copy of the run whose candor.toml raises it, so the harmless and
maths sets stay as measured and candor compare still reads the whole measurement. Numbers only.

    python long.py RUN TRIAL"""

import sys
import time
from pathlib import Path

from candor import measure, search
from candor.run import Run

measure.SETS = ("harmful_held_out",)
measure.HELD_OUT = ("harmful_held_out",)


def progress(description, total):
    count = 0

    def done():
        nonlocal count
        count += 1
        if count % 32 == 0 or count == total:
            print(f"{time.strftime('%H:%M:%S')} {description} {count}/{total}", flush=True)

    return done


folder, trial = Path(sys.argv[1]), int(sys.argv[2])
run = Run.open(folder)
assert run.answer_tokens["harmful_held_out"] == 1024, run.answer_tokens
name = f"trial-{trial}-exported-slots-{run.settings.slots.measure}"
edit = search.recorded_edit(run, trial)
result = measure.Measurement(run, edit, name, progress=progress, exported=True).measure()["harmful_held_out"]
for which, labels in (("judge", result), ("8B", result[measure.SECOND])):
    refused = sum(bool(r) for r in labels["judge_refused"])
    unsafe = sum(s == "Unsafe" for s in labels["judge_safety"])
    print(f"{name} {which}: refused {refused}, unsafe {unsafe} of {len(labels['judge_safety'])}")
print(f"cut at 1024: {sum(not f for f in result['finished'])}; seconds {round(result['seconds'])}")
