"""Negative control: pruning with too few sure beaters (g below TPE's n_below) must change the trajectory,
so the identical-trajectory check is able to fail."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import invariance as inv
for mode in ("mo", "cap"):
    full, jf, _ = inv.run(mode, prune=False)
    for g in (1, 5, 20, 21):
        p, jp, _ = inv.run(mode, prune=True, g=g)
        first = next((i for i, (a, b) in enumerate(zip(full, p)) if a != b), None)
        print(f"{mode} g={g}: first differing trial {first}; prompts judged {jp} vs {jf}")
