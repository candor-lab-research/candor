# The search's range ends: where the best edits sit, and what wider ranges cost (2026-10-27)

This is the phase-end question on the search's ranges (`history.md`, 2026-10-23). Should new runs
draw the peak layer from 0.4 rather than 0.6, and should reach go past 0.6? Numbers only. The first
part reads the four searches' journals; the second runs candor's own search code at d2043bc on the
miss-rate simulator; the third scores rude-llama's best edits moved past the ends, on the Mac. No
pods.

## Bottom line

- In every model, the top edits crowd two range ends: the peak's floor (0.6) and reach's ceiling
  (0.6). In each model, 7-8 of the 10 best one-layer edits press one or the other. 5 of the 8 kinds'
  best edits press some end, the first direction layer read among them.
- Crowding doesn't show that a better edit lies beyond an end. Most trials sit near those ends anyway
  (on the 27B per layer, 95 of 141 within the cap peak at 0.60-0.65). The simulator's landscapes
  crowd the same way, with nothing better beyond: on 16 of the 20 without thinking, the true best
  presses the peak's floor or reach's ceiling, held there by the landscapes' assumed rise in KL
  outside heretic's ranges.
- Where nothing better lies beyond today's ranges, drawing from wider ones from the start costs a
  lot. Within 10% of the true best at 600 trials, hits fall from 47% to 26% with the peak from 0.4,
  and to 8% with reach to 1.0 as well. The loss is as large where the best is truly interior (38%
  to 16% and 5%). It is the pattern phase 3 found for strength, where a fixed 0.5-3.0 halved the
  hits (`phase-3-qwen3.5-9b.md`, step 3).
- On rude-llama there is no better edit past the ends. Five of its best trials, each moved past the
  end it presses one setting at a time, raised KL in all 12 moves (by 1-61%), beyond 2 SE in 11, at
  no fewer refusals beyond noise.
- So the ranges stay as they are. Widening them on the edge alarm, as strength is, isn't needed on
  this evidence. Instead, a finalist that presses an end can be scored moved past it before it ships,
  as part 3 does, for a few trials' cost.

## 1. Where the best edits sit

`ends.py` (output in `ends.txt`) applies the search's own edge test (`search.at_edges`, within 5% of
a range end) to each kind's 10 least-KL trials within the cap. The table counts distinct trials.

| Search | One layer: the best presses | Per layer: the best presses | Top 10 pressing a peak's floor / a reach's ceiling: one layer; per layer |
| --- | --- | --- | --- |
| Llama-3.2-3B (rude) | nothing | nothing | 4 / 5; 0 / 0 |
| Qwen3.5-9B | the first direction layer read (12; all 10) | attention strength's ceiling (3.0), MLP reach's ceiling | 4 / 5; 0 / 5 |
| K2-Horizon-7B | MLP reach's ceiling | attention peak's floor | 4 / 5; 5 / 4 |
| Qwen3.8-27B | attention peak's floor | nothing | 3 / 5; 3 / 2 |

The 27B's per-layer kind had widened to strength 0.5-5.0. Trial 535, which rude-llama shipped,
presses the attention peak's floor and reach's ceiling (`phase-2-rude-llama.md`). Where the trials
within the cap sit by peak is in `evidence-2026-10-23/floor.txt`.

## 2. What the wider ranges cost where nothing better lies beyond

Each design ran 100 times on each landscape. The landscapes are the harness's (`tools/misscheck/`),
fitted to rude-llama's searches. KL rises outside the ranges those searches drew from: the peak
from 0.6 and reach up to 0.6 on every landscape, and strength outside 0.8-1.5 on those marked
penalised. "grow" and "D3grow" are candor's search as it is, without and with thinking. "Peak from
0.4" changes only the peak range; "and reach to 1.0" widens reach as well. The table gives the share
of runs whose pick is within 10% of the true best at 600 trials (95% interval):

| Landscapes | Runs | As it is | Peak from 0.4 | And reach to 1.0 |
| --- | --- | --- | --- | --- |
| Best inside 0.8-1.5, strength penalised | 700 | 47% (43-51) | 26% (22-29) | 8% (6-10) |
| Best inside 0.8-1.5, no strength penalty | 500 | 53% (49-58) | 25% (21-29) | 9% (7-12) |
| Best at attention 2.0-2.3, no strength penalty | 200 | 20% (15-26) | 3% (1-6) | 2% (1-5) |
| Best needs 2.0-2.5, strength penalised | 300 | 64% (58-69) | 26% (22-32) | 8% (5-12) |
| Best needs 2.0-2.6, no strength penalty | 300 | 71% (66-76) | 36% (31-42) | 10% (7-14) |
| Without thinking, the 4 whose best is interior | 400 | 38% (33-43) | 16% (13-20) | 5% (3-7) |
| Without thinking, the 16 whose best presses an end | 1600 | 56% (53-58) | 27% (25-29) | 9% (7-10) |
| Thinking, four landscapes, strength penalised | 400 | 61% (56-65) | 19% (16-23) | 6% (4-8) |
| Thinking, four landscapes, no strength penalty | 400 | 64% (59-69) | 17% (14-21) | 5% (3-8) |

- No run of any design beat the reference best edit, so the bar was the same for every design.
  Across the base landscape's runs, the median best KL rose from 0.0677 to 0.0764 and 0.0966,
  against a true best of 0.0562.
- Prompts judged per trial rose a little: 65.6, 67.6 and 70.8 on the first row.
- "As it is" matches phase 3's numbers for the design then built: 47% against 48%, and 61-64%
  against 65-66% pooled.
- **What this can't show:** what a wider range gains where a better edit lies beyond today's
  ranges. The landscapes are fitted to searches that drew only inside them.

## 3. rude-llama's best edits moved past the ends

`beyond.py` (output in `beyond.txt`) scores five of rude-llama's 10 best one-layer trials that press
an end, as the search scores its trials, on its 100 test prompts. Each is scored as it is, then with
the end it presses moved past the range, one setting at a time, and the two are paired by prompt in
the same session. The search was made with older code (search version 2, now 3), so these scores
aren't set beside its trials. Rescored as they are, all five repeat their search scores.

| Trial (as it is: refused, KL) | Moved past the end | Refused | KL | Paired KL gap (SE) |
| --- | --- | --- | --- | --- |
| 535 (0, 0.0778), shipped | attention peak 0.5 / 0.4 | 0 / 0 | 0.0786 / 0.0792 | +0.0007 (0.0003) / +0.0014 (0.0005) |
| | attention reach 0.8 / 1.0 | 0 / 0 | 0.0786 / 0.0786 | +0.0008 (0.0003) / +0.0007 (0.0004), a tie |
| 295 (2, 0.0777) | attention peak 0.5 / 0.4 | 1 / 1 | 0.1034 / 0.1138 | +0.0257 (0.0071) / +0.0361 (0.0096) |
| 345 (2, 0.0825) | attention peak 0.5 / 0.4 | 2 / 2 | 0.0855 / 0.0878 | +0.0030 (0.0009) / +0.0053 (0.0015) |
| 381 (2, 0.0770) | attention reach 0.8 / 1.0 | 2 / 2 | 0.0790 / 0.0795 | +0.0021 (0.0007) / +0.0025 (0.0008) |
| 463 (0, 0.0817) | MLP reach 0.8 / 1.0 | 0 / 0 | 0.1141 / 0.1313 | +0.0323 (0.0085) / +0.0496 (0.0112) |

- 535's attention edit is nearly flat (edge strength 0.996) and already spans layers 1-27, so its
  moves change little: 1-2% more KL. The narrower edits lose more: 295 33-46%, 463 40-61%.
- Limits: one model; single-setting moves with the other settings left as they are; the trials were
  chosen as best on these same test prompts, which favours them as they are.

## Files

- `ends.py`, `ends.txt`: part 1. `beyond.py`, `beyond.txt`: part 3, about 20 min on the Mac.
- `tools/misscheck/misscheck/ends_run.sh` (designs `floor-` and `ends-` in `runs.py`) and
  `ends_summary.py`: part 2, about 1.5 h on 10 cores with the references. The interior and on-an-end
  rows are `summary.table` over those landscapes; which landscapes' bests press an end is by the same
  edge test, applied to each landscape's true best.
