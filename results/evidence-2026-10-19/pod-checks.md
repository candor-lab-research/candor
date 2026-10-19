# Pod checks: the 8B judge and row normalisation (2026-10-19)

Two checks the phase-end discussion asked for (`method-evidence.md`, F104 and F67), on one A40 pod
(EU-SE-1, $0.59/h, 2 h 39 min, about $1.57 of GPU time). Numbers only: no prompt, answer or judge
reply was read. The scripts, their tables and raw outputs (labels, per-prompt KL, first-token reads
as token ids) are kept in `runs/podchecks-2026-10-19/`, outside git.

## F104: Qwen3Guard-Gen-8B on the stored held-out answers

The 8B at bf16 through candor's own judge code, on the answers `candor measure` stored for K2's
unedited model and finalists and Qwen's unedited model, finalists and heretic's Trial 161. Every
answer got a label (no "no label"); 44-60 s a set. Control: the 0.6B judged again the same way on
the pod changed 0 of its 5,292 stored labels. The 4B is the Mac's Q8_0 run (`method-evidence.md`).

Harmful held-out, refusals of 288:

| Set | Keywords | 0.6B | 4B | 8B |
| --- | --- | --- | --- | --- |
| K2 base | 287 | 287 | 285 | 286 |
| K2 205 | 23 | 3 | 6 | 7 |
| K2 207 | 15 | 1 | 3 | 2 |
| K2 265 | 14 | 1 | 1 | 2 |
| K2 114 | 9 | 3 | 9 | 9 |
| Qwen base | 282 | 279 | 280 | 280 |
| Qwen Trial 161 | 29 | 28 | 53 | 53 |
| Qwen 175 | 18 | 0 | 0 | 0 |
| Qwen 205 | 58 | 16 | 16 | 18 |

Harmless held-out, refusals of 300: the 8B finds 8 in all 2,700 (K2 base 2, 205 1, 265 1; Qwen
base 3, 205 1; the rest 0), the 0.6B 13 and the 4B 11.

Agreement, pooled (edited harmful: K2's 4 finalists, 1,152 answers; Qwen's 3 edits, 864):

| Pair | Pool | Both | First only | Second only | McNemar p | kappa |
| --- | --- | --- | --- | --- | --- | --- |
| 0.6B / 8B | K2 edited | 5 | 3 | 15 | 0.0075 | 0.35 |
| 0.6B / 8B | Qwen edited | 33 | 11 | 38 | 0.00014 | 0.55 |
| 0.6B / 8B | Unedited | 564 | 2 | 2 | 1 | 0.80 |
| 0.6B / 8B | All harmless | 7 | 6 | 1 | 0.125 | 0.67 |
| 4B / 8B | K2 edited | 14 | 5 | 6 | 1 | 0.71 |
| 4B / 8B | Qwen edited | 55 | 14 | 16 | 0.86 | 0.77 |
| 4B / 8B | Unedited | 565 | 0 | 1 | 1 | 0.95 |
| 4B / 8B | All harmless | 8 | 3 | 0 | 0.25 | 0.84 |

- The 4B and 8B agree with each other on edited models, with no systematic difference (p 1 and
  0.86). Against both, the 0.6B under-counts refusals there: 8 against the 8B's 20 on K2, 44
  against 71 on Qwen.
- Of the 53 edited answers the 8B calls refusals and the 0.6B doesn't, the 8B rates 32 Unsafe, 18
  Controversial and 3 Safe: most are answers it calls both a refusal and unsafe.
- Decisions under the 8B: K2's finalists 7, 2, 2 and 9 of 288, so all four stay within the
  harmful bar of 10 under both the 0.6B and the 8B; 114 is again closest. The rule's outcome is
  unchanged (205 and 114 pass; 207 and 265 fail on maths). Qwen's 175 refuses 0 under all three
  judges, Trial 161 53 and 205 18. No decision changes.

## F67: K2 205 and Qwen 175 scored with "full" and "pre"

Each run folder copied with only `edit.row_norm` changed to "pre"; the two differ only in their
adapters (checked). Each finalist scored at its own strengths with candor's `score_locked`, in the
order full, pre, full, pre, each score in its own session; seed 0; 28 slots (K2) and 32 (Qwen);
thinking as the search had it (K2 on at high, budget 8192; Qwen off); the search's 100 harmful and
100 harmless test prompts (KL over the 98 and 97 that the unedited model thought through). The
unedited thinking was sampled afresh on the pod, so these scores aren't repeats of the search's.

| Trial | Arm | Refused of 100, two scores | KL, two scores | Seconds a score |
| --- | --- | --- | --- | --- |
| K2 205 | full | 3, 1 | 0.0349, 0.0326 | 444, 501 |
| K2 205 | pre | 2, 1 | 0.0312, 0.0338 | 482, 456 |
| Qwen 175 | full | 0, 0 | 0.0494, 0.0507 | 130, 125 |
| Qwen 175 | pre | 1, 0 | 0.0519, 0.0523 | 127, 126 |

Prompt by prompt, with 95% bootstrap intervals over prompts:

| Trial | Pre's mean KL against full's | Same, every score against one read of the unedited model | Repeats of one arm | Per-prompt variance, pre against full over repeats |
| --- | --- | --- | --- | --- |
| K2 205 | -3.7% (-16.1% to +9.6%) | +0.9% (-6.4% to +9.2%) | -6.6% (full), +8.5% (pre) | 1.00 |
| Qwen 175 | +4.0% (-7.7% to +16.1%) | +6.0% (+0.5% to +11.1%) | +2.7% (full), +0.9% (pre) | 2.47 |

- Every pair, across arms or repeats, is a tie by candor's per-prompt test (gap within 2 SE).
  Refusals: no prompt-level difference on either (sign test p 1).
- Every score reads KL after the same unedited thinking; what moves it is reading again in a new
  session. Two reads of the unedited model differ by a mean KL of 0.0037-0.0063 on K2 and
  0.0016-0.0028 on Qwen, and a repeat's per-prompt sd is about the size of the KL on K2 (0.025-0.036
  at 0.033) and a third of it on Qwen (0.017-0.020 at 0.050).
- On K2 no change is detectable: against a common read the interval would show one of about 10%,
  and the variance ratio is 1.00. On Qwen, against a common read, pre may raise KL about 6% (+0.5%
  to +11.1%; paired t p 0.07, Wilcoxon p 0.04, sign test 53 against 44, p 0.42), one of 97 prompts
  carrying 43% of the gap; the variance ratio of 2.47 says pre changes some prompts beyond reading
  again. These intervals resample prompts, not the two sessions of each arm.
- Both trials were found by searches under "full", so a swap to "pre" at their strengths is biased
  toward "full". Whether "pre" searches better is still the one Qwen pair
  (`evidence-2026-10-14/data-report.md`).
- The "full" scores' mean KL is 10% above the search's 0.0306 on K2 and 12% below its 0.0567 on
  Qwen; they differ from the search's in the unedited thinking, the KL prompts and candor's code.

Verdict: no difference by candor's tie test on either finalist; on Qwen, pre may raise KL about 6%,
in the direction the tuning favours. Nothing here argues for leaving "full", which K2's and Qwen's
shipped edits use.
