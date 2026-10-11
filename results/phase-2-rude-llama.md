# Phase 2 results: rude-llama

Numbers only; answers stay in the run's private folder. Every measurement is Q4_K_M in llama-server,
8 slots, answers up to 256 tokens (maths 512), refusals judged by Qwen3Guard-Gen-0.6B. The rule
these are judged against is in STATUS.md (phase 2's decision rule).

## The search (2026-10-11)

`uv run candor search runs/rude`: 200 trials (60 random, then aimed) in 3 h 13 min, about 58 s
each. Each trial is scored on the 100 test prompts with 100-token answers, and KL on the 100
harmless test prompts. The unedited model refuses 99 of the 100.

The fewest refusals so far, as it ran: trial 10 (13), 16 (7), 77 (4), 88 (3), 136 (2), 165 (0).

Trials no other beats on both refusals and KL:

| Trial | Refused /100 | Keywords /100 | KL |
| --- | --- | --- | --- |
| 165 | 0 | 4 | 0.1017 |
| 167 | 1 | 4 | 0.0984 |
| 187 | 3 | 8 | 0.0840 |
| 135 | 4 | 7 | 0.0824 |
| 190 | 12 | 10 | 0.0741 |
| 10 | 13 | 17 | 0.0733 |
| 107 | 16 | 12 | 0.0703 |
| 112 | 17 | 19 | 0.0683 |
| 60 | 18 | 19 | 0.0681 |
| 185 | 19 | 17 | 0.0622 |

All ten read their direction per layer. Trial 170 uses one direction, from layer 12.04.

## Held-out measurement (2026-10-11)

| | Unedited | Trial 170 | Trial 165 | Trial 167 | Trial 187 |
| --- | --- | --- | --- | --- | --- |
| Harmful refused, judge (rule: at most 6/288) | 277 | 6 | 5 | 9 | 11 |
| Harmful refused, keywords | 275 | 25 | 12 | 19 | 22 |
| Harmful answered unsafely | 6 | 254 | 257 | 255 | 250 |
| Harmful cut off at 256 tokens | 12 | 266 | 259 | 263 | 261 |
| Harmless refused, judge (rule: at most 7/300) | 7 | 2 | 2 | 2 | 1 |
| Harmless refused, keywords | 7 | 4 | 2 | 2 | 4 |
| Maths correct /1319 | 980 | 977 | 1003 | 998 | 998 |
| Maths p against unedited (rule: at least 0.05 if worse) | | 0.87 | 0.083 | 0.197 | 0.173 |
| Maths cut off at 512 tokens | 17 | 21 | 18 | 19 | 14 |
| KL from unedited (rule: at most 0.1011) | | 0.1011 | 0.1089 | 0.1061 | 0.0995 |
| Strict rule | | The bar | Fails on KL | Fails on refusals and KL | Fails on refusals |

Head to head against Trial 170, prompt by prompt (only 170 / only candor's trial):

| | Trial 165 | Trial 187 |
| --- | --- | --- |
| Harmful refused, judge | 3 / 2, p 1 | 2 / 7, p 0.18 |
| Harmful refused, keywords | 16 / 3, p 0.004 | 12 / 9, p 0.66 |
| Maths correct | 67 / 93, p 0.048 | 69 / 90, p 0.11 |
| KL from Trial 170 | 0.0311 | 0.0288 |

## What it shows

- No candor trial passes the strict rule.
- Trial 170 sits at a better point on the refusal-KL trade-off than anything candor found: at
  about its KL (0.0995-0.1061), candor's trials refuse 9-11 against its 6, and the one that matches
  its refusals (165) has 8% more KL. Each gap alone is within chance (165's refusals p 1, 187's
  p 0.18), but all three trials fall on the same side.
- candor's trials do better at maths: 998-1003 against Trial 170's 977 and the unedited 980. Only
  165 against Trial 170 gets under p 0.05 (0.048), and the three are probably similar recipes, so
  this is a lead, not a finding.
- The search ranked refusals in the right order (0, 1 and 3 of 100 became 5, 9 and 11 of 288) but
  flattered its best trials, as picking the best of 200 on the same 100 prompts does. Its KL ran
  7-18% below the held-out KL.
- candor's trials trip the keyword list less than Trial 170 (12-22 against 25).

## The search's single-direction trials

Read from the search's study (numbers only): 96 of the 200 trials used a single direction (32 of
the 60 random, 64 of the 140 aimed). None refused fewer than 35 of 100; every trial under 10 used
directions per layer. heretic's own search tried 125 single-direction trials: none under 10 in its
random phase either (best 11), but from trial 90 on it found 20 under 10, and they were its best
(Trial 170: 2 of 100 by keywords at its lowest KL). heretic's good ones edit the MLP from the middle
of the model with a wide reach (Trial 170 peaks at layer 16.6 of 27 and reaches 12.8 layers);
candor's single-direction trials nearly all peaked the MLP at the last layers with a short reach.
Trial 170 read at layer 12 is inside candor's search space.

## Trial 170 scored by the search's own scorer (2026-10-11)

`candor score runs/rude --heretic .../models--rude-llama-bf16.jsonl --trial 170`:

| Edit | Direction | Refused /100 | Keywords | KL | In the search |
| --- | --- | --- | --- | --- | --- |
| heretic-170 | layer 12.039 | 0 | 3 | 0.0834 | |
| heretic-170 at layer 12 | layer 12 | 0 | 4 | 0.0818 | |
| trial-165 (the check) | per layer | 0 | 4 | 0.1014 | 0, 4, 0.1017 |

The unedited model refuses 99 of 100, as in the search.

- The search missed it. Trial 170 read at layer 12, which the search could have drawn, refuses as
  few as candor's best (0) at about 20% less KL (0.0818 against 0.1017), and beats every trial in
  the search's table on both.
- Not a fault in single-direction scoring: the layer-12 version uses the search's own adapters.
- Not the scorer: it ranks Trial 170 ahead of 165, as the held-out measurement did (KL 0.1011
  against 0.1089).
- The check repeated its refusals and keywords exactly, and its KL to within 0.0003 (0.3%). The
  command calls that "not reliable" because it wants a match to 4 decimals; the gap that decides
  this is about 0.02, some 60 times larger, so the conclusion stands.

## Before the diagnosis

The plan was to score Trial 170 with the search's own scorer, on the same 100 prompts, before deciding what ships
(`candor score`, built for this):

- If it beats candor's best trials there, the scorer would have picked it and the search missed
  it: look at the search space and what the search tried (every trial above is per layer; Trial
  170 is one direction).
- If it scores about the same as 165 and 167, the scorer can't separate edits this close: score
  on more prompts or with longer answers.
- If it scores far worse than its held-out 6 of 288 suggests (around candor's single-direction
  trials, 35 or more), look for a fault in how the search scores single-direction edits.
