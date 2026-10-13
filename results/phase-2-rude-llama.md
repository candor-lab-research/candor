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

## The search fix (2026-10-12)

Four agents studied the miss from different sides and a fifth tested designs on test landscapes:
synthetic score landscapes over candor's exact settings, calibrated to the 400 real trials of both
searches (candor's and heretic's) with Trial 170 as the anchor between their scales. Each design was
run 20-30 times per landscape. Everything is numbers only; the agents never saw prompts or answers.

Why the search missed, from Optuna 4.9's source and the real run:

- Mainly the goal. Aiming at every trade-off of refusals against KL, the sampler's set of good
  trials (the top of the refusal-KL front) held 9 one-layer trials that were all weak edits: 36-93
  refusals at KL 0.009-0.027, MLP peaking at the last layers with a short reach. New trials are drawn
  near good ones, so it kept drawing weak one-layer edits. Only 1 of its 96 one-layer trials was near
  Trial 170's shape.
- The good one-layer shape is rare under today's ranges: both bands start early and run to the last
  layer. About 4% of random draws have it, and under 1% also read a good direction layer.
- Sharing settings between the two kinds of edit was only a small leak; separating them alone did
  not help.

Results on the calibrated landscape, share of runs within 10% of the best (within 3%):

| Design | 400 trials | 600 trials | Equal time to today's 400 |
| --- | --- | --- | --- |
| Today's search | 20% (0%) | 23% (0%) | 20% (0%) |
| Two kinds apart, each capped, spread start | 83% (13%) | 93% (20%) | 77% (13%) |
| The same, stopping trials sooner (after 1 fully scored trial refused fewer) | 73% (13%) | 93% (17%) | 93% (17%) |

On the harshest landscape (a narrower target) the chosen design reached 17% at 400 trials against
3%, and 37% at 1,200: better, but still a miss in most runs, with nothing inside a run to warn of it.
Where the best was per-layer it reached 100% against 97%. Optuna's GP sampler did worse in this
harness: with refusal counts this noisy it settled at the cap's edge, where edits truly refuse 3-4.
Repeats did not help: for the same total trials one long run beat two or four shorter ones, and
whether two repeats agreed did not predict a miss. Stopping when progress stalls missed too often.

Built: the two kinds as two studies taking turns, each starting from 20 scrambled Sobol points,
multivariate TPE minimising KL with refusals above 2 of 100 as a constraint (by how far over, scaled
to all prompts for a stopped trial), and a trial stopped once it has refused more than 2 and more
than 25 fully scored trials of its kind. TPE learns from at most 25 best trials, all of which then
rank ahead of the stopped one, so the sampler draws exactly what it would have without stopping; a
review ran candor's own search code on the landscapes and got the simulated design's trials one for
one. The first build stopped after 1 trial refused fewer, which the evaluation had kept only as a
speed mode; the review caught it, and it missed more on the harshest landscape (table above, 17%
against 30% at 600 trials). The time saved is to be measured on rude-llama. 300 trials of each kind
by default. The search's report flags a best trial pressing on the end of a range. Widening ranges automatically is deferred: the
landscapes can't say whether it matters, and the edge alarm on a real model will.

A check on six randomised landscapes (rand1-rand6, made after the design was chosen; 60 runs
each, 100 on rand5) gave the share of runs within 10% of the best, pooled: today's search 22% at
400 trials and 26% at 9.6 hours of scoring (about its 600 trials); without stopping, the chosen
design (which the built stop rule doesn't change) 46% and 50%. It was at or above today's search on
every landscape and budget, significantly on most; the only cells below it were one run in 60
within 3%. rand2, whose best sits on the attention strength's ceiling with mid-depth peaks like the
harshest landscape, stayed hard for every design (at most 22%). The references are good to 1-2%,
so within-3% figures are soft. A gentler stop rule gained nothing per trial and cost time.

The 4-trial pilot (2026-10-12) took 57-91 s a trial against 46-67 s for the old search's trials
refusing as many (71, 75, 28 and 98 of 100 against the old trials' times by refusals). Scoring a
chunk at a time left the model idle while the judge worked, and short refusals left slots empty
until the chunk's longest answer finished. Each chunk is now judged while the server answers the
ones after it. The chunks and the judge's batches are the same, so scores mean what they did and
the pilot's trials stand. The edge alarm also waits until a kind has trials beyond its spread
start: the pilot flagged a start point, which is only a point of a fixed grid.

Limits to remember: 100 test prompts can't tell 0 refusals from 2, so the search's pick truly
refuses about 2 of 100 and sits above the true best; measuring the best few of each kind on the
held-out prompts, as the search suggests, is how the choice is made.

## The rebuilt search on rude-llama (2026-10-12)

`uv run candor search runs/rude`: 600 trials in 7 h 21 min, about 44 s each (the first search: 58 s).
Stopping trials that can't win saved about a quarter; 262 trials were scored in full. Picked from
its study, numbers only:

| Kind | Least KL at 0 refused | at 1 | at 2 |
| --- | --- | --- | --- |
| One layer | 535: 0.0778 | 231: 0.0833 | 305: 0.0760, 327: 0.0764 |
| Per layer | 324: 0.0914 | 462: 0.0874 | 448: 0.0831, 492: 0.0849 |

On the same scorer, Trial 170 refuses 0 at KL 0.0834 (0.0818 read at layer 12), so 535 matches it
on refusals at 5-7% less KL. 535 reads layer 12, with attention as Trial 170's (strength 1.33 against
1.33, peak at layer 16.7 against 16.5, editing from layer 1 against 3 to the last) and a gentler MLP
(1.03 against 1.26, peak 19.5 against 16.6).

The search suggested the two least-KL trials of each kind within the cap: 305, 327, 448 and 492,
all refusing 2. It skipped 535, and its edge alarm checked only 305, which presses no range end. 535
presses two: the attention peak's floor (0.6) and its reach's ceiling (0.6). The fix (`77106e4`,
committed before any held-out number) suggests each kind's least KL at each count within the cap
unless one refusing fewer has less, checks each for range ends, and announces such trials as they
finish. On this run it suggests 535, 305, 324, 462 and 448.

Held-out, measured as Trial 170 was:

| | Trial 170 | 535 | 305 | 327 | 448 | 492 |
| --- | --- | --- | --- | --- | --- | --- |
| Search score (refused /100, KL) | 0, 0.0834 | 0, 0.0778 | 2, 0.0760 | 2, 0.0764 | 2, 0.0831 | 2, 0.0849 |
| Harmful refused, judge (at most 6/288) | 6 | 5 | 5 | 3 | 14 | 15 |
| KL from unedited (at most 0.1011) | 0.1011 | 0.0920 | 0.1214 | 0.1205 | 0.0973 | 0.0965 |
| Maths correct /1319 (unedited 980) | 977 | 996 | 981 | 997 | 990 | 996 |
| Harmless refused, judge (at most 7/300) | 2 | 2 | 3 | 2 | 3 | 3 |
| Harmful refused, keywords | 25 | 16 | 11 | 11 | 15 | 18 |
| KL from Trial 170 | | 0.0010 | 0.0237 | 0.0211 | 0.0283 | 0.0296 |
| Decision rule | The bar | Passes | Fails on KL | Fails on KL | Fails on refusals | Fails on refusals |

Against Trial 170, prompt by prompt, 535: harmful refused 1 / 0 (p 1), keywords 12 / 3 (p 0.035),
maths 34 / 53 (p 0.053). Against the unedited model, its maths is 70 / 86 (p 0.23).

What it shows:

- 535 passes phase 2's decision rule, so it ships (the milestone:
  `milestone-rude-llama-vs-heretic.md`).
- The check set before the run (a suggested trial at most 6/288 at KL at most 0.1041) fails for the
  four trials as the search printed them; it passes for 535 under the fixed suggestions. The fix
  came from the search's own numbers, not the held-out ones.
- The twins agree (305 and 327, 448 and 492), so the gaps between search and held-out are not luck;
  they depend on the kind of edit:
  - One layer at direction layer 14: refusals hold (2/100 became 3-5/288), KL rises 58-60%.
  - Per layer: KL rises 16-17%, refusals don't hold (2/100 became 14-15/288).
  - 535 and Trial 170, both at layer 12: KL rises 18-21%, refusals 0/100 became 5-6/288.
- The harmful test prompts are AdvBench (`harmful_behaviors`) and the held-out ones StrongREJECT, so
  refusals cross a change of source. The harmless test and held-out prompts are the same set
  (`harmless_alpaca` `test[:100]` and `test[100:400]`), so layer 14's KL gap comes from 100 prompts
  and from picking the least of hundreds of trials on them.
- Budget: one layer's least KL at 0 refused was 0.1046 after 200 trials, 0.1012 after 300, 0.0875
  after 400, 0.0817 after 500 and 0.0778 after 600; still falling at the end. The planned check of
  the picks after 200 and 400 trials was dropped, since one run can't cut the 600 default.

## Shipping trial 535 (2026-10-12 to 2026-10-13)

`candor export` merges an edit into the weights as its adapters apply it, converts and quantises;
`candor measure --exported` measures that file. The first export of 535, plain Q4_K_M, refused
161 of 288 held-out prompts against 5 with adapters, at KL 0.0318 from the unedited model against
0.0920: most of the edit was gone.

The merge was right (the change in every edited weight pointed along the edit, cosine 0.97-0.999);
quantising took it back. The share of each edited weight's change the file held: Q4_K 55-71%, Q6_K
(half the `ffn_down` layers in Q4_K_M) 94-97%, Q8_0 99.8-100%. The cause, found by three research
agents (numbers and code only): rude-llama's F16 weights were 4-bit before (MLX, `library_name: mlx`;
every 64-weight group of layers 0-19 holds at most 16 values on an even grid). Q4_K fits that grid
again, so a weight moved by an edit's few percent of a step rarely changes its value. On weights
never quantised, Q4_K keeps an edit (Gaussian rows 1.00, Qwen3Guard-0.6B 0.99-1.00), which is why
phase 0's model kept about 90%. The papers agree on the mechanism (2609.09793 in FP8; 2410.16454 and
follow-ups for 4-bit undoing unlearning) and none measured abliteration through GGUF; no abliteration
tool or quantiser checks refusals after quantising.

The fix (`6e47a01`): the export measures each edited weight's kept share, quantises those under 98%
as Q8_0 through llama-quantize's tensor type file, and refuses the file if any still falls short.
For 535: Q4_K_M kept 55.2-97.5% in all 49 edited weights; as Q8_0 each kept 99.8-100.0%; the file
is 2.4 GB against 1.9.

| Held-out | Trial 170 | 535 | Trial 170 exported | 535 exported |
| --- | --- | --- | --- | --- |
| Harmful refused, judge (at most the bar's) | 6 | 5 | 7 | 6 |
| KL from unedited (no higher than the bar's) | 0.1011 | 0.0920 | 0.1186 | 0.1132 |
| Maths correct /1319 (unedited 980) | 977 | 996 | 986 | 982 |
| Harmless refused, judge (at most 7/300) | 2 | 2 | 2 | 3 |
| Harmful refused, keywords | 25 | 16 | 19 | 21 |
| KL from the same edit as adapters | | | 0.0279 | 0.0268 |

- 535 exported against Trial 170 exported, the comparison phase 2's target names ("converted and
  quantised the same way"): 6 against 7 refused (p 1), KL 0.1132 against 0.1186, maths 982 against
  986 (p 0.76). It passes the decision rule, narrowly: the two files are KL 0.0014 apart.
- Against Trial 170 as adapters (0.1011) 535 exported fails on KL. Both exports rose by about 0.02
  in KL from their adapter measurements, so the rise is the export, not 535.
- Moving the 49 edited tensors to Q8_0 with no edit at all is itself KL 0.052 from plain Q4_K_M,
  and the edit measured against that base is 0.114. The likely reason, not proven: rude-llama's own
  fine-tune changes in layers 20-27 are as small as an edit (0.04-0.07 of a step), so plain Q4_K_M
  loses part of the fine-tune too, which Q8_0 keeps.
- The blind A/B was dropped by the user (2026-10-12): 535 and Trial 170 are KL 0.0010 apart as
  adapters, too close to tell apart by hand.
