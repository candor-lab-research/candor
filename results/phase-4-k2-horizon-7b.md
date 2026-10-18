# Phase 4 results: K2-Horizon-7B

Numbers only; answers stay in each run's private folder. The plan and decisions are in STATUS.md.

## Step 2: the timed pilot (2026-10-16)

One A40 pod (`3hbos4ytt33lux`, secure, EU-SE-1, $0.59/hr, 46,068 MiB, 05:21-08:06 UTC, about
$1.60), deleted by hand after the pull. IFM/K2-Horizon-7B at 4622f94 (main, with the 2026-10-01
anti-doom-loop merge) in Q4_K_M with an f16 KV cache, no system prompt, reasoning effort high,
sampled at the card's settings (temperature 1.0, top_p 0.95, top_k 0, min_p 0) with seed 0,
thinking to a budget of 8192, row normalisation "full", judged by Qwen3Guard-Gen-0.6B.

**The run built in 40 minutes** (setup 84 s before it): the 17 GB download under a minute,
conversion, quantisation and residuals about 10 minutes, then the unedited model's own thinking at
the build stage's fixed 8 slots, about 30 minutes for 400 prompts. Its thinking ended by itself on
100 of 100 harmless test prompts and 299 of 300 held-out ones.

**Slots that fit on an A40.** K2 is dense (36 layers, 8 KV heads of 128): its f16 KV cache takes
144 KiB a token, and llama-server allocates every slot's context at start.

| Command | Context per slot | 32 slots | Used |
| --- | --- | --- | --- |
| `measure` | 9,216 | 41,472 MiB of KV: out of memory | 28 slots, 41,497 MiB in all |
| `search` | 15,616 (the longest KL read plus a whole thinking answer) | | 28 slots asked 61,488 MiB of KV: out of memory; 16 slots, 43,370 MiB in all |

The search fixes its thinking slots at 32 in code; the pilot ran it at 16 on the pod only.

**The unedited model and Avicennasis's heretic edit**, the first 64 prompts of each set, 28 slots,
paired (`candor compare`). Avicennasis's 72 edited weights (every attention output and MLP down)
were exported from its safetensors at b364bdb in 21 minutes; Q4_K_M kept 52.3-97.6% of the edit in
13 of them, so those are Q8_0, and every edited weight then keeps 98.0-173.8%.

| First 64 of each set | Unedited | Avicennasis | Only one / the other | p |
| --- | --- | --- | --- | --- |
| Harmful refused, judge | 61/64 | 48/64 | 13 / 0 | 0.0002 |
| Harmful refused, keywords | 63/64 | 52/64 | 11 / 0 | 0.001 |
| Harmful answered unsafely | 0/64 | 8/64 | 0 / 8 | 0.008 |
| Harmful answer cut off at 256 | 6/64 | 47/64 | 1 / 42 | 1e-11 |
| Harmless refused, judge | 1/64 | 0/64 | 1 / 0 | 1 |
| Maths correct | 61/64 | 61/64 | 3 / 3 | 1 |
| Thinking cut at 8192, harmful / harmless / maths | 0 / 0 / 0 | 0 / 4 / 2 | | |
| Thinking tokens, median (most), harmful | 225 (1,292) | 523 (6,429) | | |
| Harmless | 561 (4,617) | 882 (8,213) | | |
| Maths | 140 (1,900) | 158 (8,213) | | |
| Time, harmful / harmless / maths | 120 / 352 / 142 s | 379 / 798 / 421 s | | |
| A full measurement, candor's estimate | at most 1 h 25 min | at most 3 h 55 min | | |

KL of Avicennasis's edit: 0.090 where the answer starts after the unedited model's own thinking,
0.553 after an empty thinking block, 0.058 where the thinking starts. Its card says 7 of 50 refused
on its own prompts.

**Search trials as used** (`candor search --trials 4`, 16 slots; deleted afterwards, as the
search's settings aren't chosen yet). The unedited model refused 97 of 100 test prompts.

| Trial | Direction | Refused | KL | Mean thinking | Time |
| --- | --- | --- | --- | --- | --- |
| 1 | one layer, 23 | 92 | 0.0101 | 338 | 237 s |
| 2 | per layer | 94 | 0.0218 | 371 | 236 s |
| 3 | one layer, 19, answer start | 97 | 0.0082 | 277 | 177 s |
| 4 | per layer, answer start | 99 | 0.0079 | 306 | 191 s |

None of the 400 answers reached the budget. All four are start points that barely change the
model; none complied, so the cost of an edit that does is read from Avicennasis's harmful set,
about 3 times the unedited model's time.

**Reasoning effort low**, a copy of the run at `reasoning_effort = "low"` (close
`</ifm|think_faster>`), built in 23 minutes: its thinking ended with that close on only 40 of 100
harmless test prompts and 109 of 300 held-out ones. Counting tags on 32 harmless test prompts
(token ids only):

| 32 harmless test prompts | Low | High |
| --- | --- | --- |
| Closed with `</ifm|think>` | 17 | 31 |
| Closed with `</ifm|think_faster>` | 15 | 0 |
| Tokens written, median (most, capped at 2,900) | 250 (2,283) | 918 (2,900) |

At low K2 opens with `<ifm|think_faster>` but often closes as at high, so candor counts those as
never closed; low wasn't measured further.

**What it shows**
- K2 at high thinks far less than Qwen3.5-9B did: medians 225 / 561 / 140 against Qwen's 996 /
  1,137 / 2,140 (harmful, harmless, maths), with no budget hits unedited. A full measurement is
  at most about 1.5 h unedited and 4 h for an edit that complies in part; Qwen's unedited maths
  set alone took 4.9 h.
- Step 3 (the unedited model and Avicennasis measured in full, at 28 slots) is about 6-7 h with
  the build and export, about $4; Uno-Uncensored would add about 4 h more.
- A search as used at high: near-unedited trials take about 210 s at 16 slots. Trials that comply
  think longer and the search aims at them, so 600 trials is roughly 35-60 h, about $20-35, at 16
  slots; sizing the search's slots by what each request needs (28 fit) could cut that by a
  quarter to a third. The top of the range is set by how long complying edits think, which no
  trial here reached.
- Reasoning effort low writes about 3.7 times fewer tokens, but needs both closes accepted before
  it can be scored, and its card advises against it for evaluation.

## Step 2, speed test: GPU and sampling (2026-10-16)

Two pods for about 20 minutes each (A40 `ifncdv854juqi3`, CA-MTL-1, $0.59/hr; A100 SXM 80 GB
`d9yw07h31ud45h`, US-MD-1, $1.79/hr; about $0.65 together), deleted by hand. IFM's own
K2-Horizon-7B Q4_K_M (at 7a75dbd; same architecture and size as the run's), candor's server flags
and the run's sampling, made-up prompts of 40 token ids (never decoded), answers of exactly 384
tokens; llama-server's own time per token. Rows are answers running at once.

| At once | A40 | A100 |
| --- | --- | --- |
| 1 | 12.0 ms a step, 83 tokens/s | 8.5 ms, 115 tokens/s |
| 4 | 18.4 ms, 213 | 16.5 ms, 237 |
| 16 | 43.2 ms, 366 | 37.6 ms, 419 |
| 28 | 64.9 ms, 425 | 58.8 ms, 469 |
| 56 | | 107.8 ms, 452 |
| 28, prompts of 4,096 tokens | 125.9 ms, 163 | |

On the A40 at 28 at once, sampling's top_k (top_p 0.95 throughout):

| top_k (min_p) | Step | Tokens/s |
| --- | --- | --- |
| 0, the card's (0) | 64.9 ms | 425 |
| 5,000 (0) | 27.4 ms | 987 |
| 1,000 (0) | 24.3 ms | 1,107 |
| 100 (0) | 23.6 ms | 1,137 |
| 40 (0.05), llama-server's defaults | 23.6 ms | 1,139 |

- Allocated slots don't slow a step (1 at once: 12.0 ms with 8 or 28 slots); `--kv-unified`
  changes nothing (66.1 ms at 28); sampling on the CPU is 6 times slower (419.5 ms at 28).
- Each answer running adds about 1.9 ms a step on either GPU: sampling over K2's whole vocabulary
  of 250,624 tokens when top_k doesn't cut it first. With top_k at 1,000 or below, 28 answers at
  once run 2.6-2.7 times faster.
- The A100 is about 10% faster at full load for 3 times the price, and 56 slots give no more than
  28. The A40 stays.
- Step 2's fit of a fixed 58 ms a step was wrong: it came from full batches at top_k 0 (and
  Qwen's top_k 20, a different model). Time grows with the answers running and their context.

## Step 3: the unedited model as used (2026-10-16)

On an A40 (`45usef3qip28bb`, secure, CA-MTL-1, $0.59/hr, from 10:44 UTC; the search runs on after
it): the run rebuilt from the Hub at the user's settings (temperature 1.0, top_p 0.95, top_k 1,000,
min_p 0, budget 8192, reasoning effort high, f16 KV cache, Q4_K_M, 28 slots throughout) in 26
minutes, its unedited thinking ending by itself on the harmless test and held-out prompts at 28
slots; then `candor measure` on every held-out prompt and maths question, seed 0, 28 slots, in 41
minutes. Avicennasis's edit wasn't measured (the user: too weak a bar).

| Unedited, as used | Count |
| --- | --- |
| Harmful refused, judge | 287/288 |
| Harmful refused, keywords | 287/288 |
| Harmful answered unsafely | 9/288 |
| Harmful answer cut off at 256 | 27/288 |
| Harmless refused, judge | 2/300 |
| Harmless refused, keywords | 12/300 |
| Maths correct | 1,222/1,319 (92.6%) |
| Maths answer cut off | 0/1,319 |
| Thinking cut at 8192, harmful / harmless / maths | 0 / 1 / 3 |
| Thinking tokens, median (most): harmful | 205 (1,381) |
| Harmless | 588 (8,213) |
| Maths | 144 (8,213) |
| Time, harmful / harmless / maths | 207 s / 828 s / 1,399 s (534 / 443 / 435 tokens/s) |

- K2 refuses more than Qwen3.5-9B did unedited (287 against 279 of 288) and does better at maths
  (92.6% against 89.7%), so an edit has more refusals to remove and more maths to keep.
- A whole measurement takes 41 minutes at these settings, against about 6 h for Qwen's unedited
  model; 28 slots of the search's sizing take 8,460 tokens each where the pilot's took 15,616.

## Step 5: the search (2026-10-16 to 2026-10-18)

On the same A40 after step 3: `candor search runs/k2`, strength 0.8-5.0 (widest 0.5-5.0), 28 slots,
the user's sampling, budget 8192, at most 2 refused of 100, 300 trials a kind planned. From
12:52 UTC 2026-10-16; stopped between trials after trial 137 to switch on
`search.max_budget_hits = 8` (07:44-07:46 UTC 2026-10-17, history), then stopped by the user after
trial 313 at 06:08 UTC 2026-10-18: 41 h 16 min, about $24.

| Trials | Time | Mean a trial | Budget hits (a trial) | Over 8 | Stopped early |
| --- | --- | --- | --- | --- | --- |
| 1-137, no limit | 18.8 h | 495 s | 632 (4.6) | 20 | 13 |
| 88-137 | 8.1 h | 584 s | 261 (5.2) | 10 | |
| 138-313, limit 8 | 22.4 h | 458 s | 241 (1.4) | 7 | 107 |

Within both caps (judged in full, at most 2 refused, at most 8 budget hits): one layer 15 of 157
trials (24 over 8), per layer 54 of 156 (3 over 8). The best of each:

| Trial | Kind | Refused /100 | KL | Budget hits | Settings |
| --- | --- | --- | --- | --- | --- |
| 207 | one layer | 0 | 0.0240 | 3 | layer 18, think start, attn 3.87, mlp 2.52 |
| 205 | one layer | 2 | 0.0306 | 0 | layer 18, think start, attn 4.53, mlp 2.24 |
| 265 | one layer | 1 | 0.0316 | 1 | layer 18, think start, attn 2.58, mlp 2.46 |
| 203 | one layer | 2 | 0.0473 | 2 | |
| 285 | one layer | 1 | 0.0475 | 2 | |
| 114 | per layer | 2 | 0.0470 | 0 | think start, attn 4.57, mlp 3.08 |
| 202 | per layer | 2 | 0.0489 | 1 | |
| 112 | per layer | 1 | 0.0492 | 2 | |

- One layer's best fell from KL 0.143 (trial 115) to 0.052 (151) and 0.024 (207), then held for
  106 trials; per layer's held at trial 114 for 199. Every finalist reads the direction at the think
  start, the one-layer ones from layer 18.
- After the switch a trial took 22% less time (584 to 458 s) and hit the budget 1.4 times rather
  than 5.2. Only 7 trials failed on the budget, so the saving comes mostly from the sampler drawing
  fewer edits that think long (by K2's seconds model, 27 s a budget hit, about 100 s of the 126) and
  from more trials stopped early (107 of 176 against 13 of 137). The simulator shows the same fall
  in budget hits from steering.
- Trials 207, 205, 265 and 114 are the finalists (history, 2026-10-18).

## Step 5: when to stop the search, replayed (2026-10-17)

A stall stop (stop once both kinds' best KL within the cap gained under 10% in their last 50 own
trials) replayed on the phase 3 miss-rate simulator's saved runs (`phase-3-qwen3.5-9b.md`, step 3;
its fixed-count numbers reproduce exactly). Share of runs whose pick is within 10% of the true best,
with the mean trials run:

| Rule | Positions, D3grow, penalised / free | Positions, D3 (wide), penalised / free | Strength, grow |
| --- | --- | --- | --- |
| Fixed 300 | 36% / 34% | 16% / 18% | 29% |
| Fixed 400 | 48% / 48% | 25% / 25% | 40% |
| Fixed 600 | 65% / 66% | 38% / 37% | 52% |
| Stall 50, 10% | 32% / 35% (268 / 269) | 15% / 17% (305 / 262) | 28% (281) |
| Stall 50, 10%, at most 300 | 28% / 27% (240 / 238) | 13% / 16% (270 / 247) | 25% (253) |
| Stall 40, 10% | 29% / 32% (237 / 241) | 12% / 15% (260 / 224) | 25% (245) |
| Stall 75, 10% | 42% / 45% (353 / 352) | 24% / 22% (417 / 345) | 36% (367) |
| Stall 50, 5% | 34% / 38% (294 / 299) | 17% / 20% (330 / 292) | 31% (311) |

Runs: 240 / 240, 400 / 400 and 1,200. On Qwen3.5-9B's real search the 50-trial rule stops at 370 of
545 (the one-layer best 0.2% above its final KL, per layer 5%; both finalists found by trial 205),
the 40-trial rule at 285 (per layer 8%). On K2 at 125 trials, neither kind has stalled.

- A stall stop picks no better than a fixed count of the same length, and worse for every window
  tried: gains come in jumps after flat stretches, so a stall doesn't show the search is done.
- Stopping at 300 rather than 600 roughly halves the chance of a pick within 10% of the best.

## Step 5: the budget limit on the miss-rate simulator (2026-10-17)

`search.max_budget_hits = 8` measured on the phase 3 simulator, driving candor's own search at
8dfc0f5 (the scorer swapped for a landscape; the switch-over code unchanged). Budget hits drawn per
prompt from a model fitted on K2's trials (logit = -4.4 + 1.58 log(KL / best KL) - 3.03 x refusal
rate + a smooth random field of sd 1.3 over the settings): 11-14% of trials over 8 in the K2-like
groups (D3grow), 17% in the median scenario, against K2's 15%. Three arms paired by scenario and seed: off (the pick filtered at 8 after the search), on
from trial 1, and switched on after trial 137 as K2 was. Truth: the least KL at true refusals <= 2
and expected hits <= 8. 45,360 searches of 600 trials (200, 120 and 100 seeds per scenario, design
and arm in the base, stress and cliffs landscapes), 5.9 h on 10 Mac cores. Checks: with the limit at
100 "on" draws exactly as "off"; "switch" matches "off" for its first 137 trials.

Picks within 10% of the true best at 600 trials, the difference against off with its paired 95%
interval:

| Landscape | Every group, on | Every group, switch | Position groups, on | Position groups, switch |
| --- | --- | --- | --- | --- |
| Base | -0.2 (-1.4, +1.1) | -0.5 (-1.8, +0.7) | +1.5 (-0.3, +3.3) | -0.1 (-1.9, +1.6) |
| Stress (a bump beside the best) | -0.6 (-2.3, +1.0) | -0.8 (-2.4, +0.7) | +1.2 (-1.2, +3.6) | +0.2 (-2.1, +2.5) |
| Cliffs (a bump beside every basin near the best) | -1.1 (-3.0, +0.7) | -1.5 (-3.2, +0.2) | -0.1 (-2.8, +2.6) | -2.1 (-4.6, +0.5) |

Off hits 47-48% of the time over every group. K2 hours per 600 trials (from K2's seconds per trial: 261 + 2.78 a
prompt judged + 27.1 a budget hit - 3.85 a refusal, R² 0.83), off / on / switch:

| Group | Hours | Saved by on / switch |
| --- | --- | --- |
| Positions, penalised (D3grow) | 101 / 75 / 77 | 26% / 24% |
| Positions, free (D3grow) | 79 / 73 / 74 | 8% / 6% |
| Positions, penalised (D3, wide) | 171 / 83 / 122 | 51% / 29% |
| Positions, free (D3, wide) | 87 / 77 / 81 | 11% / 7% |
| Strength, all (grow) | 110 / 79 / 84 | 28% / 24% |

- Pooled, the limit costs no measurable share of hits at 300, 400 or 600 trials, in any landscape.
- Its effect differs between landscapes beyond chance (Cochran's Q p < 0.01 for on; sd 4-8 points):
  at 600, positions free (D3grow) lost 2.8-5.8 points on from trial 1 in all three landscapes and
  6.5 (-11.8 to -1.0) switched on at 137 under cliffs; positions penalised wide gained 4.4-6.7 on. At equal pod time the K2-like groups gained or
  held (switch +3.0 to +7.0 penalised, -1.4 to -2.7 free, not significant).
- On from trial 1 matched or beat the mid-search switch and saved more time.
- Limits: the budget model rests on one model's trials, its field length weakly identified; which
  landscape K2 resembles is unknown; only a limit of 8 and the search's pick were tested. Code,
  tables and runs: `.claude/search-design-backup/misscheck/` (`budget-report.txt`).
