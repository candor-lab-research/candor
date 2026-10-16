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
