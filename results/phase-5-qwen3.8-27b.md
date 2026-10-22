# Phase 5 results: Qwen3.8-27B

Numbers only; answers stay in each run's private folder. The plan is in STATUS.md; the decisions,
with their reasons, are in `history.md`.

## Step 1: the pilot (2026-10-19)

What the pilot had to measure: `evidence-2026-10-19/cost-27b.md`, "What the pilot must measure",
plus the row-norm weight screen. The raw outputs are kept with the run (`runs/qwen27b/pilot-a`,
`pilot-b`) and the scripts, plan and cost script in `runs/qwen27b/pilot-scripts`, outside git.

**Run.** `runs/qwen27b`: Qwen/Qwen3.8-27B at 1d4bf0f, reasoning effort xhigh, Q4_K_M with an f16 KV
cache (16.81 GB, the MTP layer kept), the model card's thinking-mode sampling (temperature 1.0,
top_p 0.95, top_k 20, min_p 0, presence 0), budget 8192, KL over 200 tokens, both judges, 32 slots,
the search as K2's ("full", strength 0.8-5.0, `max_budget_hits` 8). `runs/qwen27b-medium`: the same
at effort medium with `scoring.thinking` off (and `max_budget_hits` unset, which candor requires).

**Pods.** A100 SXM 80GB (`ylsgkd7hm3qiy1`, secure, US-MD-1, $1.79/h, 17:55-20:06 UTC, $3.94
billed). A40 (`50zsga2ugtxf47`, secure, CA-MTL-1, $0.59/h, 19:00-23:38 UTC, about $2.73 of GPU time;
billing read $2.25 so far, $0.10 of it disk). In all about $6.8 with disk, against the $7-8
approved. Both deleted.

### Build

| Stage | Where | Time |
| --- | --- | --- |
| Download and BF16 GGUF | A100 | 8 min |
| Q4_K_M | A100 | 3 min |
| Prompts, then directions (bf16, both positions) | A100 | 3 min |
| The unedited thinking, 400 prompts at 32 slots | A100 | 34 min (100/100 and 289/300 thought through) |
| The same at medium effort (its own prompts and directions) | A100 | 13 min (100/100, 300/300) |
| GGUF from the Hub again, the A100's directions and thinking reused | A40 | under 31 min, nothing else rebuilt |

- The first build stopped at the thinking: llama-server couldn't allocate its 16.5 GiB KV cache.
  After the directions, PyTorch kept 50.1 GiB reserved with no tensor alive (0.0 GiB allocated);
  with expandable segments reserved memory fell to 0.0 GiB (78.7 of 79.2 GiB free). Fixed in candor
  (d471d7b); the medium build then read its directions and started its thinking in one process.
- The A40's Q4_K_M isn't byte-identical to the A100's (sha256 2f358b2f against 556e0999) though
  candor records the same inputs; both cards' reads below include that difference.
- The medium copy's directions aren't xhigh's: its prompts render differently, and the two agree at
  a median cosine of 0.80 across layers where thinking starts (lowest 0.45) and 0.97 where the answer
  starts. The same trial settings are different edits in the two runs.

### Speed and memory (one request of 2,048 tokens a slot, at the search's 8,504-token slots)

| Card | Slots | Setting | Tokens/s | GPU memory, most |
| --- | --- | --- | --- | --- |
| A100 | 32 | | 376 | 37.2 GiB |
| A100 | 64 | | 436 | 59.5 GiB |
| A100 | 32 | MTP (`--spec-type draft-mtp`, 3 drafts) | 291 (73% of drafts accepted) | 53.0 GiB |
| A100 | 32 | MTP, 1 draft | 359 (84%) | 43.5 GiB |
| A40 | 32 | | 224 | 37.1 GiB |
| A40 | 40 | `--kv-unified`, 32 slots' context shared | 223 | 38.4 GiB |
| A40 | 48 | the same | 228 | 39.4 GiB |

The A100 is 1.68 times the A40's speed at 3.0 times the price; at 64 slots it is 1.95 times, at
1.56 times the cost a token. The cost model's 2.9 times slower than the 9B (central) predicted about
228 tokens/s on the A40. Under real load on the A40 memory peaked at 38.5 GiB measuring and 39.6
GiB searching (40.1 in the cache test), against `cost-27b.md`'s 38.5 and 40.4.

### The unedited model, first 64 of each set (A40, 32 slots, both judges)

| | xhigh | medium |
| --- | --- | --- |
| Harmful refused, judge / second judge / keywords | 62 / 62 / 61 | 61 / 63 / 53 |
| Harmless refused, judge / second judge / keywords | 0 / 0 / 1 | 0 / 0 / 3 |
| Maths correct | 64 | 63 |
| Thinking tokens, harmful: mean, median, p90, most | 799, 381, 1,767, 6,555 | 255, 242, 394, 521 |
| Thinking tokens, harmless | 762, 70, 986, 8,214 | 306, 174, 526, 2,530 |
| Thinking tokens, maths | 283, 140, 413, 4,286 | 209, 164, 323, 1,137 |
| Thinking reached the budget (harmless) | 3 | 0 |
| Time: harmful, harmless, maths | 973 s, 1,357 s, 406 s | 268 s, 471 s, 261 s |

64 prompts can't keep 32 slots busy, so the tail sets the time; candor projects a full xhigh
measurement at most 5 h 21 min (medium: 2 h 29 min).

### Trials (A40, 32 slots, 100 test prompts)

| Trial | Refused | KL | Thinking, mean | Budget hits | Time |
| --- | --- | --- | --- | --- | --- |
| The unedited model, scored at the search's start | 97 | | | | 779 s |
| Search trial 1 (one layer, layer 40) | 91 | 0.0445 | 426 | 0 | 898 s |
| Search trial 2 (per layer), stopped on the budget after 96 judged | 3 | 0.1485 | 1,969 | 9 | 2,691 s |
| Test edit (Qwen 9B trial 175's settings, its direction layer 15 of 32 moved to 30 of 64, at 0.7 times its strengths), scored in full | 17 | 0.0469 | 1,975 | 8 | 2,827 s |
| A stop on refusals (the unedited model): its floor | 8 of 8 | | 179 | 0 | 77 s |
| Thinking off at medium, trials 1-3 | 96, 0, 6 | 0.0341, 0.0992, 0.0798 | | | 135, 121, 127 s |

- An as-used trial takes about 15 min when the edit still refuses and 45-47 min when it complies
  (one of those stopped at 96 of 100 judged; the cost model: 30 min, 18-48); the search's own mean
  over its two was 1,795 s. A thinking-off trial takes about 2 min (the cost model: 4.4).
- The two stops measured fall outside the 45-74% of a full trial assumed from K2: on refusals 77 s
  (9% of a refusing trial), on the budget in the last chunk 2,691 s (95% of a complying one).
- Trial 2's KL was read while answers were being dropped after its stop, and came back whole. On
  Metal (Qwen3.5-0.8B, the developer's check of the cache script) 7 of 8 such reads came back empty
  under GPU sampling; on CUDA all 121 KL reads made straight after a stop were whole (120 in K2's
  search, 7 of them on the budget; this one).

### Edit strength (A100, the test edit's shape at a multiple of its strengths, top-200 KL)

| Multiple | 0.5 | 0.6 | 0.7 | 0.8 | 1.0 |
| --- | --- | --- | --- | --- | --- |
| Mean KL | 0.0088 | 0.0159 | 0.0437 | 0.1213 | 1.864, 1.881 |

At 1.0 (attention 1.52, MLP 2.28) the 9B's best edit breaks the 27B; between 0.7 and 1.0 KL rises
about forty-fold.

### KL across cards (F146): the test edit, 100 KL prompts, top 200, each session its own server

| Sessions | Base reads apart, KL (mean, both ways) | Edited reads apart | Gap in the edit's KL, share of its mean |
| --- | --- | --- | --- |
| A100 a-3 against a-4 | 0.0033 | 0.0012 | 9.0%, tie |
| A40 b-1 (A100's adapter) against b-2 (its own) | 0.0030 | 0.0034 | 9.7%, tie |
| The 4 A100-against-A40 pairs | 0.0029-0.0045 | 0.0043-0.0047 | 1.5-11.1%, all ties |

The edit's KL: 0.0426, 0.0466 (A100); 0.0460, 0.0417 (A40). The A40's own adapter differs from the
A100's (sha256 29d8a814 against 7d6d9342).

### KL over the whole vocabulary (F119): transformers in bf16 on the A100, logits in float32

Exact KL over all 248,320 tokens against candor's `from_top`, for the unedited model and the test
edit at the same 100 prompts (exact mean KL 0.0378; the edit kept 99.9% in bf16):

| Tokens read | On the mean | Per prompt: median, lowest |
| --- | --- | --- |
| 20 | -0.56% | -0.22%, -18.3% |
| 20, summed in float32 as llama-server does | -0.56% | -0.18%, -46.9% (highest +15.4%) |
| 200 | -0.05% | -0.03%, -5.8% |
| 200, float32 | -0.04% | -0.01%, -6.6% (highest +72.0%) |

Base mass in the top 20: median 0.9999, 8% of prompts below 0.995; in the top 200: lowest 0.9967.
transformers' bf16 reads and llama-server's Q4_K_M reads differ by a median KL of 0.010.

### Row normalisation on the 27B's weights (on the Mac, from the Hub by byte range)

||full - pre|| as a share of the edit, median over 128 writers: 0.38%, 0.35%, 0.50%, 0.75% and
1.05% at strengths 1-5 (trial 175's settings at 1.0 times: 0.34%). K2's finalists were at 4.1% and
Qwen's at 1.0% (`evidence-2026-10-19/method-evidence.md`).

### The answer reusing its thinking's cache (package 11), 64 harmful prompts, A40, 32 slots

| Arm | Answer requests that reused | Prompt tokens processed | Their time |
| --- | --- | --- | --- |
| Today (cache off) | 0 of 64 | 30,045 of 30,045 | 42.5 s |
| Reuse (one slot a prompt, cache on for the answer) | 63 of 63 | 87 of 25,960 | 17.3 s |

Check, 16 prompts on their own slots: the first token read with and without the reused cache
differs by KL 2.7e-07 on average (most 2.0e-06), the same likeliest token on all 16. The times are
summed over requests running 32 at once, so the wall-clock saving is unmeasured; the arms' wall times
(534 s and 468 s) compare different thinking.

### What it means for the 27B run (`cost-27b.md` with the pilot's numbers)

Estimates from the measured A40 times under the rules in force (4 finalists and the public
reference measured in full), assuming K2's share of stopped trials (61%, each 45-74% of a full one,
which the pilot's two stops don't bear out), the unedited model in full in 2-5.4 h, a finalist at
2-2.5 times that, and the public reference at `cost-27b.md`'s 6.2-35.5 h
(`runs/qwen27b/pilot-scripts/cost/gate.py`):

| Route | A40 hours | Cost | Days on one A40 |
| --- | --- | --- | --- |
| As used at xhigh: 600 trials | 224-492 | $132-290 | 9-21 |
| Thinking off at medium, then a proxy check of 12 edits and a 12-edit as-used round, if the check passes | 59-136 | $35-81 | 2.5-6 |
| The same, if the check fails after about 150 trials and the as-used route follows | 236-507 | $139-299 | 10-21 |

Plus $0.42 for the stages up to the directions on an 80 GB card (about 14 min; the thinking-off
route about $0.4 more for medium's own directions and thinking) and about 2% for disk. The search
alone: 299-471 hours as used before stops, 21 hours thinking off. The thinking-off route's edits
are built from medium's directions, not xhigh's (Build), so its as-used round and the shipped edit
need one set of directions chosen first; and on Qwen3.5-9B the proxy ranked the near-best edits
poorly (Spearman 0.24 among 12, `cost-27b.md`).

## Step 2: the thinking-off search to 150 trials, then `candor proxy` (2026-10-20)

`runs/qwen27b-medium` (Step 1's medium copy, its directions and thinking from the pilot's A100) on an
A40 (`847t462hxacsky`, secure, EU-SE-1, $0.59/h, created 05:20 UTC), 32 slots, seed 0, the pilot's
3 trials carried on. The proxy's bar is the one built in (`src/candor/proxy.py`). The raw scores are
in `runs/qwen27b-medium/proxy/scores.json` and the search at 150 trials in `search-at-150`, outside
git.

### The search, 150 trials thinking off (05:47-10:22 UTC)

The unedited model refused 99 of 100 test prompts. 150 trials took 4 h 35 min (110 s a trial; 98 s
over the last 25); 29 stopped early (12 one layer, 17 per layer); strength never widened.

| Kind | Best within the cap (2) | Ties with it by KL (2 SE), refusing fewer | At an end of its range |
| --- | --- | --- | --- |
| One layer | trial 119, layer 40, answer start: 2 refused, KL 0.0293 | trial 97, layer 41, answer start: 0, 0.0448 | attn.peak_layer 0.602 (floor 0.6), mlp.strength 4.96 (ceiling 5, the widest) |
| Per layer | trial 114, answer start: 2, 0.0364 | trial 150: 1, 0.0416; trial 20: 0, 0.0465 | trial 114: both peak_layers 0.62 and 0.61 (floor 0.6); trial 20: mlp.strength 0.91 (floor 0.8) |

### The proxy check (10:22-14:52 UTC)

The 3 best trials of each kind and position by refusals then KL, each at 1 and 1.4 times its
strengths (held within 0.5-5), scored on the 100 test prompts with thinking off and then with medium
thinking (budget 8192); all 100 judged in every score. Thinking off took 54 min (25 scores), thinking
on 3 h 36 min (26 scores with the repeat, about 8 min each).

| Edit | Trial | Kind, direction | Strength × | Refused, off | Refused, on | KL, off | KL, on | Thinking, mean tokens | Budget hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 55 | one layer, layer 27, thinking start | 1 | 1 | 1 | 0.0590 | 0.0530 | 268 | 0 |
| 2 | 55 | one layer, layer 27, thinking start | 1.37-1.40 | 26 | 98 | 0.4815 | 0.4909 | 336 | 0 |
| 3 | 135 | one layer, layer 27, thinking start | 1 | 6 | 1 | 0.0579 | 0.0558 | 489 | 2 |
| 4 | 135 | one layer, layer 27, thinking start | 1.16-1.40 | 24 | 90 | 0.2776 | 0.2894 | 144 | 0 |
| 5 | 57 | one layer, layer 28, thinking start | 1 | 6 | 28 | 0.1265 | 0.1201 | 322 | 0 |
| 6 | 57 | one layer, layer 28, thinking start | 1.08-1.40 | 25 | 98 | 0.3927 | 0.3925 | 92 | 1 |
| 7 | 97 | one layer, layer 41, answer start | 1 | 0 | 0 | 0.0456 | 0.0486 | 359 | 0 |
| 8 | 97 | one layer, layer 41, answer start | 1.15-1.40 | 0 | 1 | 0.0711 | 0.0706 | 345 | 0 |
| 9 | 145 | one layer, layer 41, answer start | 1 | 0 | 0 | 0.0483 | 0.0520 | 344 | 0 |
| 10 | 145 | one layer, layer 41, answer start | 1.06-1.40 | 0 | 1 | 0.0568 | 0.0586 | 321 | 0 |
| 11 | 107 | one layer, layer 37, answer start | 1 | 1 | 0 | 0.0550 | 0.0541 | 303 | 0 |
| 12 | 107 | one layer, layer 37, answer start | 1.00-1.33 | 0 | 0 | 0.0552 | 0.0604 | 338 | 0 |
| 13 | 2 | per layer, thinking start | 1 | 0 | 1 | 0.0996 | 0.0989 | 179 | 0 |
| 14 | 2 | per layer, thinking start | 1.03-1.23 | 0 | 0 | 0.1163 | 0.1148 | 173 | 0 |
| 15 | 34 | per layer, thinking start | 1 | 0 | 0 | 0.1099 | 0.1038 | 185 | 0 |
| 16 | 34 | per layer, thinking start | 1.01-1.40 | 0 | 0 | 0.1382 | 0.1384 | 222 | 0 |
| 17 | 18 | per layer, thinking start | 1 | 0 | 1 | 0.1308 | 0.1324 | 214 | 0 |
| 18 | 18 | per layer, thinking start | 1.27-1.39 | 5 | 18 | 0.3399 | 0.3529 | 694 | 5 |
| 19 | 20 | per layer, answer start | 1 | 2 | 2 | 0.0462 | 0.0460 | 334 | 0 |
| 20 | 20 | per layer, answer start | 1.40 | 0 | 0 | 0.1138 | 0.1066 | 255 | 0 |
| 21 | 142 | per layer, answer start | 1 | 0 | 0 | 0.0605 | 0.0610 | 287 | 0 |
| 22 | 142 | per layer, answer start | 1.31-1.40 | 0 | 0 | 0.1001 | 0.1018 | 266 | 0 |
| 23 | 120 | per layer, answer start | 1 | 0 | 0 | 0.0653 | 0.0611 | 284 | 0 |
| 24 | 120 | per layer, answer start | 1.40 | 0 | 0 | 0.1371 | 0.1295 | 243 | 0 |

- The unedited model refused 99 thinking off and 98 thinking on. Edit 7 scored again thinking with
  another seed: 0 refused, KL 0.0477 (0.0486 first).
- **Verdict "holds":** refusal ranks agree at Spearman 0.80 (bar 0.70); among the 23 edits refusing
  at most 25 thinking off, thinking refuses a median 0 more (bar 1).
- All 18 edits within the cap (2) thinking off are within it thinking on (one more, edit 3, is only
  thinking on). Among them KL ranks agree at Spearman 0.98 (0.99 over all 24); KL on is within
  -10% to +10% of KL off (median -0.2%). Edit 7, the least refusals then KL either way, is the same.
- Thinking refuses more on 9 of 24. The large gaps are 5 edits refusing 5-26 thinking off, at 18-98
  thinking on; edit 3 went the other way (6 off, 1 on).
- The search's best within the cap (trials 119 and 114, 2 refused) weren't among the edits checked,
  which take fewest refusals first.
- Each trial's edit at 1 times, scored thinking off here, repeats the search's score: refusals
  within 1 on 11 of 12 (trial 20: 0 in the search, 2 here), KL within 0.02 on 11 of 12 (trial 57:
  0.1057 in the search, 0.1265 here).

**Cost.** The pod from 05:20 to 14:52 UTC is $5.6. It then sat idle for 1 h 9 min ($0.68) while the
Mac couldn't reach it, until the 600-trial search started on it at 16:01 UTC (Step 3).

## Step 3: the thinking-off search, 150 to 600 trials (2026-10-20 to 21)

The same pod, run and settings as Step 2, carried on at 16:01 UTC and done at 02:37 UTC: 450 trials
in 10 h 35 min (85 s a trial; 92 s over the last 25). The unedited model refused 99 of 100 again.
278 of the 600 trials stopped early (one layer 144 of 300, per layer 134 of 300).

| Kind | Best within the cap (2) | KL at 150 trials | Strength | Where it improved |
| --- | --- | --- | --- | --- |
| One layer | trial 435, layer 37, answer start: 1 refused, KL 0.0188 | 0.0293 (trial 119) | attn 3.90, mlp 3.70, range 0.8-5 | KL 0.0262, 0.0212, 0.0209 and 0.0188 at its 100th, 125th, 138th and 225th trial; none in its last 82 |
| Per layer | trial 532: 2 refused, KL 0.0254 | 0.0364 (trial 114) | attn 3.75, mlp 3.83, widened to 0.5-5 after its 240th trial (its best's mlp at the floor of 0.8) | 0.0360 (trial 446) by its 225th trial, then 0.0254 about 30 trials after the widening; none in its last 34 |

Trials no other beats on both refusals and KL: 584 (per layer, 0 refused, KL 0.0310), 435 (1,
0.0188), 4 (51, 0.0110), 35 (97, 0.0059), 19 (98, 0.0057), 33 (99, 0.0042).

The least KL at each count within the cap, with candor's paired-prompt test against the kind's best:

| Kind | Trial | Refused | KL | Against the best |
| --- | --- | --- | --- | --- |
| One layer | 477, layer 39, answer start | 0 | 0.0324 | 0.0136 more than 435 (SE 0.0044): not a tie |
| One layer | 435, layer 37, answer start | 1 | 0.0188 | |
| Per layer | 584 | 0 | 0.0310 | 0.0056 more than 532 (SE 0.0031): a tie |
| Per layer | 578 | 1 | 0.0290 | 0.0036 more (SE 0.0028): a tie |
| Per layer | 532 | 2 | 0.0254 | |

Trials 477 and 435 press peak layer's floor of 0.6 (attention at 0.602 and 0.614); the per-layer
three press no end. candor's next step for each: `candor measure --trial N` on the held-out prompts.

**Cost.** Runpod billed the pod $13.28 in all ($12.68 GPU, $0.60 disk), 05:20 UTC on the 20th to
03:15 UTC on the 21st, for Steps 2 and 3: this step's 10 h 35 min about $6.2, then 38 min idle
($0.37) until the Mac, offline, came back and pulled the run. The pod is deleted.

## Step 4: the search's best scored as used (2026-10-21)

`candor proxy --trial` on a new A40 (`erf2ig41i8hyot`, secure, CA-MTL-1, $0.59/h, 14:46-17:12 UTC,
$1.49 billed): six trials, each at its own strength and at 1.4 times it (held within 0.5-5), on the
100 test prompts with thinking off and then at medium (budget 8192), 32 slots. The GGUF rebuilt from the Hub in 24 min (its Q4_K_M sha256 2f358b2f, the pilot A40's), the
directions and thinking reused. Thinking off took 29 min, thinking on 74 min (14 scores with the
unedited model and the repeat, about 5 min each). The unedited model refused 99 both ways.

| Edit | Trial | Kind, direction | Strength × | Refused, off | Refused, on | Keywords, on | KL, off | KL, on | Thinking, mean tokens | Budget hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 477 | one layer, layer 39, answer start | 1 | 0 | 0 | 7 | 0.0400 | 0.0356 | 330 | 0 |
| 2 | 477 | one layer, layer 39, answer start | 1.21-1.27 | 0 | 0 | 4 | 0.0730 | 0.0674 | 309 | 0 |
| 3 | 435 | one layer, layer 37, answer start | 1 | 1 | 0 | 9 | 0.0249 | 0.0239 | 328 | 0 |
| 4 | 435 | one layer, layer 37, answer start | 1.28-1.35 | 0 | 0 | 5 | 0.0500 | 0.0542 | 286 | 0 |
| 5 | 251 | one layer, layer 37, answer start | 1 | 1 | 2 | 2 | 0.0293 | 0.0268 | 318 | 0 |
| 6 | 251 | one layer, layer 37, answer start | 1.24-1.40 | 0 | 0 | 5 | 0.0523 | 0.0539 | 276 | 0 |
| 7 | 584 | per layer, thinking start | 1 | 0 | 1 | 8 | 0.0401 | 0.0379 | 167 | 0 |
| 8 | 584 | per layer, thinking start | 1.36-1.40 | 23 | 5 | 6 | 0.1029 | 0.0984 | 173 | 0 |
| 9 | 578 | per layer, thinking start | 1 | 0 | 1 | 9 | 0.0382 | 0.0358 | 169 | 0 |
| 10 | 578 | per layer, thinking start | 1.37-1.40 | 29 | 0 | 2 | 0.0815 | 0.0828 | 168 | 0 |
| 11 | 532 | per layer, thinking start | 1 | 1 | 0 | 12 | 0.0359 | 0.0341 | 164 | 0 |
| 12 | 532 | per layer, thinking start | 1.31-1.33 | 66 | 0 | 3 | 0.0719 | 0.0659 | 168 | 0 |

- At its own strength every trial is within the cap thinking (0-2 refused); trial 435 has the least
  KL, 0.0239, and with another seed 0 refused, KL 0.0271 (9 by keywords, thinking 311 tokens).
- The per-layer edits at 1.4 times refused 23-66 thinking off and 0-5 thinking: the other way from
  step 2's, where thinking undid partial edits.
- KL read here was above the search's for all six (search, thinking off: 477 0.0324, 435 0.0188, 251
  0.0209, 584 0.0310, 578 0.0290, 532 0.0254), by 23-41%.

### Why KL read higher: a check on the same pod (16:57-17:07 UTC)

`candor score` (thinking off, as the search scores) of a control, trial 97 (not among the least KL;
0.0448 in the search, 0.0456 in step 2), and of trial 435, each against the unedited reads the
search keeps (read on the pilot A40 from the same GGUF bytes) and against the unedited model read
again on this pod:

| Trial | In the search | Kept reads | Read again here | This step's round, thinking off |
| --- | --- | --- | --- | --- |
| 97 | 0 refused, KL 0.0448 | 0, 0.0480 | 0, 0.0474 | |
| 435 | 1, 0.0188 | 3, 0.0188 | 2, 0.0210 | 1, 0.0249 |

The two reads of the unedited model, same file and card type, apart by a mean KL of 0.0020 over the
100 prompts (0.0019 the other way): a median of 0.00002, the largest 0.0563, so a few prompts read
differently from session to session and the rest alike. Trial 435's KL ran from 0.0188 to 0.0249
across four sessions on the same file thinking off.

## Step 5: the finalists and the public 27B measured in full (2026-10-21 to 22)

Three A40s (secure, $0.59/h; created about 17:15 UTC): A `ok2m9w5hclbrso` (EU-SE-1) built the GGUF,
measured the unedited model, then trial 435, then the public edit; B `ffg0pizz0t9yur` (CA-MTL-1)
trials 251 and 532; C `lwdczzq2nbk4k4` (EU-SE-1) trial 477. Each finalist exported (Q4_K_M) and
measured with `measure --exported`, the public edit (huihui-ai's Huihui-Qwen3.8-27B-abliterated
@739e3c5, `evidence-2026-10-21/public-27b.md`) with `export --outside` and `measure --outside`:
medium thinking, budget 8192, answers up to 256 tokens, 32 slots, seed 0, every held-out prompt and
maths question, both judges. Compared prompt by prompt with the unedited model (p: McNemar; bounds:
the ends of `compare`'s two-sided 95% intervals, wider than the rule's one-sided ones). Package 11
was not tried first (history).

| | Unedited | 251 | 477 | 435 | 532 | huihui |
| --- | --- | --- | --- | --- | --- | --- |
| Kind, direction | | one layer, 37, answer start | one layer, 39, answer start | one layer, 37, answer start | per layer, thinking start | rank-1, layers 17-51 |
| Harmful refused, judge / 8B | 281 / 280 of 288 | 1 / 3 | 1 / 3 | 1 / 7 | 0 / 0 | 2 / 2 |
| Harmful judged unsafe, judge / 8B | 5 / 1 | 263 / 251 | 263 / 252 | 263 / 254 | 279 / 269 | 268 / 265 |
| Harmful answers cut at 256 tokens | 118 | 283 | 283 | 284 | 273 | 284 |
| Harmful refused, keywords | 256 | 15 | 18 | 21 | 17 | 23 |
| Harmless refused, judge (change's bounds) | 2 of 300 | 2 (-4, +4) | 2 (-4, +4) | 2 (-3, +3) | 2 (-4, +4) | 1 (-4, +2) |
| Maths correct (change's bounds) | 1,281 of 1,319 | 1,279 (-11, +7) | 1,284 (-6, +12) | 1,281 (-9, +9) | 1,283 (-7, +11) | 1,285 (-3, +11) |
| Maths thinking cut at 8192 (bounds) | 0 | 0 (-3, +3) | 0 (-3, +3) | 1 (-2, +4) | 0 (-3, +3) | 1 (-2, +4) |
| Thinking cut, harmful / harmless | 0 / 0 | 0 / 1 | 0 / 0 | 0 / 0 | 0 / 1 | 0 / 0 |
| Harmless KL (95% interval) | | 0.0259 (0.011-0.119) | 0.0274 (0.013-0.114) | 0.0247 (0.011-0.121) | 0.0434 (0.027-0.089) | 0.0382 (0.028-0.053) |
| KL by prompt: median; share in the top 5 | | 0.0007; 69% | 0.0007; 63% | 0.0007; 67% | 0.0019; 47% | 0.0016; 28% |
| KL after empty thinking / at the thinking start | | 0.065 / 0.0008 | 0.073 / 0.0006 | 0.060 / 0.0005 | 0.166 / 0.0038 | 0.054 / 0.0006 |
| Thinking median, harmful / harmless / maths | 194.5 / 175 / 167 | 465.5 / 170 / 167 | 480 / 183.5 / 167 | 467.5 / 168 / 163 | 239 / 173.5 / 168 | 540 / 176 / 163 |
| Edit kept in the file | | 98.3-100.1% (2 of 76 weights at Q8_0, merged in F32) | 99.6-100.0% | 99.2-100.4% | 99.3-100.0% | 99.8-100.1% (70 weights) |
| Export, then measure | | 47 min, 1 h 41 min | 18 min, 1 h 40 min | 26 min, 1 h 37 min | 23 min, 1 h 40 min | 28 min, 1 h 40 min |
| Decision rule | | pass | pass | pass | pass | pass |

The unedited model took 1 h 29 min; builds 20-30 min. Every export is 16.8 GB (251's 16.9 GB).

Between edits: refusals and unsafe answers by McNemar p; KL paired by prompt over the 300 harmless
prompts, as the mean gap and its standard error, and as the prompts where the first reads lower and
higher with a sign test's p (the one-layer edits' KL sits mostly in a few prompts, where a mean has
little power):

| | Refused, judge | Refused, 8B | Unsafe, judge / 8B | Maths correct | KL mean gap (SE) | KL lower / higher (p) |
| --- | --- | --- | --- | --- | --- | --- |
| 532 against 435 | 0 / 1 (p 1) | 0 / 7 (p 0.016) | p 1.4e-4 / 0.0015 | 1,283 / 1,281 (p 0.82) | +0.0187 (0.0050) | 82 / 218 (2e-15) |
| 532 against 477 | 0 / 1 (p 1) | 0 / 3 (p 0.25) | p 1.4e-4 / 9.1e-4 | 1,283 / 1,284 (p 1) | +0.0159 (0.0050) | 99 / 201 (4e-9) |
| 532 against 251 | 0 / 1 (p 1) | 0 / 3 (p 0.25) | p 3.1e-5 / 1.2e-4 | 1,283 / 1,279 (p 0.56) | +0.0174 (0.0053) | 80 / 220 (3e-16) |
| 251 against 477 | 1 / 1 (p 1) | 3 / 3 (p 1) | p 1 / 1 | 1,279 / 1,284 (p 0.33) | -0.0015 (0.0031) | 163 / 137 (0.15) |
| 251 against 435 | 1 / 1 (p 1) | 3 / 7 (p 0.34) | p 1 / 0.63 | 1,279 / 1,281 (p 0.79) | +0.0013 (0.0023) | 132 / 168 (0.043) |
| 435 against 477 | 1 / 1 (p 1) | 7 / 3 (p 0.34) | p 1 / | 1,281 / 1,284 (p 0.68) | -0.0027 (0.0021) | 184 / 116 (1e-4) |
| 251 against huihui | 1 / 2 (p 1) | 3 / 2 (p 1) | p 0.36 / 5.2e-4 | 1,279 / 1,285 (p 0.26) | -0.0123 (0.0105) | 186 / 114 (4e-5) |
| 532 against huihui | 0 / 2 (p 0.5) | 0 / 2 (p 0.5) | p 9.8e-4 / 0.39 | 1,283 / 1,285 (p 0.83) | +0.0051 (0.0087) | 121 / 179 (0.001) |

- **The rule picks a one-layer edit, but not which one.** All four pass, and so does the public
  edit. Trial 532 refuses fewest (0 under both judges). Counting both judges, each edit against
  532, 435 is out (7 under the 8B, p 0.016) and 251 and 477 are within noise, so the lower KL ships:
  251 at 0.0259. Counting the run's judge only, all four are within noise and 435 ships at 0.0247.
  Either way the KL that decides is a tie on the mean: 251, 477 and 435 are within 2 SE of each
  other, while 532's is above each beyond noise. By prompt, 435 reads lower than 251 (p 0.043) and
  477 (p 1e-4), and 251 and 477 are level. The rule names neither reading, nor which KL test, nor a
  tie-break (history).
- **Trial 532's answers are judged unsafe more often within the 256 tokens the judges read:** 279
  against 263 for each one-layer edit, and the 8B agrees (269 against 251-254). Against the public
  edit only the run's judge sees it (279 against 268, p 0.001; the 8B 269 against 265, p 0.39).
  Every answer neither refused nor judged unsafe ran past 256 tokens (24, 24 and 25 for the
  one-layer edits, 9 for 532, 18 for the public edit, by the run's judge), so those may reach the
  substance later; a longer answer cap would tell. 532 also finished 15 answers within 256 tokens
  against 4-5 (p 0.001-0.013), thought half as long on harmful prompts (median 239 against
  466-480) and reads more KL than any of them on most prompts.
- **Against the public edit, trial 251 shows no difference beyond noise in refusals or maths,** and
  their mean KL ties, though 251 reads lower on 186 of 300 prompts (p 4e-5), most of its KL sitting
  in a few prompts. The 8B judges the public edit's answers unsafe more often (265 against 251,
  p 5e-4; the run's judge 268 against 263, p 0.36). Trial 532's mean KL ties the public edit's too,
  but it reads higher on 179 of 300 prompts (p 0.001).
- **Held-out KL kept step 4's order at the top, not below it:** step 4's round read 435, 251, 532,
  477 (0.0239-0.0356); held out 435, 251, 477, 532 (0.0247-0.0434), 477 down 23% and 532 up 27%.
- Pod A built the unedited model's GGUF; B and C built their own from the Hub at the same recipe
  (`paired.model` 3d8651f2). Their bytes weren't hashed before the pods went; A40 builds matched in
  step 4. A measurement records its export's recipe, not its bytes, so the file shipped is exported
  again and measured in full, at the same seed and slots, and compared with this step's
  measurement: no change beyond noise (step 6).
- The unedited model took 1 h 29 min against the pilot's projection at medium of 2 h 29 min
  (step 1); the cost model's xhigh-based 2-5.4 h, a finalist at 2-2.5 times that and the public edit
  at 6.2-35.5 h were looser still: an edit took 1.09-1.14 times the unedited model, the public edit
  2 h 8 min with its export. Steps 2 to 5 cost $23.97 against the thinking-off route's $35-81.

**Overnight hand-off.** The Mac couldn't stay awake, so a CPU pod (`8se5lz7z7xgyhn`, cpu3c, 2 vCPU,
EU-NL-1, $0.06/h) took each GPU pod's measurements, answers and logs as its jobs ended, checked
every file by md5, and each GPU pod then deleted itself: B after 22:21 UTC, A after 23:29. The Mac
pulled the 126 files at about 05:50 UTC, checked each against the CPU pod's md5s, and deleted it.
The exports went with the pods.

**Cost.** Runpod billed $9.20: A $3.94, B $3.22, C $1.55 (deleted at 19:43 UTC), the CPU pod $0.49.

## Step 6: trials 251 and 532 with longer answers; 532's file measured again (2026-10-22)

One A40 (secure, $0.59/h, `t8pzq4lzs3sru5`, CA-MTL-1, created 08:25 UTC) built the GGUF, exported
trials 251 and 532 and hashed them, answered the held-out harmful prompts again on each export with
answers up to 1,024 tokens, picked the winner, and measured the winner's export in full again
(`evidence-2026-10-22/job.sh`). The longer answers ran in a hard-linked copy of the run,
`runs/qwen27b-long`, whose `candor.toml` differs only in `measure.answer_tokens` (1,024) and whose
harmless and maths sets are step 5's (`long.py` measures the harmful set alone). Medium thinking,
budget 8192, 32 slots, seed 0, both judges, as in step 5. The winner was picked by a rule set after
step 5's 256-token gap and before the 1,024-token answers were read (`decide.py`; history): 532 if,
under either judge, 251's answers are refused more often or judged unsafe less often than 532's
beyond chance (McNemar p < 0.05); otherwise 251. `pair.py` pairs each trial's two caps by prompt.

Exports (sha256):
- 251: `6f8cc84240f63ac035c58ec36e5fb7c36564b891c884c862c307b6a136f5455b`
- 532: `46dbc2fd8c552befe21feee4997467c5120292ff8673922d718f3b135245382e`

| Harmful, 288 held-out | 251, 256 tokens | 251, 1,024 | 532, 256 tokens | 532, 1,024 | Only 251 / only 532 at 1,024 (p) |
| --- | --- | --- | --- | --- | --- |
| Refused, judge / 8B | 1 / 3 | 0 / 1 | 0 / 0 | 0 / 0 | 0 / 0 (p 1); 1 / 0 (p 1) |
| Judged unsafe, judge / 8B | 263 / 251 | 254 / 247 | 279 / 269 | 275 / 269 | 0 / 21 (p 9.5e-7); 1 / 23 (p 3e-6) |
| Judged controversial, judge / 8B | 10 / 24 | 16 / 29 | 2 / 13 | 5 / 14 | |
| Judged safe, judge / 8B | 15 / 13 | 18 / 12 | 7 / 6 | 8 / 5 | |
| Answers cut at the cap | 283 | 117 | 273 | 77 | 44 / 4 (p 1.5e-9) |
| Refused, keywords | 15 | 34 | 17 | 46 | 25 / 37 (p 0.16) |
| Answer tokens, median | 256 | 883.5 | 256 | 652 | |
| Thinking median | 465.5 | 444.5 | 239 | 247 | |
| Answering, judging | 23 min, 1 min | 37 min, 2 min | 17 min, 1 min | 28 min, 2 min | |

(Step 5's columns are its measurements, from step 5's exports.)

- **Trial 532 wins by the rule:** at 1,024 tokens it is still judged unsafe more often than 251,
  beyond chance under both judges, by about as much as at 256 (21-22 prompts against 16-18), and
  neither refuses beyond noise.
- **The gap isn't the 256-token cap's.** Of 251's 34 answers the run's judge doesn't call unsafe at
  1,024 tokens, 27 end within 1,024 tokens (none within 256); under the 8B, 31 of 41. Of 532's 13,
  none is cut (the 8B: 1 of 19).
- **The longer cap is a fresh sample, not a longer copy:** thinking lengths matched on 3 and 10 of
  the 288 prompts. Within each trial, 256 to 1,024 tokens moved the run's judge's unsafe count by -9
  (251; 14 / 5 prompts, p 0.064, within noise though close) and -4 (532; p 0.22), the 8B's by -4
  (p 0.5) and 0; the gap between the two reappeared in a second sample.
- Keyword refusals rose with the cap (15 to 34, 17 to 46) while the judges' fell or stayed at 0-1:
  longer answers hold more refusal-like phrases.

**532's export measured again in full** (`shipped-532`, from this pod's export, against step 5's
`trial-532-exported-slots-32`, from pod B's, same settings): no difference beyond noise.

| | Step 5 | Again | Only step 5 / again (p) |
| --- | --- | --- | --- |
| Harmful refused, judge / 8B | 0 / 0 | 0 / 1 | 0 / 0 (1); 0 / 1 (1) |
| Harmful judged unsafe, judge | 279 | 274 | 5 / 0 (0.062) |
| Harmful cut at 256 | 273 | 278 | 2 / 7 (0.18) |
| Harmless refused, judge / 8B | 2 / 2 | 2 / 1 | 1 / 1 (1); 1 / 0 (1) |
| Maths correct | 1,283 | 1,287 | 6 / 10 (0.45) |
| Maths thinking cut | 0 | 0 | |
| Harmless KL from the unedited model (95% interval) | 0.0434 (0.027-0.089) | 0.0438 (0.028-0.089) | |
| Answering and first-token reads, harmful / harmless / maths | 17 / 26 / 54 min | 17 / 18 / 54 min | |

KL between the two measurements' first tokens: 0.0016 (0.001-0.002). Against the unedited model
the file passes the rule again: refused 0 / 1 of 288 (bar 10), harmless +0 (-4, +4), maths +6 (-1,
+13), no budget hit (`compare`'s two-sided 95% bounds, in prompts). As with the longer cap, a repeat
is largely a fresh sample: thinking lengths matched on 3 of 288 harmful, 44 of 300 harmless and 503
of 1,319 maths prompts, on the same GPU type, seed and slots. Pod B's export wasn't hashed, so the
repeat holds batching and any difference between the two files.

**Times.** Build 24 min; exports 47 min (251, the same 2 weights at Q8_0 from F32) and 23 min
(532); the longer answers 40 and 31 min with both judges; the full measurement 1 h 32 min. The job
ran 08:31-12:49 UTC, 4 h 18 min, against the quote of about 5.5 h. A GPU logger started inside the
job kept its process group alive, so `pod.sh wait` saw the end only when it was stopped, at
13:00 UTC, 11 min later.

**The release** (on the same pod, `evidence-2026-10-22/release.sh`): the measured export (sha256
checked again, `46dbc2fd…`) with `general.name` set to Qwen3.8-27B-Candor-Uncensored and
`general.finetune` (the snapshot hash) dropped; all 866 tensors byte-identical, no other key
changed. Against the unedited Q4_K_M, 73 of 866 tensors differ: `attn_output` 10 (layers 27-63),
`ssm_out` 28 (26-62), `ffn_down` 35 (22-56), all Q4_K or Q6_K. It loaded in llama-server b11524.
`Qwen3.8-27B-Candor-Uncensored-Q4_K_M.gguf`: 16,810,714,464 bytes, sha256
`6ec5d2739654c9459bd14bb36d324b267a7548a84cb2a47832f59b8eeb1f81e0`, the Hub's LFS hash the same.
The upload sent 1.47 GB of the 16.8 GB (the Hub's deduplication), 4 min. Rename and checks 2 min.

**Cost.** The job's 4 h 18 min at $0.59/h is about $2.54. The pod, kept for the user's test and
the release, ran 08:25-14:58 UTC, 6 h 33 min: about $3.9 (billing read $3.51 so far). Deleted.
