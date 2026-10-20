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
