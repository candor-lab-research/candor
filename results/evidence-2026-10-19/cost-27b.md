# Qwen3.8-27B: cost and time model (2026-10-19)

For the phase-end discussion. Numbers only: no prompt, answer or judge text was read. The model's own
files were read through `huggingface_hub` (config, tokenizer, file sizes; no weights). Prices are a
live Runpod read today. An independent review recomputed the model's numbers; its corrections are
folded in below and override the sections after it where they differ.

## The answer

- **The current process can't meet the gate of about a day and $20.** As used, with an as-used
  search of 600 trials and 4 finalists measured in full, the 27B run comes to about **$167-195**
  (281 A40-hours central, range $86-319; $195 if a stopped trial costs 74% of a full one, as K2
  measured before package 7, rather than the 45% assumed), or 6-22 days on one pod. The search is
  three quarters of it.
- **The cheapest route is a thinking-off search, and it is conditional.** Under the rules in force
  (4 finalists in full, the full 1,319 maths, `candor proxy` as built) it comes to about **$73**
  ($50-109). With rule changes not yet asked for (3 finalists, maths staged and sequential, no maths
  for the public reference) it is about **$53-57**. All of these hold only if a per-model proxy check
  shows thinking-off scores rank edits as thinking on does. On Qwen3.5-9B they ranked the near-best
  edits poorly (Spearman 0.24 among the 12 near-best; 0.93 with a clustered interval of 0.49-1.0
  for the answer-start edits), so the check may fail; then the as-used search applies. Expected
  cost: about $58, $86 or $114 at a 90%, 70% or 50% chance the check passes.
- **No case meets $20**, not even the cheap end on community cards ($25).
- **Unknowns the pilot measures:** the 27B's speed (scaled from the 9B: 2.9 times slower per token,
  2.4-3.2) and how long it thinks (0.7-1.5 times the 9B; the one published figure suggests about
  1.2). A pilot of about $7-8 (about 7 h on an A40 and 2.3 h on an 80 GB card) replaces both with
  measurements.
- **Context.** Phases 3 and 4 cost $30.92 and $32.90 billed (GPU and disk), 50.6 and 53.3 GPU-hours;
  K2 stopped at 313 trials, and 600 would have cost about $54. A 27B costs about 2.9 times as much
  per token.

Corrections from the review to the sections below:
- "Finalists staged" does not pick the same edit by construction: the signed rule ships the passer
  with the fewest harmful refusals, KL deciding only within noise, and measures every finalist in
  full. Staging, 3 finalists in place of 4, group-sequential maths and a public reference without
  maths are all changes to the rule signed off on 2026-10-19 and need sign-off. Group-sequential
  maths would save less than set here (a lossless edit uses 55-78% of the questions) and lowers the
  pass rate by up to 3.6 points.
- Thinking off and medium effort are one option, not two: Qwen3.8's template renders the same
  thinking-off prompt at every effort, without the system message xhigh adds.
- The pilot costs about $7.5, not $2; disk on 205-310 GB pods adds $2-3, and extra pod builds about
  $0.7.
- Wall time for the conditional plan is about 31-35 h plus the pilot, and the proxy check's verdict
  comes when about 75% of the thinking-off search is spent.
- The MTP batch figures quoted are for diffusion drafting, not MTP; package 11 is a hypothesis
  counted in the totals (-$1.9).

## 1. The model

`Qwen/Qwen3.8-27B` at 1d4bf0f (2026-08-14), `Qwen3_5ForConditionalGeneration`, llama.cpp arch `qwen35`
(candor's `gguf_architecture`):

| | Qwen3.8-27B | Qwen3.5-9B | K2-Horizon-7B |
| --- | --- | --- | --- |
| Layers (linear / full attention) | 64 (48 / 16, every 4th full) | 32 (24 / 8) | 36 dense |
| Hidden, heads, KV heads, head size | 5,120, 24, 4, 256 | | |
| Linear attention | conv 4; 16 key heads, 48 value heads, 128 each | | |
| Vocabulary, MTP layers | 248,320; 1 | 248,320 | 250,624 |
| Parameters (BF16, with vision) | 27,781,427,952; safetensors 55.56 GB | | |
| KV f16 per token (`kv_bytes_per_token`) | 64 KiB | 32 KiB | 144 KiB |
| Recurrent state per slot (`recurrent_bytes_per_slot`) | 149.6 MiB | 50.25 MiB | 0 |
| Q4_K_M | 16.55 GB without MTP, 16.81 GB with it (Hub GGUF listings); 2.86 times the 9B's 5.78 GB (`runs/qwen/gguf`) | 5.78 GB | 5.59 GB (export) |

**Slots.** candor doesn't size slots from GPU memory: `init` gives a thinking model 32, and a KV-fit
error names the setting to lower (`config.Slots`, review package 6). Each slot's context, from candor's
own formulas:
- `measure` (`measure.py`): longest prompt plus answer plus thinking room, plus 8. Maths is at most 223
  tokens, plus 42 for xhigh's system message (the template counted at xhigh, against medium), plus 512,
  plus 8,315 (8,192, a 107-byte message and 16). That makes about 9,100 tokens, or **718 MiB a slot**
  with the recurrent state.
- `search` (`search.py`): about 8,500 tokens, or **681 MiB a slot**.

With weights at 15.6 GiB, about 0.5 GiB of buffers for measuring and 3.5 GiB for searching (the judge
and torch, from K2's 41-42 GiB at 28 slots):

| GPU memory | Slots that fit, measure / search | 32 slots |
| --- | --- | --- |
| 48 GB (A40: 45.0 GiB usable, K2 pilot) | 41 / 38 | Fits: about 38.5 / 40.4 GiB |
| 32 GB (RTX 5090) | 21 / 17 | No |
| 80 GB (A100, H100) | 88 / 88 | 64 fits |
| 94-96 GB (H100 NVL, RTX PRO 6000) | 107-111 | 64 fits |

**A constraint.** The directions stage loads the model in bf16. `run.check_memory` asks for
2 x 27.78e9 x 1.1 = **61.1 GB of free GPU memory**, so every 48 GB card refuses it. Directions need an
80 GB or larger card (about 1 h), or the whole run on one. Disk: about 127 GB to build (HF 55.6, BF16
GGUF 54.7, Q4_K_M 16.5), plus about 78 GB free for each export (`check_disk`: one 16-bit copy and the
quantised file).

## 2. Throughput

Measured anchors (A40, llama-server b11524):

| | Model | Slots | Tokens/s | Source |
| --- | --- | --- | --- | --- |
| Full measurement, GPU sampling | Qwen3.5-9B | 32 | 451-452 (5.79-6.22M tokens in 3.56-3.83 h); harmless sets 365-377 | measurement files, trials 175 and 205 |
| As-used trial, KL and judge included | Qwen3.5-9B | 32 | 281-290 (568-660 s an edit at mean thinking 1,545-1,752) | `proxy/scores.json`, `trials.json` |
| Thinking-off trial | Qwen3.5-9B | 32 | 87-91 s a trial: 36 s answers, 41.5 s main KL, 4.7 s beside reads | journal; method-evidence F3, F5 |
| Speed test | Qwen3.5-9B | 32 / 64 | 662 / 760 (+15%) | phase 3, step 4 |
| Full measurement | K2-Horizon-7B | 28 | 434-534 | phase 4, step 3 |
| A100 against A40 | K2-Horizon-7B | 28 | +10% for 3 times the price | phase 4 speed test |

**The 27B against the 9B, per token, on an A40 at 32 slots: 2.9 times slower (range 2.4-3.2).** The
27B has 2.86 times the weights, 2.97 times the parameters (BF16 GGUFs 54.66 against 18.41 GB), 2.98
times the recurrent state a slot, 2 times the KV a token and 2 times the layers. The bottom of the range
holds if part of the 9B's 48 ms step (662 tok/s at 32) is fixed cost: sampling the same 248k vocabulary,
or kernel launches. The top holds if 27B steps are more compute-bound on the A40.

That gives:
- Measurement sets: about **156 tok/s** (141-188).
- As-used search trials: about **98 tok/s** (89-119).
- A single stream: about 30 tok/s (16.5 GB read at K2's measured 555 GB/s). One 8,192-token tail
  answer takes about 4 min or more by itself.

**Confidence.** Medium on an A40: same architecture, runner and sampler, but one model pair
extrapolated 3 times. Low on other GPUs: K2's A100 test is the only cross-card data, and it was limited
by sampling at top_k 0, which doesn't apply to Qwen's top_k 20. The pilot measures both.

**Thinking length, the second unknown: the 27B as used, against the 9B, 1.0 times (range 0.7-1.5).**
Nothing measures Qwen3.8-27B with candor's sampling. Published figures (DreamFast, xhigh, greedy,
budget 15,360): the base model averages 1,207 tokens on HarmBench and 12 edits average 2,331-8,715,
against the 9B base's 1,004 harmful mean (measured). Maths is about 77% of a measurement's tokens
(9B: 4.5-4.8M of 5.8-6.2M).

## 3. Prices (Runpod, read live 2026-10-19, `list-gpu-types`)

Stock is shown in brackets. "Speed" is the estimated 27B speed against an A40, a hypothesis. "Tokens
per $" is relative to an A40 on secure cloud at $0.59.

| GPU | GB | Community (stock) | Secure (stock) | Speed (hypothesis) | Tokens per $, community / secure | Directions stage |
| --- | --- | --- | --- | --- | --- | --- |
| A40 | 48 | $0.35 (none) | $0.59 (low, EU-SE-1) | 1.0x | 1.69 / 1.00 | no |
| RTX A6000 | 48 | $0.33 (none) | $0.59 (none) | 1.0-1.1x | 1.79-1.97 / 1.00-1.10 | no |
| L40 | 48 | $0.69 (low) | $0.82 (none) | 1.1-1.4x | 0.94-1.20 / 0.79-1.01 | no |
| L40S | 48 | $0.79 (none) | $1.09 (low) | 1.2-1.6x | 0.90-1.19 / 0.65-0.87 | no |
| RTX 6000 Ada | 48 | $0.74 (none) | $0.99 (low) | 1.2-1.6x | 0.96-1.28 / 0.72-0.95 | no |
| RTX PRO 5000 | 48 | $0.82 (low) | $0.96 (none) | 1.4-1.9x | 1.01-1.37 / 0.86-1.17 | no |
| RTX 5090 | 32 | $0.69 (none) | $1.19 (none) | 1.2-1.6x, 17-21 slots | 1.03-1.37 / 0.59-0.79 | no |
| A100 SXM 80GB | 80 | $1.39 (low) | $1.79 (low) | 1.7-2.4x | 0.72-1.02 / 0.56-0.79 | yes |
| A100 PCIe 80GB | 80 | $1.19 (low, host CUDA 12.4 only: the cu128 image can't run) | $1.79 (none) | | | yes |
| H100 PCIe | 80 | $1.99 (none) | $2.89 (none) | 1.9-2.6x | 0.56-0.77 / 0.39-0.53 | yes |
| H100 SXM | 80 | $2.69 (none) | $3.99 (medium) | 2.5-3.5x | 0.55-0.77 / 0.37-0.52 | yes |
| H100 NVL | 94 | $2.59 (none) | $3.19 (none) | 2.6-3.6x | 0.59-0.82 / 0.48-0.67 | yes |
| RTX PRO 6000 | 96 | $1.69 (none) | $2.49 (low) | 1.8-2.6x | 0.63-0.91 / 0.43-0.62 | yes |
| H200 SXM | 141 | $3.59 (none) | $5.29 (low) | 2.8-4.0x | 0.46-0.66 / 0.31-0.45 | yes |

No card clearly beats the A40 per dollar: faster cards buy time, not savings. Only community A40 or
A6000 would cut cost (by about 40%), and neither was in stock. The speed ranges are my estimates from
bandwidth and compute, not measurements. Disk billing wasn't read and isn't modelled.

## 4. The current process

This is as used at xhigh. candor refuses thinking off for Qwen3.8 at xhigh (findings, Thinking
models), so its search scores as K2's did. The plan: an A40 at 32 slots; `max_budget_hits` at 8; a
stopped trial costs no more GPU time (package 7); directions built on an A100 SXM community ($1.39);
4 finalists, as on K2.

Per trial:
- As used: **30 min** fully scored (18-48), **21 min** on average with stopped trials (10-41). 55% stop (K2
  stopped 61% with the limit on). A stop is assumed to cost 45% of a full trial (K2 measured 74%
  before package 7).
- Thinking off: 4.4 min fully scored (3.1-6.0), 2.8 min on average (2.1-3.7).

| Stage | A40 hours: low / central / high | Cost, central (low-high) |
| --- | --- | --- |
| Pilot: speed tests, 3 as-used and 3 thinking-off trials | 1.5 / 2.2 / 3.2 | $2.0 (1.6-2.6) |
| Build: GGUF and unedited thinking; directions on 80 GB | 1.5 / 2.2 / 3.3, plus 1 h on 80 GB | $2.7 (2.3-3.3) |
| Unedited reference, measured in full | 5.9 / 10.3 / 17.0 | $6.1 (3.5-10.0) |
| Best public release, measured in full (1.0-2.0 times an edit's tokens; Trial 161 was 1.3) | 6.2 / 13.9 / 35.5 | $8.2 (3.7-20.9) |
| Search as used, 600 trials | 101 / 207 / 406 | $122 (60-240) |
| 4 finalists in full, and exports | 26 / 44 / 72 | $26 (15-43) |
| **Total** | **144 / 281 / 539 (6.0 / 11.7 / 22.4 days)** | **$167 ($86-319)** |

## 5. The levers

Savings are for the central case. "Miss risk" is judged against the rule that the search must not
miss.

| Lever | Saving | Evidence | Miss risk |
| --- | --- | --- | --- |
| **Thinking-off search; only the best scored as used** (a 12-edit proxy check, then a 12-edit as-used round) | Search 207 h to 41.5 h (-80%), about $98 | Measured, Qwen3.5-9B only: Spearman 0.98 over 24 edits, but **0.24 among the 12 near-best**; 9 of 15 within-cap edits stayed within the cap as used | **Medium to high.** The proxy ranks near-best edits poorly, so the miss risk moves into the as-used round: score more distinct trials, not 1.4-times copies (F55). Needs the template fix (medium effort, or the template's own thinking-off prompts) and a per-model check. The simulator doesn't model proxy error: extend `tools/misscheck` first |
| Fewer trials, 600 to 300 | 103 h as used, 14 h thinking off | Measured on the simulator: picks within 10% fall from 65-66% to 34-36% (D3grow), from 38% to 16-18% (wide), from 52% to 29% (strength) | **High.** Hits roughly halve. Not defensible |
| Fewer test prompts, 100 to 50 | About 45% of as-used answer time; 6 h of a thinking-off search | Measured, F117: KL standard error is already 9-32% of KL at 100 prompts; the cap's resolution would drop to 1 in 50. Choosing prompts by item response theory (new-features #2) is Published only, and journals keep no per-prompt labels to replay it | **High** |
| Shorter search budget, 4,096 | 4-14% of complying edits' harmful thinking tokens, more in wall time | Measured on 5 held-out sets (Qwen 175, 205, Trial 161; K2 205, 265): 4,096 cuts 4.1-13.8% of tokens but changes 4-23% of prompts (11-66 of 288). 2,048 cuts 20-40% and changes 16-75% | **Medium to high**: it changes the outcome on about 1 in 10 prompts for little saving. Not worth it |
| Budget-hit cap at 8 (already in section 4's numbers) | 22% of a K2 trial (584 to 458 s) | Measured, K2; simulator: no pooled loss, up to 6.5 points in some landscapes | Low. Doesn't apply to a thinking-off search |
| Package 11: the answer reuses its thinking's KV | Estimated 5-9% of as-used generation: 100-170k prefill tokens a trial, about 5.5M a measurement; about 3 h in the nearest plan | Hypothesis (the review estimated 5-7% on K2) | None, but draws change, so it must land in the measurements change before the 27B reference |
| One pool across measurement sets (package 8's unbuilt half) | 3-5% of a measurement | Measured on 9B trial 205: slots busy 88% of the harmful set's time, 77% of the harmless set's | None |
| More slots: 64 on an 80-96 GB card; 40 on 48 GB | +15% on the 9B, 32 to 64 (speed test); about 0 on K2's A100 at 56 | Measured on 9B and K2; Hypothesis for the 27B | None, but slots are fixed per search and must match the reference. On 100-prompt trials, more slots worsen the tail |
| Cheaper or faster GPU | Cost: -40% only on community A40 or A6000 (none in stock). Time: 1.7-3.5 times faster on A100 or H100, at equal or higher cost | Prices Measured today; speeds Hypothesis | None, with one card type kept for the reference and finalists (F146) |
| **Finalists staged:** harmful and harmless for all, then maths in held-out KL order until one passes | 4 full (42.9 h) to 18.5 h; with sequential maths, 14.8 h | By construction the rule ships the lowest-KL passer, so this picks the same edit. On K2's data, 205 (lowest held-out KL) passed, so 1 maths run instead of 4 | **None** for a KL-ordered rule. Qwen's ship also weighed refusals, hence 1.3 runs expected |
| Group-sequential maths (new-features #1) | 15-50% of maths runs, set at 35% | Published (sequential testing; 2605.30315 found the best of many with 12-20% of pairs) | Low, if pre-registered with the non-inferiority margin (F116). Needs the 27B rule signed off |
| Public reference without maths | 13.9 h to 3.5 h, $6 | None needed | None (it is a reference, not a candidate). The maths comparison with it is lost |
| Drop the beside KL reads | 1.3 h (4.4%) of a thinking-off search; 0.5-1% as used | Measured, F5, K2 and Qwen: no pick reads them, and `measure` reads both | None |
| KL after an empty thinking block as the search's KL | 11 h (40%) of a thinking-off search | Measured, my replay today: within the caps, K2's 205 and 114 keep ranks 2 and 4, but 265 drops from 3 to 35; Qwen's 205 goes from 3 to 7, 175 from 29 to 33. Spearman 0.62 / 0.80 (F5) | **Medium.** A sampler steered by it draws other trials, which is untested. Not before a replay or simulator run |
| Search the two kinds on two pods at once | Time only: about half the search's wall time | Code: each kind is its own study, sampler seeded from its own trial count (`search.py`), so draws shouldn't change. Needs a split-by-kind option and a merge, checked by a replay | None, if the replay shows identical draws |
| MTP speculative decoding (`--spec-type draft-mtp`, in b11524) | Published single-stream 1.7-2.5 times on Qwen3.6-27B; likely 0.8-1.3 times at 32 slots on an A40 | Published single-stream; Uno's research shows gains shrink with concurrency (1.47x at 64 on H200, 0.72-0.77x at 16 on a 3090). Hypothesis at batch | None (draws are kept), but numerics change; edited weights may accept fewer drafts |
| Ngram or DFlash drafting (new-features #12, #13; b11524 lists `ngram-cache` and `draft-dflash`) | Published 1.5-2 times, 1.9 times at 32 (DFlash, vLLM) | Published on other models and engines | None, but numerics change; a DFlash draft for Qwen3.8 is unchecked |
| Medium effort, thinking on, as the cheap score | Perhaps 2-4 times fewer tokens than xhigh (K2's low wrote 3.7 times fewer) | Hypothesis | Same check as for thinking off, and a search about 3 times dearer than thinking off |

## 6. The cheapest defensible plan

Keep 600 trials of each search, 100 test prompts, the 8,192 budget for everything scored as used,
several finalists, and one card type for the reference and finalists. Spend less in these ways:
1. Directions on one 80 GB card, then copy `directions.pt` to the A40 run. Hub models are
   fingerprinted by commit (phase 3, step 2).
2. A thinking-off search, at medium effort or from the template's own thinking-off prompts, with the
   beside reads dropped. A proxy check of 12 distinct edits after 150 trials, kept only if the check
   holds on this model. A final as-used round of 12 edits.
3. The two kinds of search on two pods at once. The unedited reference and the public reference
   (harmful and harmless only) run on a third.
4. Finalists staged: harmful and harmless for 3, then maths in held-out KL order with a
   group-sequential non-inferiority test. Package 11 and one pool across sets, if the pilot shows
   their saving.

| Stage | A40 hours (central) | Cost, central (low-high) |
| --- | --- | --- |
| Pilot | 2.2 | $2.0 |
| Build, and directions on 80 GB | 2.2, plus 1 | $2.7 |
| Search, thinking off, 600 | 27.9 | $16.4 (12.3-21.6) |
| Proxy check: 12 edits both ways | 6.8 | $4.0 (2.4-6.4) |
| Final as-used round: 12 edits | 6.8 | $4.0 (2.4-6.4) |
| 3 finalists, harmful and harmless | 8.0 | $4.7 (2.7-7.8) |
| Maths, KL order, sequential (1.3 runs) | 6.8 | $4.0 (1.8-8.7) |
| Unedited reference (maths to the last look) | 8.7 | $5.2 (2.6-10.0) |
| Public reference, harmful and harmless | 3.5 | $2.1 (0.9-5.2) |
| 3 exports | 0.9 | $0.5 |
| Package 11 and one pool, -8% of as-used generation | -3.3 | -$1.9 |
| **Total** | **72 h; wall about 29 h (20-45) on 3 pods** | **$44 ($29-69); about $27 ($18-42) at $0.35** |

With KL read after an empty thinking block in the search (not yet defensible), it drops to about $37
on secure cloud, or $23 on community. This is the nearest plan to the gate, and it meets it only in
the low case on community cards.

**What it still needs before it's defensible:**
- **User sign-off at the phase-end discussion:** the thinking-off design for Qwen3.8; the proxy check
  with a clustered interval and the share of edits that stay within the cap (F55); the non-inferiority
  maths rule with group-sequential looks (F116); a public reference without maths.
- **Evidence:** the simulator extended with a noisy proxy, to show that a 12-edit as-used round after
  a thinking-off search keeps the hit rate (today it is untested). A replay showing the split-by-kind
  search draws the same trials.

**What the pilot must measure** (about 3 h on an A40 and 1.5 h on an 80 GB card, about $4):
1. Tokens/s for the 27B at 32 slots on the A40, and on one 80 GB card at 32 and 64 slots: a speed
   test, then the first real set. This pins the 2.4-3.2 factor and the GPU choice. If an L40S or RTX
   PRO 5000 is in stock, test one; the thinking-off search is about half prefill and favours compute.
2. Thinking lengths and budget hits of the unedited 27B at xhigh on 64 harmful, harmless and maths
   prompts (the 0.7-1.5 factor), and at medium for the cheap score.
3. Three as-used trials, at least one an edit that complies, so the long tail is timed. One trial
   stopped early, to time the stop under package 7 (set at 45% of a full trial).
4. Three thinking-off trials at medium: their time, their KL share, and the KL after an empty thinking
   block beside it.
5. Package 11 before and after on one set; MTP on and off; `--kv-unified` at 40-48 slots.
6. GPU memory at 32 slots against the estimates (38.5 GiB measuring, 40.4 GiB searching), and `-np`
   on the pod.
7. Directions built on the 80 GB card and accepted by the A40 run without rebuilding.
8. While it's on the pod: F146's same-card KL control and F119's `kl_tokens` 200 check. Both are
   cheap and are owed before the 27B reference.

The proxy check itself needs about 150 thinking-off trials first, so it is the search's first gate,
not part of the pilot. If it fails on this model, the as-used search's cost applies, and no plan comes
near the gate.

## Sources outside the repo

- Qwen3.8-27B config and file sizes: the Hub API (`Qwen/Qwen3.8-27B` at 1d4bf0f;
  `JonathanColetti/Qwen3.8-27B-Uncensored-GGUF`, `unsloth/Qwen3.8-27B-GGUF`,
  `huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF` listings, sizes only).
- Runpod prices and stock: `list-gpu-types`, read 2026-10-19 (secure and community availability, pods).
- MTP in llama.cpp, single-stream gains on Qwen3.6-27B:
  [mer.vin guide](https://mer.vin/2026/05/run-qwen-3-6-mtp-in-llama-cpp-faster-local-inference-with-built-in-speculative-decoding/),
  [aiweekly](https://aiweekly.co/alerts/llamacpp-mtp-boosts-qwen3-coding-speed-71),
  [HF Space comparison](https://victor-mtp-on-hf-endpoints.static.hf.space/index.html),
  [LlamaCppEx.MTP](https://llama-cpp-ex.hexdocs.pm/LlamaCppEx.MTP.md). The `--spec-type draft-mtp`
  flag was checked in candor's own b11524 `llama-server --help`.
- No published llama.cpp batched throughput for a 27B on an A40, A100 or H100 was found, so the speeds
  above scale candor's own 9B measurements.
