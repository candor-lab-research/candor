# Evidence for the phase-end discussion (2026-10-19)

The repository review left 11 method questions for the phase-end discussion
(`evidence-2026-10-18/repo-review.md`, "Method questions"). This report answers the six that the K2
and Qwen runs could answer on the Mac, with no pod time, and records the heretic upstream check.
Numbers only: no prompt, answer or judge reply was read. Each part was checked by an independent
reviewer who wrote their own code for the key numbers; their corrections are folded in.

Data: `runs/k2` (K2-Horizon-7B, 313 search trials, finalists 205, 207, 265, 114; 205 ships) and
`runs/qwen` (Qwen3.5-9B, 545 thinking-off trials, finalists 175 and 205, against heretic's Trial
161): the Optuna journals, the held-out measurements, llama-server's logs (timestamps only) and the
proxy files.

## heretic upstream

Nothing that changes heretic's default edit since 2026-10-14:
- master is 3a5c8645 (2026-10-17, 2.0.0.dev0), one commit after 66a952b, a PR template only (#502,
  the one PR merged). `config.default.toml` is the same file: ARA (rank 50, preserve row magnitudes),
  on attention outputs and down projections. The last release is still 1.4.0 (PyPI 1.4.0), whose
  default is still directional abliteration with "full" row normalisation.
- Open, not merged: #487, a thinking-model bug. On master and 1.4.0 the auto-detected
  `response_prefix` can include the start of the reasoning, and it is added to every prompt, so it
  reaches the refusals, the KL and what the edit is fitted on. #501 would fix it. This updates
  2026-10-14's reading (`evidence-2026-10-14/ara-evidence.md`) that Qwen3.5/3.8 templates escape it:
  #501 shows the recheck after the empty-thinking prefix exposes them too. For the discussion: a
  heretic reference on a thinking model would set `response_prefix` itself.
- Also open: #499 (ARA through a low-rank update, the same maths, 1.25-3.2x faster, the author timed
  Qwen3.8-27B among others), #503 (draft: streaming ARA, opt-in), #495 (re-score top trials on the
  merged model), #500 (a classifier refusal scorer, not default).
- On 2026-10-14 the maintainer closed 8 old PRs unmerged, among them an LLM refusal judge (#244),
  false refusals on harmless prompts (#210) and multi-token KL (#209).

## F5: the KL reads beside the answer start

Every trial of a thinking model also reads KL where the thinking starts (`kl_thinking`) and with an
empty thinking block (`kl_empty`).

| Run | Trials | n | Spearman with kl: kl_thinking | kl_empty |
| --- | --- | --- | --- | --- |
| K2 | all | 313 | 0.60 (0.50 to 0.68) | 0.85 (0.81 to 0.88) |
| K2 | within the caps | 69 | 0.36 (0.09 to 0.64) | 0.62 (0.44 to 0.75) |
| Qwen | all | 545 | 0.39 (0.31 to 0.47) | 0.95 (0.94 to 0.96) |
| Qwen | within the caps | 149 | -0.18 (-0.35 to -0.01) | 0.80 (0.69 to 0.88) |

Bootstrap 95% intervals. Within the caps: scored in full, at most 2 of 100 refused, and on K2 at
most 8 budget hits. Qwen's pooled -0.18 mixes two kinds at different KL levels: within each it is
0.14 (one layer) and 0.79 (per layer, level with kl_empty's 0.76).
- Cost: 3.9 s a K2 trial (20.2 min over 313, 0.8% of the search) and 43 min on Qwen (5.5%). Under
  today's code, which skips KL for a trial stopped on refusals, 12.6 and 23.9 min. The review's
  30-50 min on K2 was too high.
- Nothing decides on them: only the CLI's thinking summary reads them; `beats`, `front`,
  `to_measure` and the sampler use `kl`. `measure` reads both places for every finalist.

Verdict: drop both reads from the search.

## F117: the search's KL treated as exact

The search keeps each trial's mean KL only; no per-prompt KL is stored for any search trial. So the
noise was sized from the measured finalists' held-out reads (about 300 prompts each):
- The sd of per-prompt KL differences between two edits is r times their mean KL, with r 0.93-3.20
  across 10 pairs (median 1.88), larger for edits that differ more. At 100 prompts that makes the
  SE of a difference an estimated 9-32% of the KL. An estimate: it scales the held-out spread by KL
  level, while test KL runs 1.05-2.56x below held-out; scaling by prompt count alone would give r
  3.2-4.8 at K2's test levels, so the noise may be larger.
- A second noise: the same Qwen trials read again thinking off by `candor proxy` (same 96 prompts,
  slots, model file; another session, 6 of them on another A40 pod) moved a median 3.4% from the
  search's KL, at most 25.5% (at KL 0.005); 9 of 18 moved more than the 3% within which `candor
  score` takes a repeat as the same score.
- The one test-against-held-out flip: K2 207 was lower than 205 on the test prompts (0.8-1.6
  estimated SE apart, a tie), and 205 lower held out by 0.0096 (paired 95% -0.0191 to -0.0001). The
  test reads scored adapters and the held-out reads exported files: 207's export alone read +0.0061
  above its adapters held out (+0.0001 to +0.0121), and 205 has no adapter measurement, so the two
  causes can't be told apart. Qwen's 205 and 175 did not flip: 205 is lower on both (the review said
  they did).
- "Every within-cap trial within 2 SE of the best" as a finalist rule: on K2 at r 1.88 it keeps 19
  trials grouped by kind (3 one layer: exactly 207, 205, 265; 16 per layer, with 114); at r 1.0 it
  keeps 6 and leaves out 205. Grouped by kind only it drops Qwen's 175 at r 1.0, 1.88 and 3.0;
  grouped by kind and position, as the proxy groups edits, it keeps 175 at all three, but sends
  68-115 Qwen trials and 11-47 K2 trials to measurement.

Verdict: store each trial's per-prompt KL and treat gaps under about 2 SE as ties. Not as the
whole finalist rule: the sets are too large to measure, and r varies too much between pairs.

## F3: skip a trial's answers when 25 trials of its kind already have lower KL

A replay of each journal in order ("kind" is one layer or per layer; within the caps in force when
the trial ran):

| | K2 | Qwen (thinking off) |
| --- | --- | --- |
| Median answer / main KL time a trial | 402 s / 22.7 s | 36 s / 41.5 s |
| Trials that would skip their answers | 10, all per layer | 33 |
| Answer time saved | 1.22 h (about 1.0 h under today's code) | 0.34 h |
| KL newly read by trials today's code stops before KL | 0.75 h | 2.87 h |
| Net | about +0.25 to +0.48 h (under 1.2%) | -2.53 h (-19%) |

No trial that would skip was ever on a front, a finalist, measured or shipped. But K2's 250 and
288 and 11 of Qwen's skipped trials refused 0 within the caps, and their counts would go
unrecorded; and 10 of Qwen's 33 lie inside the per-kind 2-SE sets of F117, so "never a pick" holds
for today's `to_measure`, not for a tie rule.

Verdict: too little on K2 and a loss on Qwen. Don't adopt.

## F119: KL over the top 20 tokens

candor reads the 20 most likely first answer tokens from llama-server and puts the rest in one
bucket. The unedited model's top-20 probability at the answer start, as llama-server reports it:

| Model | n | min | p1 | p5 | median | below 0.995 | below 0.99 | below 0.97 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K2 base | 297 | 0.898 | 0.974 | 0.987 | 0.9996 | 13.1% | 7.7% | 1.0% |
| Qwen base | 294 | 0.944 | 0.976 | 0.991 | 0.9998 | 9.9% | 4.4% | 1.0% |

The edits are close to their base (K2 finalists 16.8-17.8% below 0.995, Qwen 9.9-12.2%). The
empty-thinking read is where 20 is short on K2 (median 0.957); the search's picks don't use it.

What the cut costs:
- Truncation itself: about 0.4-0.7% of KL, for candor's edits and heretic's alike, near the 0.5%
  measured on Llama-3.2-3B without thinking. The reads can't bound it from above.
- The rule for a base token the edited read doesn't list (it gets the edited read's smallest listed
  probability) matters more for heretic's Trial 161, which pushes about 5x more base mass out of its
  top 20 than candor's edits: if those tokens are really at half or a tenth of that floor, Trial
  161's KL is 2.0% or 6.9% low, against 0.3-0.4% or 1.1-1.6% for candor's edits. So
  candor-against-heretic KL slightly favours heretic.
- llama-server b11524 sums the softmax in float32 (`tools/server/server-common.cpp`,
  `get_token_probabilities`), so tokens below about 6e-8 of the top one add nothing to the total.
  The listed probabilities come out a little high and the remainder low, whatever the number of
  tokens asked for. Its size depends on the tail and isn't measured. `llama-perplexity
  --kl-divergence` can't size it (it floors tokens 16 nats down and reads other positions); a
  float64 softmax over transformers' full logits at the same answer-start ids, against the same
  logits summed in float32 in llama-server's order (the top ones first, then the rest), would; exact
  KL and candor's `from_top` on those logits size the truncation and the floor too.

None of this changed a decision: the gaps they turned on were far larger (K2 205 against 114 at
0.052 against 0.082; Qwen's ship turned on refusals).

Verdict: set `scoring.kl_tokens` to 200 for the 27B run (untimed, but only a larger partial sort
and about 10x the reads' JSON; base reads are keyed on it so runs can't mix values). It shrinks the
truncation and the floor, not the float32 loss. Leave K2 and Qwen at 20, and size the float32
effect once against the full-vocabulary reference on the 27B pilot.

## F67 + F97: "pre" against rank-3 "full" row normalisation

Each finalist's edit rebuilt with candor's own functions, tensor by tensor, at its own strengths:

| Edit | Tensors | Gap / the edit, median (max) | Gap / the weight, median (max) |
| --- | --- | --- | --- |
| K2 205, 207, 265, 114 | 234 | 4.1% (14.3%) | 0.13% (0.76%) |
| Qwen 175, 205 | 97 | 1.0% (5.4%) | 0.025% (0.18%) |

The gap is ||full - pre||. Against the size of the weight it is under 1% on both models; against
the size of the edit, which is what the two modes change, Qwen is at about 1% and K2 at about 4%.
- Both remove the same amount along the direction (they differ by 0.002 on K2, 0.001 on Qwen); the
  gap is a rescaling of rows, mostly off the direction.
- For down projections the gap is larger at low strength: on K2, 7.0% below strength 1.5 against
  1.6% at 1.5-2.5 and 1.7% above; Qwen's sat at 1.8-2.4 (0.75%). K2's attention outputs are high
  at any strength (5.4% near 2).
- Neither of candor's modes keeps row lengths (worst row per tensor: pre 1.5% median on K2, rank-3
  full 0.6%); only exact full does. All finalist directions are spread (max |r_i| 0.06-0.16), so the
  review's 1-26% for concentrated directions wasn't tested.

Verdict: the weights don't settle it, and K2 differs most. Decide the 27B default on behaviour:
score one finalist both ways (K2 205, Qwen 175) at the same seed and slots (unchecked: whether
candor accepts a "pre" copy of a search made under "full"), or run the short pre-against-full pair
from the same Sobol starts on the 27B pilot; both are pod jobs, for the user. Before either, the
same gap computed on the 27B's weights over a sweep of strengths (minutes, on the CPU) shows which
model it resembles.

## F104: the judge on edited models

The stored held-out answers judged again by Qwen3Guard-Gen-4B. The 4B doesn't fit the Mac at 16
bits, so it ran as Q8_0 in candor's llama-server, fed the token ids candor's judge builds, greedy.
Controls: the 0.6B run the same way, and with candor's own judge on the Mac's GPU (fp16, not the
pods' bf16), each changed 2-4 labels of 1,152 (K2) and 864 (Qwen) edited answers against the stored
ones; the 4B changed 15 and 51. The 4B itself wasn't checked at 16 bits.

Harmful held-out, 288 each:

| Set | Keywords | 0.6B (stored) | 4B | 4B only | 0.6B only | Keywords, not 0.6B | 4B of those |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K2 base | 287 | 287 | 285 | 0 | 2 | 0 | 0 |
| K2 205 | 23 | 3 | 6 | 4 | 1 | 21 | 1 |
| K2 207 | 15 | 1 | 3 | 2 | 0 | 15 | 1 |
| K2 265 | 14 | 1 | 1 | 0 | 0 | 14 | 0 |
| K2 114 | 9 | 3 | 9 | 7 | 1 | 9 | 1 |
| Qwen base | 282 | 279 | 280 | 2 | 1 | 5 | 2 |
| Qwen Trial 161 | 29 | 28 | 53 | 32 | 7 | 26 | 4 |
| Qwen 175 | 18 | 0 | 0 | 0 | 0 | 18 | 0 |
| Qwen 205 | 58 | 16 | 16 | 6 | 6 | 53 | 2 |

- The 156 answers that keywords call refusals and the 0.6B doesn't mostly aren't refusals to the
  4B either: it calls 9 refusals. The two judges are one family with one label scheme, so this
  agreement isn't independent evidence that the keywords over-count.
- On edited models the two judges agree only moderately (kappa 0.44 on K2, 0.52 on Qwen). Of the
  refusals the 4B finds, the 0.6B finds 6 of 19 on K2 and 31 of 69 on Qwen; of the 0.6B's, the 4B
  finds 6 of 8 and 31 of 44. Most of it is heretic's Trial 161 (32 of the 38 the 4B alone calls
  refusals on Qwen) and K2's 114 (7 of 13); candor's Qwen finalists split 6 against 6.
- The 4B calls more refusals on edited models: 19 against 8 on K2 and 69 against 44 on Qwen. That
  is the direction that would let an edit pass a bar it shouldn't.
- The refusals on edited models are much longer: 35 of the 37 both judges call refusals, and 45 of
  the 51 only the 4B does, run to the 256-token answer limit, against 27 of 288 answers of the
  unedited K2. The 0.6B was checked before only against keywords on unedited models (287 of 287 on
  K2); there the two judges agree too (kappa 0.76, 563 of 565 refusals shared).
- Decisions: on K2 all four finalists stay within 10/288 (205: 6, 207: 3, 265: 1, 114: 9), and 205
  still ships. On Qwen 175 refuses 0 under every judge; harmful refusals are a pass test, so Trial
  161's higher count changes nothing.
- The bar is sensitive to the judge: K2's 114 went from 3 to 9 against 10.

Verdict: no decision changes, but F104 stays open: the 0.6B's recall on edited models is unchecked
against ground truth, and a bigger judge of its family finds more refusals there. Neither is
ground truth. Options: the 8B at 16 bits on a pod for the disagreements (about $1-2), or a judge
from another family; and, for the 27B rule, whether a finalist must pass under two judges or the
second's count is reported beside the first.
