# Findings and ideas

Everything candor has learned that could shape the method, and every idea not yet acted on, each
flagged by what supports it:

- **Measured**: candor's own numbers, with the model and how many edits or prompts. One model is
  not enough to change the method; that needs several families (STATUS, "How we work").
- **Published**: a paper or others' public results, cited.
- **Hypothesis**: not tested yet; says what would test it.

The numbers behind each are in the file named. New findings are added here as they come; how an
entry is corrected, dropped or overturned is in CLAUDE.md ("Finished, dropped and replaced").

## Thinking models

- **Thinking off ranks edits as thinking on does, across the whole range.** [Measured: Qwen3.5-9B,
  24 edits refusing 1-100, 100 test prompts] Spearman 0.98, but only 0.24 among step 6's 12
  near-best edits. Thinking off costs about 90 s a trial against 10-20 min for an edit that complies
  scored as used (24 edits: 39 min against 3 h 47 min). So the search may score thinking off and
  score only its best as used. (`phase-3-qwen3.5-9b.md`, steps 5 and 6)
- **Thinking adds many refusals to edits that half-work, few to ones that work.** [Measured:
  Qwen3.5-9B, 24 edits in step 5 and 12 in step 6] Edits refusing 8-22 with thinking off refused
  3-28 more thinking (4 of the 5 at least 10 more); a median of 5 more where thinking off refuses at
  most 25 (step 6: 2). Of 15 edits within the cap of 2 thinking off, 9 stayed within it thinking;
  those that slipped at their own strength slipped by 3-5, none beyond chance on its own. (steps 5
  and 6)
- **A public K2-Horizon-7B edit refuses far more as used than its card says.** [Measured:
  K2-Horizon-7B, 64 held-out prompts; its card's count Published] Avicennasis's heretic edit refused
  48/64 as used (unedited 61/64) against 7/50 on its card. Prompts, judge and mode all differ, and
  K2 can't think off, so the gap can't be put down to thinking. (`phase-4-k2-horizon-7b.md`, step 2)
- **Qwen3.8-27B can't be scored thinking off as candor scores it, except at medium effort.**
  [Measured: its chat template] At xhigh, its default, and at low the template adds a system message
  that turning thinking off leaves out, so an answer after an empty thinking block isn't the model
  with thinking off, and candor refuses `scoring.thinking = "off"`. At medium it is a clean close.
  A cheap search of it needs medium, or scoring from the template's own thinking-off prompts (the
  phase-end discussion).
- **Two models of the same size think for very different lengths.** [Measured: K2-Horizon-7B at
  high, 64 prompts a set; Qwen3.5-9B, held-out sets] Unedited median thinking 224 / 560 / 140 tokens
  (harmful, harmless, maths) on K2 against 996 / 1,135 / 2,140 on Qwen, with no K2 budget hits (the
  full sets: 1 harmless, 3 maths; step 3); a K2 search trial that barely edits took about 210 s at
  16 slots. So a model's cost can't be quoted from another's: each needs its pilot. (step 2)
- **Extra strength pushes one-layer edits that already work past their best.** [Measured:
  Qwen3.5-9B, 3 pairs of each kind, 100 test prompts] At 1.4 times the strength, every one-layer
  edit in step 6 refused far more as used (1 to 20, 6 to 49, 0 to 49), though not with thinking off,
  and thought less. Per-layer edits held or improved (5 to 1, 1 to 1, 6 to 3), though mostly their
  feed-forward strength rose: attention, at 2.91-2.99, could rise only to its ceiling of 3.0. In
  step 5, one-layer edits that half-worked improved with it (41 to 18, 50 to 30), and trial 145
  (layer 18, KL 0.39) held (6 to 5). So strength beyond an edit's own best is not a safe correction
  for thinking. (steps 5 and 6)
- **Complying lengthens harmful thinking; how much, and whether maths thinking grows, depends on the
  edit.** [Measured: Qwen3.5-9B, 24 edits on the test prompts; Trial 161 and trials 205 and 175 on
  the held-out] In step 5, edits that still refuse thought 700-1,300 tokens on average, edits that
  comply 1,600-2,900. Harmful median: unedited 996, trial 175 1,641 (0 refused), 205 2,050.5, Trial
  161 2,973. Maths median: 175 1,874 and 205 2,426 against the unedited 2,140, budget hits level
  (187 and 174 against 184); Trial 161 3,620 and 297, and lost maths (1,183 to 1,162, p = 0.04).
  (steps 4, 5 and 7)
- **Two different effects lengthen thinking.** [Published, and Measured on 3 edits of Qwen3.5-9B] On
  harmful prompts it comes with complying, even from an edit that leaves maths untouched (DreamFast,
  12 edits of Qwen3.8-27B: base 1,207 tokens, edits 2,331-8,715). On maths it depends on the edit
  (heretic lengthened it twice, most other tools shortened it) and looks like damage to how the
  model stops thinking (the Halt Vector paper). The maths loss is budget hits, not lost skill
  (Gemma-4-12B: accuracy on answered problems within 3 points; Trial 161: 95.6% correct where its
  thinking ended by itself, against the unedited 94.8%). Of candor's two Qwen finalists, 205
  lengthened maths thinking 13% (median 2,426 against 2,140) with no more budget hits and kept
  maths; 175 shortened it. Nobody has measured how length depends on strength or layer.
  (`evidence-2026-10-15/thinking-length.md`; step 7) [Measured: K2-Horizon-7B trial 207] Not always:
  it lost 24-26 maths correct with only 5-10 more budget hits. (`phase-4-k2-horizon-7b.md`, step 5)
- **Time follows the longest thinking, not the median.** [Measured: Qwen3.5-9B] A 32-prompt pilot's
  medians put the unedited maths set at 4.3 h; it took 4.9 h. (step 4)
- **KL where the thinking starts does tell edits apart.** [Measured: Trial 161, 0.064 against the
  unedited model] Against the earlier assumption that the first thinking token barely changes.
  (step 4)
- **Keyword counts and the judge disagree, both ways.** [Measured: Qwen3.5-9B] Trial 161 rebuilt: 24
  refused by the judge, 12 by keywords. The search's best: 0-2 by the judge, 5-41 by keywords.
  [Published] Keywords with thinking off flatter thinking models badly: Qwen3.5-35B-A3B 3-5 refused
  by keywords against 33+ by a judge, Qwen3.8-27B 0 against 30.5% (DreamFast;
  `evidence-2026-10-14/ara.md`). candor decides on the judge.
- **A bigger judge finds more refusals on edited models.** [Measured: K2-Horizon-7B and Qwen3.5-9B,
  7 edited and 2 unedited held-out sets of 288, Qwen3Guard-Gen-4B at Q8_0 and 8B at bf16] On edited
  models the 4B calls 19 refusals and the 8B 20 against the 0.6B's 8 on K2, and 69 and 71 against 44
  on Qwen (0.6B against 8B: McNemar p 0.0075 and 0.00014); the 4B and 8B agree (kappa 0.71 and
  0.77). Most are on heretic's Trial 161 and K2's 114, and mostly run to the 256-token answer limit.
  Of the 156 that keywords call refusals and the 0.6B doesn't, the 4B calls 9. No decision changed,
  but K2's 114 is at 9 of 288 under both bigger judges against a bar of 10. All three judges are one
  family, so none is ground truth. (`evidence-2026-10-19/method-evidence.md` F104, `pod-checks.md`)
  Since 2026-10-19 a new run's finalists and reference are also judged by the 8B
  (`measure.second_judge`), and the 27B's harmful bar holds under both. [Hypothesis] The family
  itself may under-count; test: a judge of another family on the edited held-out answers.
- **The best public Qwen3.5-9B edit's 0/400 may not be as used.** [Published, with caveats]
  DreamFast judged trohrbaugh's heretic edit (`Qwen3.5-9B-heretic-v2`) at 0 of 400 HarmBench
  refusals, GSM8K 87.64 to 85.97; its report notes a 2,048-token cap and truncated thinking. Edited
  Qwen thinks longer than that on harmful prompts (Trial 161's median 2,973), so many answers may
  have been cut before any reply; candor reads the final answer after up to 8,192 and counts an
  unfinished one against the edit. Not comparable until measured the same way.
  (`evidence-2026-10-14/ara-evidence.md`) [Measured: the user, outside candor, 400 prompts, settings
  not recorded] Trial 175 also refused 0 of 400, so the two tie on that count; trohrbaugh's edit is
  still unmeasured as used.
- **Directions read at the thinking start mostly failed on Qwen and won on K2.** [Measured:
  Qwen3.5-9B; K2-Horizon-7B] On Qwen, 6 per-layer edits read there refused 71-100 of 100 either way
  and 6 one-layer ones 14-100, but trial 175 (one layer, layer 15) refused 0 of 100 test prompts
  with two seeds and 0 of 288 held-out as used, at KL 0.059 (steps 5-7). On K2 every finalist, and
  both kinds' bests, read there (`phase-4-k2-horizon-7b.md`, step 5). [Published] They worked on
  Qwen3-8B (arXiv 2507.03167).
- **20 tokens hold nearly all the mass where the answer starts, not all.** [Measured: K2-Horizon-7B
  and Qwen3.5-9B, about 300 held-out prompts each] The unedited models' top 20 hold a median 0.9996
  and 0.9998, but 13% and 10% of prompts fall below 0.995. Cutting there misses about 0.5% of KL; the
  floor for a base token the edit doesn't list may put heretic's Trial 161 2-7% low against 0.3-1.6%
  for candor's edits. llama-server b11524 sums the softmax in float32, so tokens below about 6e-8 of
  the top one are lost and the remainder is understated, by an unmeasured amount.
  (`evidence-2026-10-19/method-evidence.md`, F119) [Hypothesis] 200 tokens shrink the first two;
  test against transformers' full logits at the same ids on the 27B pilot, summed in float64 and in
  float32 in llama-server's order.
- **Until 2026-10-19 every sampled thinking-on answer replayed its thinking's random draws.**
  [Measured: llama-server b11524 on test models; the effect on counts a Hypothesis] llama-server
  starts its random numbers again at each request, and the answer was sent with the thinking's seed,
  so its token k was drawn with the number that drew thinking token k: the review found 300 of 300
  draws in the same slice of the odds, against 45 of 300 with another seed. The effect on counts is
  unmeasured, probably small, and shared by both sides of every pair. Each answer now has its own
  seed, and `compare` refuses to pair older measurements with newer ones; measuring one reference
  both ways would size it. (`evidence-2026-10-18/repo-review.md`, package 3)

## The search

- **Batching moves answers, though it moved no label.** [Measured: rude-llama] Two runs at the same
  settings at 8 slots gave first-token probabilities up to 0.15 nats apart on 19 of 100 prompts; at
  1 slot they were bit-identical (the slots change's checks, whose setup and reads weren't kept).
  Measured at 8 and at 16 slots, the unedited model kept every held-out label and KL 0.0000, but at
  least 6 of its 588 held-out answers and 31 of 1,319 maths answers changed, 5 of those flipping
  correct. (`history.md`, the slots change; `phase-2-rude-llama.md`, the noise floor) [Hypothesis]
  It may blur KL between edits whose KL differs little; re-reads at the same slots moved a median
  3.4% (the KL read again, below), and that entry holds the test.
- **KL depends on the machine: compare it only within one.** [Measured: K2-Horizon-7B, 3 search
  trials, 99 KL prompts] Read on the Mac, trials 45, 57 and 58 came out 2-7% lower than the search's
  reads on an A40. Per prompt the KL was spread across many prompts, with no floor or single prompt
  behind it, so the search's two reads of 0.4500 were chance. The Mac read its own Q4_K_M, whose
  quantised tensors all differ from a pod's (`phase-3-qwen3.5-9b.md`, the pod check), so the file
  and the backend are confounded. (`phase-4-k2-horizon-7b.md`, step 5)
- **More strength than heretic's 1.5 made every best edit.** [Measured: Qwen3.5-9B, and K2-Horizon]
  Qwen's best was 22 refused before the range widened to 3.0 at about trial 120, then 0-2. Every
  best since uses more than 1.5 (one layer: 2.2-2.5 on the feed-forward weights; per layer: 2.9 on
  attention). K2-Horizon: heretic's runs refused 54 at a cap of 1.5 and 7 at 5.0 (by keywords);
  candor's 22 trials inside 1.5 refused 88-100 (by its judge), and its finalists put attention at
  2.58-4.57. (`phase-3-qwen3.5-9b.md`; `evidence-2026-10-14/data-report.md`;
  `phase-4-k2-horizon-7b.md`, step 5; history, 2026-10-16)
- **The full 600 trials mattered.** [Measured: Qwen3.5-9B, one search] The best edits came at trials
  205-428 and none improved in the 117 trials after. 150 trials would have missed them all; by 300
  the one-layer best was within 1% of the final KL, the per-layer best 7-11% above it. (step 6)
- **One-layer edits give the lowest KL; 100 test prompts didn't separate the two finalists.**
  [Measured: Qwen3.5-9B] On the 100 test prompts as used, trial 205 (layer 12) refused 1 at KL 0.037
  and 175 (layer 15) 0 at 0.057; on the 288 and 300 held-out, 205 refused 16 and the two KLs were
  level (0.058 and 0.059), both below Trial 161's 0.073. Per-layer edits within the cap sat at KL
  0.085-0.100 on the test prompts. (steps 6 and 7) [Measured: K2-Horizon-7B, 4 finalists] Again:
  held-out KL 0.052-0.064 one layer, 0.082 per layer.
- **The search's best can fail the full measurement on maths, which the search doesn't score.**
  [Measured: K2-Horizon-7B, 4 finalists, every held-out prompt and maths question] Trial 207,
  first by the search's KL with 0 refused, lost 26 maths correct (p 0.019; 24 with adapters, so not
  the export); trial 265 hit the budget on 13 maths questions against 3 (p 0.021). The two that
  passed were second and fourth by the search's KL. So measuring several finalists in full is part
  of not missing, and its cost belongs in a 27B quote. (`phase-4-k2-horizon-7b.md`, step 5)
- **A cap on budget hits saves pod time and, pooled, costs no picks.** [Measured: a replay on
  K2-Horizon-7B's journal, then K2 from trial 138; the miss-rate simulator (`tools/misscheck/`),
  45,360 searches] With
  `search.max_budget_hits` at 8, K2's trials took 22% less time (584 to 458 s), mostly from the
  sampler drawing fewer edits that think long. On the simulator it cost no measurable share of picks
  pooled, but some landscapes lose up to 6.5 points, beyond chance. So: on for a thinking model
  whose pilot shows budget hits common. Owed: whether the fixed prompt order biases it.
  (`phase-4-k2-horizon-7b.md`, step 5)
- **"full" row normalisation against "pre" rests on one pair of searches.** [Measured: Qwen3.5-9B,
  69 and 70 trials from the same 40 Sobol starts, thinking off] "pre" was at 50 of 100 (KL 0.059),
  "full" at 32 (KL 0.069), both still falling. K2 used "full" untested; init still defaults to
  "pre". (`evidence-2026-10-14/data-report.md`) [Measured: the 6 K2 and Qwen finalists rebuilt both
  ways] The two differ by a median 1.0% of the edit on Qwen and 4.1% on K2 (0.025% and 0.13% of the
  weight), mostly rows rescaled off the direction; for down projections larger at low strength,
  K2's attention high at any strength; neither keeps row lengths.
  (`evidence-2026-10-19/method-evidence.md`) [Measured: K2 205 and Qwen 175, each scored twice both
  ways on one A40] On a fixed edit candor's tie test finds no difference on either, nor any in
  refusals; on Qwen pre may raise KL a little (p 0.07), the way the "full"-tuned strengths lean.
  Whether "pre" searches better still rests on the one Qwen pair.
  (`evidence-2026-10-19/pod-checks.md`)
- **Per-layer edits press the 3.0 ceiling on attention.** [Measured: Qwen3.5-9B, one search] Every
  per-layer best puts attention at 2.9-3.0, so a better per-layer edit may lie beyond it. (step 6)
- **Refusals fall gradually as strength rises; there is no cliff.** [Measured: 8 searches on 3
  models] [Published] The literature agrees. (`evidence-2026-10-14/data-report.md`,
  `lit-refusal.md`)
- **A trial's KL read again can move more than 3%.** [Measured: Qwen3.5-9B, 18 thinking-off re-reads
  by `candor proxy`, same prompts, slots and file, another session, 6 on another A40 pod] A median
  3.4%, at most 25.5% (at KL 0.005): 9 of 18 beyond the 3% within which `candor score` takes a
  repeat as the same score. rude-llama's one greedy repeat was within 0.3% (`phase-2-rude-llama.md`,
  the search fix). (`evidence-2026-10-19/method-evidence.md`, F117) [Measured: K2 205 and Qwen 175,
  two repeats each on one pod, each score in its own session, same seed and slots] Repeats moved KL
  6.6% and 8.5% on K2 and 2.7% and 0.9% on Qwen, all ties by candor's per-prompt test and all
  within the fitted 11%; but K2's refusals moved by 2 of 100 (3 to 1), so a repeat doesn't always
  stay within a refusal. With the unedited thinking sampled afresh and the code of 2026-10-19, KL
  moved from the search's record by K2 +10% and Qwen -12%. (`evidence-2026-10-19/pod-checks.md`)
- **Small KL gaps between trials are noise.**
  [Measured: K2-Horizon-7B and Qwen3.5-9B, 10 pairs of measured finalists, about 300 held-out
  prompts] At 100 test prompts the SE of a KL difference is an estimated 9-32% of the KL (scaled by
  KL level; by prompt count it would be larger); K2's 207 led 205 on the test prompts within that,
  and 205 led held out by 0.0096, of which 207's export alone may account for 0.0061. "Every
  within-cap trial within 2 SE of the best" would send 11-115 trials to measurement, too many as
  the finalist rule. Since 2026-10-19 each trial keeps its per-prompt KL, and the search shows
  which suggested finalists tie their kind's least within 2 SE, as guidance only.
  (`evidence-2026-10-19/method-evidence.md`, F117)
- **The KL reads beside the answer start decide nothing.** [Measured: K2-Horizon-7B 313 trials,
  Qwen3.5-9B 545] Within the caps `kl_empty` tracks `kl` (Spearman 0.62 and 0.80); `kl_thinking`
  less on K2 and Qwen's one-layer edits (0.36, 0.14), level on Qwen's per-layer ones (0.79 against
  0.76). No pick read either, and they cost about 0.5% (K2) and 3% (Qwen) of the search; since
  2026-10-19 only `measure` reads them, for every finalist. (`evidence-2026-10-19/method-evidence.md`,
  F5)
- **Skipping a trial's answers once 25 of its kind have lower KL saves little on K2, loses on
  Qwen.** [Measured: replays of K2-Horizon-7B's 313 and Qwen3.5-9B's 545 trials] Under 1.2% on K2,
  and a 19% loss on Qwen, where KL costs more than the answers. No skipped trial was a pick under
  today's rules, but 13 refused 0 within the caps and 10 of Qwen's 33 sit inside a 2-SE tie set.
  (`evidence-2026-10-19/method-evidence.md`, F3)
- **The simulator's cached wide-range bests were stale.** [Measured: `tools/misscheck/harness`, the
  base landscape recomputed] The backup's cache put the wide range's best one-layer edit at layer 12
  at KL 0.0839 and its best per-layer edit at 0.10277, above the normal range's 0.0683 and 0.10258
  in the same file, though the wide range holds the normal one; recomputed, the wide range gives
  0.0683 and 0.10258.
  The headline phase 2 numbers use the normal range, and no result in the docs cites the wide-range
  values. Clearing `tools/misscheck`'s `cache/` after changing `landscape.py` or `optimum.py` avoids
  it.

## Pods

- **A pod's GPU failed after 395 trials of one candor process, then CUDA wouldn't start.**
  [Hypothesis] A "full" search restarts llama-server every trial, so no server lived through them:
  either GPU memory candor's own process holds (the judge, and a GPU SVD each trial) grew, or the
  machine's GPU failed. Memory held at 24.4-24.6 GB over 12 edits scored as used, and at 16.1-16.6
  GB through two 4-hour measurements, which doesn't test 395 trials. It matters for 27B. [Measured:
  K2-Horizon-7B, A40] Through 313 "full" trials in 41 h it held at 41.0-41.9 GiB with no failure.
- **Real sets run 1.4-1.8 times slower than the speed test.** [Measured: Qwen3.5-9B, A40, 2
  edits] 444-467 tokens/s on harmful and maths sets (harmless 365-377) at 32 slots, against the
  test's 662 (32 answers of exactly 1,024 tokens): while the last long answers finish, most slots
  idle (on trial 205, busy 77-88% of a set's time), and answers run to 8,000 tokens. 8 slots: 283.
  A full measurement of a 9B edit took 3 h 36 min-3 h 53 min. (step 7)

- **Slots that fit depend on the model's KV cache, not the GPU alone.** [Measured: K2-Horizon-7B,
  A40] A dense model's f16 cache is 144 KiB a token (K2: 36 layers of 8 KV heads), so 32 slots of
  9,216 tokens need 41,472 MiB (40.5 GiB): `measure` fits 28 on an A40. The search sized every slot
  for its longest KL read plus a whole thinking answer (15.6k tokens), so it fit only 16. Since
  2026-10-16 slots are run settings (`[slots]`), each search slot is sized for its largest request,
  and `inspect` shows the cache a token takes. Qwen3.5-9B's hybrid layers (32 KiB a token, plus 50
  MiB of state a slot) never hit this. (step 2)

- **Sampling the whole vocabulary is what limits speed at full batch.** [Measured: K2-Horizon-7B,
  A40 and A100, made-up prompts] With top_k 0 (K2's card) each running answer adds about 1.9 ms a
  step on either GPU, capping llama-server near 425-470 tokens/s at 28; top_k 1,000 or less runs
  28 at once at 1,107-1,139 tokens/s. An A100 is about 10% faster for 3 times the price; 56 slots
  give no more than 28. (`phase-4-k2-horizon-7b.md`, speed test)

## Export

- **Merging into BF16 rounds tiny edits away before any quantising.** [Measured: K2-Horizon-7B
  trial 207, Qwen3.5-9B weights, a synthetic 4096x4096] Round to nearest kept 39-98% of small
  edits and Q8_0 of that no more (trial 207's two edge weights: 89.9% and 97.6%), while Q8_0 of an
  unrounded F32 merge kept 100-102%. So `export` merges a short weight again in F32 and stores it
  as Q8_0 from that, or as F32. (history, 2026-10-18)
- **Giving one tensor its own type in llama-quantize (b11524) moves others.** [Measured: toy
  models, the real binary] An override skips the counter Q4_K_M uses to pick Q4_K or Q6_K for later
  ffn_down layers: one at layer 1 changed 8 of 14 later ones. `export` now gives every tensor its
  plain type; rude-llama's shipped exports weren't affected (their overrides ran to the last layer).

## Ideas

- **Start strength wide (to 3.0 or more) rather than widening once the best presses 1.5.**
  [Measured: 2 models] The best-supported change; it is only a default setting, and K2's search
  started wide (above). Before making it the default: one more model family.
- **Measure the best public edit as a second reference.** [Hypothesis] `candor export --outside`
  then `candor measure --outside` on trohrbaugh's Qwen3.5-9B edit, as Trial 161 was, would show
  whether its 0/400 holds as used; a first look on the 100 test prompts costs minutes.
- **Thinking length by verdict on Trial 161's held-out harmful set,** from its labels and token
  counts: if prompts refused after long thinking are the longest, the model is deliberating.
  [Hypothesis; counts only, no new runs]
- **A separate edit just for thinking length** (Halt Vector, ThinkEdit, SEAL: 12-50% shorter) if
  length decides. [Published, on unedited models only]
- A stall stop [Dropped 2026-10-17: picked no better than a fixed count; phase 4, step 5]
- **Thinking length as a criterion between candidates.** [Hypothesis; 3 edits] Trial 161 lengthened
  maths thinking and lost maths through budget hits; candor's 205 lengthened it 13% with no more
  budget hits and kept maths, so length alone didn't decide. Test: on the next model,
  whether maths thinking length on the test prompts predicts held-out maths. [Measured: K2's 4
  finalists] Median maths thinking stayed at 135-149 tokens (unedited 144) in all four, yet two
  failed on maths; the search's own budget hits were 3 and 1 for those two, 0 for the two that
  passed. Test: the search's budget hits against held-out maths on the next model's finalists.
- **ARA, heretic's new default edit.** [Published: others' judged tests] It kept models intact best
  but never removed the most refusals; on Qwen3.8-27B 57.5% complied against 82.2% for a careful
  directional edit. Parked until the phase-end discussion (its condition, after Qwen, is met), or
  if keeping maths and thinking intact decides between edits. (`evidence-2026-10-14/ara.md`)
- **Search coordinates such as total edit mass and spread, and a GP sampler instead of TPE.**
  [Hypothesis, with literature] Only after an offline replay over several model families.
  (`evidence-2026-10-14/lit-search.md`)
- **Cheaper as-used scoring:** vLLM or SGLang with each edit swapped into the weights, prompt
  caching, a refusal probe, multi-fidelity optimisation, Uno's parallel decoding. [Hypothesis]
  (`history.md`, "Not in this plan"; `new-features.md`)
- **Low reasoning effort as K2's cheap score.** [Measured: K2-Horizon-7B, 32 harmless test
  prompts; the rest Hypothesis] Searching at low and scoring the best at high would take thinking
  off's place. Low wrote 3.7 times fewer tokens (median 250 against 918), but closed its
  `<ifm|think_faster>` thinking with high's `</ifm|think>` 17 times in 32, which candor counts as
  never closed, and K2's card advises against low for evaluation. Test: accept both closes at low,
  then `candor proxy`'s rank check, low against high, on a first search's best. (step 2)
- **More slots in one shared KV pool (`--kv-unified`).** [Hypothesis, from the speed test] At top_k
  1,000 a step grows only about 0.46 ms per answer (24.3 ms at 28; 12.0 at 1, at the card's top_k),
  so 48-56 answers at once might reach about 1,400 tokens/s, but f16 KV at the full budget caps K2
  at 28 slots on an A40. A pool the slots share, sized to what they use at once, could fit more,
  since most answers end far short of the budget; it changes numerics and how a full pool behaves.
  Test: time 28 against 48 slots with it, and check counts against per-slot caches on one edit.
- **A faster GPU for 27B.** [Measured on K2-Horizon-7B; Hypothesis for 27B] On K2 an A100 was about
  10% faster than an A40 at full load for 3 times the price, as sampling the whole vocabulary, not
  the GPU, was the limit (speed test). A 27B model reads 3 times the weights a step, so test again:
  a short timed pilot on an A100 and an H100 before quoting 27B.
- **3-5 refusal directions,** if a single direction fails as used. [Hypothesis]
- **Optuna 5,** once candor has a baseline. [Hypothesis] Its TPE changed multivariate bandwidth and
  multi-objective weights (`history.md`, "Not in this plan"). Test: the miss-rate simulator with
  each.
- **A 27B for the 18 GB Mac: Ternary Bonsai 2 27B with candor's adapters, unmerged.** [Hypothesis;
  PrismML's numbers Published] Its ternary GGUF is 5.95-7.21 GB, against about 16 GB for a 27B at
  Q4, and decodes 32.1 tok/s on an L4 (300 GB/s), so roughly half that on the M3 Pro (150 GB/s).
  Merging an edit and repacking to ternary would round most of it away, but candor already applies
  edits as rank-1 LoRA adapters, so the file could ship untouched with the adapters beside it. Needs
  PrismML's llama.cpp fork. Test: timed speed on the Mac, whether the fork's `--lora` works on its
  ternary types, and whether an unpacked FP16 exists for the directions.
