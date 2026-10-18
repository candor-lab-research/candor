# Findings and ideas

Everything candor has learned that could shape the method, and every idea not yet acted on, each
flagged by what supports it:

- **Measured**: candor's own numbers, with the model and how many edits or prompts. One model is
  not enough to change the method; that needs several families (STATUS, "How we work").
- **Published**: a paper or others' public results, cited.
- **Hypothesis**: not tested yet; says what would test it.

The numbers behind each are in the file named. New findings are added here as they come, and an
entry is corrected rather than deleted when later evidence changes it.

## Thinking models

- **Thinking off ranks edits as thinking on does.** [Measured: Qwen3.5-9B, 24 edits, 100 test
  prompts] Spearman 0.98. Thinking off costs about 90 s a trial against 10-20 min for an edit that
  complies scored as used (24 edits: 39 min against 3 h 47 min). So the search may score thinking
  off and score only its best as used. (`phase-3-qwen3.5-9b.md`, step 5)
- **Thinking adds refusals to edits that half-work, not to ones that fully work.** [Measured: 24
  edits, step 5; a median of 2 more on step 6's 12] Edits refusing 8-22 with thinking off refused 3-28 more thinking (4 of the 5 at least
  10 more); a median of 5 more where thinking off refuses at most 25. Of 5 edits within the cap of 2 thinking off, 4 stayed
  within it thinking (the one that slipped, a one-layer edit, went from 1 to 6). (step 5)
- **Thinking brings refusals back on K2-Horizon-7B too.** [Measured: K2-Horizon-7B, 64 held-out
  prompts; its card's count Published] Avicennasis's heretic edit refused 48/64 as used (unedited
  61/64) against 7/50 on its card, as Trial 161 on Qwen3.5-9B went from 11/100 thinking off to
  28/288 as used: a second family. (`phase-4-k2-horizon-7b.md`, step 2)
- **Two models of the same size think for very different lengths.** [Measured: K2-Horizon-7B at
  high, 64 prompts a set; Qwen3.5-9B, held-out sets] Unedited median thinking 225 / 561 / 140 tokens
  (harmful, harmless, maths) on K2 against 996 / 1,137 / 2,140 on Qwen, with no K2 budget hits; a
  K2 search trial that barely edits took about 210 s at 16 slots. So a model's cost can't be quoted
  from another's: each needs its pilot. (step 2)
- **Extra strength pushes one-layer edits that already work past their best.** [Measured:
  Qwen3.5-9B, 3 pairs of each kind, 100 test prompts] At 1.4 times the strength, every one-layer
  edit in step 6 refused far more as used (1 to 20, 6 to 49, 0 to 49), though not with thinking
  off, and thought less. Per-layer edits held or improved (5 to 1, 1 to 1, 6 to 3). In step 5,
  one-layer edits that half-worked improved with it (41 to 18, 50 to 30), and trial 145 (layer 18,
  KL 0.39) held (6 to 5). So strength beyond an edit's own best is not a safe correction for
  thinking. (steps 5 and 6)
- **Complying lengthens harmful thinking; how much, and whether maths thinking grows, depends on the
  edit.** [Measured: Qwen3.5-9B, 24 edits on the test prompts; Trial 161 and trials 205 and 175 on
  the held-out] In step 5, edits that still refuse thought 700-1,300 tokens on average, edits that
  comply 1,600-2,900. Harmful median: unedited 996, trial 175 1,648 (0 refused), 205 2,052, Trial
  161 2,980. Maths median: 175 1,874 and 205 2,426 against the unedited 2,140, budget hits level
  (187 and 174 against 184); Trial 161 3,620 and 297, and lost maths (1,183 to 1,162, p = 0.04).
  (steps 4, 5 and 7)
- **Two different effects lengthen thinking.** [Published, and Measured on 3 edits of Qwen3.5-9B]
  On harmful prompts it comes with complying, even from an edit that leaves maths untouched
  (DreamFast, 12 edits of Qwen3.8-27B: base 1,207 tokens, edits 2,331-8,715). On maths it depends on
  the edit (heretic lengthened it twice, most other tools shortened it) and looks like damage to how
  the model stops thinking (the Halt Vector paper). The maths loss is budget hits, not lost skill
  (Gemma-4-12B: accuracy on answered problems within 3 points; Trial 161: 95.6% correct where its
  thinking ended by itself, against the unedited 94.8%). candor's two finalists lengthen harmful
  thinking but not maths. Nobody has measured how length depends on strength or layer.
  (`evidence-2026-10-15/thinking-length.md`; step 7) [Measured: K2-Horizon-7B trial 207] Not
  always: it lost 24-26 maths correct with only 5-10 more budget hits. (`phase-4-k2-horizon-7b.md`,
  step 5)
- **Time follows the longest thinking, not the median.** [Measured: Qwen3.5-9B] A 32-prompt pilot's
  medians put the unedited maths set at 4.3 h; it took 4.9 h. (step 4)
- **KL where the thinking starts does tell edits apart.** [Measured: Trial 161, 0.064 against the
  unedited model] Against the earlier assumption that the first thinking token barely changes.
  (step 4)
- **Keyword counts and the judge disagree, both ways.** [Measured: Qwen3.5-9B] Trial 161 rebuilt:
  24 refused by the judge, 12 by keywords. The search's best: 0-2 by the judge, 8-41 by keywords.
  [Published] Keywords with thinking off flatter thinking models badly: Qwen3.5-35B-A3B 3-5 refused
  by keywords against 33+ by a judge, Qwen3.8-27B 0 against 30.5% (DreamFast;
  `evidence-2026-10-14/ara.md`). candor decides on the judge.
- **The best public Qwen3.5-9B edit's 0/400 may not be as used.** [Published, with caveats]
  DreamFast judged trohrbaugh's heretic edit (`Qwen3.5-9B-heretic-v2`) at 0 of 400 HarmBench
  refusals, GSM8K 87.64 to 85.97; its report notes a 2,048-token cap and truncated thinking. Edited
  Qwen thinks longer than that on harmful prompts (Trial 161's median 2,980), so many answers may have
  been cut before any reply; candor reads the final answer after up to 8,192 and counts an
  unfinished one against the edit. Not comparable until measured the same way.
  (`evidence-2026-10-14/ara-evidence.md`)
  [Measured: the user, outside candor, 400 prompts, settings not recorded] Trial 175 also refused
  0 of 400, so the two tie on that count; trohrbaugh's edit is still unmeasured as used.
- **Directions read at the thinking start mostly fail, but one gave the best edit.** [Measured:
  Qwen3.5-9B] 71-100 refused of 100 either way on 12 edits; trial 175 (one layer, layer 15) refused 0
  of 100 test prompts with two seeds and 0 of 288 held-out as used, at KL 0.059. [Published] They
  worked on Qwen3-8B (arXiv 2507.03167). (steps 6 and 7)

## The search

- **Batched answers aren't repeatable run to run.** [Measured: rude-llama, 8 slots, 100 prompts, the
  slots change's checks] Two runs at the same settings gave different answers and first-token
  probabilities, by up to 0.15 nats on 19 of 100 prompts; at 1 slot they are bit-identical. So
  noise of this size is in every measurement and search score already. [Hypothesis] It may blur
  KL between edits whose KL differs little. Test: measure one edit twice at its slots and compare
  per-prompt KL and counts. (`history.md`, the slots change)
- **KL depends on the GPU backend: compare it only within one.** [Measured: K2-Horizon-7B, 3 search
  trials, 99 KL prompts] Read on the Mac (Metal), trials 45, 57 and 58 came out 5.0%, 2.0% and 6.4%
  lower than the search's reads on an A40 (CUDA): 0.427, 0.441 and 0.089 against 0.450, 0.450 and
  0.096. Per prompt the KL was spread across many prompts, with no floor or single prompt behind it,
  so the search's two reads of 0.4500 were chance.
- **More strength than heretic's 1.5 made every best edit.** [Measured: Qwen3.5-9B, and K2-Horizon]
  Qwen's best was 22 refused before the range widened to 3.0 at about trial 120, then 0-2. Every
  best since uses more than 1.5 (one layer: 2.2-2.5 on the feed-forward weights; per layer: 2.9 on
  attention). K2-Horizon: 54 refused at a cap of 1.5, 7 at 5.0. (`phase-3-qwen3.5-9b.md`;
  `evidence-2026-10-14/data-report.md`)
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
- **Per-layer edits press the 3.0 ceiling on attention.** [Measured: Qwen3.5-9B, one search] Every
  per-layer best puts attention at 2.9-3.0, so a better per-layer edit may lie beyond it. (step 6)
- **Refusals fall gradually as strength rises; there is no cliff.** [Measured: 8 searches on 3
  models] [Published] The literature agrees. (`evidence-2026-10-14/data-report.md`,
  `lit-refusal.md`)

## Pods

- **A llama-server process failed its GPU allocations after 395 trials, then CUDA wouldn't start.**
  [Hypothesis] Either the server's memory grew trial by trial, or the machine's GPU failed. Memory
  held at 24.4-24.6 GB over 12 edits scored as used, and at 16.1-16.6 GB through two 4-hour
  measurements, which doesn't test a 395-trial server; it is logged every minute on pods, so the
  next long search will tell. It matters for 27B. [Measured: K2-Horizon-7B, A40] Through 313 trials
  in 41 h it held at 41.0-41.9 GiB with no failure, though the server restarts between some
  trials.
- **Real sets run 1.4-1.8 times slower than the speed test.** [Measured: Qwen3.5-9B, A40, 2
  edits] 444-467 tokens/s on harmful and maths sets (harmless 365-377) at 32 slots, against the
  test's 662 (32 answers of exactly 1,024 tokens): while the last long answers finish, most slots
  idle (on trial 205, busy 77-88% of a set's time), and answers run to 8,000 tokens. 8 slots: 283.
  A full measurement of a 9B edit took 3 h 36 min-3 h 53 min. (step 7)

- **Slots that fit depend on the model's KV cache, not the GPU alone.** [Measured: K2-Horizon-7B,
  A40] A dense model's f16 cache is 144 KiB a token (K2: 36 layers of 8 KV heads), so 32 slots of a
  thinking budget of 8192 need 41.5 GB: `measure` fits 28 on an A40. The search sized every slot for
  its longest KL read plus a whole thinking answer (15.6k tokens), so it fit only 16. Since
  2026-10-16 slots are run settings (`[slots]`), each search slot is sized for its largest request,
  and `inspect` shows the cache a token takes. Qwen3.5-9B's hybrid layers (32 KiB a token, plus
  50 MiB of state a slot) never hit this. (step 2)

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
  [Measured: 2 models] The best-supported change; it is only a default setting. Before making it
  the default: one more model family.
- **Measure the best public edit as a second reference.** [Hypothesis] `candor export --outside`
  then `candor measure --outside` on trohrbaugh's Qwen3.5-9B edit, as Trial 161 was, would show
  whether its 0/400 holds as used; a first look on the 100 test prompts costs minutes.
- **Thinking length by verdict on Trial 161's held-out harmful set,** from its labels and token
  counts: if prompts refused after long thinking are the longest, the model is deliberating.
  [Hypothesis; counts only, no new runs]
- **A separate edit just for thinking length** (Halt Vector, ThinkEdit, SEAL: 12-50% shorter) if
  length decides. [Published, on unedited models only]
- **A stall stop can't tell when the search is done.** [Measured: the phase 3 miss-rate simulator,
  240-1,200 runs a group; Qwen3.5-9B's search] Stopping once both kinds' best KL within the cap
  gains under 10% in 50 of their trials picks within 10% of the best no more often than a fixed
  count of the same length (about 270 trials: 15-35% of runs, against 37-66% at 600); every
  window tried did worse. Stopping at 300 rather than 600 roughly halves the hits.
  (`phase-4-k2-horizon-7b.md`, step 5)
- **Thinking length as a criterion between candidates.** [Hypothesis; 3 edits agree] The one edit
  that lengthened maths thinking (Trial 161) lost maths through budget hits; candor's two finalists
  didn't lengthen it and kept maths, so it didn't decide between them. Test: on the next model,
  whether maths thinking length on the test prompts predicts held-out maths. [Measured: K2's 4
  finalists] Median maths thinking stayed at 135-149 tokens (unedited 144) in all four, yet two
  failed on maths; the search's own budget hits were 3 and 1 for those two, 0 for the two that
  passed. Test: the search's budget hits against held-out maths on the next model's finalists.
- **Thinking that runs to the budget, as a search constraint with an early stop.** [Measured: a
  replay on K2-Horizon-7B's journal, 126 trials; built as `search.max_budget_hits`, on K2 from
  trial 138, 2026-10-17] Edits that comply hit the 8,192 cap 0-5 times in 100 and the unedited
  model 0 of 288; 17 of 126 trials hit it more than 8 times (the 11 among the first 100 took 33% of
  the time), none ever its kind's best, and judging them again at 8 takes only trials 66 and 102
  (KL 0.117) out of the cap. With hits spread evenly through the prompts, the stops would have cut
  the time 17%, nearly all from trials far over the caps; near-misses (9-14 hits) stop late. Unlike
  the refusal stop it can change later draws, in a kind with fewer edits within the caps than the
  sampler learns from (K2's one layer: 2). [Measured: the miss-rate simulator, budget hits fitted
  on K2, 45,360 searches] It costs no measurable share of picks within 10% of the best, pooled
  (at 600 trials -0.2 to -1.1 points on, -0.5 to -1.5 switched on at 137; intervals within -3.2 to
  +1.1), even with failing edits beside the best, and saves 6-26% of K2's pod time in K2-like
  landscapes; but it helps some landscapes and costs others up to 6.5 points, beyond chance.
  So: on from trial 1 for a thinking model whose pilot shows budget hits common (K2: 15% of trials
  over 8), off where they are rare and the saving small. [Measured: K2's search, 176 trials with
  it, 50 before] A trial took 22% less time (584 to 458 s) and hit the budget 1.4 times rather than
  5.2, with only 7 failing on it: the sampler steering away from long thinking, as on the
  simulator, saves more than the stop itself. Owed: whether the fixed prompt order biases it. (`phase-4-k2-horizon-7b.md`, step 5) Related:
  thinking length as a criterion, above.
- **ARA, heretic's new default edit.** [Published: others' judged tests] It kept models intact best
  but never removed the most refusals; on Qwen3.8-27B 57.5% complied against 82.2% for a careful
  directional edit. Parked until after Qwen, or if keeping maths and thinking intact decides
  between edits. (`evidence-2026-10-14/ara.md`)
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
  1,000 a step grows only about 0.5 ms per answer (10.6 ms at 1, 24.3 at 28), so 48-56 answers at
  once might reach about 1,400 tokens/s, but f16 KV at the full budget caps K2 at 28 slots on an
  A40. A pool the slots share, sized to what they use at once, could fit more, since most answers
  end far short of the budget; it changes numerics and how a full pool behaves. Test: time 28
  against 48 slots with it, and check counts against per-slot caches on one edit.
- **A faster GPU for 27B.** [Measured on K2-Horizon-7B; Hypothesis for 27B] On K2 an A100 was about
  10% faster than an A40 at full load for 3 times the price, as sampling the whole vocabulary, not
  the GPU, was the limit (speed test). A 27B model reads 3 times the weights a step, so test again:
  a short timed pilot on an A100 and an H100 before quoting 27B.
- **3-5 refusal directions,** if a single direction fails as used. [Hypothesis] **Optuna 5,** once
  candor has a baseline. [Housekeeping]
- **A 27B for the 18 GB Mac: Ternary Bonsai 2 27B with candor's adapters, unmerged.** [Hypothesis;
  PrismML's numbers Published] Its ternary GGUF is 5.95-7.21 GB, against about 16 GB for a 27B at
  Q4, and decodes 32.1 tok/s on an L4 (300 GB/s), so roughly half that on the M3 Pro (150 GB/s).
  Merging an edit and repacking to ternary would round most of it away, but candor already applies
  edits as rank-1 LoRA adapters, so the file could ship untouched with the adapters beside it. Needs
  PrismML's llama.cpp fork. Test: timed speed on the Mac, whether the fork's `--lora` works on its
  ternary types, and whether an unpacked FP16 exists for the directions.
