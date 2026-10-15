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
- **Thinking adds refusals to edits that half-work, not to ones that fully work.** [Measured: same
  24 edits] Edits refusing 8-22 with thinking off refused 3-28 more thinking (4 of the 5 at least
  10 more); a median of 5 more where thinking off refuses at most 25. Of 5 edits within the cap of 2 thinking off, 4 stayed
  within it thinking (the one that slipped, a one-layer edit, went from 1 to 6). (step 5)
- **The harder an edit pushes, the longer the model thinks.** [Measured: same 24 edits, and Trial
  161 on held-out prompts] Edits that still refuse think 700-1,300 tokens on average; per-layer
  edits that comply 1,600-2,000, none reaching the budget; one-layer edits that comply
  1,800-2,900, up to 10 of 100 reaching it. Trial 161 tripled harmful thinking (median 996 to
  2,980), took maths thinking from 2,140 to 3,620, budget hits on maths from 184 to 297 of 1,319,
  and lost maths (1,183 to 1,162 correct, p = 0.04). (steps 4 and 5)
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
- **Directions read at the thinking start mostly fail.** [Measured: Qwen3.5-9B] 71-100 refused of
  100 either way on 12 edits; but trial 175 (layer 15) refused 0 at KL 0.057, thinking off, not yet
  scored as used. [Published] They worked on Qwen3-8B (arXiv 2507.03167).

## The search

- **More strength than heretic's 1.5 made every best edit.** [Measured: Qwen3.5-9B, and K2-Horizon]
  Qwen's best was 22 refused before the range widened to 3.0 at about trial 120, then 0-2. Every
  best since uses more than 1.5 (one layer: 2.2-2.5 on the feed-forward weights; per layer: 2.9 on
  attention). K2-Horizon: 54 refused at a cap of 1.5, 7 at 5.0. (`phase-3-qwen3.5-9b.md`;
  `evidence-2026-10-14/data-report.md`)
- **The full 600 trials mattered.** [Measured: Qwen3.5-9B, one search] The best edits came at trials
  205-428 and none improved in the 117 trials after. 150 trials would have missed them all; by 300
  the one-layer best was within 1% of the final KL, the per-layer best 7-11% above it. (step 6)
- **One layer's direction at layer 12 gives the lowest KL.** [Measured: Qwen3.5-9B, thinking off
  only] Trials 205 and 417: 0-1 refused at KL 0.03, against 0.079 for the best per-layer edit and
  0.092 for Trial 161 rebuilt (24 refused). Not yet scored as used; one-layer edits slipped most
  when thinking (above). (step 6)
- **Per-layer edits press the 3.0 ceiling on attention.** [Measured: Qwen3.5-9B, one search] Every
  per-layer best puts attention at 2.9-3.0, so a better per-layer edit may lie beyond it. (step 6)
- **Refusals fall gradually as strength rises; there is no cliff.** [Measured: 8 searches on 3
  models] [Published] The literature agrees. (`evidence-2026-10-14/data-report.md`,
  `lit-refusal.md`)

## Pods

- **A llama-server process failed its GPU allocations after 395 trials, then CUDA wouldn't start.**
  [Hypothesis] Either the server's memory grew trial by trial, or the machine's GPU failed. GPU
  memory is now logged every minute on pods; growth across a long search would show the first.
  It matters for 27B, which leaves less memory spare.

## Ideas

- **Start strength wide (to 3.0 or more) rather than widening once the best presses 1.5.**
  [Measured: 2 models] The best-supported change; it is only a default setting. Before making it
  the default: one more model family.
- **A shorter search on larger models (about 300 trials).** [Hypothesis, from one search] Test: on
  the next model, see where its best edits turn up.
- **Thinking length as a criterion between candidates.** [Hypothesis] Edits that lengthen thinking
  may lose maths and reach the budget more (Trial 161 did). Test: step 7 measures both on the
  finalists.
- **ARA, heretic's new default edit.** [Published: others' judged tests] It kept models intact best
  but never removed the most refusals; on Qwen3.8-27B 57.5% complied against 82.2% for a careful
  directional edit. Parked until after Qwen, or if keeping maths and thinking intact decides
  between edits. (`evidence-2026-10-14/ara.md`)
- **Search coordinates such as total edit mass and spread, and a GP sampler instead of TPE.**
  [Hypothesis, with literature] Only after an offline replay over several model families.
  (`evidence-2026-10-14/lit-search.md`)
- **Edits at the thinking start.** [Published: one paper] [Measured: one trial works thinking off]
  Trial 175 is scored as used in step 6; probably candor's setup held them back before.
- **Cheaper as-used scoring:** vLLM or SGLang with each edit swapped into the weights, prompt
  caching, a refusal probe, multi-fidelity optimisation. [Hypothesis] (`history.md`, "Not in this
  plan")
- **A faster GPU for 27B.** [Hypothesis] On the A40, llama-server reached about 660 tokens/s,
  well below what the GPU's memory allows, so a faster card may help less than its price. Test: a
  short timed pilot on an A100 and an H100 before quoting 27B.
- **3-5 refusal directions,** if a single direction fails as used. [Hypothesis] **Optuna 5,** once
  candor has a baseline. [Housekeeping]
- **Models of more than about 36 layers with "pre" edits** pass Linux's argument limit in the
  `--lora` list. [Measured limit] "full" loads one adapter per trial; check before 27B.
