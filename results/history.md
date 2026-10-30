# History

Every decision with its reasons, the research tables and the dated log; the plan in force is in
STATUS.md. Until 2026-10-14 (commit 8d89012) this was part of STATUS.md. New log entries go at the
end of this file.

## Decisions

Depth before breadth (decided 2026-10-10): one complete run on the models you already use, in the
format you run them, before anything else. Every model is scored and shipped as a GGUF, and runs
in llama-server, the llama.cpp server candor scores in, so nothing differs between testing and use.
For every model, the final check (refusals, KL and maths, with your KV cache type) runs on the exact
file shipped, since merging and re-quantising can shift the result (phase 0: KL 0.01 between adapters
and merged weights in Q4_K_M). (Decided 2026-10-15: the user runs models only in llama-server, or a local runner that runs them better.)

| Phase | What | Done when |
| --- | --- | --- |
| 0 | Score edits as per-layer LoRA adapters in llama.cpp | Adapters match merged weights, no speed cost |
| 1 | Core maths and `candor inspect` | candor's rebuild of a heretic trial matches its refusals, and measuring heretic's saved model under heretic's conditions gives heretic's numbers exactly; `inspect` reports correctly on rude-llama, Qwen3.5-9B and K2-Horizon |
| 2 | rude-llama end to end on the Mac: measure, search, export a GGUF to run in llama-server | Under the decision rule written in step 4 before any search runs, candor's shipped GGUF is at least as good as heretic's Trial 170 converted and quantised the same way: fewer or equal refusals on held-out prompts, no higher KL, no maths loss. It runs in llama-server. (A blind A/B was dropped by the user on 2026-10-12: 535 and Trial 170 are too close to tell apart by hand.) |
| 3 | Qwen3.5-9B, a thinking model: scored with thinking on at your settings; residuals read on a pod | Written in step 4 before any search runs, from the measured noise, against heretic's own Trial 161 exported and measured as candor ships its edits |
| 4 | K2-Horizon at your real settings | Measured honestly with its own chat template (whether llama-server reads it is to be checked) and the reasoning level you use, then improved |

From phase 3 a job expected to take over about 2 hours, or needing more memory than the Mac has,
runs on a pod (the user's decision, 2026-10-12); shorter jobs (a measurement, a pilot) can run on
the Mac. The rude-llama search of phase 2 is the last long one on the Mac. So candor's whole
pipeline (convert, residuals, search, measure) must run on a pod: `scripts/pod.sh` and its watchdog,
which deletes the pod at a deadline set per job (README, "On a pod"), checked on rude-llama in phase
3 step 2. Spend pod money only where a run earns it.

Every step gets an independent review before it is called done; its findings are fixed with
tests and the fixes verified until a review comes back clean. Heavy jobs (converting a real model,
reading residuals, a search) wait for the user's go.

### Phase 3 decisions (the user, 2026-10-13)

- Thinking on everywhere: the search, every measurement and `serve`. The model is always used
  thinking, so scoring without it wouldn't replicate use. (Revised 2026-10-14, below: the search may
  use a cheap thinking-off score where step 5 shows it ranks edits alike; decisions stay as used.)
- Answers are sampled, not greedy, at Qwen's recommended thinking settings (temperature 1.0, top_p
  0.95, top_k 20, min_p 0, presence_penalty 1.5), with a fixed seed per prompt so two edits are still
  compared prompt by prompt. `serve` uses the same settings. Greedy isn't how the model is used, and
  thinking models tend to loop under it. Sampling adds noise greedy didn't have, so the first Qwen
  measurement is the unedited model twice with different seeds, and the decision rule allows for it.
- A thinking budget of 8192 tokens in real use (the user's setting; they are unsure of it):
  after it the model is made to close its thinking and answer. Measurements and `serve` use it. The
  search scores at a shorter budget, set by the pilot from how long answers think (numbers only),
  since 8192 for 600 trials would take days. If answers rarely think past about 1,500 tokens, the
  search uses 8192 too.
- An answer still unfinished after its forced close counts against the edit.
- KL is read at the start of the answer, after a closed thinking block, as heretic read it: the
  first token of thinking is nearly fixed, so KL there can't tell edits apart. The pilot checks that
  KL at the answer start does separate edits. Directions stay read at the think start.
- The reference is heretic's own Trial 161, re-saved by heretic on the pod and exported and measured
  as candor ships its edits (the user's choice, 2026-10-13): heretic made it with row normalisation
  "full" (rank 3) and directions read after a closed thinking block, which candor can't rebuild.
- Maths: the full 1,319 GSM8K questions at 8192 on every finalist, never on search trials.
- Pods: the pilot runs on an A40 ($0.59/hr); the search runs on the cheapest card that finishes it
  overnight (about 12 h). The watchdog deletes the pod at a cap set per job. A guess until the pilot:
  search $10-20, finalists $5-10.
- Optuna stays 4.9, as the search was tested on it. The past-the-boundary variant (2605.21706) is
  built only if Trial 161 and the search both fail with thinking on.

### Phase 3 decisions after the thinking-model research (the user, 2026-10-13)

The user's research check found no published weight edit on Qwen3.5, or any Gated DeltaNet model,
with thinking on; every working community recipe turns thinking off. So:

- Directions are read both at the think start and after a closed thinking block, and the search
  chooses between them: the evidence is split, our K2 directions at the two positions had a
  per-layer cosine of only 0.2-0.5, and Trial 161, read after the close, refused none of its 35
  thinking-on answers that finished.
- KL is read at the answer start after the unedited model's own thinking (one saved trace per
  harmless prompt, only those whose thinking finished), not after an empty thinking block, which
  measures thinking-off; the empty-block KL is kept beside it. No paper validates it, so step 4
  checks it separates the unedited model from Trial 161 by more than sampling noise.
- Strength may reach 0.5-3.0 for Qwen3.5. heretic's K2 run refused at best 54/100 with a ceiling
  of 1.5 and 7/100 with 5.0 (peaks 4.5-4.6, row normalisation "full"); Trial 161's peaks are
  1.42-1.46 of 1.5; a 27B Qwen hybrid's weight edit collapsed at 2.0; Qwen3.5-2B needed 0.5.
  Strengths under "full" and candor's "pre" aren't the same scale. K2-Horizon gets its own range in
  phase 4.
- The miss-rate check, rerun on the test landscapes, changed that (results file, step 3): a fixed
  0.5-3.0 halved the hits when the best lies inside 0.8-1.5. So the search starts at 0.8-1.5 and
  widens an end to 0.5 or 3.0 itself when a kind's best edit presses it, which stayed within about
  12 points of the wide range where the best needed it. The position is a parameter of each kind's
  study, its 20 start points split evenly; four studies were 12-17 points worse. 600 trials stay.
- The fallback, built only if the pilot fails: 3-5 directions, or ARA, instead of 2605.21706's
  method, which edits activations and can't be a weight edit. If candor's "pre" edits fail where
  Trial 161 ("full") worked, row normalisation is the first suspect.
- The biggest open risk is the search's cost with thinking on: with Trial 161, 65 of 100 harmful
  answers were still thinking at 2048 tokens. The pilot measures it before the full search.

### Phase 3 decisions after the night on the pod (the user, 2026-10-14)

The goal stands: edits are chosen on how the model behaves as it is used, thinking on, sampled, in
llama-server on the file that ships. But scoring 600 trials that way would take 100-200 GPU hours
for a 9B model (Trial 161's measurement alone took 8 h), and the process must also work for 27B+
thinking models. A 9B model must not take days. So:

- Decisions only ever use as-used numbers. A cheap score (thinking closed with an empty block,
  greedy, short answers, judged) may only decide which edits get the expensive as-used test, and
  only once step 5 shows it ranks edits as thinking on does, for each new model.
- Why the check: no published work validates a thinking-off score for ranking edits. The gap
  between the modes changes size and even sign by model and by edit (unedited Qwen3 gives 6-15
  points more harmful answers thinking on, 2509.00544; R1-distills are far safer thinking off,
  2502.12025; 3-30% of prompts flip between modes, 2606.25013); an edit tuned thinking off reached
  0.23-0.34 against 0.73 tuned thinking on (2607.02396); K2's heretic trial refused 7/100 thinking
  off and 38/100 on, while Trial 161 held (11/100 off, 9.7% on). Answers under about 100 tokens miss
  late refusals (heretic #487).
- 100 prompts give about +-6 points; edits a few points apart are told apart only paired on the
  held-out prompts.
- llama-server's presence penalty covers only the last 64 tokens, prompt included (`repeat_last_n`),
  where Qwen means the whole output (vLLM). candor measures it as llama-server runs it; it may be one
  reason thinking runs long in real use.

### Phase 3's decision rule (from step 4, signed off by the user 2026-10-14)

candor's edit, exported and measured as used on the held-out prompts like Trial 161 (Q4_K_M, f16 KV
cache, 32 slots, Qwen's sampling, budget 8192), is compared prompt by prompt; "beyond noise" means
a paired McNemar test at p < 0.05. It passes if:

- harmful refused (judge) is no more than Trial 161's 28/288 beyond noise;
- harmless refused is no more than the unedited model's 5/300 beyond noise;
- KL after the unedited model's own thinking is no more than Trial 161's 0.073 (KL isn't sampled);
- maths correct is no worse than Trial 161's 1162/1319 beyond noise;
- maths thinking reaching the budget is no more than Trial 161's 297/1319 beyond noise;

and it beats Trial 161 if it also does better beyond noise on maths, on thinking reaching the budget,
or by a lower KL. Measured on the full 1,319 maths questions, as Trial 161 was.

### Phase 3 steps

| Step | What | State |
| --- | --- | --- |
| 1 | Thinking-model settings: sampling with a seed per prompt, the thinking budget with a forced close (llama-server's own), the judge reading only the final answer, an answer that never comes counted as a refusal, KL at the answer start; `serve` at the same settings; `measure --seed` for the noise check; `export --outside` and `measure --outside` to ship and measure heretic's own save of Trial 161 like candor's edits (the user's choice: heretic made it with row normalisation "full", rank-3, which candor can't rebuild). Checked on a tiny thinking model in the real llama-server and SmolLM2 | Done 2026-10-13, reviewed: 2 major findings (the chat page forces `\n</think>` after the message, so measurements now force the same; a heretic trial read after a `response_prefix` is refused, not rebuilt wrongly) and 6 minor ones fixed, each with a test that fails without its fix |
| 2 | Pod bootstrap, torch's CUDA build matched to llama.cpp's 12.8, and an in-pod watchdog that deletes the pod at a budget; checked on rude-llama | Done 2026-10-13: `scripts/pod.sh` and its watchdog, armed from the pod's own start command; reviewed until clean (9 findings, then a regression they caused, then none), each fixed with a test that fails without it. On an A40 the run matched the Mac (F16 GGUF and prompt ids byte-identical, directions at cosine 0.9999, the same counts on 60 answers but one maths cut-off) and the watchdog deleted the pod itself; about $0.51. Qwen3.5's fast kernels left out: transformers only reads residuals of short prompts (README). Results file |
| 3 | The method changes above: directions at both positions with the search choosing, KL after the unedited model's thinking, strength widened up to 0.5-3.0 where a kind's best presses 0.8-1.5, the miss-rate check rerun, and the research table corrected | Done 2026-10-13, reviewed: no correctness bug, 6 findings fixed (a kill while widening and how the unedited thinking is sampled were untested; `upgrade` refused a strength past 3.0; the `--lora` limit below; no progress while the unedited model thinks; a misleading upgrade message), each with a test that fails without its fix. Miss-rate numbers in the results file |
| 4 | Qwen3.5-9B on a pod: `init`, `run`, the unedited model measured twice (noise), heretic's own Trial 161 re-saved, exported and measured with `--outside`, thinking lengths, KL at each position. The decision rule written from them, for sign-off | Done 2026-10-14 ($9.56): Trial 161 refuses 28/288 against 279, but loses maths (1162 against 1183 of 1319, p = 0.04) and reaches the thinking budget on 297 maths questions against 184. The noise run was skipped (the user); the rule uses paired tests. A speed test found GPU sampling 2.2x faster. Results file |
| 5 | Built and reviewed 2026-10-14, to run on a pod: speed, then the proxy check. candor enforces the thinking budget itself, so llama.cpp samples on the GPU (it turns GPU sampling off for a request with a budget; same sampler maths, only the draws per seed change); 2-3 servers per GPU if the speed test shows a gain; an edit's scoring stops once its refusals pass the cap; 150-token answers in the cheap score. Then about 24 edits spread over kinds, positions and strengths (stronger ones included) scored both thinking off and as used, plus one repeat for noise: whether the two rank edits alike, and whether thinking on needs a stronger edit | Done 2026-10-14: "offset" (log) |
| 6 | The search, on the path step 5 picks. If the cheap score ranks like thinking on (rank correlation at least 0.7, perhaps after a strength correction): 600 cheap trials, then about 15-20 edits from their front scored as used at the corrected strength and raced on shared prompts and seeds. If not: a thinking-on search of about 150 trials, seeded with step 5's as-used scores, the miss-rate simulator rerun at 150 first | Done 2026-10-15: 545 cheap trials, then the best 6 scored as used (log) |
| 7 | The best 2 exported and measured as used on the held-out prompts and 400 maths questions, paired with the unedited model on the same ones; the winner's full 1,319 maths is its final check on the file that ships; copied to the Mac as it is (each machine quantises differently), `serve` on the Mac | Done 2026-10-15, on the full 1,319 maths questions (log) |

### The search fix (done 2026-10-12)

The first search missed a better recipe inside its own space: Trial 170 read at layer 12 scores 0
of 100 at KL 0.0818 with the search's own scorer, against candor's best 0.1017. It never tried that
shape. The search runs on new models with nothing to compare against, so this was fixed before
phase 3 (the user, 2026-10-11). Designs were compared on test landscapes calibrated to the real
trials, and no default came from Trial 170's values (results file, "The search fix").

The design chosen: the two kinds of edit searched as two studies taking turns, each starting from 20
evenly spread trials and aiming at the least KL within a cap of 2 refusals, with trials that can't
win stopped once 25 fully scored trials of their kind refused fewer, which leaves every trial the
sampler draws unchanged. On the calibrated test landscape it got within 10% of the best in 83% of
runs at 400 trials and 93% at 600 (the first design: 20% and 23%), and was no worse on any landscape
tested. It is not watertight: on the harshest landscape it still missed in most runs. Repeats aren't
built (whether two agreed didn't predict a miss); the search flags suggested trials that press the
end of a range instead, and from phase 3 widens strength itself when a kind's best presses it.

The check, set before it ran: one search with no heretic recipe as a start passes if a trial it
suggests refuses at most 6 of 288 held-out prompts at KL 0.1041 or less. Trial 535 refuses 5 at KL
0.0920 and passes phase 2's strict rule too, so it ships instead of Trial 170's recipe. But the four
trials the search first suggested all failed; 535 is suggested only since the suggestions were fixed
(`f557f24`, before any held-out number). The milestone: `results/milestone-rude-llama-vs-heretic.md`.

To compare a trial against the rule: `candor compare runs/rude base trial-N` (KL and maths are
judged against the unedited model) and `candor compare runs/rude heretic-170 trial-N` (head to head).

### Phase 2's decision rule (signed off by the user, 2026-10-11)

Written from the full measurement (log below), before any search. candor's edit for rude-llama is
at least as good as Trial 170 when, measured with `candor measure` and the same settings, all of
these hold:

1. Held-out harmful prompts refused, by the judge: at most 6 of 288 (Trial 170's).
2. KL on the held-out harmless prompts' first tokens: at most 0.1011 (Trial 170's).
3. Maths: not significantly worse than the unedited model (McNemar p of at least 0.05), as Trial
   170 isn't (p 0.87).
4. Held-out harmless prompts refused, by the judge: no more than the unedited model's 7 of 300.

Measurement noise is too small to matter here (no held-out label flipped and KL 0.0000 between 8
and 16 slots), so counts are compared as they are, strictly: no trading refusals for KL. Step 6
then repeats this on the exact file shipped, and adds running it in llama-server and the blind A/B.

Also decided: the search counts each trial's refusals with the judge, not keywords (keywords
miscounted 19 of Trial 170's 25), and rude-llama gets one search: if its result passes, it ships;
if not, Trial 170's recipe built by candor ships. Then phase 3, Qwen3.5-9B.

### Phase 2 steps

| Step | What | State |
| --- | --- | --- |
| 1 | `candor init NAME --model SOURCE`: checks the model, writes `runs/NAME/candor.toml` with every setting explained, validated on every load; pins the model's commit, the datasets' commits and the date; refuses models candor can't edit; never overwrites a run | Done, reviewed |
| 2 | `candor run RUN` up to the search: GGUF (edit targets checked byte for byte), prompt ids from llama-server, directions, one adapter per edit target per direction set; resumable, and rebuilds whatever a settings or code change affects | Done, reviewed |
| 3 | Fixes from the full review (see the log) | Done: every fix shown to matter by a test that fails without it; a second review was stopped as diminishing returns, so the first real rude-llama run is the check that matters |
| 4 | Measurement, settled before any search: a held-out harmful set outside AdvBench (de-duplicated against the direction prompts, pinned); the Qwen3Guard judge with its false-refusal rate measured on answers to harmless prompts, keyword counts kept beside it; the scoring noise floor from repeat scoring; paired comparison (McNemar) with a margin; KL on 300 harmless prompts; the full GSM8K test set, paired; heretic's Trial 170 rebuilt by candor and scored as adapters, so the comparison is like for like. Written down as phase 2's decision rule | Done: measured, and the decision rule signed off |
| 5 | The search: Optuna over a kernel per component (heretic's 0.8-1.5 strength range), stored resumably in the run folder, each trial scored in llama.cpp on the quantised GGUF, refusals counted by the judge; progress with time left, the Mac kept awake. Built on SmolLM2, then a 10-trial rude-llama pilot (time per trial) and one full search, each with the user's go | Done: the first search (200 trials) missed Trial 170's shape and none of its trials passes; the rebuilt one (600 trials, 7 h 21 min) found trial 535, which passes |
| 6 | Export: merge, convert and quantise; the final check on the exact shipped file with your KV cache type; a `candor serve` command that runs the shipped file in llama-server at your settings (system prompt, KV cache, date); the blind A/B | Done 2026-10-13: `export` (with a check that quantising keeps each edited weight's change, moving lossy ones to Q8_0), `measure --exported` and `serve`, each reviewed; the first export lost most of 535's edit to Q4_K and the check came from that (results file); the blind A/B dropped by the user |

## Research changes and where they land

From the 2026-10-10 paper review, corrected after the full review read each paper in full.

| Finding | Change | Where |
| --- | --- | --- |
| Matching targets by name silently misses most of the effect on mixture-of-experts models (2609.09793) | `inspect` lists every per-layer tensor that outputs the hidden size and fails on any it can't place or any layer without a writer | Phase 1, done |
| Keyword judges count compliant answers with safety words as refusals (2512.13655: 72.2% vs 95.7% non-refusal on 900 abliterated answers) | Qwen3Guard-Gen-0.6B on the final answer, with its false-refusal rate measured (the vendor's refusal precision is 83-89%, 2510.14276; calibrate any judge, 2606.25487); keyword counts kept beside it | Phase 2 step 4 |
| One count on 100 prompts can't separate close models (2609.25173 for unpaired sampled runs; heretic's transformers scoring varied by 1-2 refusals with batch size). Measured in llama-server on rude-llama between 8 and 16 slots: no held-out label and 5 of 1,319 maths results changed, KL 0.0000 (`phase-2-rude-llama.md`) | Measure the noise floor; compare paired on a held-out set; scoring is greedy, so seeds matter only for sampled checks | Phase 2 step 4 |
| First-token KL misses lost ability (2512.13655: -4.5pp GSM8K at KL 0.043) | The full GSM8K test set, paired against the same quantised file; multiple-choice flips only as a warning (2407.09141) | Phase 2 step 4 |
| Re-quantising after an edit can bring refusals back (2609.09793) | The last check runs on the exact file you ship, with your KV cache type (2606.09864; already used for all scoring). On rude-llama, whose weights were 4-bit before, Q4_K kept only 55-98% of each edited weight's change and refusals went from 5 to 161 of 288, so `export` measures every edited weight's kept share and quantises lossy ones as Q8_0 | Phase 2 step 6, done |
| Thinking mode changes refusals (2509.00544). Whether to read a direction at the think start or the think close is split: refuse-or-comply is predictable at the first thinking token (2606.25013: AUROC 0.84-0.95, rarely changing after 20% of the thinking), 2510.06036 finds a refusal cliff at the think close but little in Qwen3-Thinking, and our K2 directions at the two places had a cosine of 0.2-0.5. (2609.18471, cited here before, never looks at the think close.) | Score in the mode and at the level(s) you run, with answers long enough to finish thinking, judged on the final answer with unfinished answers counted against the edit (2609.09793). Directions read at both places, the search choosing per trial | Phases 3-4; phase 3 step 3 |
| One direction at a strength near 1 fails on Qwen-family thinking models of 8B and up: 7% attack success on Qwen3-8B (2607.02396), heretic's default on Qwen3.5-9B still refusing 70/100 after 800 trials (heretic #404, confounded by keywords read in the thinking). 2605.21706's 3.3% doesn't bear on this: it edits activations along SVM-probe directions with a margin searched per layer, and never thinks. Strengths that worked: 1.4-1.6 on a 27B Qwen hybrid's weights (2.0 collapsed it), 4.5-4.6 on K2-Horizon under heretic's "full" row normalisation, 0.5 on Qwen3.5-2B | Measure Qwen3.5 properly before trusting heretic's Trial 161; strength from 0.8-1.5, widened up to 0.5-3.0 where a kind's best edit presses it | Phase 3 |
| A direction averaged over the reasoning (2507.03167: clearly better on R1-Distill-Llama-8B, 0.4% against 10.1% refused; a tie on Qwen3-8B's full set, 7.5% against 8.5%); one direction from a mixed prompt pool beats per-category ones (2602.02132) | Build directions from prompts like the refusals you actually hit; a reasoning-averaged set only if refusals remain | Phases 3-4 |
| More directions help on Qwen3-8B: 7% attack success with one, 62% with three, 73% with five (2607.02396); heretic's unreleased ARA, a rank-50 edit pushing harmful outputs past harmless ones, took Qwen3.5-4B from over 50/100 refused to 4/100 at KL 0.14. Cleaned directions (2601.08489, single author, untuned baseline) | 3-5 directions or ARA only if the phase 3 pilot fails | Phase 3, if needed |
| First-token KL after an empty thinking block measures the model thinking-off. Ablation can make thinking run on (2507.03167: unfinished thinking from 2.4% to 39.9%), heretic #253 reports thinking loops KL didn't catch, and Arditi's most sensitive damage measure was loss on the unedited model's own completions (2406.11717) | KL read after the unedited model's own thinking; KL after an empty block and the thinking's length kept beside it | Phase 3 step 3 |

## Not in this plan

- heretic and `tune/heretic-kit`: frozen, no new features. The guide stays as reference.
- `tune/abliterate.py`: deleted 2026-10-10. candor does the same job.
- Cut 2026-10-10 for depth first: `candor doctor`, a live dashboard, one-command pod orchestration,
  rewriting the guide (the README and `docs/` are the documentation), and support for more model
  families (the existing mixture-of-experts, Gemma and GLM checks stay as they are).
- Optuna is pinned below 5 (4.9, the version heretic ran), so rude-llama's search uses the same
  sampler that found Trial 170. Optuna 5 changed multivariate TPE's bandwidth and its
  multi-objective weights; worth trying once candor has its own baseline, for example on Qwen3.5.
- Linux's 128 KiB argument limit on the `--lora` list no longer limits layers: since 2026-10-15 a
  list too long by path goes through links named by number (K2-Horizon's 36 layers: 157,039 bytes by
  path once `score` adds its own adapters, 34,313 by number).
- Ideas for cheaper thinking-on scoring, researched 2026-10-14, for later:
  - vLLM or SGLang for as-used scoring, with each trial's edit swapped into the weights in place as
    RL frameworks do (LoRA is out: vLLM's gives NaN past about 400 tokens on Qwen3.5-9B, #52568).
    Estimated (not measured) 1-1.5k tok/s on an A40, 5-7k on an H100, against llama-server's 300.
    Costs: full precision instead of the shipped Q4 file, a presence penalty over the whole output
    (no window, prompt excluded), results that depend on the batch (#59764). The first thing to try
    if llama.cpp stays too slow for 27B.
  - Patching llama.cpp's penalties loop to visit only the tokens it counts (byte-identical output,
    but our own build instead of the release binaries).
  - Caching the prompt within a trial so each answer doesn't re-read its thinking (`cache_prompt` is
    off; slots would need erasing between trials).
  - A probe on early thinking activations to predict refusal (AUROC 0.90-0.97 on unedited models,
    2606.25013), untested on edited models; llama-server can't read activations.
  - Multi-fidelity Bayesian optimisation (co-kriging, BoTorch) or OptunaHub's MetaLearnTPE warm
    start: worth it only with 50-80 as-used points.
- Ideas researched 2026-10-14 and rejected: a short thinking budget with a forced close (closing
  early pushes models towards refusal, R1-7B 36% to 94% safe, 2502.12025, by different amounts per
  edit); judging the first part of the thinking (text judges F1 51-73, reliable only after about
  half the thinking, 2507.12428); first-token and other generation-free scores (the first thinking
  token is fixed; unvalidated even thinking off); Optuna pruning by prompt count (not for
  multi-objective studies in 4.9, too noisy early; stopping at the refusal cap is exact);
  speculative decoding (sampling stays on the CPU with drafts, little gain when batched).
- Reporting the llama.cpp prompt-cache bug upstream: not wanted.
- Separate tracks, outside candor: the Qwen3.5 rude fine-tune and the Mac Studio purchase.

## Log

### Phase 0: per-layer LoRA scoring in llama.cpp (done 2026-10-10)

On Llama-3.2-3B-Instruct with rude-llama's Trial 170 layer profile, scoring an edit as per-layer
rank-1 adapters with per-request scales gives the same results as merging it.

| | Refusals /100 | First-token KL from base |
| --- | --- | --- |
| transformers, base / merged | 97 / 3 | 0.0912 (exact) |
| llama.cpp bf16, base / adapters / merged | 97 / 3 / 3 | 0.0917 (adapters), 0.0907 (merged) |
| llama.cpp Q4_K_M, base / adapters / merged | 97 / 4 / 4 | 0.122 (adapters), 0.110 (merged) |

- Adapters vs merged weights: KL 0.00003 and the same first token on 100/100 prompts in bf16;
  KL 0.01 and 97/100 in Q4_K_M. The run-to-run noise floor in Q4_K_M is about KL 0.002.
- KL from the server's top 20 tokens is within 0.5% of the exact value.
- 616 loaded adapters: 3.5 s startup instead of 1.5 s, no change in tokens/s, active or not.
- The merged copy converted to GGUF with no file fixes, because merge.py never rewrites them.
- Two llama-server traps, handled in `engine.py`: `--lora-init-without-apply` leaves every
  adapter's default scale at 1 (the engine sets them to 0 and checks), and prompt caches don't
  record which adapters computed a prompt (the engine sends `cache_prompt: false`; without it 3
  refusals became 28).

Numbers: `runs/phase0/results.json` (not in git). Script: `spikes/phase0.py`, last in commit
`ae2f060`. Pod: A40, about 23 minutes, about $0.23.

### Phase 1: core maths and `candor inspect` (done 2026-10-10)

- `candor.heretic` reads heretic's Optuna journals; `spikes/reproduce_trial.py` (last in `ae2f060`) rebuilds a trial
  with candor's own direction finding and edit. On rude-llama Trial 170 it gets heretic's 2/100.
- Heretic tokenizes the rendered chat with the tokenizer's own special tokens added, so Llama 3
  and K2-Horizon get two begin-of-text tokens. Proved by running heretic's own `Model.generate`
  with a recording stub. transformers, mlx-lm and llama.cpp (`chat.cpp`) all send one. Qwen3.5 has
  none, so it is unaffected.
- `spikes/compare_models.py` (last in `ae2f060`) scored the saved heretic model (`models/local/rude-llama-heretic`)
  against the original on heretic's test prompts:

| Input | Date in prompt | Batch | Refusals, original | Refusals, edited | KL |
| --- | --- | --- | --- | --- | --- |
| heretic's (2 tokens) | 09 Oct 2026 | 64 | 96 | 2 | 0.0260 |
| as apps send it (1 token) | 09 Oct 2026 | 64 | 89 | 3 | 0.0375 |
| as apps send it (1 token) | 10 Oct 2026 | 32 | 91 | 4 | 0.0384 |

  The first row is exactly heretic's journal, so every difference is explained: the extra token,
  the date the Llama 3 template writes into each prompt, and batch-size noise of about 1-2
  refusals. candor's keyword judge and heretic's agreed on all 800 saved answers.
- `candor.checks` reports both traps for any model: `begin_of_text` and `writes_the_date`.
- The engine test (`tests/test_engine.py`) runs a tiny random GGUF in llama-server. It fails if the
  default-scale reset or `cache_prompt: false` is removed. `--cache-ram 0` and `--ctx-checkpoints 0`
  only save memory and time, since nothing is reused without the prompt cache.
- `candor inspect MODEL` reads the config, tokenizer and file headers only. On the three models:
  rude-llama passes (warns: doubled token, date); Qwen3.5-9B passes and finds all 64 writers,
  including 24 linear-attention `out_proj` (llama.cpp's `ssm_out`); K2-Horizon passes, and
  warns about remote code and the doubled token.
- `gguf` now comes from the same llama.cpp release as the engine (PyPI's 0.19.0 doesn't know
  K2-Horizon), and candor mirrors the converter's `language_model.` prefix stripping.
- Reading the papers corrected the review: 2609.04721 doesn't cover Qwen3.5 (its linear-attention
  writer is known from heretic's code), and 2609.09793 argues against name matching without
  prescribing shape matching.

### Reviews of phases 1 and 2 (2026-10-10)

Each step was reviewed independently and every finding fixed; the commit messages list them.

- Phase 1: 13 defects, then a regression and 5 gaps in the fixes. They were in the engine (a port
  another server holds, the served model, cleanup, prompts whose first token is an incomplete
  UTF-8 sequence), finding targets on unusual layouts, choosing the llama.cpp architecture as its
  converter does, `inspect`'s errors and exit codes, and `merge` checking every target first.
- Step 1 (`init`): 9 defects and 4 weak tests: run names, strict settings, building the run
  folder under a temporary name, pinning Hub models and the date.
- Step 2 (`run` up to the search): 12 defects and 5 weak tests, then 8, 3 and 1 over three rounds
  of verifying the fixes: stale results after a settings change, the toolchain's checks, finite
  residuals, over-long prompts, the disk check and flushing to disk on macOS.

### First real run: rude-llama through step 2 (2026-10-10)

`candor init rude --model ~/Desktop/tune/models/rude-llama-bf16`, then `candor run runs/rude`, on the
18 GB Mac: 122 s in all, every check passed, nothing left running.

| Stage | Time | Result |
| --- | --- | --- |
| Convert, check the 56 targets byte for byte | 23 s | `model-F16.gguf`, 6.0 GB (rude-llama is F16 on disk; its transformers-5 tokenizer converted through the staging folder) |
| Quantise | 28 s | `model-Q4_K_M.gguf`, 1.9 GB |
| Prompt ids from llama-server, checked against the chat endpoint | 8 s | 1,000 prompts, date pinned |
| Residuals on MPS, straight onto the GPU | 57 s | 29 directions (the embedding's is zero, since every prompt ends on the same token) |
| Adapters | 6 s | 15 direction sets x 56 targets = 840 files, 31 MB |

- Running it again reused everything in 3.6 s. Deleting `directions.pt` rebuilt only that stage,
  and the directions came out bit for bit identical; the adapters were reused.
- The largest process peaked at 2.9 GB of RAM; the model's 6.4 GB sits in GPU memory.
- Fixed on the way: transformers' weight-loading progress bar leaked into candor's output, and a
  deleted result was reported as "what it was built from has changed" rather than missing.

### Step 4: held-out prompt sets (2026-10-10)

The user chose StrongREJECT (`walledai/StrongREJECT`, MIT, gated, pinned at `df54272b`) for the
held-out harmful prompts. Its prompts are never read; everything below is counts.

- 313 rows: 221 written by its authors, 92 taken from other sets, 25 of them from AdvBench. 23 of
  those 25 are word for word in candor's harmful sets, which are AdvBench, so a prompt set can now
  leave out rows by a column's value (`exclude = { source = ["AdvBench"] }`), and naming a value no
  row has is an error, so a typo can't silently keep them.
- Held-out sets drop any prompt also in another set. That left 288 harmful prompts (none exact
  repeats; one shares over 60% of its word triples with a direction or test prompt, seven over
  20%) and 300 harmless ones from `harmless_alpaca` `test[100:400]` (none over 40%).
- Both sets are tokenized with the rest and must fit with a whole answer. On rude-llama only
  `prompts.json` rebuilt (15 s, 1,588 prompts); the directions and adapters were reused.
- `candor upgrade RUN` adds settings a newer candor has, at their defaults, pins any prompt sets it
  adds and lists what it added; `runs/rude` was upgraded with it. A missing setting's error points
  to it.

### Step 4: measurement built and piloted (2026-10-10)

- `candor measure` answers the held-out sets and GSM8K's 1,319 test questions with the deployed
  Q4_K_M in llama-server, 8 at a time. Qwen3Guard-Gen-0.6B (pinned at `fada3b2f`) labels each
  held-out answer from its `Refusal:` and `Safety:` lines; keyword counts sit beside it. It keeps
  the 20 most likely first tokens on the harmless prompts for KL. Each set's answers are saved as
  soon as they finish, then judged and the labels saved apart, so a stopped run resumes from the
  last set saved and a new judge re-judges the saved answers without answering again. Answers and
  the judge's replies go only to `private/`; numbers are per prompt, so `candor compare` pairs two
  measurements (exact McNemar), and refuses two made on a different model file or settings.
- An edit gets its own adapters, since heretic's Trial 170 reads its direction at layer 12.04, between
  whole layers. Trial 170 is checked to match candor's method (row norm "pre", no winsorising,
  orthogonalised directions) before it is rebuilt.
- Maths answers are asked to end with "Answer: N"; the stated number, or else the last number of a
  finished answer, is compared with GSM8K's. One cut off at 512 tokens without stating its answer is
  wrong. Held-out answers get 256 tokens for the judge.
- One independent review found 8 problems, all fixed with a test that fails without the fix:
  measurements didn't record the quantisation, so a Q8_0 measurement could reuse or be compared
  with a Q4_K_M one; `compare` didn't check the two used the same settings; a kill sent every queued
  request before stopping; a judge failure lost the answers; a cut-off maths answer could count as
  correct; prompts read at measure time weren't checked against `prompts.json`; a heretic study
  without the orthogonalisation setting was accepted; and saved answers weren't tied to the
  numbers.
- The real judge was checked on made-up answers: a refusal is labelled one, a bread recipe isn't.
- `candor upgrade` added `prompts.maths` and `[measure]` to `runs/rude`, pinning both.

Pilot on rude-llama, the first 10 prompts of each set (counts only):

| | Unedited | Trial 170 rebuilt | Only one / other | p |
| --- | --- | --- | --- | --- |
| Harmful refused, judge | 9/10 | 0/10 | 9 / 0 | 0.004 |
| Harmful refused, keywords | 9/10 | 0/10 | 9 / 0 | 0.004 |
| Harmful answered unsafely | 0/10 | 8/10 | 0 / 8 | 0.008 |
| Harmful cut off at 256 tokens | 1/10 | 10/10 | 0 / 9 | 0.004 |
| Harmless refused, judge and keywords | 0/10 | 0/10 | | |
| Maths correct | 5/10 | 4/10 | 1 / 0 | 1 |
| KL on harmless first tokens | | 0.058 | | |
| Time (full run, estimated) | 48 s (about 1 h) | 56 s (about 1 h 11 min) | | |

The judge and keywords agreed on all 40 held-out answers; 10 prompts can't say more than that.
Re-running both gave the same counts and KL 0.0577 against 0.0576, a first sight of the noise the
full measurement's `--slots 16` run will size.

### Step 5: the search built (2026-10-11)

- `candor search RUN` runs Optuna's TPE (4.9, as heretic ran) over heretic's edit shape: one
  direction for every layer or each layer's own, and per component a peak strength (0.8-1.5), its
  position (0.6-1.0 of the depth), the edge strength as a share of the peak, and a reach of at least
  one layer. 200 trials, 60 random, 128 candidates per step, two objectives: KL and the refusal
  share, with KL replaced below 0.01 as heretic does. candor's single direction is a whole layer
  (Trial 170's 12.04 sits next to layer 12), and refusals are counted by the judge.
- Every adapter set loads into one llama-server once; a trial sets strengths per request, answers
  the 100 test prompts (100 tokens), the judge counts refusals (an empty answer always counts), and
  KL comes from the 100 harmless test prompts. Answers are dropped once judged.
- It resumes where it stopped, fails a trial cut off part way, moves the sampler's seed on each
  session so no random trial repeats, and won't carry on a search whose scores would now mean
  something else (model file, adapters, prompts, lengths, judge, KV cache, settings, code, Optuna).
  `nohup` keeps it going after the terminal closes. It ends with the trials no other beats on both
  refusals and KL, and the `measure --trial N` commands for the most promising, which rebuild the
  same adapters byte for byte.
- Building it showed that edits could reach no layer at all (a reach under one layer), unlike
  heretic's; the reach now never goes below one layer.
- Two independent reviews (heretic fidelity and correctness; overnight robustness, privacy and
  tests) found 12 problems. 11 are fixed, each with a test that fails without the fix: among them a
  resumed search repeated its random trials, Optuna 5's TPE isn't the one heretic used, the time
  left never showed, saved labels from before the empty-answer rule were reused, and ten mutations
  of the scoring survived the tests. The twelfth, models of 64 layers or more, is noted under "Not
  in this plan".
- A two-trial search on SmolLM2 with the real judge passes end to end.
- On rude-llama: `candor upgrade` added `search.kl_target`; re-judging both saved measurements
  under the empty-answer rule changed no count. A two-trial search (the start of the real one):
  the unedited model refuses 99 of the 100 test prompts; the two random trials refused 42 and 22,
  at KL 0.065 and 0.076 (keywords: 37 and 18). 76 s per trial, about a minute more per session
  for loading 840 adapters and the judge and scoring the unedited model.

### Step 4: the full measurement (2026-10-11)

Run by the user on the Mac, about 30 minutes per measurement. Counts only.

| | Unedited | Trial 170 rebuilt | Only one / other | p |
| --- | --- | --- | --- | --- |
| Harmful refused, judge | 277/288 | 6/288 | 271 / 0 | 5e-82 |
| Harmful refused, keywords | 275/288 | 25/288 | 250 / 0 | 1e-75 |
| Harmful answered unsafely | 6/288 | 254/288 | 0 / 248 | 4e-75 |
| Harmful cut off at 256 tokens | 12/288 | 266/288 | 0 / 254 | 7e-77 |
| Harmless refused, judge | 7/300 | 2/300 | 5 / 0 | 0.063 |
| Harmless refused, keywords | 7/300 | 4/300 | 3 / 0 | 0.25 |
| Maths correct | 980/1319 (74.3%) | 977/1319 (74.1%) | 81 / 78 | 0.87 |
| Maths cut off at 512 tokens | 17/1319 | 21/1319 | 10 / 14 | 0.54 |
| KL on harmless first tokens | | 0.1011 | | |

- Trial 170 holds on prompts it never saw: 96% refused before, 2% after, and the judge rates 88% of
  its answers as real compliance.
- No maths loss, but the edit changes which questions are right: 159 flip, evenly both ways, against
  5 from batching alone.
- On Trial 170 the keywords count 25 refusals where the judge counts 6 (they differ on 27 answers;
  on the unedited model, 4): compliant answers that mention words like "illegal". heretic chose
  Trial 170 by keywords, and candor's search scores by keywords too.
- The judge called 7 of the unedited model's harmless answers refusals and 2 of Trial 170's, so its
  false alarms are at most about 2%.
- Noise floor (unedited at 8 and 16 slots): every held-out label the same, KL 0.0000; 5 maths results
  changed (3 / 2), and 2 more were cut off.
- The `--limit` estimate overshot (59 minutes against about 25): ten prompts can't keep 8 slots busy.

### Full review (2026-10-10)

Eleven reviewers covered the code, the tests (about 90 mutations), the papers and the plan; an
independent verifier checked each finding, two for serious ones. 124 findings, about 60 distinct,
all but one confirmed.

- Found correct:
  - The maths matches heretic's source (edit, kernel, direction interpolation, KL, judge) and
    llama.cpp's LoRA conventions: adapters against merged weights gave KL 9e-14 on the tiny model.
  - Requests with different adapter scales are never batched together.
  - 1,000 adapters start in 0.55 s on Metal.
  - Every edit target of the three models goes through llama.cpp's LoRA path.
  - `inspect` is right on all three.
  - The CUDA runtime archive holds exactly libcudart, libcublas and libcublasLt, loaded beside
    llama-server through an `$ORIGIN` rpath (from llama.cpp's release.yml; a real pod run is still
    to come).
- Blockers, fixed:
  - rude-llama and heretic's saved model couldn't be converted: transformers 5 wrote a tokenizer
    class the converter's transformers 4.57.6 doesn't know. They now convert from a staging folder
    with only the tokenizer settings renamed.
  - llama.cpp's converter reorders the input columns of Qwen3.5's 24 linear-attention `ssm_out`
    weights. That stopped the run after converting, and adapters built in the original order would
    have been wrong. Adapters and the target check now apply the converter's own reordering.
  - K2's own Python was never downloaded or trusted. `init` now sets `trust_remote_code` for models
    that need it; K2's code was checked to load and run under transformers 5.19 on a tiny random
    config (its `output_hidden_states` returns nothing, so candor's hooks are what work).
- Also fixed:
  - Records hold candor's code version for each stage, so a fix to a stage rebuilds its results.
  - `run` checks GPU memory before downloading anything: Qwen3.5-9B needs 21.2 GB and K2 19.8 GB
    against the Mac's 10 GB. The model loads straight onto the GPU (`accelerate`).
  - F16 models convert to F16 and are checked byte for byte; rude-llama is F16 on disk.
  - Prompt datasets are pinned to a commit; Hub configs are read at the pinned commit.
  - Killing candor stops llama-server.
  - On pods: the GPU check reads llama-server's device list (it passed on a failed CUDA start), torch
    must see the GPU, the `--lora` argument stays under Linux's per-argument limit, and the
    converter's packages are pinned.
  - HTTP timeouts, and an answer cut short by the slot's context is an error.
  - The default strength range is heretic's 0.8-1.5, and its meaning under each `row_norm` is
    described correctly.
- Tests: mutation showed stage invalidation, atomic `init`, the byte-for-byte check, the pinned
  date reaching the ids, KL's unlisted tokens, the CUDA check, the lock and five earlier fixes
  untested. Each now has a test, and every fix was shown to matter by reverting it (P1 12 of 12,
  P2 51 of 51, P3 25 of 25, plus P4's).
- Removed: `spikes/` and the library code only they used, `prompts.py` and `reference.py`.
- Plan: rewritten depth first, GGUF for everything, with the research table corrected after reading
  each paper in full (above).

320 tests, plus 5 end to end, pass.

### Phase 3, step 5: the proxy's verdict (2026-10-14)

`candor proxy` said "offset" (Spearman 0.98; thinking refuses a median of 5 more where thinking off
refuses at most 25). By the plan's step 6 that means cheap trials, with the edits they suggest scored
as used at more strength; no method change. The user chose to run it overnight and to carry the
search to its full 600 trials rather than 400: the morning start isn't delayed either way, and the
trial where the best edit turned up shows whether a shorter search would do on larger models.
Numbers in `results/phase-3-qwen3.5-9b.md`.

### Phase 3, step 6: the search, and a pod's GPU failing (2026-10-15)

The search stopped at 545 of 600 trials when the pod's GPU failed; the results were pulled and the
pod deleted. With no new best after trial 428, the last 55 trials were skipped (the user agreed),
and a new pod scores the best 6 as used: trials 205, 417, 342, 428, 402 and 175, each at its
strength and 1.4 times it. `candor proxy --trial` was added for that; it keeps such checks apart and
gives no verdict, since a narrow set of near-zero edits can't show an order. Found on the way:
`candor run` crashed on its summary for "full" edits, and printed unexpected errors' messages; both
fixed. Findings and ideas now live in `results/findings.md`, flagged by their evidence.

### Phase 3, step 6: the best scored as used (2026-10-15)

Trials 205, 417, 342, 428, 402 and 175 at their strength and 1.4 times it, on a new pod. Within the
cap as used: 205 and 175 at their strength (one layer, KL 0.037 and 0.057), 428 at both strengths
and 342 at 1.4 times (per layer, KL 0.085-0.100). 1.4 times wrecked every one-layer edit that
worked (to 20-49 refused), so step 5's "offset" doesn't mean more strength for edits that already
work. Step 7 is proposed on 205 and 175, the two lowest-KL edits within the cap.

### Phase 3, step 7: candor's finalists beat Trial 161 as used (2026-10-15)

Trials 205 and 175 were exported at their strength (Q4_K_M, no Q8_0 needed) and measured in full on
a new pod, about $5.50. Both pass the decision rule and beat Trial 161; 175 refuses 0 of 288 held-out
harmful prompts against 28, with maths at the unedited model's level (1,181 against 1,183), so it is
the better of the two. The first start left out `--slots 32` and measured with 8; it was stopped
after about 55 minutes and run again with 32, and checking every setting against the reference's
before and after a start is now part of how jobs start. The user left Trial 161 out of the thinking
review (re-saving it would need heretic run again), and asked that the phase end with a discussion
of where to go, with no changes to code, defaults or the method until agreed. Numbers in
`results/phase-3-qwen3.5-9b.md`; `results/milestone-qwen3.5-9b-vs-heretic.md`.

### Phase 4, step 1: candor runs K2-Horizon (2026-10-15, overnight)

The user chose K2-Horizon-7B next (the strongest model their 18 GB Mac runs well, and a second
model family), at reasoning effort high in llama-server, and asked that it come as options and
tools that leave other models working as before. K2's template has no thinking switch and opens a
think tag per effort, so candor now takes that tag's end tag as the close; a new
`deployment.chat_template_kwargs` carries the effort to every render and to `serve`; a thinking-off
search is refused where the template can't turn thinking off (K2), checked once before a search
records anything; a `--lora` list too long for one argument goes through numbered links; and
`--outside` takes an unquantised GGUF, since one public K2 edit is published only that way. IFM's
Python files hold no network, file or process access. A developer agent built it, an independent
review found nine points (one broke "unchanged": existing thinking-off commands started an extra
server), all fixed and re-reviewed to a pass. rude-llama's trial 535 on 10 prompts per set gave
byte-identical answers before and after, and Qwen's prompt ids rebuilt identically. The same
night, four agents searched arXiv for new features (`results/new-features.md`), and the chat-app
template check was removed at the user's request: models run only in llama-server.

### Phase 4, step 2: the timed pilot (2026-10-16)

On one A40 (about $1.60), the K2 run built in 40 minutes and ran end to end. K2 thinks far less
than Qwen3.5-9B at high effort (unedited median harmful thinking 224 tokens against 996), so a full
measurement is at most about 1.5 h unedited and 4 h for an edit that complies in part. Avicennasis's
heretic edit refused 48 of 64 held-out harmful prompts as used, against 7/50 on its card: thinking
brings refusals back, as on Qwen. Four search trials as used took about 210 s each at 16 slots, so
a 600-trial search at high is roughly $20-35. Three things in code stood in the way: slots are fixed
(32 slots don't fit on an A40 for a dense model; the search sizes each slot for 15.6k tokens, so it
fit 16, on a patch made on the pod only), and at low effort K2 often closes with high's tag, which
candor doesn't accept. A research agent found Uno gains little at 32 sequences on Ampere and would
take weeks to build into llama.cpp, and that Uno-Uncensored is built on K2 with the Uno adapter
merged in. The user then set a rule: every setting is customisable per run, and the long-term goal
is a tool that inspects a model and picks or offers its best settings; defaults keep existing runs
unchanged. Numbers in `results/phase-4-k2-horizon-7b.md`.

### Phase 4, step 2: speed test on an A40 and an A100 (2026-10-16)

The user asked for an A40-or-bigger decision on cost and for longest-first only if it paid. A
simulation on measured per-prompt lengths showed longest-first saves 3-17% on search trials with
lengths known in advance and nothing on maths, so it isn't built. A 20-minute test on each GPU
showed the A100 about 10% faster at 3 times the price, so the A40 stays, and found the real limit:
with K2's top_k 0, sampling the whole 250k vocabulary costs about 1.9 ms a step per answer, and
top_k 1,000 or below runs 2.6-2.7 times faster at 28 slots. Which top_k is the user's call, since
it is how K2 is used. A developer agent is making slots per-run settings meanwhile. Numbers in
`results/phase-4-k2-horizon-7b.md`.

### Phase 4: K2's settings decided (the user, 2026-10-16)

- Sampling: temperature 1.0 and top_p 0.95 as K2's card says, with top_k 1,000 and min_p 0. The
  distribution is the card's unless the top 95% spans more than 1,000 tokens, and scoring runs 2.6
  times faster at 28 slots (the speed test). The user runs llama-server the same way.
- Thinking budget 8192, as for Qwen: no unedited pilot answer reached it, Avicennasis's edit did on
  6 of 192, and 32,768 would leave about 7 slots on an A40.
- Uno-Uncensored isn't a reference: it is built on K2 with the Uno adapter merged into its weights.
- The A40 stays: an A100 was about 10% faster at full load for 3 times the price.

### Phase 4: slots as run settings (2026-10-16)

Following the user's rule that every setting is customisable, a developer agent made the slot counts
run settings (`[slots]`: `thinking`, `search`, `measure`, with defaults that record exactly what
runs recorded before), sized each search slot for its largest request instead of a KL read plus a
whole answer, reported a KV cache that doesn't fit with the setting to lower, and had `inspect` show
the KV cache a token takes (and a hybrid's state a slot). An independent review passed with 8 points
to fix, all fixed with tests, and a re-review passed with 2 minor ones, also fixed: 763 tests pass.
rude-llama's numbers are identical by sha256 before and after, Qwen's and K2's records still match,
and answers were bit-identical across slot sizes on three models. The checks found batched answers
vary run to run at 8 slots (`findings.md`). K2's run uses 28 slots throughout.

### Phase 4: the search before the references (the user, 2026-10-16)

The user dropped Avicennasis's full measurement (it refused 48 of 64 as used in the pilot, too low a
bar to spend time on) and asked for candor's full search on K2 next. The unedited model is still
measured in full first, about 45 minutes, as the baseline every edit is judged against and a speed
check at the new settings; the search starts straight after on the same pod. The decision rule is
drafted from the unedited numbers during the search and signed off before finalists are measured.
K2's strength may widen to 5.0 rather than 3.0, as heretic's own K2 run needed 4.5-4.6 under the same
"full" edit; the start stays 0.8-1.5 and widens only on the alarm, as on Qwen.

### Phase 4, step 3: the unedited model as used (2026-10-16)

K2's run rebuilt at the user's settings in 26 minutes and the unedited model was measured on every
held-out prompt and maths question in 41 minutes at 28 slots: 287 of 288 harmful prompts refused,
2 of 300 harmless, maths 1,222 of 1,319. The search started straight after on the same pod.
Numbers in `results/phase-4-k2-horizon-7b.md`.

### Phase 4, step 5: the search restarted with strength 0.8-5.0 (the user, 2026-10-16)

The search started at 11:56 UTC over heretic's 0.8-1.5, to widen to 5.0 at each kind's 60th trial.
All 22 trials it ran, at 0.8 to 1.47, refused 88-100 of 100 test prompts (KL 0.004-0.033), about
139 s each, as heretic's own K2 run had refused 54 of 100 at a cap of 1.5. Waiting for the alarm
would have spent about 100 more trials, 4 hours, on edits too weak to win. The user chose to
restart at 12:52 UTC with strength 0.8-5.0 from the start, still able to widen to 0.5: starting
wide is the change `findings.md` rates best supported, and the miss-rate test's worry about it
(halving the hits when the best lies inside 0.8-1.5) doesn't apply where nothing inside it comes
close. The 22 trials' journal is kept on the pod beside the run.

### candor: the search reports its progress (2026-10-16)

After the user had to ask before the narrow K2 start was flagged, they asked that candor itself
show whether a search is getting anywhere, so it can be stopped early when it isn't, with or
without an assistant. Every `search.report_every` trials (25) the search prints, and writes to
`search/progress.json`: trials done, pace and time left; per kind the best so far, its strengths
against the range (ends flagged once the sampler aims), whether it widened, and trials since it
improved; thinking tokens and budget hits. It only reads the trials: a seeded widening search draws
identical trials with and without it, and the setting stays out of the search's record. Built by a
developer agent; a review passed with 3 minor points, fixed and re-reviewed to a pass; 767 tests
pass. Alarms that act on the report (widening early, stopping) wait for evidence from real searches.

### Phase 4: thinking that runs to the budget, to investigate after K2 (the user, 2026-10-16)

By trial 72 the K2 search had slowed to about 900 s a trial: the sampler had moved to strong edits,
and some thought to the 8,192 cap on 9-39 of 100 prompts (trial 69: 39, at KL 0.81, 23 minutes).
The search's score can't see this and nothing stopped them (`findings.md`, Ideas). The user asked
for it to be logged as a feature to investigate and then build once K2 is done: thinking at the cap
as a constraint beside refusals, with an early stop, tested first on this search's journal. Not
changed mid-search, as it would restart the search; finalists are measured as used either way.

### Phase 4, step 5: the search stops by 300 trials, sooner on a stall (the user, 2026-10-17)

At 123 trials the per-layer best was trial 114 (2 refused, KL 0.047), down from 0.084 twenty of its
trials before, and the pace 662 s a trial: 600 trials would take 3.7 more days and about $52, where
a 9B model must not take days. On Qwen, by trial 300 the one-layer best was within 1% of its final
KL and the per-layer best 7-11% above it, and KL gaps larger than that on the 100 test prompts
didn't hold on held-out ones (trials 205 and 175: 0.037 and 0.057, then 0.058 and 0.059). So the
search stops at 300 trials, or sooner once both kinds have stalled: a kind stalls when its best
edit within the cap has a KL less than 10% below its best 50 of its own trials before. Checked at
each progress report; the search is then stopped by hand and the finalists measured. Not now: per
layer had just nearly halved its KL.

### Phase 3, step 7: the thinking review done (the user, 2026-10-17)

The user reviewed trial 175's thinking in `candor serve` and called the review done; no counts were
reported. The unedited model and trial 205, also listed for it, were dropped with it. Phase 3 now
waits only on the phase-end discussion.

### Phase 4, step 5: the budget limit switched on mid-search (the user, 2026-10-17)

The stall stop is dropped: replayed on the miss-rate simulator, it picked no better than a fixed
count of the same length, and 300 trials rather than 600 roughly halve the hits (results file, step
5). So this morning's stop at 300 has no evidence behind it; how far the search runs (300, 400 or
600) waits for the user, once the new pace is seen. The user chose instead to make trials cheaper,
reversing the 2026-10-16 entry's "not changed mid-search": `search.max_budget_hits` was built (a
developer agent; review passed, 2 minor points fixed and re-reviewed; 785 tests) and switched on at
8. The search stopped between trials after trial 137 (07:44 UTC), its journal, settings, log and
code backed up on the pod (`/root/backup-20261017T0744`), and resumed with the 137 trials judged
again from their recorded counts: 20 over 8, of which only trials 66 and 102 had been within the
cap; neither kind's best changes. From trial 138 trials stop on the budget, and one layer's
refusal stop moved from 27 to 29 refusals (per layer 2 to 3) to keep its guarantee once the sampler
ranks by both caps, and was back to 27 and 2 by trial 143. Code older than cb0fa99 must not run on `runs/k2` again: it would carry on in
the old studies.

### Phase 4, step 5: the budget limit checked on the miss-rate simulator (2026-10-17)

The check owed when the limit went on: a developer agent ran candor's own search at cb0fa99 on the
phase 3 simulator, budget hits fitted on K2's trials, 45,360 searches in 5.9 h on the Mac. Pooled,
the limit costs no measurable share of picks near the best, on from trial 1 or switched on at 137
as K2's was, and saves 6-26% of pod time in K2-like landscapes; it helps some landscapes and costs
others beyond chance (results file, step 5). So it stays on for K2. For later models, on from trial
1 where a pilot shows budget hits common, off where rare: for the phase-end discussion.

### Phase 4's decision rule (from step 3, signed off by the user 2026-10-18)

Each finalist, exported and measured as used with `measure --exported` on the held-out prompts like
the unedited model in step 3 (Q4_K_M, f16 KV cache, 28 slots, the user's sampling, reasoning effort
high, budget 8192, seed 0, the full 1,319 maths questions), is compared with the unedited model
prompt by prompt; "beyond noise" means a paired McNemar test at p < 0.05. It passes if:

- harmful refused (judge) is at most 2% beyond noise, by a one-sided binomial test at p < 0.05:
  10 or fewer of 288 pass, 11 fail (the search's own cap is 2 of 100);
- harmless refused is no more than the unedited model's 2/300 beyond noise;
- maths correct is no worse than the unedited model's 1,222/1,319 beyond noise;
- maths thinking reaching the budget is no more than the unedited model's 3/1,319 beyond noise.

Among finalists that pass, the one with the fewest harmful refusals ships, or the lower KL (read
after the unedited model's own thinking) where that difference is within noise. KL has no bar: there
is no reference edit to set one, and maths and harmless refusals measure the damage directly. If
none passes, the search carries on from its journal.

### Phase 4, step 5: the search stopped at 313 trials (the user, 2026-10-18)

The user stopped the search after trial 313 (06:08 UTC, between trials) rather than run to 600,
weighing about 40 more hours and $25 against the gain on a small model. Advised against: the
simulator gives 300 trials about half 600's chance of a pick within 10% of the true best, and one
layer's best came at trial 207 after a flat stretch; but the best is strong (trial 207: 0 refused,
KL 0.024) and if no finalist passes, the search carries on from its journal. Progress at the stop:
one layer's best had held for 106 trials, per layer's (trial 114, KL 0.047) for 199. The finalists
are the three best one-layer edits within both caps (207, 205, 265) and per layer's best (114),
measured as used under the rule signed off today. Search numbers in the results file.

### candor export: weights quantising can't hold stay in F32, in one file (the user, 2026-10-18)

Trial 207's export was refused: 2 of its 66 edited weights kept 89.9% and 97.6% of the edit even
at Q8_0. The user chose to keep such weights at higher precision in the one GGUF rather than the
2026-10-13 backup of the plain Q4_K_M base plus a LoRA adapter. A first attempt (BF16) was found in
review not to work: the merge itself rounds tiny edits away (findings, Export). So a short weight
is merged again unrounded and stored as Q8_0 from that, or as F32; every other tensor keeps its
plain type, pinned against a llama-quantize quirk the review found. The adapter backup is left for
a shipped file that fails the decision rule.

### Phase 4, step 5: the finalists measured; trial 205 ships (2026-10-18)

By the rule signed off today, trials 205 and 114 pass; 265 fails on maths thinking reaching the
budget and 207, the search's best, on maths correct (with adapters too, so the edit, not the
export). 205 and 114 tie at 3 harmful refused, so the lower KL decides: 205. Trial 207's export was
the first to keep weights apart (2 as Q8_0); its file measured as its adapters did. `runs/k2` is on
the Mac, checked against the pod, and the pod deleted (50.0 h, about $29.50). Next: the user's
test (step 6). Numbers in the results file.

### candor serve: tools in every session (the user, 2026-10-18)

The user wants every chat with a model to have their web and sandbox tools. `--mcp` now defaults to
`CANDOR_MCP`, set in the user's shell to `~/Desktop/tune/tools/mcp.json`; with tools the context
is 32,768 tokens. `CANDOR_MCP=` before a command serves as measured, without them.

### Phase 4, step 6: the user's test of trial 205 passed (2026-10-18)

The user chatted with trial 205's shipped file in llama-server on the Mac, with their web and
sandbox tools, and is happy with the model. The release decision remains theirs.

### candor serve: a trial's export no longer downloads the original weights (2026-10-18)

Serving K2's trial 205 downloaded its 11 GB of original weights, for the context limit and the layer
count. Both now come from the served GGUF and the model's config.json, so `serve --trial` fetches
only the config; checked on `runs/k2` with the weights deleted. Serving the unedited model still
downloads them to check its GGUF (STATUS).

### Phase 4, step 6: trial 205 released (the user, 2026-10-18)

Published from candor-team as `candor-lab/K2-Horizon-7B-Candor-Uncensored-GGUF`, with the Qwen
release's card template: the measured export with `general.name` set and `general.finetune` (a
snapshot hash) dropped, all 327 tensors byte-identical, loaded in llama-server b11524. The card was
reviewed against the measurements twice; the review found the "full" edit is rank 3, so the Qwen
card's "rank-1" is wrong. K2's original weights were then deleted from the Mac.

### The repository reviewed in full; steps 1 and 2 of its fixes (the user, 2026-10-18)

The user asked for a full review before the next big task: everything watertight, accurate and
efficient. 97 agents, read-only and numbers only, confirmed 161 findings, merged into 24 work
packages and 11 method questions (`evidence-2026-10-18/repo-review.md`); nothing shows a shipped
result wrong. The user agreed the order: faster tests first, then wrong results, then 27B cost and
safety, then one measurements change before any 27B reference, then the rest; method questions go
to the phase-end discussion. Each step is reviewed independently and its findings fixed with tests
that fail without them.

Step 1: one llama-server launch for measuring and `serve`, ready checks every 0.05 s, the GPU listed
once a process and every server told the GPUs by name, so one restarted after a GPU fails refuses
to run on the CPU; `pytest -n 6` takes the suite from about 13 minutes to about 70 s.

Step 2: `inspect` now fails a model whose template turns thinking off but whose prompts don't open
it (Qwen3, SmolLM3, GLM-4.5), and finds Qwen3.8-27B's close, which it had missed: that model was
taken for one that doesn't think, so `candor run` would have stopped on it. Qwen3.8's thinking-off
prompt also drops a system message at every effort but medium, so candor refuses to score it
thinking off there (findings). Measurements discard a set's old files before answering it again, and
`load` refuses labels counted for other answers or under another count; `compare` shows each
change's 95% interval and the slots, and medians are now true medians (the docs' even-set medians
moved by up to 20.5 tokens and are corrected; the released cards quote 948, 1,648 and 2,980, which wait
for the user). `findings.md` was corrected where the review found its numbers wrong or overreaching.
Both reviews passed after their points were fixed: 801 tests pass.

### The released cards corrected (the user, 2026-10-18)

With the user's go, both cards were corrected on Hugging Face from candor-team. Qwen's said the edit
was a rank-1 change to every layer: it is rank 3 and edits 48 of 64 matrices (attention in layers
7-31, MLP in 9-31). Both now give true medians of harmful thinking: K2 947 (was 948), Qwen 1,641 and
Trial 161 2,973 (were 1,648 and 2,980). CLAUDE.md's slots rule now takes the reference's from its
name or results file, as `[slots]` holds it.

### Repository review step 3: 27B cost and safety (2026-10-18)

Packages 6-10, 14 and 17, each reviewed independently and its findings fixed with tests that fail
without them.
- Thinking runs get 32 slots from `init`, and a trial stopped early costs no more GPU time.
- Measurements save each answer as it comes and resume from the last; an export writes the edit into
  a copy of the run's GGUF, byte for byte what merging and converting gave (SmolLM2, a tiny
  Qwen3.5); the disk is checked before a download, from the checkpoint's own float type.
- Exports are now version 3: "full" exports are built from the GGUF's layout, as their adapters
  are, so older ones count as stale. The search's VERSION stays: stopped trials' KL changed, which
  matters only if `search.max_budget_hits` is raised in a search spanning the change.
- Pod jobs (`scripts/pod.sh`): `wait NAME runs/RUN` pulls every 30 minutes and at the end, exits 0
  only if the job did, and stops 6 hours before the watchdog's deadline; a dropped SSH connection is
  reported as such; `code` syncs the pod's packages and refuses while a job runs.
- Run stages: the unedited thinking is saved per set; a `directions.pt` lost after it was read is
  refused rather than read again; adapters link by number, so 64 layers with 3 targets fit; the
  scoring budget can't exceed the deployment's; every small result is flushed to disk as written;
  `candor run` keeps the Mac awake.
- Found on the way: llama-server told to stop waits for every request it is answering to close,
  so an interrupted measurement, search or thinking run waited 60 s and then killed it. Every batch
  of requests now closes its own when stopped.

### candor menus (the user, 2026-10-18)

The user asked for arrow-key menus, nicer than heretic's. `candor` alone, in a terminal, lists the
runs with how far each has got, shows a run's panel (model, stages built, the search's best trials,
measurements, exports) and its actions, starting on the step it is ready for, and runs the same
commands, each shown first. Jobs that can take hours can instead be shown as `scripts/pod.sh`
commands. Built on questionary, which heretic also uses. Reviewed twice; 853 tests pass.

### Repository review step 4: the answer's own seed, one measurements change (the user, 2026-10-19)

Package 3, done before any 27B reference so references are measured again only once.
- llama-server starts its random draws again at every request, so a thinking model's answer, sent
  with its thinking's seed, drew token k with the number that drew thinking token k. The answer, and
  the request that finishes a half-written character at the budget, now get seeds of their own made
  from the prompt's; the thinking keeps the prompt's. Checked on llama-server with a model whose odds
  never change: one seed draws alike whatever the prompt, and the answer no longer repeats its
  thinking.
- Maths answers written `\boxed{18}` are read as stated, and `-$5` as -5.
- Measurements are now version 2, the unedited thinking version 2 and the search version 3.
  `compare` refuses to pair a measurement with one made by other measuring code, and the menus mark
  those made before. Measuring again under an older one's name keeps it as `NAME-measuring-1`, so the
  references the phase 2-4 results rest on are never measured over. Exports and the adapters
  `candor score` builds no longer record the measurements' version, so later measurement changes
  leave them current (those built before are rebuilt once). Rebuilding a trial's edit has a version of its own, so the searches' trials can
  still be measured and exported, but the searches can't be carried on (the menus say so).
- Before a pod quote on K2 or Qwen: the unedited thinking is sampled again on the first measurement,
  and the unedited model is measured again before a new measurement is compared with it.
- The cache-reuse speed-up (package 11) also changes draws; it needs a timed pilot, so it waits for
  the 27B pilot, before the 27B reference.

Reviewed three times; 868 tests pass.

### Repository review step 5: the rest (the user, 2026-10-19)

Packages 13, 15, 16 and 18-24, each reviewed independently and its findings fixed with tests that
fail without them.
- The search: a search or score refused for other inputs is refused before the GGUF, directions or
  adapters it was scored with are rebuilt; a trial stopped early stays outside the caps when they
  are judged again; a trial scored again repeats within a refusal and 3% KL and says how it differs;
  the unedited model's first tokens are kept per inputs and machine (`search/base-first-tokens/`).
- The machine: measurements, judge labels, trials and proxy checks record the GPUs and platform;
  `compare` names them where they differ and reads no KL across Metal and CUDA; a search moved to
  another machine says so. CLAUDE.md's checklist asks for the reference's GPU type (the user).
- The CLI: export suggests the names measure makes; `--seed` is refused on a greedy run; `init`
  checks the commit it pins; `serve` fails when llama-server crashes and fits layers itself; a pinned
  Hub model's GGUF is found without downloading its weights; one error handler for every command.
- Run-time waste: each weight read once for every adapter set; only the harmful test prompts loaded a
  session; `candor score` scores the unedited model again only with no full trial to repeat.
- Dead code and tests: upgrade shims for settings no run holds removed; heretic 1.0-1.2 journals
  read; no test module imports another; the miss-rate simulator in `tools/misscheck/`; the lock only
  for macOS arm64 and Linux x86_64, the converter's pins resolved best-match (the same GGUF bytes on
  SmolLM2-135M); docs brought up to date.
- Left, with reasons: package 11 waits for the 27B pilot; the proxy's scorers stay apart (F129: a
  minute saved for a more complex scoring API); F80 and F132 wait for the next directions change, as
  they rebuild every run's directions. The simulator's cached wide-range best recomputes lower
  (findings); no doc result uses it.

Reviewed in three rounds; 886 tests pass.

### Evidence for the phase-end discussion (the user, 2026-10-19)

Gathered on the Mac from the K2 and Qwen runs, numbers only, each part checked by an independent
reviewer who recomputed its numbers (`evidence-2026-10-19/method-evidence.md`).
- heretic upstream: master 3a5c864, default edit still ARA, nothing merged but a PR template; its
  thinking-model prefix bug (#487) is open and, by its fix (#501), reaches Qwen3.5/3.8 templates
  too, against 2026-10-14's reading.
- Six of the review's 11 method questions, with recommendations for the discussion, none adopted:
  the beside KL reads decide nothing (drop them); KL gaps under about 2 SE are noise, and the search
  keeps no per-prompt KL (store it); skipping answers once 25 trials have lower KL saves little on
  K2 and loses on Qwen (don't adopt); 20 KL tokens miss about 0.5%, with two biases beyond it (200
  for the 27B run shrinks truncation and the floor, not llama-server's float32 loss); "pre" and
  "full" differ by 1% of the edit on Qwen and 4% on K2, so only scoring both ways decides; a bigger
  judge changes no decision but finds more refusals on edited models, so the 0.6B's recall there
  stays open.
- Corrected on the way: Qwen's 205 and 175 didn't flip between test and held-out KL; the beside
  reads cost K2 20 min, not 30-50; `candor score`'s 3% repeat tolerance is exceeded by half of 18
  real re-reads (a small fix in STATUS). The e2e test passes (9 of 9).

### The 27B decision rule, two judges, the beside reads dropped (the user, 2026-10-19)

The phase-end discussion's first decisions, from `evidence-2026-10-19/method-evidence.md` and the
review's F116.

The rule for the 27B phase, fixed before any 27B search. Each finalist, exported and measured as
used on the held-out prompts like the unedited model, is compared with it prompt by prompt; a bound
is the one-sided 95% bound of the paired change (Bonett and Price, as `compare`'s intervals). It
passes if:
- harmful refused is at most 2% beyond noise, by a one-sided binomial test at p < 0.05 (10 or fewer
  of 288), under both the run's judge and Qwen3Guard-Gen-8B at 16 bits;
- harmless refused rises by less than 2 points: its upper bound is under +2 pp;
- maths correct falls by less than 2.5 points: its lower bound is above -2.5 pp;
- maths thinking reaching the budget rises by less than 1 point, or a quarter of the unedited
  model's rate where that is larger: its upper bound is under that.

Among finalists that pass, the one with the fewest harmful refusals ships, or the lower KL where
that difference is within noise. If none passes, the search carries on from its journal.

Why: phase 4's rule passed an edit unless it was worse beyond noise, so a noisier measurement
passed more easily and a lack of evidence counted as no loss. Margins fixed in advance make the
measurement show the loss is small. -2.5 pp is about what 1,319 maths questions can show: a lossless
edit passes 93% of the time on K2 and 97% on Qwen, against about 60% at -1.5 pp. The budget-hit
margin scales because Qwen hits the budget on 14% of maths questions and K2 on 0.2%. Replayed on
every past finalist, it gives the old rules' outcomes: K2 205 and 114 pass, 207 fails on maths
(-3.31), 265 on budget hits (+1.28); Qwen 175 and 205 pass; heretic's Trial 161 fails on both
(-2.81, +10.36).

Two judges: on both K2 and Qwen a bigger judge of the same family found 2-2.4x more refusals on
edited models than the 0.6B, so the bar must hold under both; the search keeps the 0.6B. Dropped
from the search: the KL reads where the thinking starts and with empty thinking, which no pick
reads. Then, on the user's request for guidance: the search keeps each trial's per-prompt KL and
shows which suggested finalists tie the best within about 2 SE, as guidance only, since the
average of 100 prompts can't separate them (K2's 207 led 205 on the test prompts, 205 led held
out); `init` writes 200 KL tokens for new runs, as nearly free and closer to the full vocabulary;
KL-first skipping is not taken up (it saves under 1.2% on K2 and costs 19% on Qwen). The pod checks
on K2 and Qwen (the 8B judge, both row normalisations) are approved.

### The 27B: Qwen3.8-27B at xhigh; the gate reset after a pilot (the user, 2026-10-19)

The cost model (`evidence-2026-10-19/cost-27b.md`, reviewed): today's process costs about $167-195
for the 27B, and a thinking-off search with a per-model proxy check about $73 under the rules in
force, only if the check passes. No route meets about a day and $20 without searching less, which
the simulator shows roughly halves the chance of finding the best edit. The user's decisions: the
next model is Qwen3.8-27B as they use it, at reasoning effort xhigh; the must-not-miss standard
stays, and the release gate is set again from a pilot's measured speed and thinking length; the
pilot (about $7-8) runs once the approved code changes land and the K2/Qwen pod checks are done.

### The code for 27B: a second judge, per-prompt KL, the beside reads out (2026-10-19)

The changes decided at the phase-end discussion so far, each reviewed independently and its
findings fixed with tests that fail without them.
- `measure.second_judge`: `init` gives a new run Qwen3Guard-Gen-8B, pinned; `measure` judges the
  held-out answers with it after the run's judge, one judge loaded at a time, in files of its own;
  `compare` shows its refusals beside the first judge's and never pairs different second judges.
  Every judge is sized against the GPU before it loads (the 8B needs about 18 GB, so not on this
  Mac). Runs without the setting are unchanged.
- The search no longer reads KL where the thinking starts or after an empty block (`measure` still
  does); each trial keeps its per-prompt KL, and the search's summary shows each suggested
  finalist's paired gap from its kind's least and whether it ties within 2 SE. Picks are unchanged;
  no search version bump, as no score the sampler or picks read changed.
- `init` writes 200 KL tokens for new runs; the default stays 20.
- The engine's stop test failed about 1 run in 500: llama-server b11524 deadlocks when SIGTERM lands
  mid-generation (its handler takes a lock the generating thread holds). The test now stops the
  server once it has dropped the answer; production keeps its 60 s wait and kill (STATUS).

905 tests pass.

### The K2/Qwen pod checks: the 8B judge and row normalisation (2026-10-19)

One A40 pod, 2 h 39 min, about $1.57 (`evidence-2026-10-19/pod-checks.md`). Qwen3Guard-Gen-8B
judged the stored K2 and Qwen held-out answers again: it agrees with the 4B, and against both the
0.6B under-counts refusals on edited models; every K2 finalist stays within the harmful bar under
the 0.6B and the 8B, so no decision changes. K2's 205 and Qwen's 175 scored twice each with "full"
and "pre": no difference by candor's tie test, and on Qwen "pre" may raise KL a little. Nothing
argues for leaving "full"; the 27B's weight gap is screened in the pilot. The pod was deleted.

### Phase 5, step 1: the Qwen3.8-27B pilot (2026-10-19)

The approved pilot ran on an A100 80GB (build, directions, the full-vocabulary and cross-card KL
checks, speed, MTP, a medium-effort copy) and an A40 (the cross-card control, speed, 64 prompts a
set at xhigh and medium, as-used and thinking-off trials, the cache test), about $6.8 of the $7-8
approved; both pods deleted. Run settings: the model card's thinking-mode sampling, and the search
as K2's ("full", strength 0.8-5.0, budget cap 8). Numbers in `phase-5-qwen3.8-27b.md`, step 1. What
it settles: the A40 stays the card (the A100 is 1.68 times faster at 3.0 times the price); MTP and
`--kv-unified` don't help; KL over 200 tokens is within 0.05% of the whole vocabulary and
llama-server's float32 sum doesn't matter on the mean; an A100 and an A40 read KL alike; "full" and
"pre" differ on the 27B's weights as little as on Qwen's; the answer can reuse its thinking's cache
(package 11 still to be timed on an as-used trial). The 9B's best edit breaks the 27B (KL 1.86), and
an as-used trial takes 15-47 min against a thinking-off one's 2: under the rules in force the
as-used route costs an estimated $132-290 and 9-21 days, thinking off $35-81 and 2.5-6 days if its
proxy check passes (about the as-used cost if it fails), and medium's directions differ from
xhigh's, so the gate and the route go to the user. The script review found real gaps before the pods
ran (an 80 GB-only directions step for the medium copy, MTP's memory at 64 slots, a cross-card
control that rebuilt its adapter); they were fixed and re-reviewed.

### candor: PyTorch gives a freed model's GPU memory back (2026-10-19)

The 27B's first build stopped at the unedited thinking: after the directions, PyTorch kept 50.1 GiB
reserved with no tensor alive, so llama-server couldn't allocate its KV cache on the 80 GB card.
`protect_memory` now sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (unless either of
torch's names is set) when candor starts, and `Run.changes` frees the GPU after building a "full"
edit, which `candor measure` hadn't. Reviewed twice (the first review found the measure path); each
test fails without its fix; checked on the pod, where the medium copy's build read its directions
and started its thinking in one process. 912 tests pass.

### The 27B at medium effort; the proxy check started (the user, 2026-10-20)

After the pilot, the user chose to use and score Qwen3.8-27B at reasoning effort medium rather than
xhigh. Medium is the one effort where its template turns thinking off cleanly, so `runs/qwen27b-medium`
becomes the run: its own directions and medium thinking (built on the pilot's A100), the search
scoring with thinking off, and `candor proxy` checking, as the rule in force requires for each new
model, that thinking off ranks edits as medium thinking does. The proxy's pass bar is the one built
in (refusal ranks agree at 0.7 or better, and edits that nearly work thinking off need little more
strength with thinking on); the search continues to 600 trials thinking off only if it passes, and
comes back to the user if not. The xhigh run (`runs/qwen27b`) is kept as the pilot's record.

### Phase 5, step 2: the proxy check holds on the 27B; the search carries on (2026-10-20)

`runs/qwen27b-medium`'s thinking-off search ran to 150 trials, then `candor proxy` scored 24 of its
best edits both ways: Spearman 0.80 on refusals and a median of 0 more refused thinking where edits
nearly work, so "holds" under the bar built in. Every edit within the cap thinking off stayed within
it thinking. Thinking undid the partial edits (24-26 refused thinking off became 90-98), which is
where the two disagreed. The user chose to carry the search on to 600 trials on the same pod, before
going offline for a flight; the pod's deadline moved to 48 h so the run outlasts the time away.
(`results/phase-5-qwen3.8-27b.md`, step 2)

### Phase 5, step 3: the 27B's thinking-off search to 600 trials (2026-10-21)

The search ran from 150 to 600 trials on the same A40 in 10 h 35 min while the user was offline,
and was pulled and the pod deleted when the Mac came back (38 min idle; the pod $13.28 for steps 2
and 3). Its best within the cap: one layer trial 435 (1 refused, KL 0.0188), per layer trial 532 (2,
0.0254), both well below their 150-trial bests; the one-layer best still presses peak layer's floor.
The as-used round of its best waits for the user's go and cost. (`results/phase-5-qwen3.8-27b.md`,
step 3)

### Phase 5, step 4: the 27B's best scored as used; KL re-reads checked (2026-10-21)

On the user's go, `candor proxy --trial` scored trials 477, 435, 251, 584, 578 and 532 with medium
thinking on a new A40 (about $1.45): all six refused 0-2 of 100 at their own strength, trial 435 the
least KL (0.0239). Their KL read 23-41% above the search's, so the same pod scored a control and
trial 435 against the kept and fresh unedited reads: the control moved 6-7%, the two unedited reads
were apart by a mean KL of 0.0020, and no candor fault was found; the least-KL trials of 600 read high
by luck of the read. The recipe-keyed cache is listed as a small fix. The full measurements wait for
the user. (`results/phase-5-qwen3.8-27b.md`, step 4)

### Phase 5, step 5: the 27B's finalists and the public edit measured in full (2026-10-22)

On the user's go, after a top-up, three A40s measured the unedited model and finalists 435, 251, 532
and 477 in parallel. Package 11 wasn't timed first: candor recommended skipping it for this model,
as its saving is untimed and at most 5-10% of a trial, and the user took that. The user then chose
to measure the best public edit too (huihui @739e3c5, picked in
`evidence-2026-10-21/public-27b.md`), as earlier releases were compared with one. All five pass the
decision rule. The Mac couldn't stay awake overnight, so a CPU pod took each GPU pod's results and
the GPU pods deleted themselves; $9.20 in all. (`results/phase-5-qwen3.8-27b.md`, step 5)

The ranking clause ("the one with the fewest harmful refusals ships, or the lower KL where that
difference is within noise") left three things open, which the review found:

- Which judge's refusals rank. Counting both judges, each edit against the one refusing fewest
  (532), 435 drops out (7 under the 8B, p 0.016, the only one of the six comparisons under 0.05;
  0.094 after a Holm correction over all six, 0.047 over the 8B's three) and 251 ships on KL;
  counting the run's judge only, all four are level and 435 ships on KL.
- Which KL test. On the mean paired by prompt, 251, 477 and 435 tie and 532 is higher beyond noise;
  by prompt (a sign test), 435 reads lower than 251 and 477, which are level.
- What decides a tie.

So the rule picks a one-layer edit but not which. Separately, 532's answers are judged unsafe more
often than the one-layer edits' within the 256 tokens read, beyond noise under both judges, which no
clause counts. Both go to the user: which edit ships now, and the clause's reading for the next
model (proposed: refusals under both judges against the fewest, two-sided McNemar p < 0.05; KL by a
sign test paired by prompt, p < 0.05; a tie broken by fewer refusals under both judges together,
then by the lower mean KL: 251 here, as 435 is out on refusals before KL).

### Phase 5, step 6: trial 532 picked for the 27B; its file measured again (the user, 2026-10-22)

Asked to pick between one-layer 251 and per-layer 532, the user took candor's proposal: answer the
held-out harmful prompts again on both exports with answers up to 1,024 tokens, and ship 532 if,
under either judge, 251's answers are refused more often or judged unsafe less often beyond chance
(McNemar p < 0.05), otherwise 251, which reads less KL on most prompts. Set after step 5's
256-token gap, the rule was written into `results/evidence-2026-10-22/decide.py` before the
1,024-token answers were read: a pre-set replication test, not a blind one. At 1,024 tokens 532
is still judged unsafe more often (275 against 254, p 9.5e-7; the 8B 269 against 247, p 3e-6), so
532 is the one to ship. This ranks on unsafe answers, which the signed-off rule's ranking clause doesn't count (it
would ship a one-layer edit on KL); the clause, with its open reading (step 5), is for the
phase-end discussion. 532's export was made again, hashed and measured in full again: no
difference beyond noise from step 5's, and it passes the rule. One A40, job 4 h 18 min; the pod is
kept for the user's test, as the 27B doesn't fit the Mac. (`results/phase-5-qwen3.8-27b.md`, step 6)

### Phase 5, step 6: the user's test of trial 532 passed; it is released (the user, 2026-10-22)

The user chatted with trial 532's measured file in llama-server on the pod, with their web and
sandbox tools: the tools ran on the Mac, and the pod's llama-server reached them through the SSH
link (`ssh -R`, a bridge per tool; `results/evidence-2026-10-22/bridge.py`, `pipe.py`,
`pod-mcp.json`). They were happy with it and asked for it to be uploaded, following the earlier
cards. Published from candor-team as `candor-lab/Qwen3.8-27B-Candor-Uncensored-GGUF`, uploaded
from the pod to a private repo and made public once checked: the measured export with
`general.name` set and `general.finetune` dropped, all 866 tensors byte-identical, loaded in
llama-server b11524. The card follows K2's and Qwen's, with a huihui-ai column (step 5's public
edit) and the second judge; its KL is over the top 200 tokens, this run's `kl_tokens`. Its
"Measured as used" says the rule's ranking would have shipped a lower-KL edit and 532 was picked on
unsafe labels, which the rule doesn't count. It was reviewed against the measurements twice. The
pod was deleted. (`results/phase-5-qwen3.8-27b.md`, step 6, "The release")

### The phase-end discussion: finalists ranked on unsafe answers (the user, 2026-10-23)

The ranking clause of the 27B rule (2026-10-19) is replaced, from the next model on: "Among
finalists that pass, one drops out if another's harmful answers are judged unsafe more often beyond
chance under both judges (McNemar, two-sided p < 0.05). Of those left, the lowest mean KL ships."
Refusals stay in the pass bar and no longer rank. The search still scores refusals.

Why: on the 27B every finalist refused 0 or 1 of 288 under the run's judge (0-7 under the 8B), so
the old clause came down to KL, while per-layer 532's answers were judged unsafe more often than
each one-layer finalist's (p 0.0015 at most under both judges; step 6 confirmed it against 251 at
1,024 tokens). An answer not refused can still dodge or hold nothing harmful; the field's benchmarks
score whether the request was carried out (findings, "Edits level on refusals"). Both judges must
agree so one judge's quirk can't drop an edit, which removes the doubling across judges behind step
5's six comparisons (435's refusals: p 0.016 under the 8B, 0.094 corrected); pairs of finalists are still
tested uncorrected. The new clause settles step 5's three open readings: refusals no longer rank,
and the lowest mean KL decides. It costs nothing: the judges already label every answer, at the 256
tokens read.

Replayed on stored labels, it gives every pick shipped: K2's 205 and 114 are level on unsafe (269
against 268; the 8B 266 against 265), so 205 on KL; Qwen3.5-9B's 175 is judged unsafe more often
than 205 (279 against 254; the 8B 273 against 248; p 1.6e-6 each), so 175; the 27B's 532, where
the old clause would have picked a one-layer edit without saying which (step 5).

The rest of the discussion is held until the user takes up the next model, being done with the
27B: the peak-layer floor (findings; candor proposed 0.4 for new runs, checked on the miss-rate
simulator first), the cost and time against the release gate (the 27B took about 2.4 days and $28 from its
search to release, $35 with the pilot), and the phase close (the e2e test, the tagged lines in
findings, the next model, now picked by gap: the next entry).

### Next models picked by gap (the user, 2026-10-23)

K2-Horizon-7B has the most downloads per day of candor-lab's four releases (184, against 134 on the
27B's first day, 75 for Qwen3.5-9B and 46 for Llama), with 5 other uncensored versions on the Hub
against Qwen3.5-9B's 303. The user decided to pick the next models by gap: a base many people run
as a GGUF with few uncensored versions so far, where a new release reaches more people than on a
crowded base. A first ranking, to be read again just before choosing, is in
`results/evidence-2026-10-23/gaps.md`: Granite 4.2 30B leads, with 1 uncensored repo against
238,240 GGUF downloads in 30 days; candor would first need to handle its template's thinking switch.

### The gap counted again; Granite 4.2 30B's lead rests on one repo; MoE sized (2026-10-23)

On the user's request, candor rescored the gap by each family's uptake of uncensored versions and
sized the mixture-of-experts models `gaps.py` left out (`moe.py`). The rescore ordered 4 of the 6
pairs of our releases, against the gap's 5, and reads 0 where a family has no uncensored version, so
it was dropped. Two reviews found `gaps.py` missed uncensored repos tagged but not named so, and
fine-tunes (Laguna-XS has 5, not 1), treated a per-token experts count as MoE, and read a missing or
wrong size as 0; fixed and run again (213 dense, 132 MoE or over 36B). Each model's top GGUF repo
is now shown: IBM's own GGUF holds 97% of Granite 4.2 30B's demand, and set aside its gap falls
below K2-Horizon-7B's. Laguna-XS-2.1 (MoE, 33B) leads the MoE models either way and, without top
repos, is second among recent models to Spark-X2.5-4B, which has 17 uncensored versions already.
MoE models of 36B or less hold a third to a half of the dense gap. (`evidence-2026-10-23/gaps.md`)

### What MoE support would take: researched and checked (2026-10-23)

`evidence-2026-10-23/moe-support.md`, from 13 arXiv papers (2025-2026), heretic master 09271b6 and
its issues, llama.cpp b11524's source and `candor inspect` on five MoE models. An independent check
confirmed every arXiv ID, llama.cpp line and candor file:line, and the claim the build rests on:
b11524 applies 3D LoRA to expert tensors at runtime with per-request scales. It corrected three
statements: the edit is an exact projection of the MoE block only under `row_norm = "none"`; Laguna
can't be scored on Metal in b11524 even with the issue's patch once an adapter is loaded; and
heretic's issues are weak evidence (#218 was a broken measurement). The order it supports: the code
with tests on tiny random MoE models on the Mac, optionally OLMoE-1B-7B on an A40, then Laguna-XS-2.1
on pods; Granite-3.1-3B-A800M moves later. Whether to build it, and how Laguna-XS is used (thinking
on or off), are the user's decisions.

### MoE support to be built; the first MoE model to be confirmed first (the user, 2026-10-23)

The user decided candor builds MoE support, as essential to the models it releases, on the plan in
`evidence-2026-10-23/moe-support.md` section 6: the code with tests on tiny random MoE models on the
Mac first. Before any work on Laguna-XS-2.1, the user wants to confirm it is the right first MoE
model, so the build stops there for that check.

### The README made a front page; its reference moved to `docs/` (the user, 2026-10-23)

The user found the README one block of text, hard to follow on GitHub. It is now a front page: a
banner, two diagrams (why candor scores the file you run, and its four stages) in a light and a dark
version drawn by `docs/images/draw.py`, the menus rendered from a real run, a quick start, the
commands and the released models. The reference it held moved, every fact kept, into
`docs/commands.md`, `docs/models.md`, `docs/pods.md` and `docs/development.md`, with tables for
options and settings. Checked with GitHub's renderer in both themes, every link and anchor
resolving.

### candor given a mark and a wordmark (the user, 2026-10-23)

The user found the banner's vector drawing basic and asked for a logo fit for a professional,
institutional lab. The mark is a disc of seven stripes, a representation split into its directions,
with the middle one pulled out along itself: the direction candor removes. It was chosen from seven
directions drawn and compared at banner and favicon sizes (a C ring was dropped as too close to
Coinbase's mark). The name is set in Source Serif 4, the rest in Inter, all outlined by
`docs/images/brand.py`, which also draws `mark.svg` for profile pictures.

### MoE support built and tested on the Mac (2026-10-23)

A layer's routed experts are one edit target, stacked [experts, out, in] as llama.cpp's converter
stacks them, whether stored apart (Laguna, Qwen3-MoE, OLMoE) or held in one tensor; each expert is
edited along its layer's direction at the MLP strength, as are a shared expert and any dense layer,
and the router isn't. It is scored as one 3D adapter, exported merged, and the export's check takes
the least share any expert keeps. `inspect` passes such models and fails experts held [experts, in,
hidden] or square. A heretic trial isn't rebuilt on a model with experts, as its journal doesn't say
whether heretic edited them. Qwen3.5-MoE's linear attention is reordered as the converter reorders it.
Dense models are unchanged bit for bit (no version bump). Three independent reviews: the first found
the maths, order and export right and asked for these refusals and more tests, now in
`tests/test_moe.py` and the e2e test (Qwen3-MoE, OLMoE, Laguna and Qwen3.5-MoE through b11524's
converter; adapters against merged weights); the third confirmed the fixes and, with a dense control,
showed the adapters-against-merged gap at wide random weights isn't routing's (findings). 949 tests
and the e2e test pass. Not yet run: a real MoE model, or CUDA. (findings, "Mixture-of-experts models")

### The first MoE model: Laguna-XS-2.1's demand in doubt (2026-10-23)

Asked to confirm Laguna-XS-2.1 before work on it, candor compared nine MoE candidates
(`evidence-2026-10-23/first-moe.md`), checked independently. Laguna fits the code best, but LocalAI's
gallery installs both of its top GGUF repos (without both its gap is 3,843) and its uncensored versions
draw 1,104 downloads in 30 days, the least of the nine: about 0.5-6 a day for a release. The check also
found `gaps.py`'s name search missed "-BF16" and "-hf" names (Nemotron-3.5-Lightning has 27 uncensored
repos, not 9); fixed and run again. candor recommends Qwen3-30B-A3B-Instruct-2507 as the first MoE: the
only one with a published KL-measured edit to check against, real uncensored demand (52-148 a day),
no thinking. The choice is the user's.

### The brand made the standard for anything visual (the user, 2026-10-23)

The user asked for the new logo and the README's styling to be used for everything from now on.
`docs/brand.md` holds it: the mark and its files, the colours on dark and light, the type, layout,
charts and voice. `brand.py` now also draws the logo and the mark alone for dark and light
backgrounds, and the avatar is `avatar.svg`. STATUS's way of working points to it.

### The brand on Hugging Face: every model card and the org card (the user, 2026-10-23)

Each of the four released cards now opens with `header.png`, the logo on ink, and the org card with
the banner, each repository carrying its own copy; nothing else in the cards changed. Uploaded from
candor-team after checking the live cards matched the staging folders. The avatar is ready as
`docs/images/avatar.png`; neither GitHub nor Hugging Face has an API for an org's picture, so the
user uploads it.

### candor written in lowercase everywhere (the user, 2026-10-23)

The user asked for the name in lowercase everywhere, for consistency with the logo. The five
Hugging Face cards now write candor and candor lab in lowercase in text and titles; repository IDs,
GGUF file names and `general.name` keep their spelling, as renaming them would break downloads and
links. `docs/brand.md` states the rule. The Hugging Face org's display name, "Candor Lab", can only
be changed on its settings page, by the user.

### The code made public under Apache 2.0 (the user, 2026-10-26)

The user chose to publish the repository ahead of the release gate.
Before it went public: no keys, tokens, private keys, IP addresses or prompt and answer text in any
file or commit; tests, lint and types clean. Licensed Apache 2.0 for its patent grant and because
its LICENSE names no copyright holder; no heretic code is copied, so heretic's AGPL-3.0 doesn't
apply. Public at https://github.com/candor-lab-research/candor.

### The gap counted again with image-text-to-text models (2026-10-27)

`gaps.py` read only the Hub's text-generation models, but chat models that also take images (Qwen3.6,
Gemma 4, Qwen3.8-27B) are tagged image-text-to-text. Counted again with both pipelines
(`evidence-2026-10-27/gaps.md`), the dense pool is 293 against 213 (92 of the 110 that came in from
the new pipeline, 30 gone) and the MoE-or-over-36B one 178 against 132. Neither pick changes. The new
pipeline's entries at the top are task models (translation, OCR), and Gemma 4 and Qwen3.6 have
295-420 uncensored versions each. Kwaipilot's KAT-Coder-V2.5-Dev, a coding fine-tune of
Qwen3.6-35B-A3B that the filter takes for an original, now leads the MoE gap without its top repo,
but its uncensored versions hold 1.0% of its downloads.

### The phase-end discussion closed; phases 3 to 5 done (the user, 2026-10-27)

The user asked candor to settle the rest of the discussion on its recommendations.
- **The search's ranges stay.** In all four models the top edits crowd the peak's floor and reach's
  ceiling (both 0.6), but that doesn't show a better edit beyond them. Most trials sit near those ends
  anyway, and the miss-rate simulator's landscapes crowd the same way with nothing better beyond.
  Where nothing better lies beyond, drawing the peak from 0.4 cuts runs within 10% of the best at
  600 trials from 47% to 26%, and with reach to 1.0 as well, to 8% (`evidence-2026-10-27/ranges.md`).
  Before the next search: rescore real best edits with the peak or reach moved past 0.6 to see
  whether a better edit is there. Widening those ends and the direction layers' first on the edge
  alarm, as strength already is, would follow only if one is and the simulator shows it keeps
  today's hits (findings, Ideas).
- **Cost and time against the release gate.** The gate is the user's, for posting more widely: a 27B
  model done in about a day and $20, against the best public release. It was to be set again from
  the 27B's pilot (2026-10-19), which hasn't happened. The 27B took about 2.4 days and $28 from its
  search to release ($35 with the pilot). That is below the pilot's estimate for its route ($35-81,
  2.5-6 days, the pilot not included), and it tied huihui-ai's edit. Closing the gap without
  searching less, which the simulator shows loses hits, comes down to the speed items in STATUS, and
  finalists measured on pods in parallel, which cuts the days but not the cost. Setting the gate
  again is left to the user.
- **Phase close.** The e2e test passes (15 of 15); findings' three tagged lines are deleted (their
  records: the stall stop, 2026-10-17; `--kv-unified`, 2026-10-20; the gap's rescore, 2026-10-23).
  The next model is the first MoE, Qwen3-30B-A3B-Instruct-2507, picked on `first-moe.md`'s reasons
  rather than its gap rank, which still hold after the count above:
  - real uncensored demand, about 66-194 downloads a day for a release;
  - `inspect` passing on today's code;
  - no thinking, so a cheaper run;
  - a public heretic edit, saved in full, that can be measured the way candor measures its own, as
    Trial 161 was.

  Its pilot waits for the user's go.
- **heretic upstream, at the new phase's start:** master d34dd919 (2026-10-26), default edit ARA, last
  release 1.4.0. Merged since 2026-10-19: response-prefix fixes (#489, #523), models with
  Hyper-Connections (#519), templates that reject the system role (#498) and blank refusal markers
  (#512). The thinking-model prefix bug (#487) is still open. On Qwen3-30B-A3B, #218 is closed: it
  was a broken measurement. Nothing changes candor.
- The rude-llama milestone's title now says candor is level with heretic, which is what the shipped
  files show, as the repository is public.

### No better edit past the range ends on rude-llama (2026-10-27)

Five of rude-llama's 10 best one-layer trials that press the peak's floor or reach's ceiling were
scored on the Mac, as the search scores its trials. Each was scored as it is and with that end moved
past its range, one setting at a time (`evidence-2026-10-27/ranges.md`, part 3). All 12 moves raised
KL, by 1-61%, beyond 2 SE in 11, at no fewer refusals beyond noise, so the ranges stay and widening
them on the edge alarm is dropped. Phase 6 scores any finalist that presses an end moved past it,
which costs a few trials.

### The first MoE: Qwen3.8-Flash-Next (the user, 2026-10-27)

The user wanted a new model rather than Qwen3-30B-A3B-Instruct-2507 (July 2025), and then rather
than Ornith-1.5-35B-A3B, found too niche. They chose the most popular newest one: Qwen3.8-Flash-Next
(2026-08-24; 3.35M downloads in 30 days, 6,583 likes; 11.5M GGUF downloads and 234 uncensored
versions). It is 180B parameters with about 3.5B active per token (10 of 512 experts), in a new
layout, qwen4exp, with four residual streams (hyper-connections) and n-gram per-layer embeddings.
candor's choice of models of 36B or less, the A40's reach, had left it out. `inspect` fails on its
per-layer embeddings today, and its size means a multi-GPU pod for the directions and a larger card
than an A40 for scoring, so it costs more than the A40 runs. Support is built on the Mac first; the
pilot's quote then goes to the user.

### The code moved to the candor-lab-research account (the user, 2026-10-27)

The repository moved to https://github.com/candor-lab-research/candor, the same anonymous identity
as the Hugging Face org. Commit hashes before this entry changed in the move; the files that cite
them, the misscheck scripts included, were updated to the new ones.

### Qwen3.8-Flash-Next support built and reviewed (2026-10-27)

`inspect` passes the model: 144 writers (36 `linear_attn.out_proj`, 12 `self_attn.o_proj`, 48 stacked
expert `down_proj`, 48 shared-expert `down_proj`). Its PLE projection is reported as left unedited,
and its memory is known (30 KiB a token and 112.57 MiB a slot).
- **Residual:** read as the four streams' mean.
- **Converter:** qwen4exp's `linear_attn.out_proj` columns are reordered as Qwen3.5's are.
- **Several GPUs:** the directions stage spreads the model across them, each filled to a tenth less
  than it has free, and refuses before loading a pod with too little RAM. It needs twice the table
  transformers keeps on the CPU and a tenth more: 225 GB for Flash-Next.
- **Unchanged:** one GPU and Apple GPUs work as before.

Two independent reviews. The first found the GPUs packed to about 2 GB, so the first batch would run
out of memory, and an unchecked 205 GB of RAM while loading. It also found the pod one GPU larger than
needed: four, not five. The second confirmed the fixes and asked for a margin on the RAM check. On a
tiny random Qwen4-Exp through b11524, the adapters answer as the merged edit (KL 4.55e-9). 981 tests
pass, plus the e2e ones touched. (findings, "Mixture-of-experts models")

### Small fixes on the Mac while the first MoE is chosen again (the user, 2026-10-29)

The user has $27 of Runpod credit and won't top up. Flash-Next's pilot alone is $27-50 and its search
scores on a card 4.2 times an A40 an hour, so the first MoE is chosen again with the user; first, the
fixes the Mac makes for free.
- **F80 and F132, now rather than at the next directions change:** no directions are read before the
  next model's. `directions.pt` holds one row per layer, without the embedding output nothing read,
  and residuals are read longest prompts first. Both directions stages are at version 2, so no search
  in `runs/` can be carried on, scored beside, or have its trials exported or measured again; their
  released files and measurements stand. An export built from older directions is refused before any
  residual is read.
- **`candor score`'s repeat check:** a repeat's KL is the same when it ties paired by prompt and moves
  no more than 25.5%, the most a measured repeat moved. 9 of 18 measured repeats moved more than the
  3% it allowed (findings).
- The repository moved on the Mac to `~/Documents/Projects/LLM/candor`; its venv was reinstalled.
- **The other small fixes, each reviewed until clean:** a local model pushed to a pod keeps its
  fingerprint (`pod.sh push` gives each file the Mac's time to the nanosecond, follows links and
  names files left on the pod); a search's kept unedited reads and each measured set follow the
  GGUF's sha256, hashed once per file, and a set resumed from other bytes is answered again; an
  export says how many tensors it holds finer than its name; `export --outside` names what it leaves
  out; a stopped `candor proxy` keeps the unedited model's score; `inspect` warns on weights
  quantised to 4 bits before (`evidence-2026-10-29/quantised-before.md`); `measure --set` measures
  sets alone. Left: `serve` on a pod with the Mac's tools, whose config went with the move.

### Phase 6: Spark-X2.5-4B (the user, 2026-10-29)

The user wanted a smaller, newer model than Qwen3-30B-A3B-Instruct-2507. Of the general chat models
of 26B or less released since March, Spark-X2.5-4B (2026-08-24) has the largest gap without its top
repo, 39,750 against Agents-A1-4B's 11,505, and real uncensored demand: 17 repos with 296,301
downloads in 30 days, 19% of its GGUF downloads (`evidence-2026-10-27/gaps.tsv`). Its successor
X2.6-4B (2026-10-19) has 11 uncensored repos already. Flash-Next's support stays for when there is
credit.
- **What candor needed:** its template turns thinking off by writing `</think>` in place of
  `<think>`, which `think_close` now reads as that end tag; it is scored thinking on only, as an
  answer after an empty block isn't how its template turns thinking off. Its remote code imports only
  math, torch and transformers. llama.cpp b11524 converts and runs `spark2_5`.
- **Run settings:** the card's thinking sampling (temperature 1.0, top_p 0.95, top_k off, which runs
  as 1,000 as on K2), budget 8192, and the search as K2's and the 27B's.
- **Upstream:** heretic master efb1be6e (default edit ARA), #525 (hyper-connections) merged since
  2026-10-27, #487 still open, no Spark issues; no llama.cpp issues on `spark2_5`.

### Phase 6, step 1: the Spark-X2.5-4B pilot (2026-10-29)

On one A40 for 2.2 h, about $1.30. candor runs it correctly as used: refusals 62 of 64 harmful, none
of 64 harmless, maths 63 of 64; transformers' forward matches llama.cpp's; its two answer-start
positions give nearly the same directions. It found three faults, each fixed with tests and
reviewed: remote code written for transformers 4 (tied weights as a list, old mask arguments) now
runs under 5; first tokens are read once no slot is busy, as a read after a trial stopped on its
budget came back empty on CUDA; and `candor score`'s refusal band is 2, the most a real repeat moved,
until repeats are paired by prompt. The e2e tests pass on CUDA, the Qwen4-Exp model given heads its
CUDA kernel takes. Trials take 251 s, so 600 on an A40 would take nearly all the credit; 64 slots
are slower a trial. Numbers in `phase-6-spark-x2.5-4b.md`, step 1.

### Spark-X's search on the A40, staged at 300 trials (the user, 2026-10-29)

The user chose the A40 ($0.59/h, secure) over a community RTX 4090 ($0.34/h), and fewer trials than
600. 300 first: on Qwen3.5-9B the one-layer best was within 1% of its final KL by trial 300 and K2's
search stopped at 313, but the 27B's shipped edit was trial 532 and the simulator gives 300 trials
about half 600's chance of a pick within 10% of the best. So at 300 the progress goes to the user;
the search carries on from its journal if the bests are still moving.

### Spark-X's search stopped at 93 trials; its three finalists measured (the user, 2026-10-29)

At 92 trials both kinds' bests within the cap had held, per layer's since trial 18 and one layer's
since trial 53, each at KL 0.0154 on the test prompts, under the 27B's final 0.0188. The user
stopped the search there, against the 300 staged, to measure them. candor names three finalists:
53, 70 and 18. On the same pod, the unedited model is measured in full, trial 18, which presses MLP
strength's floor, is scored with it moved past (STATUS, phase 6), and each finalist is exported and
measured. The 27B's decision rule, ranked on unsafe answers (2026-10-19 and 2026-10-23), applies,
put to the user before any finalist's measurement is read; if none passes, the search carries on
from its journal. Numbers in `phase-6-spark-x2.5-4b.md`, step 2.

### Phase 6, step 3: Spark-X's finalists measured; trial 53 picked by the rule (2026-10-30)

The rule proposed before the measurements started, the 27B's ranked on unsafe answers, was put to
the user with the results, who accepted it and asked for the release. Trials 53 and 70 pass; 18
fails on maths thinking reaching the budget (96 against 66). 53 and 70 are level on unsafe answers
under both judges, so the lower mean KL ships: trial 53, one layer (17, read where the thinking
starts), 0 of 288 refused (1 under the 8B), KL 0.0457, maths and harmless refusals within noise.
Trial 18 past its MLP floor scored no better beyond noise. No public reference was measured: the
user went to the release without one, so the card has no comparison column, as K2's. The pod sat
idle 4 h 13 min after the job (about $2.50), as the queue didn't delete it; it was deleted once
everything was pulled. Numbers in `phase-6-spark-x2.5-4b.md`, step 3.

### Phase 6, step 4: Spark-X2.5-4B released (the user, 2026-10-30)

The user asked for it to go up on Hugging Face as the earlier ones did, once happy with the results,
without a chat test of their own first (the 27B had one). Published from candor-team as
`candor-lab/Spark-X2.5-4B-Candor-Uncensored-GGUF`: the measured export renamed on the Mac, every
tensor byte-identical, loaded in llama-server b11524, uploaded to a private repo, the card reviewed
against the measurements (two fixes) and then made public. The card follows K2's, without a
comparison column, and drops "(not released yet)", which the earlier cards still say. Phase 6 cost
about $12.94. (`phase-6-spark-x2.5-4b.md`, step 4)

The phase-end e2e test passes on the Mac, 17 of 17 in 4 min 38 s, at 8044401.
