# History

STATUS.md as it stood before it was trimmed on 2026-10-14 (commit 053f50b): the plan in full detail, every
decision with its reasons, the research tables and the log. STATUS.md keeps what is in force; new
log entries go at the end of this file.

## Plan

Depth before breadth (decided 2026-10-10): one complete run on the models you already use, in the
format you run them, before anything else. Every model is scored and shipped as a GGUF, and runs
in llama-server, the llama.cpp server candor scores in, so nothing differs between testing and use.
For every model, the final check (refusals, KL and maths, with your KV cache type) runs on the exact
file shipped, since merging and re-quantising can shift the result (phase 0: KL 0.01 between adapters
and merged weights in Q4_K_M). (Decided 2026-10-15: the user runs models only in llama-server, or a local runner that runs them better.)

| Phase | What | Done when | State |
| --- | --- | --- | --- |
| 0 | Score edits as per-layer LoRA adapters in llama.cpp | Adapters match merged weights, no speed cost | Done |
| 1 | Core maths and `candor inspect` | candor's rebuild of a heretic trial matches its refusals, and measuring heretic's saved model under heretic's conditions gives heretic's numbers exactly; `inspect` reports correctly on rude-llama, Qwen3.5-9B and K2-Horizon | Done |
| 2 | rude-llama end to end on the Mac: measure, search, export a GGUF to run in llama-server | Under the decision rule written in step 4 before any search runs, candor's shipped GGUF is at least as good as heretic's Trial 170 converted and quantised the same way: fewer or equal refusals on held-out prompts, no higher KL, no maths loss. It runs in llama-server. (A blind A/B was dropped by the user on 2026-10-12: 535 and Trial 170 are too close to tell apart by hand.) | Done 2026-10-13: trial 535, exported, passes against Trial 170 exported the same way, and ships (`runs/rude/exports/trial-535/`); see `results/milestone-rude-llama-vs-heretic.md` |
| 3 | Qwen3.5-9B, a thinking model: scored with thinking on at your settings; residuals read on a pod | Written in step 4 before any search runs, from the measured noise, against heretic's own Trial 161 exported and measured as candor ships its edits | Steps 1-4 done 2026-10-14 (below); the decision rule awaits sign-off, then step 5 |
| 4 | K2-Horizon at your real settings | Measured honestly with its own chat template (whether llama-server reads it is to be checked) and the reasoning level you use, then improved | |

From phase 3 a job expected to take over about 2 hours, or needing more memory than the Mac has,
runs on a pod (the user's decision, 2026-10-12); shorter jobs (a measurement, a pilot) can run on
the Mac. The rude-llama search of phase 2 is the last long one on the Mac. So candor's whole
pipeline (convert, residuals, search, measure) must run on a pod: `scripts/pod.sh` and its watchdog,
which deletes the pod at a deadline set per job (README, "On a pod"), checked on rude-llama in phase
3 step 2. Runpod had $18.56 (topped up 2026-10-12), less about $0.51 for that check and $9.56 for
step 4: about $8.50 left, so the next pod run needs a top-up. Spend it only where a run earns it.

Every step gets an independent review before it is called done; its findings are fixed with
tests and the fixes verified until a review comes back clean. Heavy jobs (converting a real model,
reading residuals, a search) wait for the user's go.

Results of each phase, numbers only, are in `results/` (phase 2: `results/phase-2-rude-llama.md`;
phase 3: `results/phase-3-qwen3.5-9b.md`).

### Next

1. Phase 3, step 5: `candor proxy` on a pod (started 2026-10-14, about $2-3); its verdict picks
   step 6's path, whose price the user sees first (a top-up of about $10 needed).
2. Search evidence (2026-10-14, reports in `results/evidence-2026-10-14/`). The hypothesis that
   refusal falls off a cliff, so a coarse sweep should find where it collapses, is dropped: across 8
   searches on 3 models and in the literature refusals fall gradually with strength. Qwen3.5-9B's good
   edits sit only at the 1.5 strength ceiling, as K2-Horizon's did until heretic's cap went to 5.0
   (best 54 refused, then 7). heretic itself stalls at 70/100 on Qwen3.5-9B (#404, 4 runs), so
   Trial 161's 11 was an outlier. Candidates, each built only with proof across model families (the
   user, 2026-10-14):
   - a wider strength range from the start (K2; abliterix's Qwen3.5/3.6-27B ranges of 1-5 and 1-6;
     heretic PR #52); this run's widening to 3.0 near trial 120 is the first Qwen test;
   - heretic's newer default edit, ARA (4/100 on Qwen3.5-4B where abliteration stays above 50, PR
     #211): find out what it is first;
   - coordinates such as total edit mass and spread, and a GP sampler instead of TPE (general
     optimisation literature, untested here): only after an offline replay over several model
     families, 20+ seeds per arm, as `lit-search.md` sets out;
   - edits at the thinking start never refused under 97 in candor, but worked on Qwen3-8B
     (2507.03167): probably candor's setup; look later.
3. Not blocking, each when there is a reason:
   - 64 slots for a thinking model's trials: 15% faster than 32 with GPU sampling (results file);
   - sampling moved to the GPU on 2026-10-14 draws differently for a seed, so measurements made
     before it pair with later ones prompt by prompt, not draw by draw; `answering` records don't
     say which, as recording it would have thrown away the unedited model's saved thinking;
   - an interrupted `candor proxy` scores the unedited model again for the mode it stopped in;
   - one search on a pod with the attention peak's floor and its reach's ceiling widened (535 presses
     both, at 0.6); wider ranges become the default only if its held-out best beats 535's;
   - whether the search should score on more or other prompts: per-layer trials' refusals and
     layer-14 trials' KL didn't hold on the held-out prompts (results file);
   - the search scoring on the base it ships, with the edited tensors at Q8_0 there too, so search
     and export match by design (needed only where the run's quantisation loses edits, as on
     rude-llama);
   - a warning in `inspect` when a model's weights were quantised before, as rude-llama's were;
   - the export's file name says Q4_K_M when some tensors are Q8_0;
   - a local model copied to a pod gets a new fingerprint (rsync from macOS keeps file times only to
     the second), so a run moved between machines rebuilds every stage; Hub models are unaffected.

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
  overnight (about 12 h). The watchdog deletes the pod at a cap set per job. Expect a top-up beyond
  $18.56 (a guess until the pilot: search $10-20, finalists $5-10).
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
| 5 | Built and reviewed 2026-10-14, to run on a pod: speed, then the proxy check. candor enforces the thinking budget itself, so llama.cpp samples on the GPU (it turns GPU sampling off for a request with a budget; same sampler maths, only the draws per seed change); 2-3 servers per GPU if the speed test shows a gain; an edit's scoring stops once its refusals pass the cap; 150-token answers in the cheap score. Then about 24 edits spread over kinds, positions and strengths (stronger ones included) scored both thinking off and as used, plus one repeat for noise: whether the two rank edits alike, and whether thinking on needs a stronger edit | |
| 6 | The search, on the path step 5 picks. If the cheap score ranks like thinking on (rank correlation at least 0.7, perhaps after a strength correction): 600 cheap trials, then about 15-20 edits from their front scored as used at the corrected strength and raced on shared prompts and seeds. If not: a thinking-on search of about 150 trials, seeded with step 5's as-used scores, the miss-rate simulator rerun at 150 first | |
| 7 | The best 2 exported and measured as used on the held-out prompts and 400 maths questions, paired with the unedited model on the same ones; the winner's full 1,319 maths is its final check on the file that ships; copied to the Mac as it is (each machine quantises differently), `serve` on the Mac | |

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
(`77106e4`, before any held-out number). The milestone: `results/milestone-rude-llama-vs-heretic.md`.

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

Measurement noise is too small to matter here (0 flips on 588 held-out prompts and KL 0.0000 between
8 and 16 slots), so counts are compared as they are, strictly: no trading refusals for KL. Step 6
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
| One count on 100 prompts can't separate close models (2609.25173 for unpaired sampled runs; heretic's transformers scoring varied by 1-2 refusals with batch size). Measured in llama-server on rude-llama: 0 of 588 held-out answers and 5 of 1,319 maths answers changed between 8 and 16 slots, KL 0.0000 | Measure the noise floor; compare paired on a held-out set; scoring is greedy, so seeds matter only for sampled checks | Phase 2 step 4 |
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
- Cut 2026-10-10 for depth first: `candor doctor`, a live dashboard, one-command pod
  orchestration, rewriting the guide (the README is the documentation), and support for more
  model families (the existing mixture-of-experts, Gemma and GLM checks stay as they are).
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
`2bb6bba`. Pod: A40, about 23 minutes, about $0.23.

### Phase 1: core maths and `candor inspect` (done 2026-10-10)

- `candor.heretic` reads heretic's Optuna journals; `spikes/reproduce_trial.py` (last in `2bb6bba`) rebuilds a trial
  with candor's own direction finding and edit. On rude-llama Trial 170 it gets heretic's 2/100.
- Heretic tokenizes the rendered chat with the tokenizer's own special tokens added, so Llama 3
  and K2-Horizon get two begin-of-text tokens. Proved by running heretic's own `Model.generate`
  with a recording stub. transformers, mlx-lm and llama.cpp (`chat.cpp`) all send one. Qwen3.5 has
  none, so it is unaffected.
- `spikes/compare_models.py` (last in `2bb6bba`) scored the saved heretic model (`models/local/rude-llama-heretic`)
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
- Noise floor (unedited at 8 and 16 slots): identical on every held-out answer, KL 0.0000; 5 maths
  answers changed (3 / 2), and 2 more were cut off.
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
than Qwen3.5-9B at high effort (unedited median harmful thinking 225 tokens against 996), so a full
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
refusal stop moves from 29 to 50 refusals (per layer 2 to 3) to keep its guarantee once the sampler
ranks by both caps. Code older than 8dfc0f5 must not run on `runs/k2` again: it would carry on in
the old studies.
