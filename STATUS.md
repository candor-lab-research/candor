# Status

This file is the single plan for candor. Anything not listed under Plan is not being worked on.

## Plan

Depth before breadth (decided 2026-10-10): one complete run on the models you already use, in the
format you run them, before anything else. Every model is scored and shipped as a GGUF, and runs
in llama-server, the llama.cpp server candor scores in, so nothing differs between testing and use.
For every model, the final check (refusals, KL and maths, with your KV cache type) runs on the exact
file shipped, since merging and re-quantising can shift the result (phase 0: KL 0.01 between adapters
and merged weights in Q4_K_M). (Decided 2026-10-11: the user isn't tied to Bionic; Bionic or LM Studio stay possible, unchecked.)

| Phase | What | Done when | State |
| --- | --- | --- | --- |
| 0 | Score edits as per-layer LoRA adapters in llama.cpp | Adapters match merged weights, no speed cost | Done |
| 1 | Core maths and `candor inspect` | candor's rebuild of a heretic trial matches its refusals, and measuring heretic's saved model under heretic's conditions gives heretic's numbers exactly; `inspect` reports correctly on rude-llama, Qwen3.5-9B and K2-Horizon | Done |
| 2 | rude-llama end to end on the Mac: measure, search, export a GGUF to run in llama-server | Under the decision rule written in step 4 before any search runs, candor's shipped GGUF is at least as good as heretic's Trial 170 converted and quantised the same way: fewer or equal refusals on held-out prompts, no higher KL, no maths loss. It runs in llama-server, and you prefer it in a blind A/B | In progress: steps 1-5 done; no search trial passes yet |
| 3 | Qwen3.5-9B, a thinking model: scored with thinking on at your settings; residuals read on a pod | Defined at the start of the phase, against heretic's Trial 161 | |
| 4 | K2-Horizon at your real settings | Measured honestly with its own chat template (Bionic couldn't parse it; llama-server's reader is to be checked) and the reasoning level you use, then improved | |

Pods only where the Mac can't cope: reading residuals for Qwen3.5-9B and K2 (16-bit, 18-19 GB).
Phase 3 builds a pod bootstrap script and an in-pod watchdog that deletes the pod at a budget, and
checks them on rude-llama first (about $0.30). Top up Runpod before phase 3 (about $3.55 left).

Every step gets an independent review before it is called done; its findings are fixed with
tests and the fixes verified until a review comes back clean. Heavy jobs on the Mac (converting
a real model, reading residuals, a search) wait for the user's go.

Results of each phase, numbers only, are in `results/` (phase 2: `results/phase-2-rude-llama.md`).

### Next: a search that doesn't miss recipes it could find

The search missed a better recipe inside its own space: Trial 170 read at layer 12 scores 0 of 100
at KL 0.0818 with the search's own scorer, against candor's best 0.1017 (results file). It is not a
fault in scoring single-direction edits, nor a scorer that can't tell them apart: the search never
tried that shape. The search will run on new models with nothing to compare against, so this is
fixed before phase 3 (the user, 2026-10-11). rude-llama ships Trial 170's recipe built by candor.

How the fix is chosen and checked, decided before any of it runs:

1. Designs are compared by how often they miss the best on test landscapes calibrated to the real
   trial numbers, where today's search must be shown to reproduce the miss. Variants guard against
   fitting this one case: the best in the per-layer family, the best just outside today's ranges, a
   narrower target, and randomly generated landscapes. The chosen design must be no worse than
   today's anywhere and better on average.
2. The fix targets the cause (how the search treats the two kinds of edit, and where it spends its
   trials), not this answer: no default comes from Trial 170's values.
3. The check on rude-llama is one search with no heretic recipe as a starting point and no range
   narrowed towards it. It passes if it finds, by itself, a recipe scoring at least as well as
   Trial 170 read at layer 12 (0 of 100 at KL 0.0818 or less, with the search's scorer). If it
   fails, the answer goes back to the test landscapes; settings are not tweaked until it passes.
4. Independent repeats that report whether they agree are built in, as the warning that works on a
   new model with no reference.

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
| 5 | The search: Optuna over a kernel per component (heretic's 0.8-1.5 strength range), stored resumably in the run folder, each trial scored in llama.cpp on the quantised GGUF, refusals counted by the judge; progress with time left, the Mac kept awake. Built on SmolLM2, then a 10-trial rude-llama pilot (time per trial) and one full search, each with the user's go | Done: 200 trials in 3 h 13 min; trials 165, 167 and 187 measured, none passes (see Next) |
| 6 | Export: merge, convert and quantise; the final check on the exact shipped file with your KV cache type; a `candor serve` command that runs the shipped file in llama-server at your settings (system prompt, KV cache, date); the blind A/B | |

## Research changes and where they land

From the 2026-10-10 paper review, corrected after the full review read each paper in full.

| Finding | Change | Where |
| --- | --- | --- |
| Matching targets by name silently misses most of the effect on mixture-of-experts models (2609.09793) | `inspect` lists every per-layer tensor that outputs the hidden size and fails on any it can't place or any layer without a writer | Phase 1, done |
| Keyword judges count compliant answers with safety words as refusals (2512.13655: 72.2% vs 95.7% non-refusal on 900 abliterated answers) | Qwen3Guard-Gen-0.6B on the final answer, with its false-refusal rate measured (the vendor's refusal precision is 83-89%, 2510.14276; calibrate any judge, 2606.25487); keyword counts kept beside it | Phase 2 step 4 |
| One count on 100 prompts can't separate close models (2609.25173 for unpaired sampled runs; heretic's transformers scoring varied by 1-2 refusals with batch size). Measured in llama-server on rude-llama: 0 of 588 held-out answers and 5 of 1,319 maths answers changed between 8 and 16 slots, KL 0.0000 | Measure the noise floor; compare paired on a held-out set; scoring is greedy, so seeds matter only for sampled checks | Phase 2 step 4 |
| First-token KL misses lost ability (2512.13655: -4.5pp GSM8K at KL 0.043) | The full GSM8K test set, paired against the same quantised file; multiple-choice flips only as a warning (2407.09141) | Phase 2 step 4 |
| Re-quantising after an edit can bring refusals back (2609.09793) | The last check runs on the exact file you ship, with your KV cache type (2606.09864; already used for all scoring) | Phase 2 step 6 |
| Thinking mode changes refusals (2509.00544); the think start and think close are decision points in distilled reasoning models (2609.18471, 2510.06036) | Score in the mode and at the level(s) you run, with answers long enough to finish thinking, judged on the final answer with unfinished answers counted separately (2609.09793). candor's directions are already read at the think start | Phases 3-4 |
| A plain single-direction edit barely worked on Qwen3.5-9B (2605.21706: 3.3%, against 92-97% for an edit pushed a fixed margin past the refusal boundary) | Measure Qwen3.5 properly before trusting heretic's Trial 161; plan a past-the-boundary variant | Phase 3 |
| A direction averaged over the reasoning (2507.03167: better on R1-Distill-Llama-8B, worse on Qwen3-8B); one direction from a mixed prompt pool beats per-category ones (2602.02132) | Build directions from prompts like the refusals you actually hit; a reasoning-averaged set only if refusals remain | Phases 3-4 |
| Multi-direction edits (2511.08379; about 10% on Qwen3.5-9B in 2605.21706) and cleaned directions (2601.08489, single author, untuned baseline) | Not planned | |

## Not in this plan

- heretic and `tune/heretic-kit`: frozen, no new features. The guide stays as reference.
- `tune/abliterate.py`: deleted 2026-10-10. candor does the same job.
- Cut 2026-10-10 for depth first: `candor doctor`, a live dashboard, one-command pod
  orchestration, rewriting the guide (the README is the documentation), and support for more
  model families (the existing mixture-of-experts, Gemma and GLM checks stay as they are).
- Optuna is pinned below 5 (4.9, the version heretic ran), so rude-llama's search uses the same
  sampler that found Trial 170. Optuna 5 changed multivariate TPE's bandwidth and its
  multi-objective weights; worth trying once candor has its own baseline, for example on Qwen3.5.
- Models of 64 layers or more: the `--lora` list of every adapter set passes Linux's 128 KiB
  argument limit (about 34 KB for rude-llama's 28 layers, 46 KB for 32). Not needed for the three
  models in this plan; shorter adapter names would fix it.
- Deferred to phase 3: settings for thinking models (template kwargs, reasoning level, sampling,
  longer answers) and choosing torch's CUDA build to match llama.cpp's 12.8.
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
  with a recording stub. transformers, mlx-lm, Bionic's MLX engine (`model_kit.py`) and llama.cpp
  (`chat.cpp`) all send one. Qwen3.5 has none, so it is unaffected.
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
  including 24 linear-attention `out_proj` (llama.cpp's `ssm_out`); K2-Horizon fails because Bionic
  can't parse its chat template (lines 89, 91, 127, 129, 628), and warns about remote code and
  the doubled token.
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
