# Status

The single plan for candor: anything not listed here isn't being worked on. Each phase's numbers
are in `results/`; what candor has found and the ideas parked are in `results/findings.md`; past
decisions with their reasons, the research tables and the log are in `results/history.md`.

## Plan

Depth before breadth: one complete run on the models you use, in the format you run them, before
anything else. Every model is scored and shipped as a GGUF in llama-server, and the final check
(refusals, KL, maths, your KV cache type) runs on the exact file shipped.

| Phase | What | State |
| --- | --- | --- |
| 0 | Score edits as per-layer LoRA adapters in llama.cpp | Done |
| 1 | Core maths and `candor inspect` | Done |
| 2 | rude-llama end to end: search, export, measure against heretic's Trial 170 | Done 2026-10-13: trial 535 passes and ships (`results/milestone-rude-llama-vs-heretic.md`) |
| 3 | Qwen3.5-9B, a thinking model, scored as used against heretic's Trial 161 | Step 6 running (below); `results/phase-3-qwen3.5-9b.md` |
| 4 | K2-Horizon at your real settings, with its own chat template and reasoning level | |

Qwen3.5-9B is a testbed: the process must work, affordably, for larger thinking models used with
thinking on (next, e.g. Qwen3.8-27B). A 9B model must not take days.

## How we work

- Jobs over about 2 hours, or too big for the Mac, run on a pod (`scripts/pod.sh`; its watchdog
  deletes the pod at a deadline set per job). Time 2-3 real units before quoting; heavy jobs and new
  spend wait for the user's go. Runpod credit: check before each pod job.
- Every step gets an independent review; findings are fixed with tests until a review is clean.
- The method changes only with evidence that holds across model families, not on one run.
  Findings and ideas go in `results/findings.md` as they come, each flagged measured, published
  or hypothesis.
- At each phase start, check heretic upstream as a reference point, not to copy: master's commit
  and default edit, PRs merged since the last check, issues on the model family (commands in
  `results/evidence-2026-10-14/ara-missed.md`). Tools checked 2026-10-14: heretic master 66a952b
  (2.0.0.dev0, default edit ARA); last release 1.4.0.

## Now: phase 3, step 6

- The search stopped at 545 of 600 trials, thinking off, when its pod's GPU failed; no new best
  after trial 428. Best: one layer at layer 12, 0-1 refused at KL 0.03; per layer, 0-2 at
  0.079-0.084 (Trial 161 rebuilt: 24 at 0.092).
- Trials 205, 417, 342, 428, 402 and 175 are being scored as used, at their strength and 1.4 times
  it, on pod `oi41qp8xyncyy6` (from 07:58 on 2026-10-15, about 3.5 h; the watchdog deletes it at
  15:40).

## Next

1. When the scoring ends: the best 2 for step 7, quoted first (a top-up needed).
2. Not blocking, each when there is a reason:
   - speed: 64 slots (15% faster than 32), weights cached on the GPU for "full" trials, the proxy's
     adapters loaded into one server;
   - measurements before 2026-10-14 pair with later ones by prompt, not by draw (sampling moved to
     the GPU);
   - an interrupted `candor proxy` scores the unedited model again;
   - the search prints "Rebuilding scores/trial/edit.gguf" on every "full" trial;
   - the search scoring on the base it ships, edited tensors at Q8_0 too (only where quantisation
     loses edits, as on rude-llama);
   - whether the search should score more prompts (per-layer refusals and layer-14 KL didn't hold
     on rude-llama's held-out prompts);
   - rude-llama: one search with the attention peak's floor and reach's ceiling widened (535 presses
     both);
   - `inspect` warning when weights were quantised before; the export's file name when some tensors
     are Q8_0; a local model copied to a pod rebuilding every stage.

## Phase 3: in force

- Decisions use as-used numbers only: thinking on; Qwen's sampling (temperature 1.0, top_p 0.95,
  top_k 20, min_p 0, presence penalty 1.5) with a seed per prompt; a thinking budget of 8192, then a
  forced close; an unfinished answer counts against the edit; the judge reads the final answer.
- KL is read where the answer starts after the unedited model's own thinking; KL after an empty
  thinking block is kept beside it.
- A cheap thinking-off score may only choose which edits get the as-used test, and only once
  `candor proxy` shows it ranks edits as thinking on does, for each new model.
- The reference is heretic's Trial 161, re-saved by heretic, exported and measured as candor ships
  its edits. Maths: the full 1,319 GSM8K questions on finalists only.
- Strength starts at 0.8-1.5 and widens to 3.0 when a kind's best edit presses it. Directions are
  read at the think start and after a closed thinking block; the search chooses. Optuna 4.9.

### Decision rule (signed off by the user 2026-10-14)

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

### Steps

| Step | What | State |
| --- | --- | --- |
| 1-5 | Settings, pods, Qwen3.5-9B and Trial 161 measured as used, the decision rule; the "full" edit, a thinking-off search, `candor proxy` ("offset") | Done 2026-10-14 (history, results file) |
| 6 | The search, thinking off, to 545 trials; its best 6 scored as used at their strength and 1.4 times it | Scoring as used: running |
| 7 | The best 2 exported and measured as used (held-out prompts, 400 maths, then the winner's 1,319); copied to the Mac; `serve`. Then the user reads the thinking of the unedited model, candor's best and Trial 161 on about 5 requests and 5 maths problems each, and reports counts only (working, weighing whether to answer, circling, garbled; changed its mind; finished; maths right) for `findings.md` | |

## Parked, to review later

Every idea not acted on, with what supports it (measured, published or a hypothesis), is in
`results/findings.md`, beside what candor has found so far.

## Not in this plan

- `tune/heretic-kit` is frozen (our kit; heretic itself is checked at each phase start).
- Cut for depth first: `candor doctor`, a live dashboard, one-command pod orchestration, rewriting
  the guide, more model families.
- Reporting the llama.cpp prompt-cache bug upstream.
- Separate tracks: the Qwen3.5 rude fine-tune, the Mac Studio purchase.
