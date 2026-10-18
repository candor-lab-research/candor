# Status

The single plan for candor: anything not listed here isn't being worked on. Each phase's numbers
are in `results/`; what candor has found and the ideas parked are in `results/findings.md`; outside
work kept for later in `results/new-features.md`; past decisions with their reasons, the research
tables and the log are in `results/history.md`.

## Plan

Depth before breadth: one complete run on the models you use, in the format you run them, before
anything else. Every model is scored and shipped as a GGUF in llama-server, and the final check
(refusals, KL, maths, your KV cache type) runs on the exact file shipped.

| Phase | What | State |
| --- | --- | --- |
| 0 | Score edits as per-layer LoRA adapters in llama.cpp | Done |
| 1 | Core maths and `candor inspect` | Done |
| 2 | rude-llama end to end: search, export, measure against heretic's Trial 170 | Done 2026-10-13: trial 535 passes and ships (`results/milestone-rude-llama-vs-heretic.md`) |
| 3 | Qwen3.5-9B, a thinking model, scored as used against heretic's Trial 161 | Measured 2026-10-15: trial 175 beats Trial 161 (`results/milestone-qwen3.5-9b-vs-heretic.md`); the phase-end discussion remains |
| 4 | K2-Horizon-7B at your real settings: llama-server, reasoning effort high | Released 2026-10-18: trial 205 passes the decision rule and the user's test (`results/phase-4-k2-horizon-7b.md`); the phase-end discussion remains |

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

## Now: between phases, waiting for the user

- Phase 3: trial 175 beats Trial 161 (0/288 refused, maths at the unedited model's level);
  `results/phase-3-qwen3.5-9b.md`. The phase-end discussion remains.
- Phase 4: trial 205 released 2026-10-18 as `candor-lab/K2-Horizon-7B-Candor-Uncensored-GGUF`
  after passing the decision rule and the user's test; numbers in
  `results/phase-4-k2-horizon-7b.md`.
- 20 ideas from arXiv are ranked in `results/new-features.md`.
- The repository review (2026-10-18): 24 work packages in the order agreed
  (`results/evidence-2026-10-18/repo-review.md`). Steps 1 (faster tests, one server launch), 2
  (the thinking check, measurements that can't pair stale labels, intervals in `compare`, findings
  corrected) and 3 (27B cost and safety: packages 6-10, 14 and 17) done; step 4, the one
  measurements change before any 27B reference, next.
- `candor` alone opens menus over the runs and their commands (the user, 2026-10-18; README, Use).

## Next

1. The repository review's steps 4-5, then the phase-end discussion: where to go next, from
   `results/findings.md`, `results/new-features.md` (chiefly the search's cost for 27B) and the
   review's 11 method questions, including how to score Qwen3.8-27B thinking off; no method
   changes until agreed.
2. Phase 4, later: low effort's two closes; Uno not now. The budget limit
   (`search.max_budget_hits`, on K2 from trial 138) costs no measurable hits on the simulator and
   stays on (results file, step 5).
3. Not blocking, each when there is a reason:
   - speed: 64 slots (15% faster than 32), weights cached on the GPU for "full" trials, the proxy's
     adapters loaded into one server;
   - measurements before 2026-10-14 pair with later ones by prompt, not by draw (sampling moved to
     the GPU);
   - the search scoring on the base it ships, edited tensors at Q8_0 too (only where quantisation
     loses edits, as on rude-llama), and whether it should score more prompts (per-layer refusals
     and layer-14 KL didn't hold on rude-llama's held-out prompts);
   - rude-llama: one search with the attention peak's floor and reach's ceiling widened (535 presses
     both);
   - llama-server's chat endpoint may count prompt tokens against the thinking budget (seen on a
     tiny model only); K2's budget message was written for a close that starts with a newline;
   - small fixes: an interrupted `candor proxy` scores the unedited model again; the search prints
     "Rebuilding scores/trial/edit.gguf" every "full" trial; serving the unedited model downloads the
     original weights to check its GGUF (the Hub commit and float type need only metadata); an
     `inspect` warning when weights were quantised before; the export's file name when some tensors
     are Q8_0 or F32; a local model copied to a pod rebuilds every stage.

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
- The decision rule, signed off 2026-10-14 (text in `results/history.md`): both finalists pass it
  and beat Trial 161.

### Steps

| Step | What | State |
| --- | --- | --- |
| 1-7 | Settings, pods, Qwen3.5-9B and Trial 161 measured as used, the decision rule; the "full" edit, a thinking-off search to 545 trials, `candor proxy` ("offset"); the search's best 6 scored as used; trials 205 and 175 exported, measured in full and copied to the Mac; the user's thinking review of trial 175 | Done 2026-10-15, review 2026-10-17 (history, results file, milestone) |

## Phase 4: K2-Horizon-7B

As used: llama-server, reasoning effort high (the user, 2026-10-15); temperature 1.0, top_p 0.95,
top_k 1,000, min_p 0, budget 8192 (the user, 2026-10-16); 28 slots on an A40. K2 can't turn
thinking off, so candor refuses a thinking-off search for it. References: the unedited model and
Avicennasis's heretic edit refused 48/64 as used in the pilot (7/50 on its card), too weak a bar to
measure in full, and Uno-Uncensored is built on K2 with the Uno adapter merged in: the reference is
the unedited model (the user, 2026-10-16). Strength 0.8-5.0 from the start (the user, 2026-10-16:
22 trials at 0.8-1.5 all refused 88-100 of 100; heretic's K2 run needed 4.5-4.6).

| Step | What | State |
| --- | --- | --- |
| 1-6 | Code for K2 and the timed pilot; the unedited model as used (287/288 refused, maths 1,222/1,319); the decision rule; the search, stopped by the user after 313 trials (budget limit 8 from trial 138); finalists 207, 205, 265 and 114 measured: 205 and 114 pass, 205 ships; the user's test with tools; released | Done 2026-10-18 (history, results file) |

## Not in this plan

- `tune/heretic-kit` is frozen (our kit; heretic itself is checked at each phase start).
- Cut for depth first: `candor doctor`, a live dashboard, one-command pod orchestration, rewriting
  the guide, more model families.
- Reporting the llama.cpp prompt-cache bug upstream.
- Separate tracks: the Qwen3.5 rude fine-tune, the Mac Studio purchase.
