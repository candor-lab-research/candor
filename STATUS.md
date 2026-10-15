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
| 3 | Qwen3.5-9B, a thinking model, scored as used against heretic's Trial 161 | Measured 2026-10-15: trial 175 beats Trial 161 (`results/milestone-qwen3.5-9b-vs-heretic.md`); the user's thinking review and the phase-end discussion remain |
| 4 | K2-Horizon-7B at your real settings: llama-server, reasoning effort high | Step 1 (code) done 2026-10-15; step 2's pilot next (below) |

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
  `results/phase-3-qwen3.5-9b.md`. The thinking review and the phase-end discussion remain.
- Phase 4 step 1 done: candor runs K2-Horizon as options that leave other models unchanged
  (rude-llama's answers byte-identical before and after, Qwen's prompt ids identical). No pod is
  running; Runpod credit about $2.50.
- 20 ideas from arXiv are ranked in `results/new-features.md`.

## Next

1. The user's thinking review, on the Mac (`candor serve`): the unedited model, trial 175 and
   trial 205 (Trial 161 left out, the user 2026-10-15), counts only. Optional now.
2. The phase-end discussion with the user: where to go next and what to try, from
   `results/findings.md` and `results/new-features.md` (chiefly the search's cost for 27B). No
   changes to code, defaults or the method until agreed.
3. Phase 4, step 2: a timed pilot on a pod (about $2-3, needs a Runpod top-up and the user's go).
4. Not blocking, each when there is a reason:
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
   - `candor measure` defaults to 8 slots where thinking runs use 32 (step 7 started with 8);
   - `candor serve` downloads the model's original weights only to fingerprint the run;
   - llama-server's chat endpoint may count prompt tokens against the thinking budget (seen on a
     tiny model only; check on a real GGUF);
   - K2's thinking budget message was written for a close that starts with a newline;
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

### Decision rule

Signed off 2026-10-14 (text in `results/history.md`) and applied in step 7: both finalists pass
and beat Trial 161.

### Steps

| Step | What | State |
| --- | --- | --- |
| 1-7 | Settings, pods, Qwen3.5-9B and Trial 161 measured as used, the decision rule; the "full" edit, a thinking-off search to 545 trials, `candor proxy` ("offset"); the search's best 6 scored as used; trials 205 and 175 exported, measured in full and copied to the Mac | Done 2026-10-15 (history, results file, milestone) |
| 7, review | The user reads the thinking of the unedited model, trial 175 and trial 205 in `candor serve` on about 5 requests and 5 maths problems each, and reports counts only (working, weighing whether to answer, circling, garbled; changed its mind; finished; maths right) for `findings.md` | |

## Phase 4: K2-Horizon-7B

As used: llama-server, reasoning effort high (the user, 2026-10-15), K2's sampling (temperature
1.0, top_p 0.95), budget 8192. K2 can't turn thinking off, so candor refuses a thinking-off search
for it; step 2 shows whether an as-used search is affordable. References: the unedited model and
the two public edits, geantendormi's Uno-Uncensored (GGUF only, claims 0% refusals) and
Avicennasis's heretic edit (7/50 refused on its own prompts).

| Step | What | State |
| --- | --- | --- |
| 1 | Code: K2's thinking, `deployment.chat_template_kwargs`, the `--lora` list limit, `--outside` GGUF files, IFM's remote code read (no network, file or process access) | Done 2026-10-15 |
| 2 | Timed pilot on a pod: build the run, time the unedited model thinking, a real quote | Next, about $2-3 |
| 3 | The unedited model and both public edits measured as used | About $10-14, a guess |
| 4 | Decision rule from step 3, signed off by the user before any search | |
| 5 | Search, the best scored as used, finalists exported and measured in full | About $15-25, a guess |
| 6 | The user's test in llama-server, then a release decision | |

## Parked, to review later

Every idea not acted on, with what supports it (measured, published or a hypothesis), is in
`results/findings.md`, beside what candor has found so far. Outside work kept for later (papers,
tools) is in `results/new-features.md`.

## Not in this plan

- `tune/heretic-kit` is frozen (our kit; heretic itself is checked at each phase start).
- Cut for depth first: `candor doctor`, a live dashboard, one-command pod orchestration, rewriting
  the guide, more model families.
- Reporting the llama.cpp prompt-cache bug upstream.
- Separate tracks: the Qwen3.5 rude fine-tune, the Mac Studio purchase.
