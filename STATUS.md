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
| 0-5 | LoRA scoring in llama.cpp, the core maths and `inspect`; rude-llama, Qwen3.5-9B, K2-Horizon-7B and Qwen3.8-27B searched, measured as used and released | Done 2026-10-27 (history; `results/phase-2` to `phase-5`, the milestones) |
| 6 | Spark-X2.5-4B, smaller and newer, the largest recent gap (the user, 2026-10-29); Flash-Next's support kept for when there is credit | Pilot done (`results/phase-6-spark-x2.5-4b.md`, step 1); search stopped at 93 trials (step 2); its three finalists being measured |

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
  `results/evidence-2026-10-14/ara-missed.md`). Tools checked 2026-10-27: heretic master d34dd919
  (default edit ARA); last release 1.4.0; its thinking-model prefix bug (#487) is open.
- At each phase end, run the e2e test (`CANDOR_E2E=1 uv run pytest tests/test_e2e.py`): nothing else
  runs it.
- Next models are picked by gap (the user, 2026-10-23): `results/evidence-2026-10-23/gaps.md`,
  counted again in `evidence-2026-10-27/gaps.md`.
- Anything visual (README, docs, diagrams, model cards, pages, charts) follows `docs/brand.md` (the
  user, 2026-10-23).

## Now: Spark-X2.5-4B's finalists measured

- The same A40 (`sk7e6bm640dlrt`), from 21:57 UTC on 2026-10-29, runs `/root/finalists.sh`: the
  unedited model measured in full, trial 18 scored past its MLP floor, then trials 53, 70 and 18
  exported and measured; about 9-11 h, $5.50-6.50. The watchdog deletes it at 19:44 GMT on
  2026-10-30; `pod.sh wait finalists` pulls every 30 minutes.
- The decision rule is the 27B's, ranked on unsafe answers (history, 2026-10-19 and 2026-10-23),
  put to the user before any finalist's measurement is read.

## Next

1. Phase 6: the reference, the best public edit, measured with `--outside` as candor ships its own,
   on the user's go (the most downloaded uncensored GGUF is soyaakinohara's Heretic-jp, 317,078).
2. Phase 4, later: low effort's two closes; Uno not now. The budget limit
   (`search.max_budget_hits`, on K2 from trial 138) costs no measurable hits on the simulator and
   stays on (`phase-4-k2-horizon-7b.md`, step 5).
3. Not blocking, each when there is a reason:
   - speed: weights cached on the GPU for "full" trials, the proxy's
     adapters loaded into one server, package 11 (the answer reusing its thinking's cache, untimed;
     skipped for the 27B);
   - a reference measured before 2026-10-19 is measured again before a new measurement is compared
     with it (`compare` refuses the pair; `docs/commands.md`);
   - the search scoring on the base it ships, edited tensors at Q8_0 too (only where quantisation
     loses edits, as on rude-llama), and whether it should score more prompts (per-layer refusals
     and layer-14 KL didn't hold on rude-llama's held-out prompts);
   - llama-server's chat endpoint may count prompt tokens against the thinking budget (seen on a
     tiny model only); K2's budget message was written for a close that starts with a newline;
   - measurements have no DEPENDS entry, so a change to the directions or adapters code answers even
     the unedited model's sets again (seen at the 2026-10-29 directions change);
   - `serve` on a pod with the Mac's tools (step 6 bridged them by hand,
     `evidence-2026-10-22/bridge.py`; the tools are at `~/Documents/Projects/LLM/tune/tools`);
   - MoE: memory rows for Laguna and Qwen3.5-MoE, Granite's fused `input_linear`, NemotronH's layers
     (`evidence-2026-10-23/moe-support.md`, section 4), each with its model;
   - heretic 2.x journals aren't read (they record no refusals for a trial);
   - llama-server b11524 can deadlock when stopped mid-answer (about 1 stop in 500; history): `stop`
     then waits 60 s and kills it, losing nothing;
   - `candor score`'s refusals of a repeat paired by prompt, as its KL is, in place of the band of 2
     (one real repeat moved 2; findings), keeping each trial's refusal labels;
   - the Qwen4-Exp e2e test's tiny model needed heads of 32 for CUDA's gated delta net; a Flash-Next
     pilot re-runs it.

## In force

- Decisions use as-used numbers only: a thinking model with thinking on, its card's sampling with a
  seed per prompt, a thinking budget of 8192, then a forced close; an unfinished answer counts
  against the edit; the judge reads the final answer.
- A cheap thinking-off score may only choose which edits get the as-used test, and only once
  `candor proxy` shows it ranks edits as thinking on does, for each new model.
- References are exported and measured as candor ships its edits. Maths: the full 1,319 GSM8K
  questions on finalists only.
- The decision rule: the 27B's (2026-10-19), finalists ranked on unsafe answers from the next
  model on (both in `results/history.md`). Optuna stays 4.9.

## Not in this plan

- `tune/heretic-kit` is frozen (our kit; heretic itself is checked at each phase start).
- Cut for depth first: `candor doctor`, a live dashboard, one-command pod orchestration, rewriting
  the guide.
- Reporting the llama.cpp prompt-cache bug upstream.
- Separate tracks: the Qwen3.5 rude fine-tune, the Mac Studio purchase.
