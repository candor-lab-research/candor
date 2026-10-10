# Status

This file is the single plan for candor. Anything not listed under Plan is not being worked on.

## Plan

| Phase | What | Done when | State |
| --- | --- | --- | --- |
| 0 | Score edits as per-layer LoRA adapters in llama.cpp | Adapters match merged weights, no speed cost | Done |
| 1 | Core maths, and `candor init` / `candor inspect` | Rebuilds heretic's numbers exactly (done); `inspect` reports correctly on rude-llama, Qwen3.5-9B and K2-Horizon | In progress |
| 2 | `candor run` on the Mac: search scored in llama.cpp, then export | On rude-llama it finds an edit at least as good as heretic's Trial 170 measured the real way (at most 4/100 refusals, KL at most 0.038, no maths loss), and the file loads in Bionic | Next |
| 3 | The same run on a Runpod pod with one command and a budget limit | Qwen3.5-9B end to end under a set budget, pod deleted automatically | |
| 4 | Thinking models, scored at your real Bionic settings | K2-Horizon measured honestly at your reasoning level and sampling, then improved | |
| 5 | Docs, `candor doctor`, live dashboard, the guide rewritten around candor | You run a new model without me | |

Pods come before thinking models because K2-Horizon is too big for the Mac and needs several GPU
runs.

### Next three steps

1. An engine test against a tiny model in the Mac's llama-server (seconds, no real model).
2. `candor inspect`: begin-of-text and date checks (written), whether Bionic can parse the chat
   template, remote code, hidden states, llama.cpp support, and edit targets found by shape.
3. `candor init`: one TOML file per run with every setting, search ranges included.

## Research changes and where they land

From the 2026-10-10 paper review. Each paper gets read before its change is built.

| Finding | Change | Phase |
| --- | --- | --- |
| Find edit targets by tensor shape, not name; Qwen3.5's linear attention needs its own | `inspect` lists targets by shape and fails if any layer is uncovered | 1 |
| Keyword judges miscount refusals, worst on reasoning output | Judge with Qwen3Guard-Gen-0.6B on the final answer; keep keywords as a cross-check, compared by agreement counts only | 2 |
| 100 prompts are too noisy to rank trials | Search on 100 with fixed seeds, re-score the best trials on 300-400 held-out prompts | 2 |
| First-token KL doesn't predict lost ability | Final candidates must pass a GSM8K subset and a multiple-choice flip check | 2 |
| Re-quantising after an edit can bring refusals back | The last check runs on the exact file you ship, with your KV cache type | 2 |
| Thinking models decide during reasoning | Directions from thinking-on runs (averaged over reasoning, think start, think close); never score with thinking off; score at the highest level you use; allow strengths above 1 | 4 |
| Multi-direction edits, cleaned directions | Optional, only if phase 4 falls short (most evidence is from steering, not weight edits) | Later |

## Not in this plan

- heretic and `tune/heretic-kit`: frozen, no new features. The guide stays as reference until phase 5.
- `tune/abliterate.py`: deleted 2026-10-10. candor does the same job.
- The guide's section 14 "plan for K2": replaced by phase 4.
- Reporting the llama.cpp prompt-cache bug upstream: not wanted.
- Separate tracks, outside candor: the Qwen3.5 rude fine-tune, the Mac Studio purchase, and the
  optional check of how many begin-of-text tokens Bionic's GGUF engine sends.

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
  record which adapters computed a prompt (the engine uses `--cache-ram 0` and
  `cache_prompt: false`; without it 3 refusals became 28).

Numbers: `runs/phase0/results.json` (not in git). Pod: A40, about 23 minutes, about $0.23.

### Phase 1 so far (2026-10-10)

- `candor.heretic` reads heretic's Optuna journals; `spikes/reproduce_trial.py` rebuilds a trial
  with candor's own direction finding and edit. On rude-llama Trial 170 it gets heretic's 2/100.
- Heretic tokenizes the rendered chat with the tokenizer's own special tokens added, so Llama 3
  and K2-Horizon get two begin-of-text tokens. Proved by running heretic's own `Model.generate`
  with a recording stub. transformers, mlx-lm, Bionic's MLX engine (`model_kit.py`) and llama.cpp
  (`chat.cpp`) all send one. Qwen3.5 has none, so it is unaffected.
- `spikes/compare_models.py` scored the saved heretic model (`models/local/rude-llama-heretic`)
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
