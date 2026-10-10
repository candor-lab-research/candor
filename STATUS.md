# Status

## Phase 0: per-layer LoRA scoring in llama.cpp (done 2026-10-10)

The design works. On Llama-3.2-3B-Instruct with rude-llama's Trial 170 layer profile, scoring an
edit as per-layer rank-1 adapters with per-request scales gives the same results as merging it.

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

Two llama-server traps, both handled in `engine.py`:

1. `--lora-init-without-apply` leaves every adapter's default scale at 1, so a request naming no
   adapters gets all of them. The engine sets the defaults to 0 and reads them back.
2. Prompt caches don't record which adapters computed a prompt, so reusing one mixes an unedited
   prompt with an edited answer (3 refusals became 28). The engine runs with `--cache-ram 0` and
   `cache_prompt: false`.

Numbers: `runs/phase0/results.json` (not in git). Pod: A40, about 23 minutes, about $0.23.

## Phase 1: core maths vs heretic (in progress)

Done 2026-10-10:

- `candor.heretic` reads heretic's Optuna journals; `spikes/reproduce_trial.py` rebuilds a trial
  with candor's own direction finding and edit. On rude-llama Trial 170 it gets heretic's 2/100.
- Heretic tokenizes the rendered chat with the tokenizer's own special tokens added, so Llama 3
  and K2-Horizon get two begin-of-text tokens. Proved by running heretic's own `Model.generate`
  with a recording stub: its input is exactly one extra token in front of the canonical input.
  transformers, mlx-lm, Bionic's MLX engine (`model_kit.py`) and llama.cpp (`chat.cpp`) all send
  one. Qwen3.5 has none, so it is unaffected.
- `spikes/compare_models.py` scored the saved heretic model (`models/local/rude-llama-heretic`)
  against the original with heretic's test prompts:

| Input | Date in prompt | Batch | Refusals, original | Refusals, edited | KL |
| --- | --- | --- | --- | --- | --- |
| heretic's (2 tokens) | 09 Oct 2026 | 64 | 96 | 2 | 0.0260 |
| as apps send it (1 token) | 09 Oct 2026 | 64 | 89 | 3 | 0.0375 |
| as apps send it (1 token) | 10 Oct 2026 | 32 | 91 | 4 | 0.0384 |

  The first row is exactly heretic's journal, so every difference is explained: the extra token,
  the date the Llama 3 template writes into each prompt, and batch-size noise of about 1-2
  refusals. candor's refusal judge and heretic's agreed on all 800 saved answers.
- `candor.checks` reports both traps for any model: `begin_of_text` (what transformers, llama.cpp
  and heretic each send) and `writes_the_date`.

Next:

- An engine integration test with a tiny GGUF on the Mac's llama-server.
- The first CLI commands: `candor init`, `candor inspect`.
