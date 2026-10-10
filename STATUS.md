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

Done 2026-10-10: `spikes/reproduce_trial.py` rebuilds a heretic trial from its journal with
candor's own direction finding and edit, then scores it the way heretic does. rude-llama Trial 170,
on the Mac (MPS, float16, about 3 minutes):

| | heretic | candor, heretic's tokens | candor |
| --- | --- | --- | --- |
| Begin-of-text tokens per prompt | 2 | 2 | 1 |
| Refusals, original | 96 | 97 | 91 |
| Refusals, edited | 2 | 2 | 2 |
| KL | 0.026 | 0.030 | 0.044 |

candor reproduces heretic once it is fed the same tokens. Heretic lets the tokenizer add its own
begin-of-text token in front of the chat template's, so Llama 3 sees two. With the single token
it's really run with, Trial 170's KL is 0.044.

Next:

- An engine integration test with a tiny GGUF on the Mac's llama-server.
- The first CLI commands: `candor init`, `candor inspect`.
