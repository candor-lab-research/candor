# Status

This file is the single plan for candor. Anything not listed under Plan is not being worked on.

## Plan

| Phase | What | Done when | State |
| --- | --- | --- | --- |
| 0 | Score edits as per-layer LoRA adapters in llama.cpp | Adapters match merged weights, no speed cost | Done |
| 1 | Core maths and `candor inspect` | candor's rebuild of a heretic trial matches its refusals, and measuring heretic's saved model under heretic's conditions gives heretic's numbers exactly; `inspect` reports correctly on rude-llama, Qwen3.5-9B and K2-Horizon | Done |
| 2 | `candor init` and `candor run` on the Mac: search scored in llama.cpp, then export | On rude-llama, measured side by side with heretic's saved Trial 170 model under identical conditions (same prompts, pinned date, one begin-of-text token, same engine), candor's model has no more refusals, no higher KL and no maths loss, and it loads in Bionic | In progress |
| 3 | The same run on a Runpod pod with one command and a budget limit | Qwen3.5-9B end to end under a set budget, pod deleted automatically | |
| 4 | Thinking models, scored at your real Bionic settings | K2-Horizon measured honestly at your reasoning level and sampling, then improved | |
| 5 | Docs, `candor doctor`, live dashboard, the guide rewritten around candor | You run a new model without me | |

Pods come before thinking models because K2-Horizon is too big for the Mac and needs several GPU
runs.

### Next three steps

1. Done: `candor init NAME --model SOURCE` checks the model, then writes `runs/NAME/candor.toml`
   with every setting the run uses, each with its explanation as a comment, and validates it on
   every load. It pins the date for templates that write one, refuses models candor can't edit,
   and never overwrites a run. Tested on rude-llama, Qwen3.5-9B, K2-Horizon and a refused
   mixture-of-experts model.
2. Done (2026-10-10, review verified clean): `candor run RUN` takes a run through every stage up to the
   search, resuming where it stopped: GGUF (every edit target checked byte for byte), prompt ids
   from llama-server (proven equal to its chat endpoint, the pinned date proven to take effect,
   lengths checked against the model's context), directions from transformers fed those ids
   (finite), and one adapter per edit target for each direction set. Each result records what it
   was built from, so changing a setting rebuilds what it affects and everything after it.
   SmolLM2-135M: 60 targets, 16 direction sets, 960 adapters, 1,000 prompts in about 20 s.
3. The search loop scored in llama.cpp (Optuna over the per-layer strengths), with the classifier
   judge and keyword cross-check; then the held-out re-score, the maths gate, export and the final
   check on the shipped file.

## Research changes and where they land

From the 2026-10-10 paper review. Each paper gets read before its change is built.

| Finding | Change | Phase |
| --- | --- | --- |
| Matching targets by name silently misses most of the effect on mixture-of-experts models (2609.09793) | `inspect` lists every per-layer tensor that outputs the hidden size and fails on any it can't place or any layer without a writer (done) | 1 |
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
- The engine test (`tests/test_engine.py`) runs a tiny random GGUF in llama-server. It fails if the
  default-scale reset or `cache_prompt: false` is removed. `--cache-ram 0` is kept as a second guard
  that no test has yet shown to be needed.
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

### Review of phase 1 (2026-10-10)

An independent review found 13 defects; all are fixed, each with a test (84 tests now):

- Engine: refused a port another server already holds (it would have scored the wrong model),
  checks the served model path, cleans up a server whose start fails, kills a server that won't
  stop, and gets first-token probabilities for prompts whose first token is an incomplete UTF-8
  sequence (llama-server records none for those). It bans that token and asks again: the reported
  probabilities come from the raw logits, which a test confirms the ban leaves unchanged.
- Targets: weights stored without a `.weight` suffix (fused experts) are found; an extra
  token-prediction layer past the model's layer count is accepted and reported; two stacks that
  both fit are reported; mixture-of-experts models fail clearly instead of half-mapping; norms on
  block outputs (Gemma 2/3, OLMo 2) are warned about, since the edit can't remove the direction
  exactly there.
- llama.cpp support is read from a table of the architectures llama.cpp's converter registers
  (`scripts/llama_cpp_architectures.py`, 280 at b11524), not guessed from `model_type`. Mistral
  was wrongly rejected before.
- `inspect` gives a clear message instead of a traceback for missing models, gated repos, base
  models and configs without sizes (exit code 2 when it can't inspect, 1 when a check fails),
  resolves templates given as a dict, reads remote code declared in the tokenizer config, and
  checks the template with a system message and several turns.
- Only the weight files a model's index lists are read or merged, so a stray
  `consolidated.safetensors` is never shipped unedited; FP8 weights are refused by `merge`.
- K2's template fails in Bionic only because of its `is sameas` tests; its `{% generation %}`
  block parses fine. The guide is corrected.

A second review of those fixes found one regression and five gaps, all now fixed (93 tests):

- The template check rendered a system message, so templates that deliberately refuse one
  (Gemma 2, Mistral v0.1) were reported as unreadable by Bionic. Only conversations transformers
  can render are now tried in Bionic's parser.
- `merge` now checks every target before writing anything: missing names, non-float dtypes and
  quantisation scales (an INT8 weight with a `weight_scale` lost its edit silently). It also
  leaves out weights in other formats such as `pytorch_model.bin`. `inspect` fails on quantised
  writers.
- GLM-4's output norms (`post_self_attn_layernorm`, `post_mlp_layernorm`) are now warned about.
- The llama.cpp architecture is chosen exactly as the converter does: the first top-level entry,
  replaced by `text_config.architectures` except for the five architectures the converter keeps
  at top level (also read from its source).
- `inspect` exits with 3, not 2 (typer's usage-error code), when it can't inspect a model,
  reports unexpected errors as a message, and handles a template that can't render one message.
- Known and accepted: another process could take the port in the instant between candor's check
  and llama-server binding it; the served-model check catches that unless it serves the same file.

### Review of `candor init` (2026-10-10)

A review found 9 defects and 4 weak tests; all fixed (161 tests):

- Run names must be one plain folder name, so `init ../x` can't write outside the runs folder;
  `--runs ~/...` is expanded (it used to create a folder literally named `~`).
- The settings file must spell out every setting: a missing one is an error, not a silent
  default, so a later candor release can't quietly change an existing run.
- Numbers are strict (no `true` or `"200"` for 200, no infinity), the seed is limited to what
  the search can use, and names can't be empty.
- A run folder is built under a temporary name and renamed into place, so a failure leaves
  nothing behind to block the next `init`; files are read and written as UTF-8.
- Hub models are pinned to the commit they were checked at (`revision`), so later steps use
  exactly those weights and that template.
- The date is pinned only when the template's one date is a `date_string` it accepts from outside,
  in that call's own format; otherwise `init` warns that a run must finish the day it starts.

### Review of step 2 (2026-10-10)

The review found 12 defects and 5 weak tests; all fixed (189 tests, 3 end to end), with each key
fix shown to matter by reintroducing the bug and watching its test fail:

- Every stage reused its result even after the settings it was built from changed (a new
  `row_norm` left old adapters in use). Each result now records its inputs and the records of
  the stages before it; a change rebuilds it and everything after it, and says so.
- The toolchain trusted a converter environment whose install had failed, didn't check binary
  versions, would have left the CUDA runtime in a separate folder from llama-server on pods (likely
  a silent fall back to the CPU), had no download timeout and no lock against two first runs.
- The pinned-date check couldn't fail on the day of `init`; it now renders a made-up date.
- Residuals and directions are checked to be finite (float16 on the Mac can overflow).
- Over-long prompts failed with a bare HTTP 400; they are now checked against the model's context
  and the scoring server is sized from the longest prompt.
- The disk check used a wrong ratio; it now uses the real parameter count.

Verification of those fixes took three more rounds: 8 new defects introduced by the fixes (the
length check skipped on reuse, too much rebuilt after a quantisation change, a server that won't
start re-downloaded on every run, a damaged record crashing instead of rebuilding, no sync before
a record is trusted, an oversized checking server, old GGUF files never cleared, two small toolchain
gaps), then 3 (macOS needs F_FULLFSYNC, identical files across release archives, a damaged install),
then 1 (advice to delete a user's own llama.cpp folder). The final round was clean: 201 tests and
3 end-to-end tests pass, and each key fix was shown to matter by reintroducing the bug.

Still unverified, for phase 3: installing the real CUDA release on a pod (the 594 MB runtime
archive's file list has not been checked against the binaries archive).
