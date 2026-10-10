# candor

Removes a language model's refusal direction. It is being built to score every candidate edit the
way you actually run the model: in llama.cpp, on the quantised GGUF you ship, at your settings.

How it works:

1. Read the residual stream for harmful and harmless prompts with transformers, once, and take
   the difference of means per layer. The prompts are tokenized by llama-server itself, so the
   directions see exactly the input that scoring and your apps use.
2. Turn the edit into rank-1 LoRA adapters, one per edit target (each weight that writes into the
   residual stream) for each candidate direction, in the layout llama.cpp's converter gives them.
3. Load them all into llama-server and score each candidate by setting per-layer strengths per
   request, so nothing is re-converted between trials. (Being built: phase 2, step 5.)
4. Merge the chosen edit straight into the safetensors files, copying every other file untouched,
   then convert and quantise it to the GGUF you run. (`merge.py` exists; the export stage is phase 2,
   step 6.)

Answers to harmful prompts are only ever written to a run's `private/` folder. Commands print
counts and measurements, never generated text.

Status: early development; the plan and what is done are in [STATUS.md](STATUS.md).

## Use

```
uv sync
uv run candor inspect ~/Desktop/tune/models/rude-llama-bf16   # a local folder or a Hugging Face id
uv run candor init rude --model ~/Desktop/tune/models/rude-llama-bf16   # writes runs/rude/candor.toml
uv run candor run runs/rude                                    # every stage built so far
uv run candor upgrade runs/rude                                # after updating candor: add new settings
uv run candor measure runs/rude                                # the unedited model, on held-out prompts
uv run candor measure runs/rude --heretic STUDY.jsonl --trial 170   # a heretic trial, rebuilt
uv run candor compare runs/rude base heretic-170               # prompt by prompt
```

- `inspect` reads only the config, tokenizer and file headers, so it is quick and needs no GPU. Its
  check that Bionic and LM Studio can parse the chat template needs Node.js and a one-time
  `npm install --prefix src/candor/js`; without them that check is reported as skipped.
- `init` runs the same checks and refuses a model candor can't edit. `candor.toml` holds every
  setting the run uses, each explained by a comment; edit it freely, and every later command checks
  it. Hugging Face models and the prompt datasets are pinned to the commits that were checked, and
  local models are stored by their full path. A model that only loads with the Python files it ships
  (K2-Horizon) gets `trust_remote_code = true`, which runs that code at the pinned commit.
- `run` resumes where it stopped. Each result in the run folder records what it was built from,
  including the version of candor's code for its stage, so changing a setting or upgrading candor
  rebuilds what it affects and everything after it, and says so.
- `upgrade` adds the settings a newer candor has and a run's `candor.toml` lacks, at their defaults,
  and lists them. candor never fills in a missing setting by itself.
- The held-out harmful prompts, which the search never sees, come from StrongREJECT without the
  rows it took from AdvBench. It is gated: accept its terms on its Hugging Face page, then
  `hf auth login`.
- `measure` answers the held-out prompts and GSM8K's 1,319 maths questions with the deployed GGUF
  in llama-server. Qwen3Guard-Gen-0.6B (pinned) labels each held-out answer as a refusal or not, with
  keyword counts beside it, and the first-token probabilities on the harmless prompts give KL
  divergence. It keeps one number per prompt in `measurements/NAME/`; answers and the judge's
  replies go only to `private/`. `--limit N` measures the first N prompts of each set and estimates
  the full run's time; a stopped measurement resumes; the Mac is kept awake.
- `compare A B` pairs two measurements prompt by prompt: each count, how many prompts went each
  way, and the exact McNemar p-value. Measuring the same edit twice with different `--slots` shows
  what batching noise alone does.
- Reading residuals needs the model in 16 bits on the GPU. `run` checks this before downloading or
  converting anything: rude-llama fits on an 18 GB Mac; Qwen3.5-9B and K2-Horizon need a pod.

A run folder holds `candor.toml`, `inspect.json`, `gguf/` (the model as an unquantised GGUF of its
own float type, the deployment quantisation, and `convert.log`), `prompts.json`, `directions.pt`,
`adapters/`, `measurements/`, `private/` and `logs/`, each result beside a `.inputs.json` record. llama.cpp itself (release
b11524: the binaries, checked by sha256, and its converter in its own environment, pinned by
`src/candor/data/converter-constraints.txt`) is fetched once into `~/.cache/candor`, or
`CANDOR_CACHE`; `CANDOR_LLAMA_BIN` points at an existing b11524 build instead.

## Development

```
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

`CANDOR_E2E=1 uv run pytest tests/test_e2e.py` also runs small real and tiny random models end to
end: conversion with llama.cpp's converter (including a tokenizer saved by transformers 5 and a
tiny Qwen3.5), the edit-target and tokenization checks, and `candor init` and `candor run`,
including resuming and a settings change. The first run downloads about 270 MB of SmolLM2-135M, and
llama.cpp and its converter environment if candor hasn't fetched them yet.

The engine tests run a tiny random model in the llama-server candor uses: `CANDOR_LLAMA_BIN`'s, or
the one in candor's cache. Run candor once to install it; without it those tests are skipped.

The one-off experiments behind the log in STATUS.md (phase 0 on a pod, rebuilding a heretic trial,
scoring heretic's saved model) were in `spikes/`, last in commit `2bb6bba`.
