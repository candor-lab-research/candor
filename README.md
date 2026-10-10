# candor

Removes a language model's refusal direction. It is being built to score every candidate edit the
way you actually run the model: in llama.cpp, on the quantised GGUF, at your reasoning level and
sampling.

How it works:

1. Read the residual stream for harmful and harmless prompts with transformers, once, and take
   the difference of means per layer. The prompts are tokenized by llama-server itself, so the
   directions see exactly the input that scoring and your apps use.
2. Turn the edit into rank-1 LoRA adapters, one per edit target (each weight that writes into the
   residual stream) for each candidate direction.
3. Load them all into llama-server and score each candidate by setting per-layer strengths per
   request, so nothing is re-converted between trials. (Being built: phase 2, step 3.)
4. Merge the chosen edit straight into the safetensors files, copying every other file untouched.
   (Being built: phase 2, step 5.)

Answers to harmful prompts are only ever written to a run's `private/` folder. Commands print
counts and measurements, never generated text.

Status: early development; the plan and what is done are in [STATUS.md](STATUS.md).

## Use

```
uv sync
uv run candor inspect Qwen/Qwen3.5-9B          # check a model; a Hugging Face id or a local folder
uv run candor init qwen --model Qwen/Qwen3.5-9B  # start a run: writes runs/qwen/candor.toml
uv run candor run runs/qwen                    # carry the run through every stage built so far
```

- `inspect` reads only the config, tokenizer and file headers, so it is quick and needs no GPU. Its
  check that Bionic and LM Studio can parse the chat template needs Node.js and a one-time
  `npm install --prefix src/candor/js`; without them that check is reported as skipped.
- `init` runs the same checks and refuses a model candor can't edit. `candor.toml` holds every
  setting the run uses, each explained by a comment; edit it freely, and every later command checks
  it. Hugging Face models are pinned to the commit that was checked.
- `run` resumes where it stopped. Each result in the run folder records what it was built from, so
  changing a setting rebuilds what it affects and everything after it, and says so.

A run folder holds `candor.toml`, `inspect.json`, `gguf/` (the converted model), `prompts.json`,
`directions.pt`, `adapters/` and `logs/`. llama.cpp itself (release b11524: the binaries, checked
by sha256, and its converter in its own environment) is fetched once into `~/.cache/candor`, or
`CANDOR_CACHE`; `CANDOR_LLAMA_BIN` points at an existing b11524 build instead.

## Development

```
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

`CANDOR_E2E=1 uv run pytest tests/test_e2e.py` also runs a small real model (SmolLM2-135M) end to
end: conversion with llama.cpp's converter, the edit-target and tokenization checks, and
`candor init` and `candor run`, including resuming and a settings change. The first run downloads
about 270 MB and sets up the converter.

`spikes/` holds the one-off experiments behind the log in STATUS.md (phase 0 on a pod, rebuilding a
heretic trial, scoring heretic's saved model); they aren't part of the library.

The engine tests run a tiny random model in llama.cpp's `llama-server`, release b11524. They look
for it at `~/Desktop/tune/llama.cpp/llama-b11524/llama-server`, or wherever `CANDOR_LLAMA_SERVER`
points, and are skipped if it isn't there.
