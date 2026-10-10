# candor

Removes a language model's refusal direction. It is being built to score every candidate edit the
way you actually run the model: in llama.cpp, on the quantised GGUF, at your reasoning level and
sampling.

How it works:

1. Read the residual stream for harmful and harmless prompts with transformers, once, and take
   the difference of means per layer.
2. Turn the edit into one rank-1 LoRA adapter per layer and component.
3. Load them all into llama-server and score each candidate by setting per-layer strengths per
   request, so nothing is re-converted between trials.
4. Merge the chosen edit straight into the safetensors files, copying every other file untouched.

Answers to harmful prompts are only ever written to a run's `private/` folder. Commands print
counts and measurements, never generated text.

Status: early development, see [STATUS.md](STATUS.md).

## Use

```
uv sync
uv run candor inspect Qwen/Qwen3.5-9B        # or a local model folder
```

`inspect` reads only the config, tokenizer and file headers, so it is quick and needs no GPU. Its
check that Bionic and LM Studio can parse the chat template needs Node.js and a one-time
`npm install --prefix src/candor/js`; without them that check is reported as skipped.

## Development

```
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

`CANDOR_E2E=1 uv run pytest tests/test_e2e.py` also converts a small real model (SmolLM2-135M)
with llama.cpp's converter and checks targets and tokenization end to end; the first run downloads
about 270 MB and sets up the converter.

The engine tests run a tiny random model in llama.cpp's `llama-server`, release b11524. They look
for it at `~/Desktop/tune/llama.cpp/llama-b11524/llama-server`, or wherever `CANDOR_LLAMA_SERVER`
points, and are skipped if it isn't there.
