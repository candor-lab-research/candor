# candor

Removes a language model's refusal direction, and scores every candidate edit the way you
actually run the model: in llama.cpp, on the quantised GGUF, at your reasoning level and sampling.

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

## Development

```
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run pyright
```
