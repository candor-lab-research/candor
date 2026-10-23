# Development

```
uv sync
uv run pytest -n 6
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

## Tests

**Engine tests** run a tiny random model in the llama-server candor uses: `CANDOR_LLAMA_BIN`'s, or the
one in candor's cache. Run candor once to install it; without it those tests are skipped.

**End to end.** `CANDOR_E2E=1 uv run pytest tests/test_e2e.py` also runs small real and tiny random
models end to end:

- conversion with llama.cpp's converter, including a tokenizer saved by transformers 5 and a tiny
  Qwen3.5;
- the edit-target and tokenization checks;
- `init`, `run` (resuming and a settings change), `measure` and `compare` (the judge standing in for
  the 8B second judge), `search`, `score` and `export` on SmolLM2-135M;
- a real thinking model's budget on Qwen3.5-0.8B, Qwen3.8-27B's `init`, and the judge.

The first run downloads about 3.6 GB (Qwen3.5-0.8B 1.8 GB, the judge 1.5 GB, SmolLM2-135M 0.27 GB),
and llama.cpp and its converter environment if candor hasn't fetched them yet. Nothing else runs it;
`STATUS.md` says when.

## Tools and history

- `tools/misscheck/` is the miss-rate simulator the search's design is judged on. Its README says how
  to run each part and which candor commit each result used.
- The one-off experiments behind the log in `results/history.md` (phase 0 on a pod, rebuilding a
  heretic trial, scoring heretic's saved model) were in `spikes/`, last in commit `2bb6bba`.

## Images

- `docs/images/draw.py` draws the README's diagrams, in a light and a dark version each.
- `docs/images/brand.py` draws the mark (`mark.svg`, also the profile picture) and the banner. Their
  type is outlined from Source Serif 4 and Inter, under the SIL Open Font License, fetched at a
  pinned commit and checked by sha256.

```
uv run python docs/images/draw.py
uv run --with fonttools --with uharfbuzz python docs/images/brand.py
```
