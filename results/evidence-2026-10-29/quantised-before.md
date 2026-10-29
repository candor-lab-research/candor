# Weights quantised before: what `inspect`'s check reads (2026-10-29)

`quantised-before.py` runs `candor inspect`'s statistic on Hub models at their current commit, by
ranged reads alone: the first rows (at least 4,096 values) of the 4 edit targets `inspect` picks
across the layers. Each row is cut into blocks of 32 raw 16-bit values; zeros of either sign are left
out, and only blocks holding more than 16 values other than zero count. A weight is flagged when the
median block holds 16 distinct values or fewer. Numbers in `quantised-before.tsv`; no weights kept.

## Bottom line

- **Weights never quantised read 32, and none is flagged.** 9 BF16 models, 36 weights: every median
  32, the lowest block 27 (Qwen3.5-9B, Qwen3-4B, Granite 4.2 8B, Mistral-7B-Instruct-v0.1,
  Phi-4-mini-instruct, deepseek-coder-6.7b-instruct, Qwen2.5-32B-Instruct,
  Qwen3-30B-A3B-Instruct-2507's experts and Qwen3.6-35B-A3B's stacked and shared experts).
- **Weights 4-bit before read far below 16.** unsloth's gpt-oss-20b-BF16, whose experts were MXFP4:
  its three expert `down_proj` weights read medians 12, 13 and 8 (lowest 10, 8, 6), all flagged; its
  attention `o_proj`, never 4-bit, reads 32 and isn't.
- rude-llama, whose MLX 4-bit origin cost most of an edit at Q4_K, held at most 16 values in every
  64-weight group of layers 0-19 (`phase-2-rude-llama.md`, the export); its local folder is gone, so
  it wasn't read here.

| Model | Commit | Weights read | Median | Lowest | Flagged |
| --- | --- | --- | --- | --- | --- |
| 9 BF16 models above | `quantised-before.tsv` | 36 | 32 | 27 | 0 of 36 |
| gpt-oss-20b-BF16, experts | 5a6fe4c0 | 3 | 8-13 | 6-10 | 3 of 3 |
| gpt-oss-20b-BF16, attention | 5a6fe4c0 | 1 | 32 | 29 | 0 of 1 |

Not tested: 8-bit or FP8 origins (Q8_0 holds them, so no warning is wanted) and weights mostly zero
(the zero rule is there for them; a unit test covers one).

Repeat: `uv run python results/evidence-2026-10-29/quantised-before.py MODEL ...` from the repository.
