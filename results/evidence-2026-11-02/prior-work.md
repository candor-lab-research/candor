# Prior work, checked before candor calls anything new (2026-11-02)

One agent's web search of arXiv, Hugging Face cards and the code and issues of heretic, abliterix
and heretic-gguf. Each paper's title and date were then checked against arXiv, each card and repo
against its page, and heretic's ARA and SOMA against its repo. No prompt lists or model outputs
were opened. Most of the cards' numbers are self-reported.

## What candor does that others have done

- **Scoring the search on the quantised GGUF in llama.cpp: done before.** heretic-gguf
  ([MoriNoNushi/heretic-gguf](https://github.com/MoriNoNushi/heretic-gguf), created 2026-08-09,
  AGPL-3.0) runs heretic's Optuna search on a quantised GGUF, each trial a rank-1 LoRA GGUF swapped
  into a running llama.cpp, with KL against the quantised model. MorinoNushi's
  GLM-5.3-Flash-Heretic-Abliterated-LoRA-V2-GGUF (2026-09-12) was searched in llama-server on a
  4-bit quant, and khimaros's gemma-4-E2B hierophant LoRA (2026-11-01) is built from a quant's own
  weights. Per-request LoRA scales are llama-server's own (llama.cpp PR 10994, 2025-01-02). candor
  setting strengths per request, rather than writing an adapter per trial, is a detail.
- **Quantisation and refusals.** Nothing found that measures quantising undoing an abliteration.
  "Quality Is Not a Safety Proxy Under Quantization" (arXiv 2606.10154, 2026-06-08): quantising cut
  unedited models' refusals by 12-68 points in 10 of 51 rows while quality held. "Mind the Gap"
  (arXiv 2505.23786, ICML 2025): GGUF rounding can be rigged to change behaviour.
- **Thinking models as used: partly.**
  - "Where Do Reasoning Models Refuse?" (arXiv 2507.03167, 2025-07-03) ablates the refusal
    direction in 4 reasoning models thinking on, judges the final answer with StrongREJECT and
    scores unfinished thinking 0. It evaluates; it doesn't search.
  - MorinoNushi's GLM-5.3-Flash V2 and V3 cards search thinking on, capped at 1,280 tokens, with
    refusals by keyword after the end tag and KL on the first token.
  - DreamFast evaluates thinking on to 8,096 tokens and counts unfinished thinking. heretic instead
    pre-fills a closed thinking block. A forced close is s1's budget forcing (arXiv 2501.19393).
  - Nothing found for KL read after the model's own thinking, or budget hits as a search cap.
- **Measurement: each part exists, not the whole.**
  - DreamFast's abliterlitics cards (from 2026-07, e.g. DreamFast/Gemma4-e4b-abliterlitics)
    measure 7-23 public releases per base: HarmBench with an LLM judge, GSM8K, first-token KL,
    thinking on, all in BF16 (one release converted back from its GGUF), with no significance tests
    and no rule set before the results.
  - orcarouter's cards (e.g. DeepSeek-V4-Flash-Vision-Uncensored, 2026-09-02) report StrongREJECT,
    GSM8K and exact McNemar tests, on their own releases, with refusals by phrase.
  - arXiv 2512.13655 (2025-12-15) compares 4 tools on 16 models, refusals by keyword, with
    intervals but no paired tests. arXiv 2607.17427 (2026-07-19) pre-registers its measure, of side
    effects rather than refusals.
  - LLM judges: abliterix (optional, Gemini), heretic (a refusal classifier); Qwen3Guard's own
    Refusal label. The UGI leaderboard keeps its questions private and publishes no judging method.

**What candor may say:** that it is not aware of anyone measuring other people's releases as they
are downloaded and used (the GGUF, thinking on to a budget) with held-out prompts, two LLM judges,
paired tests and a rule fixed before the results. Nothing else it does is new.

## Methods beyond one direction

| Method | Source | In short |
| --- | --- | --- |
| Concept cones | arXiv 2502.17420 (2025-02-24, ICML 2025) | Refusal is mediated by several independent directions and cones, found by gradient search |
| Several directions by self-organising maps (SOMA) | heretic PR 196, closed unmerged | 4 directions on gemma-3-12b: 3/100 at KL 0.08 against heretic's 0.16, self-reported |
| Per-category directions | arXiv 2602.02132 (2026-02-02) | Categories have distinct directions, but steering along any gives nearly the same refusal trade-off |
| Projected, and norm-preserving biprojected, abliteration (MPOA) | grimjim's two HF blogs (2025); an option in heretic | The direction cleared of its part along the harmless mean; weight rows keep their norms |
| Arbitrary-Rank Ablation (ARA) | heretic PR 211; heretic 2.0.0's default (2026-11-01) | No direction: each module's matrix optimised so harmless outputs stay and harmful ones move to them |
| Surgical Refusal Ablation | arXiv 2601.08489 (2026-01-13) | The direction cleaned of its overlap with capabilities to protect |
| Refusal cliff | arXiv 2510.06036 (2025-10-07) | Reasoning models hold a refusal intention through the thinking and drop it at the end; attention heads named |
| Conditional activation steering (CAST) | arXiv 2409.05907 (2024-09-06) | Steering applied only when the input matches a condition, at inference |

heretic (and heretic-gguf) are AGPL-3.0, candor Apache 2.0: candor can't take their code.
