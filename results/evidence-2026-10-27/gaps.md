# The gap counted again with image-text-to-text models (2026-10-27)

`gaps.py` and `moe.py` (`evidence-2026-10-23/`) took candidates from the Hub's text-generation models
only, but chat models that also take images are tagged image-text-to-text, as Qwen3.6's, Gemma 4's
and Qwen3.8-27B's are. Here the same filters and counts run on both pipelines (`gaps.tsv` 09:53 UTC,
`moe.tsv` 10:01 UTC): 3,213 models (3,214 for `moe.py`), 293 dense of 36B or less and 178 MoE or over
36B, against 213 and 132 on 2026-10-23. The pools aren't the old ones plus the new pipeline: of the
dense, 110 came in (92 image-text-to-text, 18 from four days of change in the top lists) and 30 left;
of the MoE or over 36B, 51 came in (37 and 14) and 5 left. Numbers only; no weights, no pods.

## Bottom line

Neither pick changes. The new entries at the top are task models (from the new pipeline) or
fine-tunes taken for originals (text-generation, from the lists' change), and the two newest general
chat families brought in are crowded.

- **Dense:** the shortlist in `evidence-2026-10-23/gaps.md` holds but for one: Granite 4.2 30B gap
  81,683 (2,657 without IBM's own GGUF), Spark-X2.5-4B 85,342 (39,750), Granite 4.2 3B 30,089
  (9,651), Agents-A1-4B 24,862 (11,505), Granite 4.2 8B 21,198 (1,058). Spark-X2.5-1.7B has left the
  pool.
- **First MoE:** Qwen3-30B-A3B-Instruct-2507 stays the recommendation (`evidence-2026-10-23/first-moe.md`):
  gap 18,663 (5,671), 25 uncensored repos with 151,474 downloads in 30 days, 24% of the family's. By
  `first-moe.md`'s reckoning a release would draw about 66-194 downloads a day.

## What the re-run adds

- **Crowded bases:** Gemma 4 31B (gap 1,004, 420 uncensored repos), Gemma 4 26B-A4B (4,812, 394),
  Qwen3.6-27B (6,997, 295), Qwen3.6-35B-A3B (9,310, 364).
- **Task models, no uncensored versions:** TranslateGemma 4B and 12B (gaps 111,814 and 63,832),
  MedGemma 1.5 4B, and OCR models (LightOnOCR-3-4B, chandra-ocr-2, HunyuanOCR, Unlimited-OCR). They
  are not general chat models, so set aside.
- **Fine-tunes the filter takes for originals:** some name their base model only in the card, not
  in a `base_model:finetune:` tag (ReadyArt's gemma-4-31B-it-scotoma-2). Others are fine-tunes of
  their own org's earlier fine-tunes, which the filter allows (FINAL-Bench's Darwin-27B-RSI, logic65's
  Whittle-Qwen-3.8-35B-A3B). Kwaipilot's KAT-Coder-V2.5-Dev, a coding model
  built on Qwen3.6-35B-A3B, now leads the MoE gap without its top repo (131,343; 36,295). But its 8
  uncensored repos hold 1.0% of its downloads, and a release would draw about 5-44 downloads a day. It
  passes `candor inspect` and thinks before answering.
- Laguna-XS-2.1 still leads the MoE gap (281,258; 35,520), and its uncensored versions still take
  0.07%: about 0.5-6 a day for a release.
