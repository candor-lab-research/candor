# Which models to release next: the gap (2026-10-23)

Counted again on 2026-10-27 with image-text-to-text models, the picks holding: `evidence-2026-10-27/gaps.md`.

The user's decision (`history.md`, 2026-10-23): pick the next models by gap, a base many people run
as a GGUF with few uncensored versions so far, rather than racing on the most crowded bases.

## Our four releases

Hugging Face's download counts for each repo as of 12:30 UTC, divided by the time since the repo
was created. Each repo holds one GGUF, so no repo counts several files. The gaps are from
`gaps.tsv` (14:48 UTC).

| Release | Live (days) | Downloads | Per day | Gap | Gap rank (of 213) | Gap without the top repo |
| --- | --- | --- | --- | --- | --- | --- |
| K2-Horizon-7B | 4.94 | 907 | 184 | 10,559 | 24 | 5,002 |
| Qwen3.8-27B | 0.90 | 120 | 134 | 39,034 | 6 | 28,244 |
| Qwen3.5-9B | 7.30 | 544 | 75 | 5,583 | 43 | 2,371 |
| Llama-3.2-3B | 9.18 | 425 | 46 | 4,902 | 54 | 3,242 |

The gap orders 5 of the 6 pairs as their downloads per day do: the exception is K2 against the 27B,
whose rate is from its first day, when downloads are usually highest. Without each base's top GGUF
repo it orders 4, Qwen3.5-9B against Llama also out of order. Four releases can't validate a
ranking: the next release's downloads against its gap is the test (`findings.md`).

The gap assumes an equal share. On the two crowded Qwen bases our releases took 3-7 times less than
that of their uncensored downloads (0.07% of Qwen3.5-9B's against 0.24%; 0.03% of the 27B's against
0.19%); on K2 more (24% against 17%) and on Llama about the same (1.3% against 1.0%). Ours are GGUF
downloads scaled to 30 days, theirs Hugging Face's 30-day counts in every format, which lowers our
share everywhere; the 27B's first-day rate raises it.

## The gap

`gaps.py` writes `gaps.tsv`. The candidates are the 1,000 most downloaded and the 1,000 most trending
text-generation models on the Hub, 1,680 in all. 213 are left once each is required to be an
original (not a quantisation, merge, adapter, test model or someone else's fine-tune), a chat model,
an architecture llama.cpp converts, dense by its config.json (no total-experts count above 1), and
of a known size of at most 36B parameters. Over 30 days:

- **GGUF demand:** downloads of the GGUF repos that name the model as the base they quantise, not
  counting uncensored ones.
- **Uncensored supply:** repos whose names or tags say uncensored ("uncensor", "abliterat",
  "heretic", "derestrict", "decensor" or "refusal" in the name; tagged uncensored or abliterated),
  found by the model's name (also without a "-BF16" or "-hf" ending, or a first word repeating its
  org), as GGUFs of it or as fine-tunes of it (ours left out).
- **Gap:** demand divided by (1 + uncensored repos), which is roughly what a new entrant could reach
  if it took an equal share. **Without the top repo:** the same with the largest GGUF repo set aside,
  as one repo can hold most of a model's downloads (over half on 106 of the 165 models with at least
  10,000, 90% or more on 14).

## The shortlist: released in the last 12 months, by gap

| Model | Params (B) | GGUF downloads | Top repo | Uncensored repos | Their downloads | Gap | Without the top repo | `candor inspect` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ibm-granite/granite-4.2-30b | 29.3 | 238,240 | 97% | 2 | 477 | 79,413 | 2,463 | thinking fails |
| XHToken/Spark-X2.5-4B | 4.1 | 1,414,642 | 49% | 17 | 201,386 | 78,591 | 40,400 | thinking fails; remote code |
| XHToken/Spark-X2.5-1.7B | 1.7 | 228,925 | 98% | 4 | 2,140 | 45,785 | 794 | thinking fails; remote code |
| ibm-granite/granite-4.2-3b | 3.7 | 378,974 | 67% | 12 | 6,761 | 29,152 | 9,600 | thinking fails |
| InternScience/Agents-A1-4B | 4.5 | 404,859 | 54% | 15 | 5,625 | 25,304 | 11,588 | passes |
| ibm-granite/granite-4.2-8b | 8.8 | 262,539 | 96% | 12 | 8,356 | 20,195 | 888 | thinking fails |

Qwen3.8-27B, already released, would rank fourth. Granite's and Spark's "thinking fails" means
candor doesn't yet handle their chat templates' thinking switch (the template turns thinking off,
but the model can still open it itself): per-model code, like K2's step 1, not a blocker.

Granite 4.2 30B's lead rests on one repo: IBM's own GGUF (24 likes) has 230,852 of its GGUF
downloads, six times the base model's 38,763; bartowski's has 2,806. Set aside, its gap is 2,463,
below K2-Horizon-7B's 5,002. Without top repos the recent dense leaders not yet released are
Spark-X2.5-4B (17 uncensored repos already, with 201,386 downloads) and Agents-A1-4B (15, with
5,625).

## Mixture-of-experts models

`moe.py` runs the same count on what `gaps.py` leaves out: 132 models that are MoE (by the same
total-experts test) or dense over 36B (`moe.tsv`). candor's MoE support was built on 2026-10-23
(`moe-support.md`, `history.md`); no real MoE model is measured yet.

| Group | Models | GGUF downloads | Uncensored downloads | Sum of gaps | Without top repos |
| --- | --- | --- | --- | --- | --- |
| Dense, 36B or less | 213 | 44,172,968 | 27,489,488 | 1,128,162 | 411,884 |
| MoE, 36B or less | 48 | 9,288,491 | 1,302,705 | 601,187 | 137,066 |
| MoE, 36-130B | 18 | 1,879,612 | 429,155 | 119,955 | 38,021 |
| MoE, over 130B | 48 | 1,364,412 | 2,541,727 | 246,644 | 54,058 |
| Dense, over 36B | 18 | 491,672 | 559,547 | 47,600 | 21,133 |

MoE models of 36B or less hold a third to a half of the dense models' gap (53%, or 33% without top
repos): a fifth of the GGUF downloads, with thin uncensored supply (14% of those against 62%).
Uncensored counts cover every format while demand is GGUF only, and a search for one model also
finds later versions' repos (GLM-5's finds GLM-5.3's), which is why the over-130B row shows more
uncensored downloads than GGUF ones.

poolside/Laguna-XS-2.1 (33.4B, 256 experts, 8 active) leads the MoE models either way. 88% of its
1,723,672 GGUF downloads are one repo, mudler's APEX GGUF (28 likes), with 1,510,641, three times
the base model's 499,646; poolside's own GGUF (100 likes) has 189,971. LocalAI's model gallery
installs both, so much of the demand may be automated: without both its gap is 3,843
(`first-moe.md`). Its 5 uncensored versions have 1,104 downloads between them, the only GGUF 564.
Without the top repo its gap is 35,505, more than any recent model's but Spark-X2.5-4B's. Its sibling Laguna-S-2.1 (117.6B) has 25 uncensored
versions with 45,547 downloads. transformers has Laguna natively (`moe-support.md`). The next MoE
gaps without top repos are Mellum2-12B-A2.5B-Instruct (18,686; a coding model) and Ornith-1.5-35B-A3B
(14,069; 84 uncensored repos); Nemotron-3.5-Lightning-30B-A3B has 27 uncensored repos (11,348
downloads), so 3,974.

## Limits

- The counts include bots and mirrors, and can't show unique users; one repo can hold most of a
  model's downloads, as on Granite 4.2 and Laguna-XS. Old models keep high GGUF counts
  (Mistral-7B-Instruct-v0.1 had 44,246 with no uncensored version), which is why the shortlist
  takes only the last 12 months.
- An uncensored repo that is neither named after the model, nor a GGUF or fine-tune of it, nor named
  or tagged uncensored is missed. HF search returns at most 1,000 results, so counts on the most
  crowded bases are lower bounds.
- A gap closes: popular new bases get uncensored versions within days, so it needs reading again
  just before a model is chosen.
- Nothing here measures how often a model refuses. The unedited model measured as used shows
  whether an edit has anything to give.
