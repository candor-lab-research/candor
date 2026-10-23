# Which models to release next: the gap (2026-10-23)

The user's decision (`history.md`, 2026-10-23): pick the next models by gap, a base many people run
as a GGUF with few uncensored versions so far, rather than racing on the most crowded bases.

## Our four releases

Hugging Face's download counts for each repo as of 12:30 UTC, divided by the time since the repo
was created. Each repo holds one GGUF, so no repo counts several files.

| Release | Live (days) | Downloads | Per day | Gap | Gap rank (of 242) |
| --- | --- | --- | --- | --- | --- |
| K2-Horizon-7B | 4.94 | 907 | 184 | 10,559 | 30 |
| Qwen3.8-27B | 0.90 | 120 | 134 | 68,046 | 4 |
| Qwen3.5-9B | 7.30 | 544 | 75 | 7,658 | 40 |
| Llama-3.2-3B | 9.18 | 425 | 46 | 5,348 | 56 |

The gap orders 5 of the 6 pairs as their downloads per day do. The exception is K2 against the 27B,
and the 27B's rate is from its first day, when downloads are usually highest. Four releases can't
validate a ranking: the next release's downloads against its gap is the test (`findings.md`).

## The gap

`gaps.py` writes `gaps.tsv`. The candidates are the 1,000 most downloaded and the 1,000 most trending
text-generation models on the Hub, 1,680 in all. 242 are left once each is required to be an
original, a chat model, an architecture llama.cpp converts, dense by its config.json, and at most 36B
parameters. Over 30 days:

- **GGUF demand:** downloads of the GGUF repos that name the model as the base they quantise, not
  counting uncensored ones.
- **Uncensored supply:** repos whose names contain the model's name and "uncensor", "abliterat",
  "heretic", "derestrict", "decensor" or "refusal", plus uncensored GGUFs of it (ours left out).
- **Gap:** demand divided by (1 + uncensored repos), which is roughly what a new entrant could reach
  if it took an equal share.

## The shortlist: released in the last 12 months, by gap

| Model | Params (B) | GGUF downloads | Uncensored repos | Their downloads | Gap | `candor inspect` |
| --- | --- | --- | --- | --- | --- | --- |
| ibm-granite/granite-4.2-30b | 29.3 | 238,240 | 1 | 459 | 119,120 | thinking fails |
| XHToken/Spark-X2.5-4B | 4.1 | 1,414,642 | 16 | 200,494 | 83,214 | thinking fails; remote code |
| XHToken/Spark-X2.5-1.7B | 1.7 | 228,925 | 3 | 2,068 | 57,231 | thinking fails; remote code |
| ibm-granite/granite-4.2-3b | 3.7 | 378,974 | 12 | 6,761 | 29,152 | thinking fails |
| InternScience/Agents-A1-4B | 4.5 | 404,859 | 15 | 5,625 | 25,304 | passes |
| ibm-granite/granite-4.2-8b | 8.8 | 262,539 | 12 | 8,356 | 20,195 | thinking fails |

Qwen3.8-27B, already released, would rank third. Granite's and Spark's "thinking fails" means
candor doesn't yet handle their chat templates' thinking switch (the template turns thinking off,
but the model can still open it itself). That's per-model code, like K2's step 1, not a blocker.
Granite 4.2 30B has the largest gap by a wide margin: one uncensored repo with 459 downloads,
against 238,240 GGUF downloads.

## Limits

- The counts include bots and mirrors, and can't show unique users. Old models keep high GGUF
  counts (Mistral-7B-v0.1 had 254,255 with no uncensored version), which is why the shortlist
  takes only the last 12 months.
- The uncensored supply is found by name, so a repo named without the base's name is missed. HF
  search returns at most 1,000 results, so counts on the most crowded bases are lower bounds.
- A gap closes: popular new bases get uncensored versions within days, so it needs reading again
  just before a model is chosen.
- Nothing here measures how often a model refuses. The unedited model measured as used shows
  whether an edit has anything to give.
