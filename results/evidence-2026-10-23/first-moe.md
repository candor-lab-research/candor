# The first MoE release: Laguna-XS-2.1 against the next candidates (2026-10-23)

Is poolside/Laguna-XS-2.1 the right first MoE release, against the MoE models of 36B or so after it in
`moe.tsv`? Gaps from `moe.tsv` (14:48 UTC); downloads (30 days unless said), likes and sizes from Hugging
Face's API (14:00 UTC), Ollama's from its pages (all time); `candor inspect` with the MoE support
built on 2026-10-23. No weights, no pods. Refusal counts are each card's, on its own prompts.

## Bottom line

- Laguna-XS-2.1 is first on fit, not on demand. It leads the gap without its top repo (35,505), but
  that rests on poolside's GGUF, which LocalAI's model gallery installs as it does mudler's; without
  both it is 3,843, level with Nemotron's 3,974. Its uncensored versions draw the least of any
  candidate (1,104 in 30 days, the best 564): a release would draw about 0.5-6 downloads a day.
- Qwen3-30B-A3B-Instruct-2507 is the better first MoE: the only one with a published, KL-measured
  edit to check candor's against (heretic's per-expert LoRA, 11/100 at KL 0.0679), real uncensored
  demand (120,163 downloads in 30 days over 26 versions: about 50-150 a day for a release), `inspect`
  passing on today's code, no thinking (a cheaper run) and Apache-2.0. Against it: crowded, and older.
- poolside's use policy doesn't bind the open weights; its trademark guidelines forbid candor's usual
  repo name.

## The candidates

| Model | Params (B) | Gap (without top repo) | Top repo: share; all-time downloads per like | Base: downloads; likes | Uncensored repos (share of downloads) | Published unedited refusals | `inspect` today |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Laguna-XS-2.1 | 33.4 | 287,279 (35,505) | 88%; 180,692 | 499,646; 258 | 5 (0.06%) | 98/100 [1] | passes |
| Mellum2-12B-A2.5B-Instruct | 12.1 | 46,833 (18,686) | 60%; 55,352 | 14,332; 91 | 4, all of Mellum2-Thinking (0.8%) | 23.1% harmful on HarmBench [2] | thinking warns |
| Mellum2.1-12B-A2.5B-Thinking | 12.1 | 67,144 (4,414) | 93%; 1,344 | 5,355; 175 | 2 (0.7%) | 43/44 off, 44/44 on [3]; 8.5% [2] | thinking fails |
| Ornith-1.5-35B-A3B | 36.0 | 37,181 (14,069) | 62%; 10,374 | 260,233; 750 | 84 (6.1%) | none found | passes |
| Nemotron-3.5-Lightning-30B-A3B | 32.9 | 10,410 (3,974) | 62%; 16,478 | 742,789; 246 | 27 (3.7%) | none found; predecessor 2.2 of 10 [7] | targets, thinking fail |
| Ling-3.0-tiny | 7.9 | 22,355 (5,081) | 77%; 4,327 | 11,426; 548 | 13 (4.6%) | 98/100, 35/100 [4] | remote code |
| LFM2.5-24B-A2B | 24.0 | 10,161 (4,470) | 56%; 702 | 18,100; 571 | 28 (19.0%) | 96-99/100 [5] | GGUF names fail |
| Qwen3-30B-A3B-Instruct-2507 | 30.5 | 13,447 (5,149) | 62%; 10,784 | 1,073,700; 850 | 26 (24.9%) | 99/100 [6]; 4.0 of 10 [7] | passes |
| K2-Horizon-MoVA-36B-A4B | 37.4 | 9,662 (4,743) | 51%; 339 | 10,289; 421 | 7 (25.2%) | 96/102, 98/100 [8] | remote code |

Share: uncensored over GGUF plus uncensored downloads (`moe.tsv`). [1] is abliterix's count on an NVFP4
card, its settings unstated (abliterix's defaults: 100 harmful_behaviors prompts, 100-token answers,
keyword markers, so likely with the template's thinking off); 6 of the 9 candidates have counts of
96-99. Cards: [1] qpqpqpqpqpqqqq's NVFP4,
[2] JetBrains' (harmful rate), [3] alexwkleung's, [4] SC117's and Securelayer7's, [5] OS-Software's,
[8] InfinimindCreations' and darkc0de's; [6] `moe-support.md`; [7] UGI leaderboard willingness. Ornith's
top repo is another fine-tune's GGUF (Tiel-Coder). Hy-Memory-30B-A3B is out: a memory writer, not chat.

## 1. Demand

- Laguna's top repo, mudler's APEX GGUF, has 5,059,389 all-time downloads and 28 likes: 180,692 per
  like, matched by the 27B's second repo (178,519) and below LM Studio's 27B MLX repos (91,186-593,783),
  so downloads per like don't separate bots from apps. LocalAI's gallery installs both of Laguna's top
  repos (7 entries pull mudler's APEX; its default `laguna-xs-2.1` pulls poolside's Q4_K_M); for
  Nemotron only ggml-org's, its top repo.
- 89% of the rest is poolside's own GGUF (189,971; 100 likes; 6,286 per like). Without it too, Laguna's
  gap is 3,843, below Ornith's 8,882 and level with Nemotron's 3,974 without its top repo: the lead
  rests on poolside's repo being real use. The base has 499,646 downloads in 30 days (2,350 all-time
  per like).
- Ollama, uncounted on Hugging Face (all time): Nemotron-3.5-Lightning 257.2K since August (at least
  3,100 a day), Laguna-XS-2.1 140.8K since June (at least 1,100 a day), Ornith-1.5 about 417K over four
  sizes, LFM2.5 about 264K over two; Mellum, Ling and K2-Horizon have no page.
- Mellum's demand is JetBrains' own repos (96% of Mellum2's in its top two, 93% of Mellum2.1's). Mellum2.1's official GGUF has 35
  times its base's downloads, Mellum2's Q8_0 10 times: the pattern `gaps.md` discounted on Granite 4.2.
- **Measured** (HF, 30 days): uncensored versions take 0.06% of Laguna-XS's downloads, the least of the
  nine, against 18.0-57.7% on our four releases' bases (`gaps.tsv`). It isn't a coding-agent effect:
  Ornith-1.5-35B-A3B, built for coding and agents, has 6.1% (206,174 over 84 repos). **Hypothesis:**
  Laguna-XS's GGUF demand is largely app-driven (LocalAI's gallery installs both top repos).
- **What a release would draw:** our releases took from 0.03% (the 27B, 532 rivals) to 24% (K2, 5) of
  their base's uncensored downloads, so a share between Llama-3.2's 1.3% and an equal one gives: Laguna
  about 0.5-6 a day, Nemotron 5-14, Ling 7-36, Ornith 81-89, Qwen3-Coder-30B 49-113, Qwen3-30B-2507 52-148
  (`moe.tsv`'s uncensored downloads over 30 days).

## 2. Use, existing uncensored versions, unedited refusals

- **Laguna-XS-2.1:** "software engineering and agentic coding" (card). dealignai's CRACK GGUF claims 0
  refusals on HarmBench-320, MMLU 73.54% to 73.63% at Q4_K_M; an abliterix NVFP4 98/100 to 4/100;
  neither gives KL, which with a measurement as used would be candor's edge.
- **Nemotron-3.5:** agents, chatbots, RAG, instruction following (card). mlasli's heretic: 0% at KL
  0.0397 (7/50 in its run); dealignai's CRACK: 0/240 on HarmBench, MMLU down 4.1 points.
- **Mellum:** coding. Mellum2.1's one public uncensored GGUF still refuses 14/44 thinking off and 17/44
  on, the weakest incumbent here. Mellum2-Instruct refuses less (23.1% harmful, beside Qwen3.5-9B's
  20.9% in the same table, a base candor found worth editing) and has no version of its own.
- **Others:** Ornith, agentic coding (SC117 9/100 at KL 0.0107); LFM2.5, on-device agents (0-1/100 at KL
  0.017-0.042); Ling 15/100 at KL 0.0677; K2-MoVA 1/102 at KL 0.359, or 56/100 at KL 0.0127.
- **Unedited refusals:** most published counts are high (table); the lowest are one of Ling's (35/100)
  and Mellum2-Instruct's; none for Ornith-1.5 or Nemotron-3.5. Measured as used decides.

## 3. Fit for candor

| Model | Experts | Thinking: template / publisher | A40: Q4_K_M + f16 KV, 32 slots x 8,504 | Residuals (2 x params x 1.1) | llama.cpp at b11524 |
| --- | --- | --- | --- | --- | --- |
| Laguna-XS-2.1 | 256 per expert, top-8, shared; 3 sliding : 1 full | off / on | 20.27 + 15.2 GB | 73.6 GB, A100 | Metal fails (#30896, open); YaRN fix after b11524 (#30331) |
| Mellum2, 2.1 | 64 per expert, top-8, no shared | none, on / same | 8.07 + 5.3 GB | 26.6 GB, A40 | Metal NaN at Q4_K_M (#30579), its fix slowing CUDA 8 times (#30704, open); Q5_0/Q8_0 rows |
| Ornith-1.5 | 256 fused, top-8, gated shared; 3 linear : 1 full | on / on | about 21.6 + 5.6 GB | 79.1 GB, A100 | candor's KV and linear-attention state sizes for qwen35moe |
| Nemotron-3.5 | 128 per expert, top-6, shared; 23 Mamba-2, 6 attention | on / on | 25.48 + 1.7 GB | 72.4 GB, A100 | wrong pre-tokenizer before b11813 (fixed by #30960, 2026-10-22; issue #30894); Q5_0/Q8_0 rows |
| Ling-3.0-tiny | 128 per expert, top-8, shared; 3 KDA : 1 MLA | on / on | 4.92 + about 1.9 GB | 17.4 GB | remote code |
| LFM2.5-24B | 64 per expert, top-4; 30 conv, 10 attention | always on | 14.53 + 5.6 GB | 52.8 GB, A100 | co-batched slots corrupt conv state (#29002, open) |
| Qwen3-30B-2507 | 128 per expert, top-8 | none | 18.63 + 26.8 GB, tight | 67.2 GB, A100 | none found |
| K2-MoVA | 100 per expert, top-8, shared; value experts | always, "high" | 22.37 + 1.67 GB a slot: about 12 slots | 82.3 GB, over 80 | remote code |

Sources: configs, indexes, templates, GGUF files; Laguna's, Ornith's and Qwen's KV and residuals from
`moe-support.md`.

- Laguna is closest to the code being built: per expert, native in transformers 5.19, Q4_K_M rows clean
  (`moe-support.md`), `inspect` passing with 10,064 writers. Its template defaults to no thinking; its
  `generation_config.json` and card turn it on. #27134 (open) misfiled its thinking-off answer on b10360
  (not reproduced on master). Metal is moot for the 30B-class models: their Q4_K_M (18.6-25.5 GB) don't
  fit the 18 GiB Mac.
- Shared by all: b11524's fast CUDA kernels need an even inner size (`mmvf.cu:794`, `mmf.cu:140`), so a
  rank-1 expert adapter takes the general path (a stream sync) on NVIDIA, asserts on AMD (#30576): time it.

## 4. Licence and use policy

- **Laguna:** OpenMDW-1.1 (read) grants use "without restriction", asks only that the licence and notices
  be kept, and restricts no outputs. The card says use "should be" consistent with poolside's AUP and
  advises against circumventing guardrails without equivalent mitigations, as NVIDIA's card does.
- **The AUP** (2026-04-27; Wayback copy of 2026-06-16) covers poolside's "Products" and forbids
  circumventing "safeguards or safety mitigations" and sexually explicit content. Since July its URL
  redirects to the Terms of Use (2026-07-21), which say poolside models "are not Products".
- **Trademark guidelines** (2026-06-29): no poolside mark (product names included) as a name's first word
  or hyphenated with other words, which "Laguna-XS-2.1-Candor-Uncensored-GGUF" does, as other Laguna GGUF
  repos do; the compliant form is relational ("... for Laguna-XS-2.1"). Section 6 also bars a poolside
  mark on a page "whose content or activities would violate Poolside's Acceptable Use Policy" (binding
  again if poolside republishes it), and section 14 asks for a "not affiliated with or endorsed by
  Poolside, Inc." disclaimer. NVIDIA's trademark terms for an "NVIDIA-Nemotron-..." name weren't checked.
- **Others:** Mellum, Qwen, K2 Apache-2.0; Ornith, Ling MIT; Nemotron OpenMDW-1.1; LFM Open License 1.0
  (commercial use under $10M revenue). Our Llama-3.2-3B release already sits under a licence making
  Meta's AUP binding (clause 1.b.iv); none of these does. Account risk is low and about equal: nothing
  needs an identity, and Hugging Face hosts 5 uncensored Laguna-XS and 25 Laguna-S versions.

## 5. Cost and time

- **Laguna** (`moe-support.md` section 6): code on the Mac, 2-3 days plus review, $0; A100 stage $1-1.5;
  A40 pilot $3-5; thinking off about $25-35 in all, thinking on about $35 (the 27B's route).
- **Nemotron:** similar pods plus the NemotronH code (four rows of `moe-support.md`'s table), thinking
  work and b11813 or later. **Mellum:** no A100 stage. **K2-MoVA:** over 80 GB, 12 slots. **LFM:** #29002.

## Recommendation

1. **Qwen3-30B-A3B-Instruct-2507** first: the one MoE with a published, KL-measured edit (heretic's
   per-expert LoRA, 11/100 at KL 0.0679), so candor's MoE edit is checked against public practice before
   it goes where none exists, as Trial 161 checked the thinking-model route on Qwen3.5-9B; real
   uncensored demand (about 50-150 downloads a day for a release); Qwen3-MoE, the layout candor's tests
   cover, `inspect` passing today; no thinking, so a cheaper run; Apache-2.0. Against: 26 uncensored
   versions already, and from July 2025.
2. **Laguna-XS-2.1** only if its demand proves real: the best fit (per expert, native in transformers,
   clean Q4_K_M, an unrestricted licence) but about 0.5-6 downloads a day on today's evidence. Write down
   its expected downloads a day before committing.
3. **Nemotron-3.5-Lightning-30B-A3B:** the broadest use, 27 uncensored versions (11,348 downloads), more
   code (the NemotronH target fixes, thinking work, Q5_0/Q8_0 rows) and b11813 or later.

What would change it: a heretic or other public reference for Laguna; Laguna's GGUF demand shown to be
people rather than LocalAI's gallery; the unedited model measured as used refusing far less than its
published count. Before any Laguna run: thinking on or off, a build after b11524 (#30331, master's Metal
guard), a compliant name, the gap read again.
