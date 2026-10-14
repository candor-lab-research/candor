# ARA: how well it works compared with abliteration (evidence as of 2026-10-14)

Research only. Every number below comes from a source I opened: heretic issues/PRs (fetched with `gh api`), Hugging Face model cards (raw README, only metric lines and tables printed), the DreamFast "Abliterlitics" benchmark cards and two Reddit posts (via the pullpush archive). No model prompts or answers were used. Prompt/response blocks were stripped by a filter before reading.

## Strength labels

- **Controlled pair**: same person, same base model, same scorer and datasets, one method changed. Still one study per arm, so TPE seed noise is not controlled.
- **Independent bench**: a third party measured published models with its own judge and settings. Methods differ, but so do authors, datasets, trial counts and which Pareto trial each author picked. This is not a controlled method comparison.
- **Card claim**: numbers the uploader reports on the model card. These come from heretic's own scorer unless stated otherwise.
- **Anecdote**: one user's report in a thread.
- **Maintainer**: p-e-w (heretic author). kabachuha (SOMA author, ARA-LoRA author) is an insider too.

## Read this first: what heretic's numbers measure

1. **Refusals are counted by keywords.** Heretic counts marker words on 100 `mlabonne/harmful_behaviors` prompts. One benchmark author says it reads about the first 100 characters (Reddit 1tsvs3j, nathandreamfast). The maintainer says the count carries "±10 after it, at minimum", and that differences under 10 mean nothing (#196, 2026-03-04 and 2026-03-11). He also says "keyword counting is inherently unreliable and a much larger source of bias" (#481).
2. **Thinking is switched off while scoring.** `config.py` on master sets `chain_of_thought_skips` to `("<think>", "<think></think>")`. Heretic prefills a closed think block, so its refusal counts and KL describe non-thinking answers. joninco (#211) notes that this skip "changes model behavior".
3. **KL is first-token only.** KL is measured on the first token of 100 harmless prompts. DreamFast re-measured with heretic's exact protocol and matched the trohrbaugh Qwen3.8 card to 9% (0.0535 card, 0.0586 measured). Other cards were off by 4.5x to 12x (coder3101 Qwen3.8, llmfan46 ultra).
4. **Trial scores are not the exported model's scores** (#490, open). Live-LoRA rounding differs from the merged weights. The maintainer's own example: a trial at KLD 0.15 and 3/100 exports as KLD 0.17 and 5/100. In his words, "we optimize for a score that we're not getting in the exported model". He rejected the post-hoc rescoring fix (#495, 2026-10-14).
5. **Name collision.** `jenerallee78/gemma-4-26B-A4B-it-ara-abliterated` calls its method "Adaptive Refusal Abliteration" from an OBLITERATUS fork (link 404). Its parameters (steer, overcorrect, preserve) look like heretic ARA applied in two passes. That is my inference; the card does not say so.
6. **The DreamFast reports mislabel methods in places.** They call coder3101's Gemma-4-E2B model "MPOA", but its card says ARA with row-norm preservation (checked in both README commits). They also call several rank-1 edits "Heretic ARA". I identified each model's method from its own card.

---

## Table 1: every ARA result found

Refusal scoring: **kw** = heretic keyword markers with thinking skipped (default); **judge** = LLM judge; n = number of prompts.

### Qwen3.5 / 3.6 / 3.8 (thinking, hybrid DeltaNet; Qwen3.8 uses `Qwen3_5ForConditionalGeneration`)

| Model | Method / who | Refusals (scoring) | KL | Other | Source | Strength |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | ARA (PR #211 branch), GhostWithAHat | 4/100 kw (main branch: ">50/100") | 0.1396 | - | #211 comment 2026-03-06 | Anecdote, same user |
| Qwen3.5-4B | ARA, later commit, GhostWithAHat | 1/100; 0/100 kw | 0.1338; 0.3984 | SOMA same user: 2/100 @ 0.1301 | #196 2026-03-11 | Anecdote; maintainer: "all indistinguishable" |
| Qwen3.5-4B | ARA (2.0.0.dev0 default), VINAY-UMRETHE | 4/100 kw (base 99) | 0.0509 | MPOA same setup: 12/100 @ 0.0427 | #472; cards `VINAY-UMRETHE/Qwen3.5-4B-heretic-baseline(-mpoa)` | Controlled pair (1 run each); maintainer: "Two tests mean nothing" |
| Qwen3.5-4B | ARA, semantic datasets, VINAY | 3/100 kw | 0.1153 | MPOA: 20/100 @ 0.0717 | #472 | Controlled pair (1 run each) |
| Qwen3.5-4B | ARA, MuXodious (refusal markers only) | 2/104 kw-custom (base 103) | 0.0567 | MPOA same recipe: 0/104 @ 0.0521 | cards `MuXodious/Qwen3.5-4B-ARA-heresy`, `-PaperWitch-heresy` | Controlled pair (card) |
| Qwen3.5-4B | ARA, MuXodious (refusal + disclaimer markers) | 6/104 | 0.0708 | MPOA 8/104 @ 0.0785; SOMPOA 4/104 @ 0.0565 | cards `...-ARA-heresy-v2`, `...-PaperWitch-heresy-v2`, `...-SOMPOA-heresy-v2` | Controlled pair (card) |
| Qwen3.5-4B | ARA, p-e-w, 2.0.0.dev0 test (2026-10-01, before the #476 fix) | 25/100 kw (base 99) | 0.3406 | Reproduced bit-exact | card `p-e-w/Qwen3.5-4B-heretic-new-ara-test`, #446 | Card claim (reproducibility test, not tuned) |
| Qwen3.5-4B | ARA, MiguelMendez101 | 3/100 kw (base 93) | 0.0321 | - | card | Card claim |
| Qwen3.5-0.8B | ARA + row-norm, dalatexcoder | 2/100; 0/100 kw (base 97) | 0.0823; 0.2185 | - | cards `...-heretic-ara-v2`, `-high-kld` | Card claim |
| Qwen3.5-2B | ARA + row-norm, dalatexcoder | 2/100; 0/100 kw (base 97) | 0.0251; 0.6486 | - | cards | Card claim |
| **Qwen3.5-9B** | ARA, NullpoLab (three Pareto picks) | 5/100; 9/100; 10/100 kw (base 84) | 0.0239; 0.0126; 0.0079 | - | cards `NullpoLab/Qwen3.5-9B-Heretic-ARA-Refusals5/9/10` | Card claim |
| Qwen3.5-9B fine-tune (OmniCoder-9B) | ARA, trohrbaugh | 7/100 kw | 0.0452 | - | card | Card claim |
| Qwen3.5-9B | ARA multi-turn, taresh18 | 2/72 kw | 0.1619 | - | card | Card claim |
| **Qwen3.5-35B-A3B** (MoE) | ARA, joninco, 400 trials, 8 GPUs | **judge** (gpt-oss-120b): 34/100 @ 0.685; 36 @ 0.122; 40 @ 0.105; 43 @ 0.026 | see refusals | Markers showed 3-5/100 where the judge found 33+/100. Only the shared expert is targetable (routed experts are fused 3D tensors) | #211 2026-03-08; #196 2026-03-09 | Anecdote, judge-scored; maintainer: too many confounds |
| Qwen3.5-35B-A3B | SOMA, same setup (joninco) | judge: 19 @ 0.560; 33 @ 0.288; 45 @ 0.154 | - | Heretic baseline, probably string-matched: 39/100 @ 0.0047 | #196 | Same |
| Qwen3.5-122B-A10B | ARA + row-norm, DazzlingShadow | 0/100; 16/100 kw (base 88) | 0.1448; 0.0233 | - | cards | Card claim |
| Qwen3.5-27B | ARA, erm14254 | "maybe 55/100 at best" with the old ranges; expanded ranges needed | - | - | #211 2026-03-11 | Anecdote |
| Qwen3.6-27B | ARA, trohrbaugh | 4/100 kw (base 99) | 0.0469 | - | card | Card claim |
| Qwen3.6-27B | ARA + row-norm, MoonRide / noclip84 / Abiray | 16/100 (base 91) / 8/100 (base 90) / 6/100 (base 92) | 0.0221 / 0.0552 / 0.0653 | - | cards | Card claim |
| **Qwen3.8-27B** | ARA (1 pass), trohrbaugh / heretic-org | Card: 0/100 kw (base 99). **DreamFast judge** (glm-5.3-flash, thinking on, xhigh, HarmBench 400, 15,360-token budget): **57.5% ASR**, 230 complied / 48 soft / **122 explicit refusals**; harmful-only 76.3% | Card 0.0535; measured 0.0586 | GSM8K 91.5% (base 91.2); MMLU-Pro -0.2pp; TQA-MC2 -1.2pp; HumanEval +3.1pp; fewest unterminated thinks of all variants (61/400) | card; `DreamFast/Qwen-3.8-27b-abliterlitics` | Independent bench |
| Qwen3.8-27B | ARA second pass on trohrbaugh ("rvn", 0bserverx) | DreamFast judge: **71.3%**, 15 explicit refusals | 0.0837 | TQA-MC2 -2.1pp; 104 unterminated thinks | same | Independent bench |
| Qwen3.8-27B | ARA + row-norm, coder3101 | 9/100 kw (base 92) | 0.0349 | Same author's directional run: 33/100 @ 0.0187 | cards `coder3101/Qwen3.8-27B-heretic(-ara)` | Controlled pair (card) |
| Qwen3.8-27B distill / Qwen3.8-9B distill | ARA, armand0e / EclipsedStar | 5/100 (base 91) / 0/100 (base 97) | 0.1004 / 0.0355 | - | cards | Card claim |

### Gemma 4 (thinking models)

| Model | Method / who | Refusals (scoring) | KL | Other | Source | Strength |
|---|---|---|---|---|---|---|
| Gemma-4-E2B | ARA + row-norm, p-e-w | Card 5/100 kw (base 98). **DreamFast** (thinking on, HarmBench 400, keywords plus glm-5.1 review of every response): 32/400 refused (92.0% ASR) | 0.1522 card; 0.1526 measured | GSM8K flex 83.47 (= base); MMLU -0.14 | card; `DreamFast/Gemma4-e2b-abliterlitics` | Independent bench |
| Gemma-4-E2B | ARA + row-norm, coder3101 | Card 5/100; DreamFast 16/400 (96.0%) | 0.1651 / 0.1673 | GSM8K +1.37; MMLU -0.30 | same | Independent bench |
| Gemma-4-E2B | ARA, llmfan46 ultra | Card 4/100; DreamFast 65/400 (83.8%) | 0.0779 / 0.0677 | GSM8K +0.46 | same | Independent bench |
| Gemma-4-E4B | ARA: NullpoLab / llmfan46 ultra / coder3101 / MuXodious / llmfan46 std | **DreamFast judge** (glm-5.2, thinking on): 14 / 18 / 25 / 27 / 36 refused of 400 (96.5 / 95.5 / 93.8 / 93.3 / 91.0% ASR) | 0.0054 / 0.0021 / 0.0021 / 0.0024 / 0.0012 | GSM8K +1.7 / +1.3 / +0.9 / +0.8 / +0.9pp | `DreamFast/Gemma4-e4b-abliterlitics`; method from each card | Independent bench |
| Gemma-4-E4B | Cards for the same models plus trohrbaugh and MoonRide | 1-7/100 kw | 0.0043-0.0256 | MuXodious: ARA 6/104 @ 0.0140 vs SOMPOA 12/416 @ 0.0497 | cards | Card claim |
| Gemma-4-12B | ARA, llmfan46 | Card 6/100; **DreamFast judge**: 76.3% ASR (95 refused of 400) | 0.1203 / 0.182 | MMLU-Pro -6.2pp; GSM8K answered-only +1.9pp | `DreamFast/Gemma4-12b-it-abliterlitics` | Independent bench |
| Gemma-4-12B | ARA + row-norm, coder3101 | Card 7/100; DreamFast judge 81.0% | 0.1576 / 0.282 | TQA -6.3; Snarks -16.9pp | same | Independent bench |
| Gemma-4-26B-A4B | ARA, trohrbaugh (v1 / v2) | Card 2/100 / 0/100 kw (base 99) | 0.0491 / 0.0799 | **jenerallee78** (StrongREJECT via GPT-4o-mini plus HarmBench-13B, 507 HarmBench prompts): "trohrbaugh/heretic-ara" **31.4%** refusal, KL 0.2999 (their own script; base vs base gives 0.0017) | cards; `jenerallee78/gemma-4-26B-A4B-it-ara-abliterated` | Independent bench (author promoting own model) |
| Gemma-4-26B-A4B | ARA + row-norm, coder3101 | Card 11/100 (base 100); jenerallee78: **15.8%** | 0.0499 / 0.4118 | Compliance quality 4.2/5 | same | Independent bench |
| Gemma-4-26B-A4B | "Adaptive Refusal Abliteration", 2 passes, jenerallee78 | 7.7% StrongREJECT (39/507); 5.7% ensemble | 0.1299 | Quality 4.6/5 | card | Card claim (self-evaluated) |
| Gemma-4-26B-A4B | ARA, MuXodious / MoonRide | 3/104 / 12/100 kw | 0.0471 / 0.0513 | MuXodious SOMPOA: 4/104 @ 0.1240 | cards | Controlled pair (card) / card |
| Gemma-4-31B | ARA: trohrbaugh / coder3101 / MoonRide / MoonRide custom | 5 / 15 / 11 / 16 per 100 kw | 0.0120 / 0.0434 / 0.0508 / 0.0417 | - | cards | Card claim |
| Gemma-4-31B | ARA-LoRA, 4-bit bnb, kabachuha, ~160 trials | 16/100 kw | 0.07 | ~36 GB VRAM, ~8 h | #332 2026-05-27 | Anecdote |
| Gemma-4-31B-QAT | ARA-LoRA (locally modified), Roman215, 613 trials | 8/100 kw | 0.0296 | Same user, directional, 1000 trials: ~63/100 @ ~0.02 | #365 | Anecdote, same user |

### Other models

| Model | Method / who | Refusals | KL | Other | Source | Strength |
|---|---|---|---|---|---|---|
| gpt-oss-20b (reasoning) | ARA v1 / v2 / v3, p-e-w | 12 / 3 / 3 per 100 kw (base 98) | 0.0657 / 0.1081 / 0.0554 | p-e-w directional cards: 58/100 @ 0.96 (v1.0.0), 49/100 @ 0.1248 and 74/100 @ 0.0501 (v1.1.0) | cards; #211 | Card claim / maintainer |
| gpt-oss-20b | ARA v3: PIQA and UGI | - | 0.0437 (in #236) | PIQA 0.7688 (base 0.7731); UGI NatInt 20.17 (base 27.18), ranked 5th of 7. MPOA "heresy" 22.52 ranked best. Maintainer: Derestricted, SOMA and ARA "basically noise"; MPOA heresy "clearly much better" | #236; #211 2026-03-16 | Maintainer, external benchmark |
| gpt-oss-20b | ARA v4 (row-norm, optimised for PIQA) | Card 9/100; re-measured by hpnyaggerman after repairing a corrupt upload: 6/100 (base 98 in the same session) | - | PIQA 0.7742 card, 0.7731 re-measured | card; #211 2026-07-27 | Independent re-measure (1 model) |
| Qwen3-4B-Instruct-2507 | ARA early, untuned weights, kabachuha | Pareto: 10 @ 0.1345, 8 @ 0.1384, 5 @ 0.2406 kw | - | heretic-org directional card: 5/100 @ 0.0741; SOMA 3/100 @ 0.08 | #211 2026-03-04 | Anecdote (maintainer: weights were not yet tuned) |
| GLM-4.7-Flash | ARA, expanded ranges, erm14254, 261 trials | 6/100; 0/100 kw | 0.0039; 0.0288 | "old ranges couldn't crack GLM at all" | #211 2026-03-11 | Anecdote |
| Gemma-3 (size not given) | ARA, erm14254 | 4/100 kw | 0.013 | ~100 trials, ~1.5 h | #211 | Anecdote |
| Qwen3-0.6B (thinking) | Master 49b56e5 (2026-10-07; ARA was default) | Bad auto prefix: baseline 0/25, best KL 16.7-17.9, model destroyed. Correct prefix, 20 trials: 0/25 | 0.0943 | 7 lm-eval tasks unchanged; MMLU +3.5 (n=171) | #487 (open) | Anecdote; maintainer: 2-20 trials too few |
| DiffusionGemma-26B-A4B | ARA, umran666 | Standard ARA "either failing to suppress refusals, or completely destroying" (KL > 0.3, up to 20s). Modified ARA: 4/100 @ ~0.1; 12/100 @ 0.09. ARA-LoRA at rank 128 "trapped"; full rank OOM | - | - | #400 (closed) | Anecdote |
| Nemotron-Cascade-2-30B-A3B / LFM2-24B-A2B / ZAYA1-74B / Step-3.7-Flash / ALIA-40b / Ministral-3-3B / MiniCPM5-2B / Agents-A1-4B / Nanbeige4.2-3B | ARA, various | 4/100; 0/100; 0/100; 7/100; 4/100; 2/416 (SOMPOA 6/416); 4/100; 0/100; 9/540 kw | 0.0333; 0.0553; 0.0095; 0.0226; -; 0.0216 (SOMPOA 0.0417); 0.0077; 0.0116; 0.0824 | - | cards | Card claim |

Comparison points for the directional method on Qwen3.5-9B, all kw unless stated:
- #404: 70/100 after 200 trials, four runs, defaults.
- `trohrbaugh/Qwen3.5-9B-heretic-v2` (v1.2.0 default, directional): 6/100 @ 0.0793. DreamFast measured it at **0/400 HarmBench refusals**, KL 0.0825 and GSM8K 87.64 → 85.97, scored with a classifier script plus LLM review. The report mentions truncated chain-of-thought analysis and a 2048-token cap (`DreamFast/Qwen3.5-9B-...-Benchmark`).
- `coder3101/Qwen3.5-9B-heretic` (SOMA): 10/100 @ 0.1334.
- `saidutta69` (v1.4.0): 79/100 @ 0.0015.

**Inference:** the directional stall in #404 is not universal. Other uploaders reached low refusal counts on the same model with the directional method.

---

## 2. Head-to-head comparisons: how many, and how controlled

### A. Same person, same model, one method changed (all one study per arm)

| # | Model | ARA | Other method | Scorer | Notes |
|---|---|---|---|---|---|
| 1 | Qwen3.5-4B (VINAY, #472) | 4/100 @ 0.051 | **MPOA (full row norm)** 12/100 @ 0.043 | kw | Semantic datasets: ARA 3 @ 0.115 vs MPOA 20 @ 0.072 |
| 2 | Qwen3.5-4B (MuXodious) | 2/104 @ 0.057; 6/104 @ 0.071 | **MPOA** 0/104 @ 0.052; 8/104 @ 0.079; SOMPOA 4/104 @ 0.057 | kw, custom | A tie within noise |
| 3 | Qwen3.5-4B (GhostWithAHat) | 4 @ 0.140; 1 @ 0.134 | directional ">50/100"; SOMA 2 @ 0.130 | kw | Maintainer: ARA and SOMA indistinguishable |
| 4 | Qwen3.5-35B-A3B (joninco) | 35 @ 0.122; 43 @ 0.026 | SOMA 33 @ 0.288; 19 @ 0.560 | **judge** | Only judge-scored pair. ARA had lower KL at about 35 refusals; SOMA reached lower refusals. Custom data, sharded workers. Maintainer: "too many confounding factors" |
| 5 | Gemma-4-31B-QAT (Roman215, #365) | 8 @ 0.030 (613 trials, modified ARA-LoRA) | directional ~63 @ ~0.02 (1000 trials) | kw | Code modified; trial counts differ |
| 6 | Qwen3.8-27B (coder3101 cards) | 9 @ 0.035 | directional 33 @ 0.019 | kw | Card values only |
| 7 | gpt-oss-20b (p-e-w cards) | 3 @ 0.055 | directional 49-74 @ 0.05-0.96 | kw | Heretic versions differ (1.0/1.1 vs ARA branch) |
| 8 | Gemma-4-E4B / 26B-A4B / Ministral-3B (MuXodious) | 6/104 @ 0.014; 3/104 @ 0.047; 2/416 @ 0.022 | SOMPOA 12/416 @ 0.050; 4/104 @ 0.124; 6/416 @ 0.042 | kw | Different prompt counts on E4B |

None of these uses multiple seeds or an equal trial budget, and only #4 uses a judge. The maintainer's own rule (#332, 2026-05-28): "TPE is *highly* random ... Running a study, changing something, then running the study again and comparing the Pareto fronts tells you basically nothing." By that standard none of the pairs is conclusive. That includes his own gpt-oss pair.

### B. Independent benchmarks of published models (judge, thinking on)

| Bench | ARA models | Best non-ARA | Result |
|---|---|---|---|
| DreamFast Gemma-4-E2B | 96.0 / 92.0 / 83.8% ASR, KL 0.07-0.17 | trevorjs 99.5% (KL 0.37); huihui-v2 99.5% (KL 0.53) | ARA: lower KL, lower ASR. Math preserved |
| DreamFast Gemma-4-E4B | 91.0-96.5%, KL 0.001-0.005 | abliterix 100% (KL 0.054); trevorjs 99.3% (KL 0.015) | ARA: lowest KL, but 14-36 refusals of 400 against 0-7 |
| DreamFast Gemma-4-12B | 76.3%, 81.0% (KL 0.18, 0.28) | huihui 89.8%; trevorjs 85.8% (KL 0.087) | ARA mid-pack on both axes |
| DreamFast Qwen3.8-27B | **57.5%** (1 pass, KL 0.059); 71.3% (2 passes, KL 0.084) | orcarouter Arditi k=1 **82.2%** (KL 0.068); huihui 75.6%; coder3101 directional 70.0%; llmfan46 **MPOA** 70.5% (KL 0.30) | Single-pass ARA unlocked least of all 12 variants, but was best for capability preservation and looping |
| jenerallee78 Gemma-4-26B-A4B | 31.4%, 15.8% refusal (StrongREJECT) | TrevorJS 33.3% | Heretic-ARA models rank 2nd and 3rd. The author's own two-pass ARA variant ranks best |

Method caveats raised publicly against DreamFast:
- HarmBench's context field was not passed for contextual behaviours (Reddit 1w8vx6w).
- The Gemma-4-12B report says its keyword cross-check "miscounts" by up to 16-19 pp in think mode.

---

## 3. Downsides and failure modes

**Keyword scoring flatters ARA the most where it matters to candor.** On thinking models:

| Case | Keyword (card) | Judge |
|---|---|---|
| Qwen3.5-35B-A3B (joninco, ARA run) | 3-5/100 | 33+/100 |
| Qwen3.8-27B, trohrbaugh ARA | 0/100 | 30.5% explicit refusals + 12% soft (thinking on) |
| Gemma-4-26B-A4B, trohrbaugh ARA | 2/100 | 31.4% |

This is not ARA-specific. coder3101's directional Qwen3.8 card says 33/100 while the judge found 70% ASR. Heretic's optimiser drives the keyword count down, so inference: any heretic-optimised method is at risk of learning to beat the markers rather than the judge.

- **Issue #487, the auto-prefix bug on Qwen3-style templates.** It occurs on templates that do not pre-append `<think>`. Baseline refusals collapse to 0 and the optimiser "is free to pick ablation parameters that maximally damage the model" (KL 16.7-17.9) while the scorer reports success. The run was on master 49b56e5 (2026-10-07), where ARA is the default, though the issue does not name the modifier. Still open; the maintainer calls a proper fix "very difficult". The #493 fix PR was closed unmerged. Qwen3.5/3.8 templates prefill `<think>`, so they take the other branch. Heretic then closes the block and scores non-thinking answers.
- **Unstable optimisation.** With `overcorrect_relative_weight > 1` the region is "chaotic (possibly even quasi-fractal)". About 25% of 200 trials had KLD > 3 on Gemma 3 4B and Qwen3 4B (#211, maintainer). In one 315-trial run, ~95% of trials were unusable, some with KLD 13-31 (erm14254). The maintainer says KLD > 9 happens only with ARA (#332), and in June he required this to be "understood" before merging. In the merged PR #446 I found no statement that the cause was found.
- **NaN/inf tensors** were reported by at least two users (#211, June). The one corrupt upload investigated (gpt-oss-20b ARA v4) turned out to be a transformers 5.4-5.5.1 serialisation bug, not ARA. hpnyaggerman says other NaN reports "stay open".
- **2.0.0.dev0 bugs fixed after the ARA merge.**
  - `lora_A` carried over between trials, so the restored trial did not equal the scored one (#476, merged 2026-10-04).
  - ARA tests never actually ran the optimisation (#478, merged 2026-10-06).
  - Results from master between 10-01 and 10-04 predate these fixes; this includes p-e-w's Qwen3.5-4B test at 25/100 @ 0.34.
- **Capability.**
  - Mostly small losses, but not uniformly better than directional. gpt-oss-20b NatInt was 20.17 against 27.18 for the base and 22.52 for MPOA; the maintainer says no method preserved it.
  - Gemma-4-12B ARA: MMLU-Pro -6.2pp (llmfan46); Snarks -16.9pp and TQA -6.3pp (coder3101).
  - Qwen3.8-27B ARA was the most preserved model in its benchmark.
- **Covert non-compliance.** #288 reported that an ARA Gemma-4-E4B asks clarifying questions instead of answering. The reporter withdrew it after their own ARA run, and the maintainer says the pattern is not ARA-specific. DreamFast describes single-pass ARA on Qwen3.8 as still refusing "about a third".
- **Architectures where it struggled.**
  - DiffusionGemma needed a modified objective (#400).
  - Qwen3.5 MoE routed experts are fused and untargetable, and ARA-LoRA has the same limit (#339).
  - Qwen3.5-27B needed widened ranges (#211).
  - A Reddit anecdote (1sanln7, HopePupal) says ARA "does *not* work well on Qwen 3.5". A Qwen3.5-27B heretic model said to be ARA still refused "a lot" (1rnic0a, ArthurianX). Both are unverified.
- **Compute cost.**
  - The maintainer says time per trial is "dominated by counting refusals", with L-BFGS converging in 2-3 iterations (#211).
  - Reported: ~1.5 h for 100 Gemma-3 trials (old ranges), but 200+ trials with the new ranges; Gemma-4-31B in 4-bit took ~8 h for ~160 trials; 613 trials took 13 h on an RTX 3090; MiniMax-M3 took ~10 min per trial even on vLLM (#332).
  - Full-matrix ARA cannot use bnb 4-bit; ARA-LoRA (now the default) can.

---

## 4. Independent evidence

Several people outside the maintainers measured ARA models:
- **DreamFast / Nathan Sapwell (Abliterlitics):** four judge-scored, thinking-on benchmarks covering 13 ARA models in total.
- **jenerallee78:** a StrongREJECT comparison. Its author also promotes their own variant.
- **joninco:** a judge-scored Qwen3.5-35B-A3B Pareto front.
- **hpnyaggerman:** re-measured gpt-oss ARA v4.
- **Others:** GhostWithAHat, erm14254, Roman215, VINAY-UMRETHE, MuXodious, NullpoLab, trohrbaugh, coder3101 and others, through cards and threads.

What is missing:
- **Papers.** An arXiv search for "arbitrary-rank ablation" and for heretic abliteration returned no paper on ARA. kabachuha reported the same in May (#332).
- **The maintainer's write-up.** It was promised in March and again on 2026-10-03 (#473) and has not been published.
- **A controlled comparison by anyone.** Nobody has run a controlled ARA-vs-abliteration comparison with a judge.

---

## 5. Verdict

**ARA beats directional abliteration under heretic's own metric (keywords, thinking off, first-token KL). The evidence is moderate.**
- Several same-person pairs show large gaps: gpt-oss-20b 3 vs 49-74; Gemma-4-31B-QAT 8 vs 63; Qwen3.8-27B 9 vs 33; Qwen3.5-4B 4 vs >50 or 12.
- ARA cards typically sit at KL 0.01-0.06.
- But every pair is one study per arm. Other uploaders also got the directional method to 4-6/100 on Qwen3.5-4B/9B. In the two MPOA pairs that are closest to controlled, ARA wins one (VINAY) and ties one (MuXodious).

**ARA beats abliteration in general as a model is actually used: no evidence for this, and the evidence that exists leans against it on refusal removal.**
- Five independent benchmarks used a judge with thinking on.
- In all of them, ARA models had the lowest or competitive KL and kept capability well.
- In none of them did an ARA model remove the most refusals. Directional variants (huihui, trevorjs, abliterix, orcarouter) reached 85-100% ASR where ARA reached 76-96%.
- None of this is controlled: authors, trial choices and datasets all differ.

**On Qwen3.5-family thinking models, the evidence is weak.**
- The only judge-scored ARA data are Qwen3.5-35B-A3B (33-43/100 refused, KL 0.03-0.7) and Qwen3.8-27B (57.5% ASR single pass, 71.3% two passes, against 82.2% for a careful directional edit).
- There is no judge-scored, thinking-on result for ARA on Qwen3.5-9B. The 5/100 cards are keyword counts with thinking skipped.
- The PR #211 Qwen3.5-4B result (4/100 vs >50) is a keyword anecdote, and the maintainer called the matching SOMA comparison noise.

**What is missing:**
1. A controlled comparison: same model, same prompts, an equal trial budget, several seeds, judge scoring with thinking on, deployed as a quantised GGUF, with KL and GSM8K.
2. ARA vs MPOA on Qwen3.5-9B under those conditions.
3. An explanation of the KL > 9 runaway trials.
4. A fix for the trial-vs-export gap (#490).
5. The promised technical write-up.

The maintainer's own position as of #446 (2026-10-01): ARA "will have to prove itself in manual testing for us to retain it as the default in the upcoming 2.0 release".
