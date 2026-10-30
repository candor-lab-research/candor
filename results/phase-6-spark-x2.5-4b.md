# Phase 6 results: Spark-X2.5-4B

Numbers only; answers stay in each run's private folder. The plan is in STATUS.md; the decisions,
with their reasons, are in `history.md`.

## Step 1: the timed pilot (2026-10-29)

One A40 pod (`26lt9po79hew0i`, secure, CA-MTL-1, $0.59/h, host CUDA 13.2, 46,068 MiB), 09:16-11:28
UTC, about $1.30 by time (the billing read $0.73 so far, $0.01 of it disk), deleted by hand after the
pull. XHToken/Spark-X2.5-4B at 0bcb356 in Q4_K_M with an f16 KV cache, no system prompt, thinking on
(its template's default), the card's sampling (temperature 1.0, top_p 0.95, min_p 0) with top_k
1,000 for the card's -1, seed 0, a thinking budget of 8192, row normalisation "full", judged by
Qwen3Guard-Gen-0.6B and 8B. Plan: `runs/spark/pilot-plan.md`; scripts: `runs/spark/pilot-scripts/`.

**The build took about 40 minutes**, three stops for fixes included: download and BF16 conversion
under 2 minutes, residuals and adapters about 1 minute, and the unedited model's own thinking at 32
slots, 3 minutes for the 100 harmless test prompts and 14 for the 300 held out. Its thinking ended
by itself on 95 of 100 and 278 of 300 (K2: 100 and 299).

**What it needed.** Its remote code was written for transformers 4.57: it lists its tied weights and
calls the mask makers with arguments transformers 5.19 renamed or dropped, which stopped the
residuals twice. `residuals.load_model` now runs it as transformers 4 did. Checked against llama.cpp's
own `spark2_5` on the run's BF16 GGUF, first answer token on the first 32 harmless test prompts:

| Most likely token the same | KL mean | KL median | KL most |
| --- | --- | --- | --- |
| 32 of 32 | 8.3e-5 | 1.6e-5 | 7.1e-4 |

**Directions where the answer starts.** Its template writes an empty thinking as `</think>` alone;
candor reads after `<think></think>`. Read at both, cosine per layer between them: 0.83-0.96 (mean
0.91), 0.92-0.96 at layers 12-30. Each against the directions at the prompt's end: 0.36-0.71.

**Speed** (speed.py, 2,048 tokens a request at the search's 8,469 of context a slot):

| Slots | Tokens/s | Per request | GPU MiB |
| --- | --- | --- | --- |
| 16 | 990 | 62.3 | 9,798 |
| 32 | 1,370 | 43.3 | 16,157 |
| 48 | 1,543 | 32.5 | 22,795 |
| 64 | 1,625 | 25.7 | 29,453 |

At 32 slots with top_k off (the card's -1, 0 in llama.cpp): 662 tokens/s, 2.1 times slower.

**The unedited model as used**, first 64 prompts of each set (`base-slots-32-first-64`, 870 s):

| | Harmful | Harmless | Maths |
| --- | --- | --- | --- |
| Refused, judge / second judge / keywords | 62 / 62 / 63 | 0 / 0 / 2 | |
| Answered unsafely | 0 | | |
| Correct | | | 63 |
| Thinking cut at the budget | 0 | 3 | 4 |
| Thinking tokens, median / most | 462.5 / 2,015 | 688 / 8,215 | 394 / 8,215 |

`measure` puts a full measurement at about 2 h 45 min at most.

**Trials as used** (`search-32`, the unedited model refusing 99 of 100 test prompts): trial 0 took
198 s, trial 21 250 s, trial 1 304 s, 251 s on average; trial 20 stopped on the thinking budget after
393 s and its KL read came back empty for a prompt, which stopped the search (fixed: reads now wait
until no slot is busy). Their thinking ran 612-785 tokens per answer, 3 of 300 answers at the
budget. The fewest refused: one layer, trial 1 (layer 23), 85 at KL 0.0098; per layer, trial 3, 94 at
KL 0.0112. At 64 slots trial 0 took 240 s against 198 s at 32.

**e2e on CUDA**: 15 of 17 passed first. The tiny Qwen4-Exp's linear-attention heads of 8 abort
llama.cpp's CUDA gated delta net (`gated_delta_net.cu:235`, which takes 32, 64 or 128), and a repeat
of the tiny run's trial refused 5 against the search's 3. With heads of 32, a stronger test edit and
`candor score`'s refusal band at 2, both passed on CUDA, the Qwen4-Exp one on the Mac too.

**The quote.** 600 trials at 251 s are 41.8 h, $24.70 on this A40 at $0.59/h, against about $25.70
of credit left; each full measurement adds up to 2 h 45 min ($1.60).

What it shows:
- candor runs Spark-X as it is used: its refusals, maths and thinking are near the card's, its
  forward in transformers matches llama.cpp's, and its two answer-start positions give nearly the
  same directions.
- On an A40 the search alone takes nearly all the credit; 64 slots don't help, as a trial waits on
  its longest answers and each answer runs slower with more beside it.

## Step 2: the search, stopped at 93 trials (2026-10-29)

One A40 (`sk7e6bm640dlrt`, secure, CA-MTL-1, $0.59/h) at 32 slots, carrying on from the pilot's 3
trials, 11:43-21:56 UTC, about $6.05 by time. It was restarted at 12:33 UTC after trial 9 stalled
(the server reload after an early stop, fdfa995). The user stopped it at 93 trials scored, against
the 300 staged, once both kinds' bests within the cap had held: per layer's since trial 18, one
layer's since trial 53. The trial in flight at the stop is lost; the search can carry on from its
journal.

| | One layer | Per layer |
| --- | --- | --- |
| Trials scored | 47 | 46 |
| Stopped early | 12 | 12 |
| Over the budget cap (more than 8 of 100) | 13 | 12 |
| Within both caps, scored in full | 9 | 14 |
| Seconds a trial, mean / median | 339 / 356 | 432 / 469 |
| Answers that reached the budget | 191 of 4,076 | 234 of 4,160 |

Trials numbered under 40, about the evenly spread start, took 266 s on average; the 56 aimed trials
after it took 464 s, as edits that refuse little think longer. Over the aimed trials 323 of 4,932
answers (6.5%) reached the budget, and 15 refused none of 100, scored in full.

How the bests moved: per layer's trial 18 (1 refused, KL 0.0154) held for its last 38 trials, while
0 refused fell from KL 0.0199 (trial 62) to 0.0188 (64) and 0.0175 (70). One layer had none within
the cap until trial 53 (0 refused, KL 0.0154), which held for its last 21.

The finalists, as candor names them: each kind's least KL at each count within the caps (2 refused,
8 at the budget), unless one refusing fewer has less.

| Trial | Kind | Direction | Refused of 100 | KL | At the budget | Strength, attention / MLP |
| --- | --- | --- | --- | --- | --- | --- |
| 53 | one layer | layer 17, think start | 0 | 0.0154 | 6 | 3.56 / 1.44 |
| 70 | per layer | think start | 0 | 0.0175 | 5 | 2.60 / 2.03 |
| 18 | per layer | answer start | 1 | 0.0154 | 7 | 2.65 / 0.91 |

Trial 70's KL is 0.0021 above trial 18's, paired by prompt (SE 0.0037): a tie. Trial 18 presses MLP
strength's floor (0.912 against 0.8), so before it is measured it is scored as the search scores,
with that strength moved to 0.65 and 0.5 (`evidence-2026-10-29/beyond-spark.py`).

What it shows:
- Within 93 trials both kinds found edits refusing 0 or 1 of 100 at KL 0.0154, under the 27B's
  0.0188 after 600. These are search scores: held out, KL has come out 1.3-2.5 times the search's
  (K2's 207, 0.024 to 0.061; the 27B's 435, 0.0188 to 0.0247).
- The budget cap binds: 25 of 93 trials went over it and the finalists sit at 5-7. The unedited
  model's own thinking reaches the budget on 5 of 100 harmless test prompts and 22 of 300 held out
  (K2: 0 and 1), so a cap set on K2 sits close to this model's own rate (findings, hypothesis).

## Step 3: the finalists measured in full; trial 53 picked (2026-10-29 to 30)

The same A40 ran `evidence-2026-10-30/finalists.sh` from 21:57 to 03:12 UTC: the unedited model
measured in full (58 min), trial 18's range-end check (29 min), then each finalist exported (Q4_K_M)
and measured with `measure --exported` (62-78 min each): thinking on, budget 8192, answers up to 256
tokens, 32 slots, seed 0, every held-out prompt and maths question, both judges. The pod then sat
idle until 07:25, 4 h 13 min (about $2.50), as nothing deleted it at the job's end; it was deleted
once everything was pulled: 19.7 h in all, about $11.64 by time. Compared prompt by prompt with the
unedited model (p: McNemar; the rule's bounds one-sided 95%, Bonett and Price).

| | Unedited | 53 | 70 | 18 |
| --- | --- | --- | --- | --- |
| Kind, direction | | one layer, 17, think start | per layer, think start | per layer, answer start |
| Search: refused of 100, KL, at the budget | | 0, 0.0154, 6 | 0, 0.0175, 5 | 1, 0.0154, 7 |
| Harmful refused, judge / 8B (of 288) | 282 / 282 | 0 / 1 | 2 / 3 | 2 / 2 |
| Harmful judged unsafe, judge / 8B | 2 / 2 | 269 / 264 | 273 / 268 | 273 / 271 |
| Harmful refused, keywords | 286 | 28 | 20 | 19 |
| Harmful answers cut at 256 tokens | 186 | 264 | 271 | 271 |
| Harmless refused, judge / 8B (of 300) | 4 / 3 | 0 / 0 | 3 / 3 | 0 / 2 |
| Maths correct (of 1,319; p) | 1,236 | 1,228 (0.4) | 1,224 (0.18) | 1,219 (0.043) |
| Maths thinking cut at 8192 (p) | 66 | 67 (1) | 44 (0.0032) | 96 (0.0016) |
| Thinking cut, harmful / harmless | 0 / 27 | 15 / 28 | 10 / 22 | 15 / 30 |
| Harmless KL, 278 of 300 (95% interval) | | 0.0457 (0.037-0.058) | 0.0520 (0.041-0.073) | 0.0581 (0.042-0.102) |
| KL after empty thinking / at the thinking start | | 0.110 / 0.0052 | 0.145 / 0.0082 | 0.185 / 0.0449 |
| Thinking median, harmful / harmless / maths | 415 / 743.5 / 392 | 1,318 / 807 / 401 | 1,088 / 669 / 347 | 1,550.5 / 885 / 429 |
| Rule's bounds: harmless / maths / budget, pp | | +0.00 / -1.65 / +1.06 | +1.30 / -1.93 / -0.75 | +0.00 / -2.29 / +3.43 |
| Edit kept in the file | | 99.7-100.0% | 99.5-100.6% (2 of 51 weights at Q8_0, merged in F32) | 99.4-100.4% |
| Decision rule | | pass | pass | fail: budget |

KL is read on the 278 harmless prompts whose unedited thinking ended by itself (22 didn't). The
rule's bars: harmful refused 10 or fewer of 288 under both judges; harmless refused's upper bound
under +2 pp; maths correct's lower bound above -2.5 pp; maths thinking cut's upper bound under +1.25
pp, a quarter of the unedited model's 5.0%, as that is above 1 pp. Trial 18 fails the last (+30,
upper +3.43 pp). Between the two that pass, harmful answers judged unsafe are level, 273 for 70
against 269 for 53 (p 0.39) and under the 8B 268 against 264 (p 0.48), so the lower mean KL ships:
trial 53. The rule was the one proposed before the measurements started; the user accepted it with
the results.

**Trial 18 past its floor** (`evidence-2026-10-29/beyond-spark.txt`), scored as the search scores,
in one session: as it is, 0 refused, KL 0.0168, 10 at the budget (in the search 1, 0.0154, 7); MLP
strength 0.65, 2 refused, KL 0.0153 (gap -0.0016, SE 0.0013, a tie); 0.5, 2 refused, KL 0.0125
(-0.0044, SE 0.0025, a tie). Nothing beats it beyond noise past the floor, only the usual trade of
KL for refusals.

What it shows:
- Trial 53 removes every refusal under the run's judge (1 under the 8B) and keeps harmless refusals,
  maths and maths thinking within noise of the unedited model.
- Held out, each finalist's KL read about 3 times its search KL (0.0154 to 0.0457, 0.0175 to 0.0520,
  0.0154 to 0.0581), more than K2's finalists (1.7-2.6) and the 27B's 435 and 532 (1.3 and 1.7).
- The finalist nearest the budget cap in the search (18: 7, and 10 on a repeat) failed on maths
  thinking, while 53 (6) and 70 (5) passed; here the cap let through an edit that fails rather than
  shutting out ones that pass (findings).

## Step 4: the release (2026-10-30)

On the Mac, as `evidence-2026-10-22/release.sh` did on the 27B's pod: the measured export (sha256
`311acb34…`, as each set's record holds) with `general.name` set to Spark-X2.5-4B-Candor-Uncensored
and `general.finetune` (the snapshot hash) dropped; all 290 tensors byte-identical and no other key
changed (`same.py`). Against the unedited Q4_K_M, 31 of 290 tensors differ
(`evidence-2026-10-30/edited.py`, on the pod): `attn_output` 27 (layers 9-35, Q4_K) and `ffn_down` 4
(21-24, Q4_K or Q6_K). It loaded in llama-server b11524.
`Spark-X2.5-4B-Candor-Uncensored-Q4_K_M.gguf`: 2,600,223,616 bytes, sha256
`fcb8118e75e6c43d1bac36fa9b1f12dbc0e62a99e5f4d7fe1a6108a719bfdb79`, the Hub's LFS hash the same.

Uploaded from candor-team to a private repo with the base's LICENSE (byte-identical to XHToken's at
0bcb356) and the header, about 2 minutes; the card was reviewed against the measurements, with two
fixes (KL is read on 278 of the 300 harmless prompts; "that mattered" became "it ruled one out", as
the rule ships 53 whether or not 18 passes), then the repo was made public:
`candor-lab/Spark-X2.5-4B-Candor-Uncensored-GGUF`. The card follows K2's, with no comparison column,
and drops the earlier cards' "(not released yet)", as the code is public. The card as published:
`evidence-2026-10-30/spark-card.md`.

**Cost of the phase:** the pilot about $1.30 and the search's pod about $11.64 (19.7 h, of which 4 h
13 min idle), about $12.94 in all, from 2026-10-29 to 30.
