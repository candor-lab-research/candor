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
