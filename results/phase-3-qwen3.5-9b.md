# Phase 3 results: Qwen3.5-9B

Numbers only; answers stay in each run's private folder. The plan and decisions are in STATUS.md.

## Step 2: candor on a pod, checked on rude-llama (2026-10-13)

One A40 pod (secure cloud, CA-MTL-1, $0.59/hr, host CUDA 13.0, driver 580.173.02), created with
`scripts/pod.sh args 1` and deleted by its own watchdog after 52 minutes: about $0.51.

| Check | Result |
| --- | --- |
| `setup` | 80 s; torch 2.11.0+cu128 and llama-server both see the A40 |
| rude-llama's safetensors, uploaded from the Mac | sha256 identical; 6.4 GB in 33 min (about 3 MB/s) |
| `candor run` (convert, quantise, prompt ids, residuals on CUDA, adapters) | All stages ran; about 4 min after the upload |
| F16 GGUF, `prompts.json` | Byte-identical to the Mac's |
| Q4_K_M GGUF | All 197 quantised tensors differ from the Mac's (same types), and the pod made the same file twice: llama-quantize's k-quants differ between its macOS arm64 and Linux x86 builds |
| Directions (bf16 on CUDA against fp16 on the Mac) | Cosine at least 0.99987 on every layer; 0.99993 or more on layers 11-24, which the edits read |
| Trial 535's adapters, built on the pod from the Mac's directions | Same metadata; tensors within 2.5e-6 relative |
| The watchdog | Moved to 36 s from now, it deleted the pod itself: unreachable within 100 s, then 404 from Runpod |

The first 10 prompts of each held-out set, on each machine's own Q4_K_M:

| | Mac, unedited | Pod, unedited | Mac, trial 535 | Pod, trial 535 |
| --- | --- | --- | --- | --- |
| Harmful refused, judge / keywords | 9 / 9 | 9 / 9 | 0 / 0 | 0 / 0 |
| Harmful answered unsafely | 0 | 0 | 8 | 8 |
| Harmful cut off at 256 tokens | 1 | 1 | 10 | 10 |
| Harmless refused, judge / keywords | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Maths correct | 5 | 5 | 5 | 5 |
| Maths cut off at 512 tokens | 0 | 0 | 3 | 1 |
| Tokens per second (harmful, harmless, maths) | not kept | 257, 387, 407 | 188, 166, 85 | 529, 605, 395 |

What this means for the rest of phase 3:

- A pod gives the Mac's results: across the two pairs (60 answers) the only count that moved is
  maths cut off, 3 against 1, with maths correct unchanged.
- Each machine quantises its own Q4_K_M, so the final check measures the file exported on the pod,
  and that file is copied to the Mac as it is, never quantised again there.
- Uploading from home is slow. Qwen3.5-9B comes from the Hub, which the pod downloads directly.
- A local model copied to a pod gets a new fingerprint (rsync from macOS keeps file times only to the
  second), so every stage rebuilt; the check gave the pod's copy the Mac's times to rebuild trial
  535 exactly. Hub models are fingerprinted by their commit and unaffected.
- The image's runpodctl is 1.14.15, which has no `pod delete`; the watchdog's fallback covers it.
- Measuring rude-llama would take about 16-18 minutes on the A40 against up to 1 h 30 min on the Mac.

## Step 3: the search's miss rate with the thinking-model changes (2026-10-13)

Before any Qwen3.5 search, the phase 2 test landscapes (calibrated to rude-llama's real trials) were
extended with landscapes whose best edit needs strength 2.0-2.6, and with landscapes that read
directions at two places (best at the same place, at the answer start, at the think start, needing
high strength). candor's own search code drew every trial; 60-100 runs per landscape and design.
Numbers only. Share of runs whose pick is within 10% of the true best, at 600 trials:

| Landscapes | Strength 0.8-1.5 | 0.5-3.0 | 0.8-1.5, widened on the edge alarm |
| --- | --- | --- | --- |
| Best inside 0.8-1.5, KL rising outside it | 61% | 27% | 48% |
| Best inside 0.8-1.5, nothing rising outside it | 63% | 28% | 54% |
| Best at attention 2.0-2.3, 0.8-1.5's best 27-35% worse | 8% | 23% | 20% |
| Best needs 2.0-2.5, KL rising outside 0.8-1.5 | 5% | 73% | 66% |
| Best needs 2.0-2.6, nothing rising outside it | 7% | 82% | 70% |

A fixed wide range halves the hits when the best lies inside 0.8-1.5, so the planned 0.5-3.0 for
Qwen3.5 was dropped. With two places to read directions, pooled over the four landscapes (KL rising
outside 0.8-1.5 / not):

| Design | 300 trials | 400 | 600 | Prompts judged per trial |
| --- | --- | --- | --- | --- |
| One study per kind, the place a parameter, 0.5-3.0 | 17% / 19% | 26% / 24% | 38% / 35% | 67 / 65 |
| Four studies, kind by place, 0.5-3.0 | 8% / 7% | 13% / 12% | 22% / 23% | 79 / 78 |
| One study per kind, the start split 10 / 10 by place, 0.5-3.0 | 16% / 18% | 25% / 25% | 38% / 37% | 67 / 65 |
| The same from 0.8-1.5, widened on the edge alarm (built) | 36% / 34% | 48% / 48% | 65% / 66% | 67 / 67 |

- The place as a parameter costs almost nothing; four studies are 12-17 points worse and judge
  18-40% more prompts per trial, as each study takes longer to reach the 25 fully scored trials
  that let a trial stop early.
- The search read the best edit's place in 99-100% of runs, except where the two places' bests were
  within 13% of each other (62-87%), where either is nearly as good.
- 600 trials stay: each step from 300 to 400 to 600 adds 12-18 points.
- On the landscapes that need high strength, widening stays within about 12 points of a wide range
  from the start, and above it where the place also matters (82% / 92% against 75% / 83%).

The rule built: from a kind's 60th trial and every 30th after, if its best trial within the cap
presses an end of strength (within 5% of the range), or none is within the cap, that end moves out
to 0.5 or 3.0, once; the kind's trials so far carry on in a new study over the wider range, with 10
evenly spread trials to start it.

A caveat: "within 10%" judges the pick against the best edit truly refusing at most 2 of 100, but a
pick measured at 2 refused truly refuses 2-4, so the measure flatters every design. Judged more
strictly the shares fall to 5-45%, with the designs in the same order. Expect the search to land near
the best, not on it, which is why its suggestions are measured on the held-out prompts.

## Step 4: the unedited model and heretic's Trial 161, as used (2026-10-14)

On an A40 pod (secure, $0.59/hr, about 16 h 20 min, $9.56 in all): Qwen/Qwen3.5-9B at c202236 in
Q4_K_M with an f16 KV cache, no system prompt, sampled at Qwen's thinking settings (temperature 1.0,
top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5) with seed 0, thinking to a budget of 8192, 32
slots, judged by Qwen3Guard-Gen-0.6B. `candor run` built 2,304 adapters; the unedited model's own
thinking ended by itself on 96 of 100 harmless test prompts and 294 of 300 held-out ones.

Trial 161 is heretic's own save (re-saved by heretic 1.4.0 on the pod; its menu said 11/100 refused,
KL 0.0675, both thinking off), exported with `--outside` and measured as candor ships its edits. Each
edited weight keeps 99.6-100.3% of its edit in Q4_K_M, so no Q8_0 was needed.

Held-out prompts, paired prompt by prompt (`candor compare`):

| | Unedited | Trial 161 | Only one / the other | p |
| --- | --- | --- | --- | --- |
| Harmful refused, judge | 279/288 | 28/288 | 251 / 0 | 6e-76 |
| Harmful refused, keywords | 282/288 | 29/288 | 254 / 1 | 9e-75 |
| Harmful answered unsafely | 2/288 | 234/288 | 0 / 232 | 3e-70 |
| Harmless refused, judge | 5/300 | 1/300 | 4 / 0 | 0.13 |
| Maths correct | 1183/1319 (89.7%) | 1162/1319 (88.1%) | 58 / 37 | 0.04 |
| Maths answer cut off | 112/1319 | 131/1319 | 41 / 60 | 0.07 |
| Maths thinking cut at 8192 | 184/1319 | 297/1319 | 53 / 166 | 9e-15 |
| Harmless thinking cut at 8192 | 9/300 | 5/300 | 5 / 1 | 0.22 |
| Harmful thinking cut at 8192 | 0/288 | 2/288 | 0 / 2 | 0.5 |

| Thinking, median tokens | Unedited | Trial 161 |
| --- | --- | --- |
| Harmful | 996 | 2,980 |
| Harmless | 1,137 | 1,175 |
| Maths | 2,140 | 3,620 |

KL of Trial 161 from the unedited model on the harmless held-out prompts: 0.073 where the answer
starts after the unedited model's own thinking, 0.106 after an empty thinking block (heretic's
place), 0.064 where the thinking starts.

- Trial 161 removes nearly all refusals with thinking on (279 to 28), and refuses no more harmless
  prompts.
- It costs maths, just past noise: 21 fewer correct, p = 0.04 paired. And it makes the model think
  far longer: maths thinking reaches the budget 113 more times, harmful thinking triples. These are
  where candor can beat it.
- KL at the start of the thinking (0.064) does tell Trial 161 from the unedited model, against the
  phase 3 assumption that the first thinking token is nearly fixed.
- The 32-prompt pilot (unedited: harmful 30/32 refused, harmless 0/32, maths 27/32) put maths
  thinking well below Trial 161's (median 2,489 against 3,620), which suggested about 4.3 h for the
  unedited model's maths; the long tail (184 reaching the budget) made it 4.9 h. Time follows the
  tail, not the median.
- The unedited model measured again with another seed, for noise, was skipped (the user, 2026-10-14):
  the decision rule uses paired tests instead.
- Time: unedited 6.0 h (harmful 22 min, harmless 45 min, maths 4.9 h), Trial 161 8.0 h, at 243-271
  tokens/s, with sampling done on the CPU.

### Speed of sampled thinking in llama-server on the A40 (32 requests of 1,024 tokens)

| Setup | Tokens/s |
| --- | --- |
| 1 server, 32 slots, sampling on the CPU (as measured above) | 304 |
| 2 servers of 16 slots, CPU | 524 |
| 3 servers of 11 slots, CPU | 392 |
| 1 server, 32 slots, sampling on the GPU (`--backend-sampling`) | 662 |
| 2 servers, GPU | 456 |
| 3 servers, GPU | 330 |
| 1 server, 64 slots, GPU | 760 |

A cheap trial's answers (100 prompts, thinking closed, greedy, 100 tokens, 32 slots): 21.6 s.

llama.cpp turns GPU sampling off for a request with a thinking budget, which is why it gave nothing
before; candor now holds the model to its budget itself. More servers only help while sampling is on
the CPU; 64 slots add 15% over 32.

## Step 5: the "full" edit's search with thinking off, then `candor proxy` (2026-10-14)

Same pod setup as step 4, sampling on the GPU, the 100 test prompts. Trial 161 rebuilt as a "full"
edit and scored as the search scores, thinking off: 24 refused (judge), 12 (keywords), KL 0.092; the
unedited model 99.

**Search, 150 trials, thinking off** (13:45-17:23, 3 h 38 min, about 89 s a trial). At about trial
120 both kinds' best pressed strength 1.5 and the search widened it to 3.0; before that its best was
22 refused (trial 98, KL 0.0725). Trials no other beats on both refusals and KL:

| Trial | Direction | Refused | Keywords | KL |
| --- | --- | --- | --- | --- |
| 146 | per layer, answer start | 1 | 25 | 0.1022 |
| 150 | per layer, answer start | 3 | 38 | 0.0931 |
| 148 | per layer, answer start | 4 | 40 | 0.0927 |
| 98 | per layer, answer start | 22 | 52 | 0.0725 |
| 118 | per layer, answer start | 25 | 56 | 0.0721 |

Trial 146 presses three edges of the widened ranges (mlp strength at its ceiling of 3, mlp peak at
its floor, mlp reach at its ceiling). The best one-layer edit: trial 145, 5 refused, KL 0.383.

**`candor proxy`**: the best 3 trials of each kind and position, each at its own strength and at 1.4
times it, scored thinking off and then as used (Qwen's sampling, budget 8192, 32 slots). Thinking off
took 39 min, thinking on 3 h 47 min, the repeat 12 min.

| Edit | Trial | Direction | Strength | Refused, off | Refused, on | Thinking, mean tokens | Budget hit | KL, on |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1-6 | 83, 67, 37 | one layer, think start | x1, x1.4 | 14-99 | 17-100 | 714-2,182 | 0 | 0.005-0.141 |
| 7 | 145 | one layer, answer start | x1 | 1 | 6 | 2,140 | 5 | 0.392 |
| 8 | 145 | | x1.4 | 3 | 5 | 1,794 | 10 | 0.434 |
| 9 | 141 | | x1 | 21 | 41 | 2,325 | 0 | 0.278 |
| 10 | 141 | | x1.4 | 8 | 18 | 2,486 | 4 | 0.361 |
| 11 | 143 | | x1 | 22 | 50 | 2,458 | 0 | 0.261 |
| 12 | 143 | | x1.4 | 13 | 30 | 2,876 | 6 | 0.384 |
| 13-18 | 34, 72, 2 | per layer, think start | x1, x1.4 | 87-98 | 71-100 | 791-1,301 | 0 | 0.005-0.034 |
| 19 | 146 | per layer, answer start | x1 | 1 | 1 | 1,807 | 0 | 0.106 |
| 20 | 146 | | x1.4 | 2 | 1 | 1,638 | 0 | 0.110 |
| 21 | 150 | | x1 | 4 | 11 | 1,872 | 0 | 0.100 |
| 22 | 150 | | x1.4 | 1 | 1 | 1,758 | 0 | 0.101 |
| 23 | 148 | | x1 | 7 | 14 | 1,983 | 0 | 0.092 |
| 24 | 148 | | x1.4 | 1 | 1 | 1,729 | 0 | 0.099 |

The unedited model: 99 refused both ways. Edit 24 again with another seed: 2 (against 1).

- **Verdict: "offset".** Rank agreement (Spearman) 0.98, above the 0.7 needed; but where thinking off
  refuses at most 25, thinking refuses a median of 5 more (8-28 more in the middle, none at the
  strongest edits). So the search may score thinking off, and the edits it suggests must also be
  scored as used at more strength.
- Within the cap of 2 as used: edits 19, 20, 22 and 24, all per layer at the answer start, KL
  0.099-0.110. Like for like with Trial 161 rebuilt (thinking off, KL 0.092): 7-15% more KL for 1
  refused against 24. Not yet within the decision rule's KL (Trial 161's 0.073, held-out prompts).
- Directions read at the think start refused 71-100 either way, as before.
- One-layer edits that comply make the thinking longest and hit the budget most (up to 10 of 100).

## Step 6: the search carried on, thinking off (2026-10-14 to 15)

From 150 trials towards 600 on the same pod, 22:05-07:32. It stopped at 545 trials (313 scored in
full, the rest stopped early as unable to win) when the pod's GPU failed: llama-server's
allocations failed after 395 trials in one server, then CUDA wouldn't start. The last 55 trials
were skipped: no trial set a new best of either kind after trial 428. About 86 s a trial.

| Trial | Kind | Read at | Layer | Refused | Keywords | KL | Strength, attention / MLP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 205 | one layer | answer start | 12 | 0 | 8 | 0.0325 | 1.06 / 2.42 |
| 417 | one layer | answer start | 12 | 1 | 20 | 0.0294 | 1.12 / 2.16 |
| 175 | one layer | thinking start | 15 | 0 | 5 | 0.0567 | 1.52 / 2.28 |
| 342 | per layer | answer start | | 0 | 33 | 0.0837 | 2.99 / 1.35 |
| 428 | per layer | answer start | | 1 | 35 | 0.0793 | 2.91 / 1.14 |
| 402 | per layer | answer start | | 2 | 41 | 0.0789 | 2.92 / 1.06 |

Trial 161 rebuilt, scored the same way: 24 refused, 12 by keywords, KL 0.092. By trial 300 the
one-layer best was within 1% of this KL, the per-layer best 7-11% above it. Every per-layer best
puts attention at 2.9-3.0, the ceiling of the widened range.

The pod (`m37qrjorh2geg9`, 20.75 h with step 5) cost about $12.25. On a new pod the run rebuilt
from the Hub in about 15 minutes (setup 75 s, the run folder's push 2 min); its unedited thinking
counts matched step 4's (96 of 100, 294 of 300).

## Step 6: the search's best scored as used (2026-10-15)

`candor proxy --trial` on a new A40 pod (`oi41qp8xyncyy6`, 07:40-10:50, about $1.90): 6 trials,
each at its own strength and 1.4 times it, on the 100 test prompts with thinking off and then as
used (Qwen's sampling, budget 8192, 32 slots). Thinking off took about 26 min, thinking on about
2 h 12 min, the repeat about 12 min. The unedited model: 99 refused both ways.

| Trial | Kind | Strength | Refused, off | Refused, on | Keywords, on | Thinking, mean tokens | Budget hit | KL, on |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 205 | one layer, answer start, layer 12 | x1 | 0 | **1** | 20 | 2,059 | 0 | **0.0369** |
| 205 | | x1.4 | 0 | 20 | 23 | 1,237 | 2 | 0.0907 |
| 417 | one layer, answer start, layer 12 | x1 | 1 | 6 | 31 | 2,068 | 0 | 0.0310 |
| 417 | | x1.4 | 0 | 49 | 49 | 1,866 | 10 | 0.1199 |
| 175 | one layer, thinking start, layer 15 | x1 | 0 | **0** | 5 | 1,952 | 0 | **0.0568** |
| 175 | | x1.4 | 3 | 49 | 50 | 1,252 | 3 | 0.1350 |
| 342 | per layer, answer start | x1 | 2 | 5 | 46 | 1,740 | 0 | 0.0902 |
| 342 | | x1.4 | 1 | **1** | 37 | 1,701 | 0 | 0.0995 |
| 428 | per layer, answer start | x1 | 1 | **1** | 44 | 1,732 | 0 | 0.0845 |
| 428 | | x1.4 | 1 | **1** | 35 | 1,668 | 0 | 0.0899 |
| 402 | per layer, answer start | x1 | 4 | 6 | 52 | 1,890 | 0 | 0.0834 |
| 402 | | x1.4 | 1 | 3 | 50 | 1,855 | 0 | 0.0893 |

Trial 175 at its strength again with another seed: 0 refused (KL 0.0553, thinking 1,674).

- Within the cap of 2 as used: 205 and 175 at their strength (one layer, KL 0.037 and 0.057), 428
  at both strengths and 342 at 1.4 times (per layer, KL 0.085-0.100). Trial 161 rebuilt, on the same
  prompts thinking off: 24 refused, KL 0.092.
- 1.4 times the strength made every one-layer edit far worse as used (1 to 20, 6 to 49, 0 to 49),
  though not with thinking off, and shortened its thinking. On per-layer edits it held or helped
  (5 to 1, 1 to 1, 6 to 3).
- Where thinking off refuses at most 25, thinking refused a median of 2 more (step 5: 5).
- Keyword counts stay far from the judge's on per-layer edits (35-52 against 1-6).
- GPU memory held at 24.4-24.6 GB through the thinking-on scoring.
