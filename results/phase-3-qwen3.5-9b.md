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
