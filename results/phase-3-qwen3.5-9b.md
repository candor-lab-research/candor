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
