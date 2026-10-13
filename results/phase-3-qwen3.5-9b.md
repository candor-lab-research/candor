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
