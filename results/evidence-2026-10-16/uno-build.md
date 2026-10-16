# Uno for K2-Horizon-7B in candor: what it would take (research, 2026-10-16)

**Recommendation: don't build yet. Run a quick check first** (about 1 h on step 2's A40 pilot pod, about $1-2). The
method is real and the K2 adapter exists, but at 32 concurrent sequences on Ampere the gain is small or below 1x, and
only llama.cpp keeps candor's measure-as-used rule. Building it there is weeks of work on a fork.

## 1. How Uno decodes [Published + read in code]
- **Same network, gated LoRA, no extra head.** Each AR matrix (q,k,v,o,gate,up,down, all 36 layers) gets a rank-128 LoRA,
  applied only to "noise" rows, never to the seed row (paper Sec 3.1-3.2; `nano_vllm_uno/layers/linear.py:181-226`).
- **One cycle = 2 forwards of B rows per sequence** (`engine/two_pass_decoding.py:100-109, 151-197`):
  - Draft `[last token, B-1 random tokens]`, LoRA on rows 1+. Row 0 (base weights) samples the next token exactly; rows
    1+ give the draft marginals q. Noise KV is dropped, seed KV kept (`:273-277, :319-327`).
  - Verify `[clean, d1..dB-1]` with base weights only: accept with min(1,p/q), else draw from norm(max(0,p-q)), plus a
    bonus token if all pass (`:56-97, :584-640`). 2 to B+1 tokens per cycle; TPF = tokens / (2 x cycles), at most
    (B+1)/2 (Sec 4.2; `generation.py:120`).
- **Attention stays causal, even inside the draft block:** a block-causal mask in training (Sec 3.2), `causal=True` at
  inference (`layers/attention.py:628-665`). No bidirectional mask. Noise tokens are uniform over ids [1, 250624).
- **Lossless under sampling** (temperature, top-k, top-p): the same filter goes on p and q (`two_pass_decoding.py:642-760`).
  No min_p or penalties. Lossless means the same distribution, not the same tokens: a seed gives different text than AR.

## 2. Engines [Read in code/PRs]
- **IFM's `nano_vllm_uno`** (Nano-vLLM fork, Python, offline `LLM.generate`, no HTTP server). K2 native in BF16
  (`engine/model_runner.py:102-104`); FP8 only on Hopper. The Linear sampler needs FA2; the Tree sampler (the 2.71 TPF
  figure) needs FA3, which is Hopper-only (`README.md:115`). **No per-request seed** (global torch RNG), no min_p.
- **SGLang** native UNO, merged 2026-09-03 (sgl-project/sglang#37667, a paper author). **Requires FA3**
  (`srt/arg_groups/speculative_hook.py:603`), so not on an A40. Refuses min_p, penalties, logprobs, per-request LoRA and
  multi-LoRA (`srt/speculative/uno_validation.py:11-50`, `speculative_hook.py:579`).
- **vLLM**: PR #55947 (Model Runner V2) open, not merged (RFC #55267). Ampere-tested; refuses request LoRA.
- **llama.cpp: nothing** on Uno, gated LoRA or diffusion-augmented decoding (issues, PRs, discussions; head f12aaa3).
  - Reusable: target-preserving rejection sampling (`common/sampling.cpp:733-840`, used when temp > 0,
    `tools/server/server-context.cpp:507-510, 4530-4543`), `seq_rm` rollback, multi-slot speculation; per-token LoRA
    precedent in `src/models/granite-switch.cpp:192-220`; all 7 targets go through `build_lora_mm`
    (`src/models/k2-horizon.cpp:248-291`).
  - Doesn't fit: every speculative type (`common/common.h:174-187`: draft, EAGLE-3, MTP, DFlash, DSpark, n-gram) drafts
    in a separate `ctx_dft` with its own KV; Uno drafts in the target's. LoRA covers the whole batch
    (`src/llama-graph.cpp:1615-1652`).

## 3. The adapter files [HF listing, read]
- IFM/K2-Horizon-7B-Uno at sha 669f041: `adapter_model.safetensors` (1,396,763,616 B, 504 fp32 tensors, about 0.35B
  params) plus small json files. Plain PEFT LoRA: r=128, alpha 8192 (scale 64; the paper's text says 256), 7 targets,
  no head, no extra tensors.
- **GGUF:** `convert_lora_to_gguf.py` with `conversion/k2_horizon.py` (no q/k permute for dense) should give a normal
  llama.cpp LoRA, about 0.7 GB at F16 [Hypothesis]. No new tensor types, but **new graph logic**: a per-token gate.
- **Training base:** K2-Horizon-7B rev `586b03f` (2026-09-03), pinned in `examples/uno_8B/run_inference.sh:7-9`. HF
  `main` became that checkpoint plus a merged FTPO "anti-doom-loop" LoRA on 2026-10-01 (`merge_info.json`).

## 4. Edits [Published + Hypothesis]
- **Still lossless with edited weights: yes, by construction.** The verifier is the edited base; the draft is the edited
  base plus the LoRA. Only acceptance (speed) can change.
- **Base drift in the paper:** RL plus a merge on top of the SFT checkpoint the adapter was trained on took average TPF
  from 2.25 to 2.10 (-6%, Table 8). Maths fell more (GSM8K 2.66 to 1.95, AIME-26 2.22 to 1.83). Qwen adapters trained on
  mismatched data still speed it up (Sec 5.2).
- [Hypothesis] A rank-1 refusal edit moves the base less than RL: expect at most about a 6% drop, maybe more on harmful
  prompts. The quick check measures it.

## 5. Speed that matters [Published; H200 BF16 unless noted]
- **K2 TPF, Linear sampler (the only one on an A40):** about 1.8-1.9 at B=4, 2.1-2.2 at B=8, also on AIME and AA-LCR at
  temp 1 and top_p 0.95 (Table 6). That is 3.6-4.4 tokens per 8-16 rows: **2.2-3.6x the compute per token** of AR.
- **K2 throughput:** 5,255 vs 3,577 tok/s at batch 64 (1.47x); 383 vs 176 at batch 1 (2.2x) (Table 7). Synthetic
  1K-in/8K-out runs at the measured TPF (Sec 5). Qwen3-8B at 32 streams: 1.74x (Table 18).
- **Independent vLLM on an RTX 3090** (sm_86, same GA102 class as the A40), Qwen3-8B BF16 (PR #55947): 1.65-1.85x at 1
  stream, 1.13-1.46x at 4, **0.72-0.77x at 16 (slower than plain)**. H100: 1.74x at 32.
- **K2 Uno in vLLM on a 3090** (FP8 W8A16): 154 tok/s at 1 stream, 869 at 32; no plain baseline (Zenodo 22652610).
- [Hypothesis] llama-server Uno at 32 slots on an A40: about 1.0-1.3x, more in the tail.

## 6. Build options
- **(a) Uno in llama-server, about 3-4 weeks on a fork, upstreaming uncertain:**
  - Convert the adapter: 0.5 d. Per-token LoRA gate (graph input in `build_lora_mm`, ubatch field, CUDA-graph safe): 2-3 d.
  - Slot loop drafting inside `ctx_tgt` (noise rows, `seq_rm`, q, verify with the existing rejection sampler, bonus),
    with the forced thinking close, prompt cache and checkpoints: 5-8 d.
  - GPU top-k/top-p for about 15 rows per sequence per cycle over 250,624 tokens: 3-5 d. Lossless tests: 2 d.
  - Risk: no gain at 32 slots. Upside: about 2x single-stream on the user's Mac.
- **(b) IFM's engine or vLLM for the search, final measurement in llama-server (3-5 d plus a rank check).** A proxy
  that breaks comparability:
  - BF16, not Q4_K_M; no min_p (candor's default is 0.05); no per-request seed in nano-vllm.
  - K2 KV is 144 KiB per token at 16-bit. Next to 19 GB of weights on a 48 GB A40, about 18 sequences of 9.2K fit, not 32.
  - Edits can be patched into loaded weights in place. Much of any gain is the engine's, not Uno's.
- **(c1) Quick check, 30-60 min on the step-2 pod.** IFM's repo, `ATTENTION_BACKEND=fa2`, K2 at 32 concurrent with
  `--diffusion-block-size` 1 (plain AR, same engine), 4 and 8; on rev 586b03f, on main, and with one edit. Record TPF and
  tok/s only via the Python API, never printing `generations` (`inference.py:142`). Compare with llama-server `-np 32`.
- **(c2) llama-server's own speculation** (n-gram, or a K2-3.7B draft with the same vocabulary; the 0.9B's differs):
  no build, already lossless, unlikely to pay at 32 slots. **(c3)** Wait for vLLM #55947 or a llama.cpp port.

## Side findings for candor
- **geantendormi's "K2-Horizon-7B-Uno-Uncensored" merged the Uno adapter into the base weights before ablating** (card:
  adapter fused in, rank-2 projection on layers 13-32, IQ4_XS). That makes it a different AR model from K2-Horizon-7B,
  with no speedup in llama.cpp. It is weaker as a K2 reference. hermitdave's and mlx-community's "merged" Uno MLX files
  did the same.
- Check that the K2 revision candor reads directions from matches the GGUF it measures (GGUFs 2026-09-22, main 2026-10-01).

Sources: arXiv 2609.04010v1; github.com/ifm-ai/uno @ fdfa90d; HF IFM/K2-Horizon-7B-Uno, IFM/K2-Horizon-7B (rev 586b03f,
merge_info.json), IFM/K2-Horizon-7B-GGUF; sglang#37667; vllm#55267, #55947; brntech/vllm-uno; zenodo 22652610; llama.cpp f12aaa3.
