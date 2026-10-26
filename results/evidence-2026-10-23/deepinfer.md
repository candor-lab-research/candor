# DeepInfer review (2026-10-23)

`deepseek-ai/DeepInfer` at 9bd9cf2 (2026-10-21): all its docs, the batch-invariance and KL tests,
the sampler, the model's config and its residual-stream code read; about 55,000 lines of Python and
5,900 of C++.

## What it is

DeepSeek's production inference engine, released for one model, DeepSeek-V4.1-Flash, on NVIDIA
SM100 and Huawei Ascend 950 only. The release is offline batch inference: the KV cache system,
prefill/decode disaggregation and load balancing that make it a service are not included. Settings
are about 57 environment variables. Kernels come from DeepSeek's own libraries (FlashMLA, DeepGEMM,
DeepEP, DeepSelect, TileKernels) plus in-tree TileLang kernels.

`doc/inference_deployment_qa.md` explains the production system: separate vision, prefill and decode
machines; decode at EP64-128 ("Wide-EP"); KV cache kept 3 days for agent traffic and 7 for chat on
about 100 PiB of 3FS SSD; expert load balancing settings and their measured imbalance (1.045-1.090);
a quadratic cost model that sizes each prefill chunk; SWA bounded replay (recomputing only the last
128 tokens, an approximation); FP4 for the main KV, the indexer and routed experts; and DSpark,
speculative decoding whose confidence head picks which draft tokens to verify (+5.4% throughput at
60 sequences per rank, -1.5% at 16). The stated scale: 155 trillion tokens a day from one cluster,
US$38 revenue per GPU-hour at off-peak prices.

## What is worth knowing for candor

1. **Batch-invariant inference, tested as exact.** With `DEEPINFER_BATCH_INVARIANT=1` the kernels
   switch to deterministic, batch-invariant modes, and sampling draws from a hash of (sequence seed, position) instead of
   a running RNG. `tests/models/test_batch_invariant.py` requires 64 sequences generated together to
   match each sequence generated alone in tokens, logprobs and top-5, exactly;
   `tests/models/test_kl.py` requires KL 0 between decoding and a full prefill of the same tokens.
   This is the published fix for the noise candor measured (findings, "Batching moves answers" and
   "A trial's KL read again"). llama.cpp has no such mode that we could find; vLLM has one. Idea in
   findings.
2. **It is not a runner candor can use.** One model, SM100 or Ascend 950 only, no GGUF, no server.
3. **DeepSeek-V4.1-Flash is not a candor target.** 763B stored parameters by HF's count (routed
   experts packed as FP4); 40 layers, 384 routed experts with 6 active. Its residual stream is four
   streams mixed before and after every sub-layer (mHC, `hc_mult` 4), and Engram adds n-gram
   embedding tables at layers 1 and 14 as a third writer to the stream, so directional ablation's
   single-stream assumption would need rethinking. candor's llama.cpp list (commit 86a2835) has
   `DeepseekV4ForCausalLM` but not `DeepseekV41ForCausalLM`, which is why the gap count left it out,
   and the gap is already filled: 11 uncensored or abliterated versions in HF's top 50 results for
   its name.
4. **Evaluation practice we already follow.** Generation and scoring are separate so a run can be
   scored again; summaries report rule-vs-judge agreement (960 of 960 AIME rollouts) and refuse a
   score with missing rollouts. One caveat on their numbers: the documented judge is DeepSeek's own
   API model, and HLE (42.6%) is judged by it alone.
