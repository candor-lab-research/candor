# Literature: refusal dose-response, spread, and thinking models

Web research only, 2026-10-14. Every source below was opened (abstract or full HTML) and checked for the claim attributed to it. "Informal" means a blog post or forum write-up, not a paper. Preprints from 2026 are not peer reviewed unless a venue is given.

## Q1. Dose-response: strength and number/choice of layers

### What has actually been measured

| Source | Models | What was varied | Shape reported |
|---|---|---|---|
| Rimsky (Panickssery) et al., CAA, ACL 2024, [2312.06681](https://arxiv.org/abs/2312.06681) | Llama 2 Chat 7B, 13B | Steering multiplier -1, 0, +1 at one layer (13 for 7B); layer sweep | Monotone, diminishing returns (13B multiple choice: 0.56 / 0.78 / 0.86). Open-ended refusal on 7B barely moved (4.60 / 4.62 / 4.62). Larger multipliers degrade text. |
| Cyberey & Evans, COLM 2025, [2504.17130](https://arxiv.org/abs/2504.17130) | Gemma, Llama 2/3.1, Qwen, Yi; R1-Distill-Qwen 1.5B/7B/32B | Coefficient -1 to 1 in 0.2 steps, one selected layer | Refusal rises gradually with the coefficient; about 0.5 at 0. At -1: Gemma-2B 0.94 to 0.02, Qwen-7B 0.94 to 0.16 (less complete). |
| Ali et al., [2507.11771](https://arxiv.org/abs/2507.11771) | Llama 2 7B, 13B, 70B | CAA at every layer | Peak at about 0.4 of depth; effect falls with model size (exponential fit); negative steering stronger than positive. No curve shape reported. |
| Marshall, Scherlis, Belrose (EleutherAI), [2411.09003](https://arxiv.org/abs/2411.09003) | 10 models (Llama 3 8B/70B, Qwen 1.8-72B, Yi 6B/34B, Gemma 2B/7B) + RWKV v5 | ACE parameter alpha at one mid layer (alpha 0 = harmless mean, 1 = harmful mean) | Plain CAA is "non-standardized" (dose depends on the prompt). ACE is "nearly perfectly standardized at alpha = -0.5 and 1.5" on Llama 3 8B, i.e. pushing past the harmless mean gives uniform compliance. "Poor layer choice either produces no effect, or causes the model to produce nonsense." Directional ablation on RWKV gives nonsense. |
| Hadetskyi, Pasquini, Sorokin, [2607.02714](https://arxiv.org/abs/2607.02714) (2026, under review) | 24 models, 0.6B to 1T, dense and MoE | Fraction of layers abliterated (5-30%, six levels) spread over 25-95% depth; alpha > 1 used for MoE | "Some models collapse monotonously, some chaotically, and some remain unmoved at any level." No sharp threshold described. Uniform spread beats norm-based layer selection by up to about 70 pp. Qwen3-8B among most susceptible (-38.9 pp), Qwen3-14B nearly resistant (-1.4 pp). Safety-training type (DPO vs GRPO) predicts susceptibility better than size. Refusal judged by string matching with `<think>` stripped. |
| Saadatinia et al., CircuitSteer, [2608.05732](https://arxiv.org/abs/2608.05732) (2026) | Llama-3.1-8B-Instruct, AdvBench n=100 | lambda at -1, -3 (multi-layer) vs -1, -3, -5 (single-layer CAA) | Multi-layer: 89% to 78% (-1) to 0% (-3). Single-layer CAA: 84% (-1), 92% (-3), 69% (-5). Only two or three points; cannot show a cliff, but shows single-layer strength alone does little. |
| Petrov, [2603.22061](https://arxiv.org/abs/2603.22061) (2026) | Qwen3.5-2B | Weight 0.3 to 1.2 per single layer | With an unmatched harmless baseline, single layers reach 0/10 refusals at w = 0.5-0.8. With topic-matched contrast, 9-10/10 at every weight and layer. n = 10 prompts per cell; thinking mode not stated. |
| Cao et al., [2606.08682](https://arxiv.org/abs/2606.08682) (2026) | Qwen3.5-27B | Strength of an emergent-misalignment vector (not refusal) | Threshold then collapse: near zero below a critical strength, rises, then falls at high strength. Analogy only; the text is internally inconsistent on layers. |

### Model and size differences

- Arditi et al. (NeurIPS 2024, [2406.11717](https://arxiv.org/abs/2406.11717)): one direction, ablated at every layer and position at full strength (no strength sweep), works on 13 models from 1.8B to 72B. Direction picked from a per-layer, per-position search, excluding the last 20% of layers.
- Hadetskyi et al.: within one family the response varies widely (Qwen3-8B vs Qwen3-14B above). Size is not the main driver.
- Ali et al.: CAA effect shrinks with Llama 2 size.
- Comparative analysis of abliteration tools ([2512.13655](https://arxiv.org/abs/2512.13655)): heretic with 50 TPE trials left 2 to 54 refusals out of 100 across 8 models (Qwen3-8B 25/100, KL 0.21). Single runs, no sensitivity analysis.

### Bottom line for Q1

- No published study sweeps ablation strength or spread finely enough to show, or rule out, a cliff. The sweeps that exist show graded, monotone, saturating change at a single layer, and model-dependent (monotone, erratic or flat) change across layer fractions.
- Supported: spreading the edit across depth beats concentrating it (Hadetskyi; CircuitSteer; grimjim, informal). Pushing past the direction can help (ACE alpha < 0; alpha > 1 for MoE in Hadetskyi). Direction choice (source layer, contrast set) can matter more than strength (Petrov; Marshall).
- Inference, not a finding: refusal is a per-prompt yes/no, so the population rate is the spread of per-prompt flip points. Tan et al. ([2407.12404](https://arxiv.org/abs/2407.12404); Llama-2-7B, Qwen-1.5-14B) find steerability "highly variable across different inputs", often bimodal, with many inputs moving the wrong way. That predicts graded population curves for a fixed direction unless flip points happen to cluster.

## Q2. Why an edit might need to be strong and broad

- Self-repair (general, not refusal-specific). McGrath et al. ([2307.15771](https://arxiv.org/abs/2307.15771)): in a 7B Chinchilla model, downstream layers restore about 70% of the logit loss after a single attention layer is ablated (middle layers, factual recall). Rushing & Nanda (ICML 2024, [2402.15390](https://arxiv.org/abs/2402.15390)): self-repair across families and sizes after head ablation; imperfect and noisy.
- Refusal redundancy across layers. hdhurve (LessWrong, July 2026, informal, Llama-3.1-8B, JailbreakBench n=100): ablating each layer's own direction in every layer except 12 matches ablating all 32 (98% vs 97% attack success); no single layer is necessary. grimjim (HF blog, informal, Gemma 3 12B): "Safety refusal persisted when early layers were not modified"; "Fewer layers allow more self-repair". No refusal counts given.
- Refusal is more than one direction. Wollschläger et al. (ICML 2025, [2502.17420](https://arxiv.org/abs/2502.17420); Qwen2.5 1.5-14B, Gemma 2 2B/9B, Llama-3 8B): refusal cones up to five dimensions; ablating the top 4+ independent directions beats the single difference-of-means direction on Gemma 2 2B; ablating difference-of-means directions from several layers does not help; larger Qwen2.5 models appear to support higher-dimensional cones (authors' interpretation). Piras et al. (AAAI 2026, [2511.08379](https://arxiv.org/abs/2511.08379)): multiple SOM directions beat one. Abu Shairah et al. ([2505.19056](https://arxiv.org/abs/2505.19056)): when refusal is spread across more token positions by training, abliteration lowers refusal by at most 10% vs 70-80% for baselines.
- Counter-evidence. Arditi et al. and Yamaguchi et al. (below) get large drops with one direction at strength 1 (but at all layers). Nanfack et al. ([2603.04355](https://arxiv.org/abs/2603.04355)) find an optimal-transport edit at 1-2 layers at 40-60% depth beats full-network direction ablation on Llama 2 and Qwen2.5 7-32B; shallow layers do almost nothing. So breadth is not always required, and the "where" matters.
- Strength of evidence: self-repair is well established in general; for refusal specifically the layer-redundancy evidence is a mix of one systematic preprint (Hadetskyi) and informal posts. Nothing measures self-repair of refusal after a partial weight edit directly.

## Q3. Thinking models

### Where the decision forms

- Yamaguchi, Etheridge, Arditi, "Where Do Reasoning Models Refuse?" ([2507.03167](https://arxiv.org/abs/2507.03167); R1-Distill-Llama-8B, R1-Distill-Qwen-7B, Qwen3-8B, GPT-OSS-20B): fixing the CoT sharply reduces outcome variance; in distilled models the opening CoT sentence can fully decide refusal vs compliance; in GPT-OSS-20B the outcome settles gradually through the CoT.
- Yin et al., "Refusal Falls off a Cliff" ([2510.06036](https://arxiv.org/abs/2510.06036); QwQ, Qwen3-Thinking, Phi-4-Reasoning, R1 distills and others): a linear probe shows refusal intent rising during thinking and, in poorly aligned models, dropping sharply at the end-of-thinking template; well-aligned models (Qwen3-Thinking, Phi) show little or no drop. Ablating about 3% of "refusal suppression" heads (deep layers) brings attack success below 10%. The "cliff" here is over token position, not edit strength.
- Chan, Yong, Bach ([2507.12428](https://arxiv.org/abs/2507.12428)): linear probes on early CoT activations predict final response safety.
- Zhao et al., CoT Hijacking ([2510.26418](https://arxiv.org/abs/2510.26418)): the refusal signal is low dimensional and dilutes over long benign reasoning.
- Yang et al., "Beyond a Single Direction" ([2605.26772](https://arxiv.org/abs/2605.26772), 2026; R1-Distill-Llama-8B, n=100): steering reverses refusal in 39% (end-of-instruction) / 43% (end-of-thinking) with the CoT held fixed, 70% with no CoT, 94% when the CoT is regenerated under steering; the compliant CoT alone keeps 48% with no steering. "Steering does not directly flip the final decision but instead biases the CoT generation process."

### Editing or steering at the thinking start

- Yamaguchi et al.: directions from the end-of-prompt template tokens (before the CoT) and from CoT tokens, each ablated at every layer and position. Qwen3-8B refusal (n=487): 60.0% to 8.5% (end-of-prompt) and 7.5% (CoT). The two directions have "generally quite low" cosine similarity. Ablation is "less reliable than on non-reasoning models", costing 5-30% relative on reasoning benchmarks. Exact template tokens and Qwen3 thinking setting are not stated.
- Cyberey & Evans: in R1 distills, a vector read at the position after `<think>` controls whether thinking is skipped (the main censorship route there); negative multiples lift censorship.
- García-Ferrero et al. (LREC 2026, [2512.16602](https://arxiv.org/abs/2512.16602)): prompt-end directions, steered at 2-16 layers on every decoding step, cut political refusals on Qwen3-Next-80B-A3B-Thinking (hybrid Gated DeltaNet, like Qwen3.5) and on a 4B thinking model (81.5% to 34.7%).
- Token-level think-start interventions work in the refusal-inducing direction: SAFEPATH (NeurIPS 2025, [2505.14667](https://arxiv.org/abs/2505.14667)) and Thinking Intervention ([2503.24370](https://arxiv.org/abs/2503.24370)).
- No study compares a think-start direction with an answer-start direction as a weight edit, and none covers Qwen3.5-9B.

### Tool-specific note (not literature)

heretic reads residuals at the last prompt position, after any `response_prefix` (source, `get_residuals`). Open issue [p-e-w/heretic#487](https://github.com/p-e-w/heretic/issues/487) (Oct 2026): for Qwen3-style templates the auto-detected prefix can leave the model in reasoning mode, so directions are taken "in the wrong state" and reasoning text is scored as non-refusal. Worth ruling out before reading the think-start failure as a property of the model. Current heretic source also edits `linear_attn.out_proj` (DeltaNet layers) alongside `o_proj` and `down_proj`.

## Verdict

- (a) Threshold or cliff in strength and spread: not established. No paper measures it at the needed resolution; the sweeps that exist are graded or model-dependent, and per-input variability argues for graded aggregate curves. The literature does support "broad beats narrow", "past the direction can help" and "direction choice matters as much as strength". The 50 to 11 jump fits these, but a cliff is unproven and would need a controlled sweep (hold direction and profile fixed, step strength; then step spread; repeat on a second model).
- (b) Think-start editing: the literature does not predict total failure. The closest result (Qwen3-8B) shows a pre-CoT direction working almost as well as a CoT direction when ablated everywhere, although the two directions differ. Reasoning models are harder because the CoT re-derives and reinforces refusal. A 97-98/100 result on Qwen3.5-9B is more likely setup-specific (exact position, layer, scoring of reasoning text, Qwen3.5 architecture) than a general law. Evidence for Qwen3.5 specifically is absent.
