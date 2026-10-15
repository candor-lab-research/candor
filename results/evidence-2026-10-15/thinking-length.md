# Why refusal edits lengthen thinking, and what to do about it

This is research only, done 2026-10-15. Each claim carries one of three flags:
- **Measured elsewhere**: others' numbers.
- **Published explanation**: what authors or maintainers say, untested.
- **Hypothesis**: mine, untested.

"My computation" means a rank correlation I worked out from a published table. candor's numbers come from the brief and `phase-3-qwen3.5-9b.md`.

**In short.** Two different things are happening, and they need different fixes:
- **Longer thinking on harmful prompts.** This comes from the change in behaviour itself. Nearly every edit measured shows it, including a conditional edit that leaves other prompts untouched. Evidence: strong.
- **Longer thinking on maths.** This varies by edit. Some edits lengthen maths thinking (heretic's twice), others shorten it, and it looks like collateral damage to the machinery that stops thinking. Evidence: moderate.
- No paper measures thinking length against abliteration strength or layer, or on Qwen3.5.

## Why

1. **A model that complies has more to write, and refusing was quick.** [Measured elsewhere]
   - On Qwen3.8-27B (DreamFast, 400 HarmBench prompts, greedy, 15,360 budget) the base model averages 1,207 tokens because it refuses fast. Twelve edits average 2,331-8,715.
   - On the same prompts, the base leaves 1 thinking block unclosed. The edits leave 23-179.
   - The conditional edit (one gated neuron per MLP, lowest KL at 0.039, GSM8K unchanged) still leaves 133 unclosed. So no general damage is needed for this.
   - On Qwen3.6-27B (same architecture as candor's model), five edits raise harmful-prompt reasoning from 3,067 to 8,237-10,194 characters.
   - candor's own numbers fit: harmful thinking triples (996 to 2,980) while harmless barely moves (1,137 to 1,175).
   - One partial exception: on Qwen3.5-27B with a 2,048 cap, heretic's edit hit the cap 0 times, like the base, against 30-34 for two other edits.
2. **The model still sees the harm but can no longer refuse, so it keeps deliberating.** [Published explanation]
   - Zhao et al. find harmfulness and refusal are separate directions. Jailbreaks and adversarial fine-tuning lower refusal but leave the model's internal judgement of harm intact. These were non-reasoning models, measured.
   - Yang et al. find the chain of thought re-derives refusal: steering flips 39% of refusals with the CoT held fixed and 70% with no CoT.
   - Pipis et al. find loops arise when the step that makes progress is hard and an easy cyclic step is available.
   - The account itself is put forward by DreamFast ("adversarial-harm deliberation does not converge") and by a heretic user (#288, later withdrawn). Nobody has measured it inside an edited reasoning model.
   - candor's 24 edits fit it. Edits that refuse nearly everything think 700-1,300 tokens. Among the 12 answer-start edits refusing 1-50 of 100, thinking rises with the refusals left (rho 0.89, my computation): half-working edits think longest. DreamFast's edits point the other way (unclosed blocks against explicit refusals rho -0.5, 11 edits, my computation), so the shape is unsettled.
3. **The edit damages how the model stops thinking. This is the likely cause on maths, which is not harmful.** [Measured elsewhere for the parts; Hypothesis that refusal edits do it]
   - The Halt Vector paper (R1-Distill-Qwen-7B) found a direction whose negative steering stops the model ending its thinking. A weight edit fitted to one scalar let the off-axis dimensions drift, still within the model's own range. The model then thought 43% longer, looped, and stopped emitting `</think>`. A control direction of the same size changed nothing.
   - On Qwen3-1.7B, an informal LessWrong post finds one MLP at layer 27 of 28 gives about 40% of the `</think>` logit. Ablating the top 4 MLPs leaves the model unable to stop.
   - Yamaguchi et al.'s ablation raised the share of thinking that never finished: GPT-OSS-20B 2.4% to 39.9% (end-of-prompt direction), Qwen3-8B 0.5% to 2.7% (CoT direction) but 0.2% for the end-of-prompt direction.
   - Generic noise does not do this: quantised reasoning models "do not exhibit increased output lengths" (Liu et al.).
   - The effect depends on the edit. On Qwen3.6-27B, most edits shortened GSM8K thinking (base 68.2% over budget, huihui 23.0%) but heretic's lengthened it (74.5%). The base rate of 68.2% at 7,168 tokens suggests a protocol quirk.
   - On Gemma-4-12B (12 variants), length tracks KL: harmful-prompt tokens rho 0.91, harmful unclosed blocks 0.76, GSM8K unclosed blocks 0.52 (my computation).
4. **Abliteration makes even non-reasoning models wordier.** [Measured elsewhere] On a task with no refusals, abliterated Gemma-4-26B-A4B and Qwen3-30B-A3B-Instruct gave longer justifications, replicated in both families ("Not a Scalpel").

## Fixes

- **Strength.** [Measured: candor only]
  - At 1.4x, all 3 per-layer pairs thought less (for example 1,983 to 1,729) and refused the same or fewer. One-layer pairs: 1 of 3 shorter.
  - In the brief, one edit at 1.4x thought less (1,237) but refused far more (20 against 1).
  - Nobody has published a curve of strength against thinking length.
- **Scope and layers.** [Measured: candor; Published explanation for late layers]
  - In candor's 24 edits, edits that touch every layer (per layer) think 1,600-2,000 tokens and never hit the budget. One-layer edits that comply think 1,800-2,900 and hit it up to 10 of 100 times.
  - The termination-circuit and Halt Vector results put the stopping machinery in late layers. Steering early layers there cost the most accuracy.
- **Edit type.** [Measured elsewhere, confounded]
  - ARA left the fewest unclosed blocks of the directional and ARA edits on Qwen3.8 (61 of 400; only a merge of fine-tunes had fewer, 23), but it also refused most (122). Compare at equal compliance before crediting ARA.
  - The conditional edit still left 133 unclosed.
  - OBLITERATUS advertises "CoT-aware" ablation (orthogonalised against reasoning directions), without numbers. Its Qwen3.8 model left the most unclosed blocks (179), and its Gemma-12B model lost 13.9pp of GSM8K headline, mostly to loops. Which preset it used is unverified.
- **Cleaner directions.** [Measured elsewhere, not on thinking length] SRA projects capability and style directions out of the refusal vector. On Qwen3-VL-4B it took first-token KL from 2.088 to 0.044 at 0% refusals. Thinking was not measured.
- **A second edit just for length.** [Measured elsewhere on base models, never after abliteration]
  - The Halt Vector, fitted as a LoRA from 24 problems: about 24% less thinking at held accuracy on 5 benchmarks, and it fixes non-termination. It works only when the whole activation is rebuilt with off-axis dimensions pinned.
  - ThinkEdit edits about 4% of heads (0.2% of parameters) along a length direction.
  - SEAL: 11.8-50.4% fewer tokens, with up to 11% better accuracy.
  - Reflection Steering: 16.9% fewer tokens.
- **Sampling and budget.** [Measured elsewhere]
  - Higher temperature cuts loops but leaves output long; Pipis et al. call it a stopgap.
  - DreamFast's loop rates are greedy, which they call worst case. candor, sampled with a presence penalty, has Trial 161 hitting the budget on only 2 of 288 harmful prompts.
  - At a 15,360 budget, no Qwen3.8 edit lost GSM8K. At 8,096, one edit hit the budget on 48% of harmful prompts.
  - The Qwen3.5 model card advises presence_penalty 0-2 against endless repetition. llama-server applies it over only the last 64 tokens (`history.md`).
- **Selection criteria.** [Published explanation] heretic users report:
  - Trials with first-token KL under 0.05 that degrade over longer outputs (#209, not merged).
  - Loops on merged models that KL misses, and a proposed KL on reasoning prompts as a probe (#253).
  - The maintainer replied that unedited Qwen3.5 also loops "on many occasions".

## Is it harmful

- **Maths: yes when the budget binds, little otherwise.** [Measured elsewhere and by candor]
  - On Gemma-4-12B, GSM8K unclosed blocks rose from 7.6% (base) to 19.6-38.2% for 4 edits, costing 10-29pp of headline against base. Accuracy on answered problems stayed within 3pp of base, so the loss is loops, not lost skill.
  - On Qwen3.8 at 15,360 there was no loss. On Qwen3.6, heretic's edit lost 2.4pp even on answered problems.
  - Thinking that never closes is mostly wrong: 6.6% correct against 90.3% for thinking that closes (R1-Distill-Qwen-7B, AIME). The overlong tail costs even unedited models accuracy (Halt Vector).
  - Yamaguchi et al.: AIME 2025, GPQA and BBEH fell 5-30% relative after ablation.
  - candor: Trial 161 lost 21 of 1,319 (p = 0.04), with 113 more budget hits.
- **Loops: common under greedy decoding, rare with candor's sampling.** [Measured]
  - Greedy: 6-45% of harmful-prompt thinking never closed on Qwen3.8 edits (23-179 of 400; base 1).
  - candor: 2 of 288 harmful (0.7%), but 297 of 1,319 maths hits (22.5%; base 14.0%).
- **Cost: 2-7 times the tokens on harmful prompts.** [Measured]
  - Mean tokens: 1.9-7.2x on Qwen3.8.
  - Reasoning characters: 2.7-3.3x on Qwen3.6.
  - Median thinking: 3.0x on candor's model, plus 69% on maths.
  - Wall time follows the longest thinking (candor, step 4).

## What candor could test

All of these are [Hypothesis], cheapest first. None needs reading a text: scripts print counts only.
1. **Thinking length by verdict.** For Trial 161's held-out harmful set, compare thinking length for refused and complied prompts from the existing labels and token counts. If prompts that refuse after thinking are the longest, deliberation (Why 2) is supported. No new generation needed.
2. **Late layers.** Across the 24 edits and the search's trials, relate maths and harmless thinking to each edit's weight on the last ~15% of layers. Then score one edit whose window ends before them. Tests Why 3.
3. **Direction overlap.** On Qwen3.5-9B, build a "halt" direction (difference of means, activations just before `</think>` against mid-thinking, on maths). Take its cosine with candor's refusal directions layer by layer. If they overlap, project it out of the refusal direction before editing.
4. **Report maths three ways on the finalists:** accuracy, budget hits, and accuracy on answered problems. This separates loops from lost skill, and puts thinking length beside KL as the tie-breaker (the idea already in findings).
5. **Penalty window.** Re-run maths for the unedited model and one edit with a whole-output presence penalty, to see how much of the budget-hit rise is llama-server's 64-token window. This is a measurement, not something to ship.
6. **If length still decides:** add a separate length edit after the refusal edit (Halt Vector or ThinkEdit style). This is the most work, and the evidence comes from base models only.

## Nothing known

- How thinking length responds to edit strength, layer or rank in any reasoning model.
- Any study of thinking length on Qwen3.5 or 3.6 apart from DreamFast's benchmarks.
- Whether refusal directions overlap with length or stopping directions.
- Why heretic lengthens maths thinking where huihui shortens it.

## Sources

- Refusal in reasoning models: Yamaguchi, Etheridge, Arditi, Where Do Reasoning Models Refuse?, arXiv 2507.03167 (App. I.1, I.3); Zhao et al., LLMs Encode Harmfulness and Refusal Separately, 2507.11878; Yang et al., Beyond a Single Direction, 2605.26772; Zhao et al., Chain-of-Thought Hijacking, 2510.26418 (refusal weakens as traces lengthen)
- Loops and stopping: Pipis et al., Wait, Wait, Wait... Why Do Reasoning Models Loop?, 2512.12895 (ICML 2026); Jayabahu, Adeleke, The Halt Vector, 2608.28859; Dutta, The Termination Circuit, LessWrong 2026-07-11 (informal); Oladri et al., Token Budget Saturation, 2607.21433
- Length control: Sun, Yan, Weng, ThinkEdit, 2503.22048; Chen et al., SEAL, 2504.07986; Hu et al., Reflection Steering, 2608.25542
- Edits and side effects: Liu et al., Quantization Hurts Reasoning?, 2504.04823; Fafuła, Abliteration Is Not a Scalpel, 2607.17427; Cristofano, Surgical Refusal Ablation, 2601.08489; Young, Comparative Analysis of LLM Abliteration Methods, 2512.13655 (non-thinking models; GSM8K most sensitive, up to -18.8pp)
- DreamFast benchmarks on Hugging Face: `DreamFast/Qwen-3.8-27b-abliterlitics`, `Gemma4-12b-it-abliterlitics`, `Qwen3.6-27B-...-Benchmark`, `Qwen3.5-27B-...-Benchmark`
- heretic issues #209, #253, #288; OBLITERATUS README; Qwen/Qwen3.5-9B model card (sampling)

## The no-prompts rule

There were four slips:
- An early filter on heretic #253 printed about 4 short lines that were fragments of a quoted prompt, plus one sentence describing what a model's thinking contained. I then tightened the filter.
- Greps of two DreamFast cards printed one sentence and one clause describing what models produced or debated.
- One search summary quoted a one-word loop phrase.

None of this is above, and every raw download has been deleted.
