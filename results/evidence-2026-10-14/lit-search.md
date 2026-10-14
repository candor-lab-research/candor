# Search strategies for abliteration edits: literature and practice check

Date: 2026-10-14. Web research only; no project files were opened. Every source below was opened and checked; code facts come from reading the source at the stated version. "Strength" ratings: **strong** = peer-reviewed or code fact; **moderate** = multi-seed practitioner experiment; **weak** = single run, anecdote or tool docs.

---

## Q1. How tools and papers choose edit settings

### heretic (github.com/p-e-w/heretic)

Code facts (strong):
- Sampler: `TPESampler(n_startup_trials=..., n_ei_candidates=128, multivariate=True, seed=...)`, two objectives, both minimised (`src/heretic/main.py`, master and v1.4.0).
- Trial budget: releases v1.0.1 to v1.4.0 (Nov 2025 to Jun 2026) default `n_trials = 200`, `n_startup_trials = 60` (random). Master after PR #446 (Oct 2026) defaults to `n_trials = 100`, `n_startup_trials = 30`, and the default modifier is now ARA, not abliteration.
- Abliteration ranges (master `modifiers/abliteration.py`): `direction_scope` in {global, per layer}; `direction_index` in [0.4L, 0.9L]; per component `max_weight` in [0.8, 1.5] for attention and [-0.25, 1.5] clamped at 0 for the MLP (PR #387, Jun 2026; it was [0.8, 1.5] for both up to v1.4.0); `max_weight_position` in [0.6L, 1.0L]; `min_weight` as a fraction of max in [0, 1]; `min_weight_distance` in [1, 0.6L]. The code comment says the ranges "are not set in stone and might have to be adjusted for future models".
- `row_normalization = "full"` with `full_normalization_lora_rank = 3` is the default since v1.3.0 (v1.2.0 had "none").
- Objective: v1.2.0 to v1.4.0 replaced KL by a scaled refusal score below `kl_divergence_target = 0.01` (PR #73). Master's plugin scorers return raw refusal rate and raw KL; I found no target logic there. Worth checking which variant the candor search mirrors.

What practitioners report (mostly weak; the multi-seed items are moderate):
- **#404** (Jul 2026): Qwen3.5-9B, default settings, best 70/100 after 200 trials, repeated 4 times; Qwen3.5-0.8B reached under 10. Maintainer: "Running further trials won't magically bring the refusals down" and "ARA and SOMA give better results."
- **PR #211** comment (Mar 2026): Qwen3.5-4B on main, "Best trials still refuse more than 50 of 100"; with ARA 4/100 at KL 0.14.
- **PR #52** comment (Jan 2026): Qwen3-VL-30B-A3B-Thinking "minimum refusal count >50 with 100 initial tries and 400 total tries"; single digits after orthogonalised direction + magnitude preservation.
- **PR #52** (the PR that added "full" row normalisation): the implementer found that with the 1.5 cap "The actual strength of the abliteration ... still leaves something to be desired"; an earlier version searched `max_weight` up to 15 on a log scale; the final recommendation was "Extend max weight to 2". The merged default kept 1.5. Same thread: "it really struggles to learn the relationship between `direction_index` and `max_weight_position`", and the best low-refusal results came at "a very high trial index".
- **PR #52 / PR #73**: the two-objective search was "wasting a lot of trials minimizing the KL divergence to tiny values like 0.002"; maintainer: "there is an inherent bias towards a KL divergence of 0, because that's what you get by simply doing nothing." The KL-target trick was the fix.
- **#88** (Dec 2025): an 800-trial run gave a worse Pareto front than a 200-trial run on rnj-1. The maintainer put it down to run-to-run variance of randomised TPE: "essentially nothing can be said about optimization behavior with any certainty, unless hundreds or even thousands of studies are conducted." The posted front had a gap: KL about 0.27 at 40-43 refusals, then about 0.03 at 48 refusals.
- **#202** (Jun 2026, moderate): 3 seeds x 150 trials (30 startup) on Qwen3-4B. The best refusal tip varied by seed: 4 / 14 / 6 and 5 / 11 / 9 for the two arms. Maintainer: "TPE doesn't converge monotonously and can get stuck if the confidence becomes too high".
- **#333** (May-Jun 2026): a two-phase search (layer range first) was proposed and rejected: "All parameters are interrelated". A commenter claimed two-phase search "guarantees" a local minimum (opinion, no data). A 3-seed x 150-trial test of narrowing `direction_index` before the search (moderate) helped on Qwen3-4B (floor 3 vs 6), hurt slightly on SmolLM3-3B (2 vs 3), and was worse in the low-refusal region of granite-4.0-micro. So narrowing the space up front helped on one model and hurt on two.
- **PR #312**: the maintainer rejected skipping refusal counting on high-KL trials, because those trials "tell the optimizer where to look".
- **#490** (Oct 2026): trial scores differ from the exported model (example: 3 to 5 refusals, KL 0.15 to 0.17), because live-LoRA bf16 rounding differs from merged weights. The selection signal is noisy and biased.
- **#487** (Oct 2026): a bad thinking-model prefix gave 0 refusals while the model was destroyed (KL 16-18). The keyword scorer counts broken output as compliance.

### abliterix (github.com/wuwangzhang1216/abliterix), a heretic derivative
Same kernel and the same multivariate TPE, with configurable ranges (code fact). The defaults match heretic (200/60), but the shipped Qwen configs use far fewer trials:
- Qwen3.5-9B: 50 trials, 15 warmup, default range [0.8, 1.5], `weight_normalization = "none"`, LLM judge. README: 2/200 refusals at KL 0.0105.
- Qwen3.5-27B: 25 trials, `strength_range = [1.0, 5.0]`. Qwen3.6-27B: 30 trials with attention [1, 6]. Its model card: the pass-1 winner had max_weight about 5.17 at 16/100; widening to [1, 8] with another seed gave 17/100; a second pass reached 10/100; about 4.5 min per trial on an A100.
- The settings docs describe narrow sweet spots: on gpt-oss-20b "all winning trials had min_frac < 0.34", and capping the range "raises the warmup hit rate dramatically" (about "10x more likely to land in the productive region"). That is the opposite profile shape to the Qwen3.5-9B corner (flat profile).
- Qwen3.5-2B A/B report: an 8-point 1-D strength grid {0.4, ..., 2.0}, selected on dev with a KL guardrail, then a locked test.
- Weak: different judge and normalisation, so the numbers do not compare with heretic's. These are configs and single runs.

### Others
- **Arditi et al. 2024** (NeurIPS 2024, arXiv 2406.11717), strong: an exhaustive sweep over all |I| x L difference-in-means candidates on 32 harmful + 32 harmless validation prompts. They pick the minimum bypass score subject to induce > 0, KL < 0.1 and layer < 0.8L. The projection is applied at full strength with no scale factor.
- **FailSpy abliterator**: an exhaustive direction sweep (`find_best_refusal_dir`), with the layer chosen by hand. **mlabonne blog**: 20 per-layer candidates on 4 prompts, picked by human inspection. Neither has a strength knob.
- **grimjim, norm-preserving biprojected** (HF blog): manual. Layers chosen by a heuristic "quality = snr x (1 - cos_sim)" read off charts, scale 1.0, layers 11-41 on Gemma-3-12B, and "Future work might formalize optimal layer selection". No search, no trial count.
- **SOM directions** (Piras et al., AAAI 2026, arXiv 2511.08379): TPE picks k of 16 directions, with 128 trials (k <= 3) or 512 (k > 3), the first 25% random, and HarmBench ASR as the objective; about 8 h per search.
- **Young 2025** (arXiv 2512.13655): ran heretic at 50 trials per model, below its 200 default. The paper does not report the startup setting, so if the default of 60 was kept, all 50 trials were random. Results varied widely (Qwen2.5-7B: 42/100 at KL 1.65), with one run per model and about 45 min per model.
- **OBLITERATUS** (HF space): optional TPE (0-200 trials), 2-20 prompts per trial ("faster but noisier"), plus a manual "Strength Sweep" tab.

**Q1 bottom line.** No tool or paper justifies its trial budget; budgets run from 25 to 800. The only multi-seed evidence shows large seed-to-seed variance at 150 trials. Standard abliteration on the Qwen3.5 family has repeatedly been reported stuck at 50 or more refusals in heretic. Winners at the edge of the range, and widening the range, recur in practice.

---

## Q2. Optimiser evidence

Established:
- **Bergstra & Bengio 2012** (JMLR 13:281-305): random search beats grid search because only a few hyperparameters matter, and which ones matter varies by dataset, so grid search is "a poor choice for configuring algorithms for new data sets". This is a direct caution for a fixed strength x spread grid that must work across models. **Bergstra et al. 2011** (NeurIPS 24) introduced TPE.
- **Watanabe 2023** (arXiv 2304.11127 v4): TPE is "an exploitative BO method", "inclined to search locally". Without the prior, "No exploration happens after one of the modals are found", and "Restarting is especially important for non-global methods such as TPE". In its benchmarks HEBO "consistently outperforms our TPE except for the 30D problems" and TuRBO "exhibits very strong performance on the benchmark functions". The budget was 200 evaluations with 10 seeds, single-objective only.
- **BBO Challenge 2020** (Turner et al., arXiv 2104.10201): 128 evaluations (16 x 8) on real ML tuning tasks. Efficiency relative to random search: hyperopt TPE 3.7x, GpyOpt 7.6x, Ax 11.8x, scikit-optimize 16.5x, pySOT 18.3x, TuRBO 21.5x, pycma 1.7x. The authors conclude "using a GP model leads to better performance than using a TPE" and that local trust-region methods do best out of the box. 6 of the top 10 teams used TuRBO; two of the top 10 switched to differential evolution in the final batches "to improve convergence close to the best point"; the Optuna developers' own entry was built on TuRBO. This is single-objective with smooth-ish landscapes.
- **HEBO** (Cowen-Rivers et al., JAIR, arXiv 2012.03826): "heteroscedasticity and non-stationarity pose significant challenges"; it uses input and output warping, and won BBO 2020. **TuRBO** (Eriksson et al., NeurIPS 2019, arXiv 1910.01739): local trust regions fix the over-exploration of global models.
- **Optuna docs** (samplers index): TPE is recommended for 100-1000 trials. GPSampler works up to about 500 and supports multi-objective. CmaEsSampler needs 1000-10000 and has no multi-objective support. **Optuna AutoSampler** (OptunaHub source): for multi-objective, fewer than 250 completed trials and an all-numeric space it uses GPSampler, and TPE only when categorical or conditional parameters are present. heretic's `direction_scope` categorical sends it to TPE, so a purely numeric search would get GP.
- **MOTPE** (Ozaki et al., JAIR 73, 2022) is the multi-objective TPE in Optuna. The practitioner evidence above shows it drifts towards the "do nothing", low-KL end in this problem.
- **Initial designs** (Bossek, Doerr, Kerschke, GECCO 2020, arXiv 2003.13826): on BBOB, "small initial budgets using Halton sampling seem preferable", but the landscape is "rather unstructured", so there is no universal rule. This argues against assuming a large front-loaded sweep pays off in general.
- **Practice guidance**: the Google Research tuning playbook uses quasi-random search for exploration and BO for exploitation, and says "A search space is suspicious if the best point sampled from it is close to its boundary." The LIBSVM guide (Hsu, Chang, Lin) recommends a coarse grid, then a finer grid on the best region, justified because "there are only two parameters".
- **Multi-fidelity**: successive halving (arXiv 1502.07943), Hyperband (JMLR 2018, arXiv 1603.06560), and BOHB (ICML 2018, arXiv 1807.01774, also the source of Optuna's multivariate TPE). Here the fidelity could be the number of prompts. This is untested for abliteration.

Not established (evidence absent):
- No controlled study shows TPE specifically failing on narrow optima, cliffs, or optima at the range edge. The only statement I found is in tool docs (IBM ado; weak): "Issues may arise if the optimal region is not sampled in the initial points and this region is disjoint from other regions with 'good' performance."
- No head-to-head of TPE against grid, sweep, GP or CMA-ES on abliteration exists.

---

## Q3. Cheaper ways to locate a threshold

- **Look Before You Steer** (arXiv 2609.22782): α* is the minimum coefficient that shifts the refusal rate by at least 0.10, found on a grid {0, 0.25, 0.5, 1, 2, 3, 5, 10, 20} with 100 prompts per point. Up to 46% of features never reach the threshold and are right-censored.
- **Steer Like the LLM** (Heyman & Vandeputte, arXiv 2605.03907): uses binary search over a global steering coefficient to hit a target coherence, then interpolates. I could not read the appendix with the algorithm's details.
- **Expert-Aware Refusal Steering** (Marbut et al., arXiv 2606.04160; MoE models): the effect is "extremely sensitive to tuning the coefficient", ASR is non-monotonic in the coefficient, and outputs are "hovering right on the cusp". This is a direct risk for bisection.
- In abliteration: abliterix's 1-D strength grid (above), and the OBLITERATUS sweep tab. Arditi, FailSpy and mlabonne sweep the direction (layer), not the strength.
- **Not found:** any paper or tool that bisects a total-strength parameter to locate refusal collapse and then tunes the rest. Evidence absent.
- **Caveat:** keyword refusal scorers count broken output as non-refusal (heretic #487), so a bisection on refusals alone can find a false collapse. It has to be constrained by KL.

---

## Parameterisation note (arithmetic from heretic's code, not an experiment)

In heretic, layer i is edited only if |i - position| <= distance, with position in [0.6L, L] and distance in [1, 0.6L]. Reaching layer 0 needs position <= distance <= 0.6L, which is the single point position = distance = 0.6L. "Strong and nearly flat over all layers" therefore needs four coordinates at their edges per component: position at its minimum, distance, min_weight fraction and strength at their maximums.

Under uniform startup sampling, landing in the outer 10% of each of 3 such dimensions has probability 10^-3, so 60 random trials hit it about 6% of the time. This is illustrative only.

The corner may therefore be an artefact of the coordinates. In coordinates such as total edit mass, spread and centre, the same edit is an ordinary interior point. abliterix's gpt-oss winners had sharp peaks, so the winning shape changes by model, and fixing the profile is unsafe. Check whether candor's profile parameterisation matches heretic's before relying on this.

---

## Verdict

Established:
- With heretic's defaults, TPE varies widely from run to run at 100-200 trials.
- It drifts towards "do nothing" edits in two-objective mode.
- It struggles with parameter interactions.
- In general benchmarks it is beaten by GP and trust-region BO at about 128 evaluations.
- A best point at the range edge is a standard warning sign.
- heretic's 1.5 ceiling under "full" was questioned by the person who implemented "full".

Not established:
- That a coarse sweep of strength x spread followed by refinement beats TPE. There is no head-to-head in abliteration.
- The general literature argues against grids beyond about 2 important dimensions and against assuming which dimensions matter.
- Narrowing the space up front in heretic helped one model and hurt two.
- Part of the Qwen3.5 plateau may be the method or range, not the search (#404, PR #211).

Suggestion: the evidence supports front-loading structured, space-filling exploration over a widened strength range, preferably in a reparameterised space, followed by model-based or local refinement. It does not support replacing the multi-dimensional search with a 2-D grid.

## A fair offline test

0. **Zero-cost retro check** (numbers only): from the existing trial log, compute each trial's total edit mass per component and a spread measure. Test whether refusals are a monotone function of total mass (isotonic fit, rank correlation), and count how many trials ever came near the corner. This is confounded by TPE's non-uniform sampling and covers one model.
1. **One-time tabular benchmark**, HPOBench style (Eggensperger et al., NeurIPS D&B 2021, arXiv 2109.06716): evaluate a large scrambled-Sobol pool over the widened, full-dimensional space on a cheap same-family model plus at least 2 other families. Use the exact objective and record refusals and KL. Pilot-time the trial cost first.
2. **Simulate strategies on the pool**: proposals snap to the nearest pool point; avoid smoothing surrogates, because they erase cliffs. Use at least 20 seeds per arm at budgets 50/100/150/200. Arms:
   - heretic TPE in the original range;
   - TPE in the widened range;
   - GPSampler;
   - Sobol/random;
   - sweep-then-refine, counting the sweep's own trials.

   Running the TPE-widened and GP arms separates "the sweep helps" from "the wider range or a better sampler helps".
3. **Pre-registered metrics**:
   - trials to first reach the target (for example refusals <= 15 at KL <= a cap), shown as a survival curve;
   - success probability at budget 100;
   - Pareto hypervolume against a fixed reference point.

   Success means winning in the median and in success rate on most models, including at least one non-Qwen model.
4. **Live confirmation**: run the top 2 arms on the 9B with at least 3 seeds each, re-scoring the picks on held-out prompts and with merged weights (#490). Then confirm, without retuning, on the 27B.
