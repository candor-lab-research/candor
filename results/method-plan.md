# Beyond one direction: the method plan (2026-11-02)

The user wants candor's method made more and more sophisticated, once the search predicts the full
measurement (STATUS, Now). This is the plan for that, not yet work: a change to what candor ships
needs evidence across this kind of model first, and goes to the user at a phase end (history).
Sources are in `evidence-2026-11-02/prior-work.md`.

## Where the method stands

- **Borrowed:** the refusal direction as a difference of means (Arditi et al. 2024), removed from
  the weights that write to the residual stream; heretic's edit shape (a strength kernel over
  layers, attention and MLP apart), its "full" row normalisation at rank 3, and Optuna TPE.
- **candor's own is how it scores and measures:** every trial in llama-server on the GGUF it ships,
  thinking on to a budget with a forced close, only the final answer judged, KL after the model's
  own thinking, budget hits capped, then held-out measurement under a rule fixed before the
  results. Only the whole of this is new (prior work).
- **The edit today:** one direction, read at the thinking start or the answer start, either one
  direction for every layer ("one layer") or one per layer, at rank 1 (rank 3 with "full").

## The candidates

In order of fit with candor and cost. [Published] is what the source claims, [Measured] candor's
own, [Hypothesis] ours.

1. **Projected and norm-preserving abliteration (MPOA).** [Published] The direction cleared of its
   part along the harmless mean, and each weight row kept at its norm. *Fit:* changes the direction
   and the rows' lengths only, so it stays a rank-1 or rank-3 adapter and scoring is unchanged;
   "full" already keeps rows' lengths roughly. *The cheapest to try.*
2. **Several directions, a refusal subspace.** [Published: concept cones; SOMA, self-reported, KL
   halved on gemma-3-12b at the same refusals] Ablate k directions in place of one. *Fit:* a rank-k
   adapter, linear in each direction's strength, so per-request strengths still work; the search
   gains k or a strength per direction. *Open:* how to find them (clustering, as SOMA; principal
   components of per-prompt differences; gradient search, as the paper).
3. **Edits aimed at the thinking.** [Published: the refusal cliff, a refusal intention dropped at
   the thinking's end, traced to attention heads; "Where Do Reasoning Models Refuse?", the thinking
   sets the refusal] [Measured: directions read at the thinking start mostly failed on Qwen3.5-9B
   and won on K2 and Spark-X (findings)] Directions read where the decision is made, such as the thinking's
   last tokens, or edits to the heads that carry it. *Fit:* directions are read at two places
   already, a third is a setting; heads are a new target. [Hypothesis] Judging this needs scoring
   as used, which no one else searches with: the likeliest place for something new. One form:
   ARA's kind of objective (5), fitted on module inputs and outputs read inside the thinking or
   at its end rather than at the prompt's last token. ARA's own lineage, as we read it (its paper
   isn't out): per-matrix edits fitted to keep some outputs and change others, as ROME and MEMIT
   edit facts (arXiv 2202.05262, 2210.07229), and harmful representations rerouted beside a
   retain loss, as circuit breakers do (arXiv 2406.04313), turned to removing refusals.
4. **Cleaned directions.** [Published: Surgical Refusal Ablation] The direction cleared of its
   overlap with what to protect, such as maths. *Fit:* rank 1, with prompts for each thing
   protected. Aimed at a known weak spot: twice the search's best lost on maths or its thinking's
   budget (K2's 207, Spark-X's 18).
5. **Arbitrary-Rank Ablation (ARA), heretic's new default.** [Published, read in heretic 2.0.0's
   code] No direction: for each attention and MLP output matrix in a range of layers, its inputs
   and outputs at the prompt's last token, for 400 harmless and 400 harmful prompts, set a small
   optimisation (L-BFGS) of a rank-50 change: harmless outputs kept, harmful ones pulled to their
   nearest harmless outputs and pushed from their old ones, rows kept at their norms. A trial sets
   6 values (the layer range, the three goals' weights, the neighbours counted) and optimises
   afresh. *Fit:* the change is a rank-50 adapter on candor's own targets, so it takes the path
   "full" trials take (an adapter built per trial, the server restarted), and `--outside` rebuilds
   a public ARA edit exactly. heretic is AGPL-3.0, so candor measures heretic's ARA saves or writes
   it afresh from the description. [Hypothesis] Fitted at one token, while reasoning models decide in
   the thinking (the refusal cliff), and with 50 dimensions a matrix to fit 400 prompts, it may
   read better on the search's prompts than held out. *Needed either way:* from 2026-11-01
   "against heretic" means against ARA. [Published] rockerBOO's Qwen3.5-9B ARA edit (heretic
   2.0.0.dev0, 2026-10-20) reports 2/100 at KL 0.0744 by heretic's own measure, not comparable with
   candor's. trohrbaugh's 27B ARA edit, made 2026-08-14 before ARA was finished, complied on 76.3%
   of DreamFast's harmful set, huihui's on 98.0% (`evidence-2026-10-21/public-27b.md`).
6. **Per-category directions.** [Published] They change how a model refuses more than whether:
   low priority.
7. **Conditional steering (CAST).** Applied at inference only, so it can't ship as one GGUF in
   llama.cpp: out.

## What comes first

1. **A yardstick that predicts the measurement** (STATUS, Now). Spark-X's held-out KL read about
   3 times its search KL, 5 times on the median prompt, so comparing methods by search scores may
   compare the gap. The first check is running (`evidence-2026-11-02/kl-gap.py`).
2. **A method bench on the Mac, free:** two small models measured in full, one thinking (such as
   Spark-X2.5-1.7B) and one not (rude-llama, measured on the Mac in phase 2), and a rule fixed
   before any result for "method B beats A": held-out refusals within the bar, held-out KL lower
   beyond noise, maths and budget hits within the bars, on both.
3. **heretic 2.x read** (STATUS: its journals aren't), so its current methods can be measured as
   candor measures its own.
4. **A seam for methods:** the direction and the edit chosen by a setting, today's the default, so
   old runs stay as they are (settings per model).

Then the candidates in the order 1, 2, 3, 4, with ARA measured as soon as heretic 2.x is read, each
on the bench first and the winner on one pod model. The bench and candidates 1, 2 and 4 cost
nothing on the Mac; candidate 3's heads, ARA and the pod model cost pod time, quoted first.
