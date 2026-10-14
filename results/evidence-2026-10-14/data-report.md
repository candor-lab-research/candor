# Does refusal fall off a cliff? Evidence from eight searches on three models

Numbers only: every figure below comes from Optuna journals (trial numbers, parameters, refusal counts, KL), `inputs.json` and config keys. No prompt, answer or judge output was opened.

Rerun (for example once the running Qwen "full" search reaches 150 trials; copy its `study.jsonl` and `inputs.json` into one folder first):

    cd ~/Desktop/candor && uv run python <scratchpad>/evidence/analyse.py <folder>/study.jsonl [...]

The script copies each journal to a temporary folder before Optuna opens it and drops a half-written last line, so it is safe on a live journal.

## Data and how it was read

| study | model | tool, refusal counter | row norm | strength range | trials | startup | stopped early |
|---|---|---|---|---|---|---|---|
| heretic-rude | rude-llama (Llama-3.2-3B fine-tune, 28 layers) | heretic, keywords | pre | 0.8-1.5 | 200 | 60 random | 0 |
| candor-rude-search-v1 | rude-llama | candor v1 (multi-objective, like heretic), judge | pre (1) | 0.8-1.5 | 200 | 60 random | 0 |
| candor-rude-search | rude-llama | candor now (least KL with refusals <= 2), judge | pre (1) | 0.8-1.5 | 600 | 40 Sobol | 338 |
| heretic-k2-run1 | K2-Horizon-7B (36 layers) | heretic, keywords | full | 0.8-1.5 | 200 | 60 random | 0 |
| heretic-k2-run2 | K2-Horizon-7B | heretic, keywords | full | **0.8-5.0** | 200 | 60 random | 0 |
| heretic-qwen | Qwen3.5-9B (32 layers) | heretic, keywords | full | 0.8-1.5 | 199 (1 failed) | 60 random | 0 |
| candor-qwen-pre | Qwen3.5-9B | candor now, judge, thinking off | pre | 0.8-1.5 | 69 (1 failed) | 40 Sobol | 0 |
| candor-qwen-full | Qwen3.5-9B | candor now, judge, thinking off | full | 0.8-1.5 | 70 (still running) | 40 Sobol | 0 |

(1) From `runs/rude/candor.toml`; the search's `inputs.json` holds only a digest of the adapters.

Checked against the code, not taken on trust:
- heretic's two objectives are `[KL, refusals / base]`, with the first replaced by `refusal share x 0.01` when KL is under 0.01 (`heretic/evaluator.py:get_score`); raw `refusals` and `kl_divergence` are user attrs. Its refusals are keyword matches; candor's are judge labels. Its KL is first-token KL; candor's is read over 20 tokens. **KL is comparable only within a study, refusals only roughly across tools.**
- candor's search now is **single-objective**: least KL subject to judged refusals <= 2 (`search.py`, `violation`, `constraints_func`), not multi-objective. candor v1 (`search-v1`) was multi-objective like heretic.
- The two K2 journals are distinct studies: different seeds, and run2 (the top-level file) drew strength from 0.8 to 5.0, run1 (`run1-ceiling-1.5`) from 0.8 to 1.5.
- heretic's settings were mapped to candor's as in `measure.py` / `search.py`: peak layer and reach divided by layers - 1, edge strength = heretic's sampled `min_weight` (already a fraction of `max_weight`), direction layer = `direction_index / (layers - 1)` for "global" trials. All 100 harmful test prompts in every study; base refusals 96 (heretic-rude), 99 (candor), 100 (K2, heretic-qwen).
- K2 and Qwen heretic runs read directions after a response prefix (answer start); heretic-rude did not.
- The two candor Qwen searches have the same seed and space, so **their 40 startup points are the same edits**, scored under two row norms. They are not independent samples of the space.

Stopped trials: candor stops a trial once it refuses more than the 25th best of its kind. Its count is a lower bound. Stopped trials never count towards a best; elsewhere they stand in at their count scaled to 100 prompts (as candor's own `violation` does), marked "s". Only candor-rude-search has any (338 of 600).

Distance: root-mean-square difference of the settings, each in units of its own range (8 kernel settings, plus the direction layer for one-layer edits), only between trials of the same kind (one layer or per layer, and position). 0.1 means the settings differ by 10% of their ranges on average.

Dose: an edit's mean strength over all layers (candor's `Kernel.at` averaged), attn plus mlp. It folds strength, edge strength and reach into one number: "how much edit".

"Startup" trials are the ones drawn before TPE aims: random in heretic and candor v1, Sobol-spread in candor now. Only they sample the space evenly; the rest follow the sampler, so their distribution says where TPE looked, not what the space holds.

## Q1-Q2. How fast each search found its best

| study | best (KL) @25 | @50 | @100 | @150 | end | best first reached at | within 5 at | within 10 at |
|---|---|---|---|---|---|---|---|---|
| heretic-rude | 5 | 2 | 2 | 2 | 2 (0.026) | 46 (random) | 13 | 13 |
| candor-rude-search-v1 | 7 | 7 | 3 | 2 | 0 (0.102) | 165 | 77 | 16 |
| candor-rude-search | 10 | 1 | 0 | 0 | 0 (0.078) | 81 of 600 | 38 | 6 |
| heretic-k2-run1 | 70 | 69 | 63 | 54 | 54 (0.057) | 111 | 111 | 91 |
| heretic-k2-run2 | 15 | 8 | 7 | 7 | 7 (0.084) | 68 | 50 (random) | 23 |
| heretic-qwen | 56 | 56 | 49 | 46 | 11 (0.068) | 161 | 159 | 159 |
| candor-qwen-pre | 72 | 70 | | | 50 (0.059) | 62 of 69 | 60 | 60 |
| candor-qwen-full | 74 | 70 | | | 32 (0.069) | 64 of 70 | 64 | 58 |

The searches do not all aim at fewest refusals: heretic trades refusals against KL, and candor now looks for the least KL within 2 refusals (on rude it first got within the cap at trial 50 and found its least-KL edit at 305; no Qwen trial is within the cap yet). Both candor Qwen searches were still improving at their last trials, so their "trials to best" is open.

heretic-qwen's drop was not a single lucky trial: from trial 155 to 176, 12 trials refused 33 or fewer (11 to 33), after 150 trials whose best was 46. TPE moved into a new region and stayed there for about 20 trials.

## Q3. Cliff or slope

| study | startup trials refusing under half of base | best -> median of its 5 nearest (distance) | nearest trial refusing halfway back to base | median refused by distance from best: <0.1 / <0.15 / <0.2 / <0.3 / >=0.3 |
|---|---|---|---|---|
| heretic-rude | 25/60 (42%) | 2 -> 5 (0.10) | 0.20 (86) | 5 / 5 / 20 / 16 / 69 |
| candor-rude-search-v1 | 17/60 (28%) | 0 -> 2 (0.14) | 0.32 (55) | - / 2 / 4 / 14 / 34 |
| candor-rude-search | 11/40 (28%) | 0 -> 9 (0.08) | 0.19 (62s) | 6 / 6 / 13 / 12 / 90 |
| heretic-k2-run1 (to 1.5) | 0/60 | 54 -> 65 (0.11) | 0.19 (78) | 66 / 63 / 71 / 74 / 86 |
| heretic-k2-run2 (to 5.0) | 17/60 (28%) | 7 -> 13 (0.07) | 0.17 (61) | 12 / 12 / 20 / 34 / 66 |
| heretic-qwen | 0/60 | 11 -> 19 (0.10) | 0.20 (99) | 22 / 28 / 44 / 59 / 98 |
| candor-qwen-pre | 0/40 | 50 -> 55 (0.10) | 0.30 (78) | 59 / 58 / 55 / 69 / 78 |
| candor-qwen-full | 0/40 | 32 -> 39 (0.10) | 0.25 (73) | 39 / 40 / 51 / 73 / 78 |

What the data show:
- **Evenly spread trials.** On rude-llama and on K2 with strength up to 5.0, 28-42% of random or Sobol trials refused under half of base. On Qwen (all three searches) and on K2 with strength up to 1.5, none did (0 of 40-60). The distributions of startup trials pile up at 90-100% of base on Qwen and K2-run1; on rude and K2-run2 they spread across the whole range. (The bimodality coefficient is above 0.555 everywhere, but it is driven by that pile-up, so it says nothing about a gap.)
- **Around the best.** On every study the median refusals rise steadily with distance from the best. Sharp jumps at short range are rare wherever they can be counted: of the pairs closer than 0.1 that involve a trial within 5 of the best, 2 of 579 on candor-rude refused halfway back to base (one a 0-refusal edit beside a stopped trial at about 75, mostly a change of mlp peak layer by 0.18 of its range), and 0 of 27, 8 and 11 on heretic-rude, candor-rude-v1 and K2-run2. Qwen has only 3-4 such pairs per search, none a jump: too few to tell. The radius at which a trial refuses halfway back to base is similar everywhere: 0.17-0.20 on heretic-rude, candor-rude, K2-run2 and heretic-qwen; 0.25-0.32 on candor-rude-v1 and the candor Qwen searches. Qwen's bottom is steeper, though: within 0.15 its neighbours have climbed 12-19% of the way back to base (heretic-qwen 11 -> 28; candor-qwen-full 32 -> 40), against 3-6% on rude and K2-run2 (2 -> 5, 0 -> 6, 7 -> 12). These are medians of 2-9 trials.
- **Along the main axis.** A dose measure (total, or one component's) is among the two strongest rank correlates of refusals in every study (Spearman -0.45 to -0.78), except heretic-qwen's startup trials, where it comes third (-0.55) after direction layer (-0.63) and kind (-0.60). The other strong correlates are the kind of edit (per layer refuses less on average everywhere, rho -0.15 to -0.73, though heretic-qwen's best edits are one-layer) and, on candor-Qwen, position (-0.74). heretic-qwen trial 161's two nearest neighbours differ from it mainly by lower attn strength (about 1.32-1.35 against 1.46) and refuse 15 and 30; the next three differ mainly in attn reach, attn edge strength or the peak layers and refuse 18-27. Within the best trial's kind, median refusals fall progressively across tenths of dose rather than in one step:

| study, best kind | median refused by tenth of dose, tenths 1-10 (dose at which tenths 6 and 10 start) |
|---|---|
| heretic-rude, one layer | 92 86 75 52 66 20 24 12 9 4 (1.34, 1.76) |
| candor-rude-search, one layer | 100 97 64 38 17 19 12 8 4 1 (1.75, 2.06) |
| heretic-k2-run1, per layer | 98 86 83 89 77 74 71 69 69 64 (0.97, 1.30) |
| heretic-k2-run2, per layer | 82 68 68 77 60 63 47 28 18 16 (1.79, 3.18) |
| heretic-qwen, one layer | 100 96 98 98 93 90 78 55 50 30 (1.16, 1.92) |
| candor-qwen-full, per layer answer start | 98 84 84 77 76 70 74 50 41 33 (1.12, 1.96) |

  A falling logistic step fitted to refusals against dose explains 0.00-0.14 more variance (R2) than a straight line within the best kind (heretic-qwen 0.78 against 0.68; K2-run2 0.53 against 0.48; heretic-rude 0.76 against 0.71; candor-rude-v1 0.85 against 0.71; the candor Qwen searches 0.89-0.95 against 0.89-0.94). heretic-qwen's one-layer curve stays at 90 or more up to dose about 1.2 and then falls over the top four tenths; candor-qwen-full's per-layer curve falls gradually from dose 0.7. Neither falls in one step. (Step fits on the startup trials alone came out degenerate in six of eight studies: steps 0.01-0.04 of the dose range wide with R2 0.17-0.35, fitted to one or two outliers. They are in the full tables but mean nothing.)
- **The trade-off.** More dose costs KL everywhere (dose against KL +0.43 to +0.81) and refusals and KL move against each other (-0.60 to -0.96). It is tightest on Qwen (-0.87 to -0.96).

## Q4. Where the good trials sit

Good = the lowest tenth of fully scored trials by refusals (its threshold varies by study: <= 4 on heretic-rude, <= 48 on heretic-qwen, <= 66 on K2-run1).

| study | good refused | startup trials that were good | best trial's settings as a share of each range (0 = floor, 1 = ceiling) | ends the good trials press on (share of good trials within 5% of the end) |
|---|---|---|---|---|
| heretic-rude | <= 4 | 2/60 | strength .75/.66, peak .03/.03, edge .79/.46, reach .82/.77, dir .09 | attn edge ceiling 32%, mlp reach ceiling 18%, attn peak floor 11%; strength ceiling 4% |
| candor-rude-search-v1 | <= 5 | 0/60 | strength .70/.51, peak .15/.51, edge .89/.70, reach .98/.73 | attn strength ceiling 30%, attn edge 20%, attn reach 20% |
| candor-rude-search | <= 0 | 0/40 | strength .75/.32, peak .05/.30, edge 1.00/.68, reach .97/.85, dir .08 | attn peak floor 30%, attn edge ceiling 30%, attn reach 26%, strength 19%/22% |
| heretic-k2-run1 | <= 66 | 0/60 | strength .09/.88, peak .64/.29, edge .75/.91, reach .31/.76 | mlp strength 1.34-1.50 (span 22% of range), 19% at ceiling |
| heretic-k2-run2 | <= 15 | 2/60 | strength .91/.88 (4.6/4.5), peak .90/.02, edge .14/.98, reach .39/.34 | attn peak ceiling 43%, mlp strength ceiling 29%, mlp peak floor 29% |
| heretic-qwen | <= 48 | 0/60 | strength .94/.89, peak .22/.03, edge .98/.97, reach .89/.56, dir .55 | mlp edge ceiling 55%, attn edge 40%, mlp peak floor 30%, attn strength ceiling 25% |
| candor-qwen-pre | <= 70 | 0/40 | strength .63/.39, peak .12/.00, edge .77/.77, reach .73/.76 | (7 trials) attn peak 0.62-0.68 |
| candor-qwen-full | <= 69 | 0/40 | strength .72/.57, peak .10/.31, edge .96/.98, reach .81/.88 | (7 trials) attn and mlp edge ceiling 43% each |

(attn/mlp; "dir" = direction layer, one-layer edits only.)

What the data show: on every model the good edits lean towards a flat band (edge strength near 1 in at least one component: good-trial medians 0.81-0.96, except candor-qwen-pre at 0.77) that peaks early (one component's peak near the 0.6 floor: medians 0.63-0.69, except K2-run1 at 0.75-0.84). Strength is less clear-cut than the trial-161 story suggests. K2 is plainly strength-limited (run1: good mlp strength squeezed into 1.34-1.50; run2: good mlp strength 3.4-5.0, attn 1.7-4.9). heretic-qwen's good edits sit only a little higher than rude-llama's (good-trial medians 1.41/1.37 against 1.31-1.39 attn and 1.15-1.25 mlp on rude; candor-qwen-full 1.39/1.20), with a similar share within 5% of the 1.5 ceiling (15-25% against 0-30% on rude), though heretic-qwen's best is at 0.89-0.94 of the range and rude's bests at 0.32-0.75. So, apart from strength, trial 161 sits in the corner good edits drift towards on the other models too; what sets Qwen apart is that nothing outside that corner was good.

## Q5. candor's Qwen searches: position and kind

| study | kind | trials (startup + aimed) | fewest refused | median refused |
|---|---|---|---|---|
| candor-qwen-pre | per layer, answer start | 24 (10 + 14) | 50 | 74 |
| candor-qwen-pre | one layer, answer start | 22 (10 + 12) | 85 | 90 |
| candor-qwen-pre | per layer, think start | 10 (10 + 0) | 97 | 98 |
| candor-qwen-pre | one layer, think start | 13 (10 + 3) | 98 | 99 |
| candor-qwen-full | per layer, answer start | 24 (10 + 14) | 32 | 74 |
| candor-qwen-full | one layer, answer start | 22 (10 + 12) | 84 | 91 |
| candor-qwen-full | per layer, think start | 11 (10 + 1) | 97 | 98 |
| candor-qwen-full | one layer, think start | 13 (10 + 3) | 98 | 99 |

What the data show: no think-start edit ever refused fewer than 97 of 100, across 23-24 per search (the 20 startup ones being the same edits in both searches), with doses up to 1.5-2.2. Position is the strongest single correlate of refusals in both searches (-0.74). Per layer beats one layer in candor; in heretic-qwen it is the other way round (one layer 11, per layer 46). candor's one-layer trials have not been near heretic's best region: their 12 best answer-start ones in candor-qwen-full read directions at 0.74-0.87 of depth (two at 0.68), while heretic's good one-layer edits read at 0.57-0.80 and its trial 161 at 0.67 with dose 2.14; the closest candor-qwen-full one-layer trial is 0.26 from trial 161 (outside the ~0.2 radius above), and none reached a dose above 1.61.

## Q6. How often a random trial lands near the best

| study | best | best among startup | startup within 5 of best | random trials for 50% / 90% odds (within 5) | startup within 10 | later (TPE) trials within 5 |
|---|---|---|---|---|---|---|
| heretic-rude | 2 | 2 | 3/60 | 14 / 45 | 5/60 | 39/140 |
| candor-rude-search-v1 | 0 | 7 | 0/60 | > 14 / > 45 | 2/60 | 21/140 |
| candor-rude-search | 0 | 4 | 1/40 | 28 / 91 | 2/40 | 182/560 |
| heretic-k2-run1 | 54 | 69 | 0/60 | > 14 / > 45 | 0/60 | 5/140 |
| heretic-k2-run2 | 7 | 8 | 1/60 | 42 / 138 | 2/60 | 13/140 |
| heretic-qwen | 11 | 55 | 0/60 | > 14 / > 45 | 0/60 | 2/139 |
| candor-qwen-pre | 50 | 72 | 0/40 | > 9 / > 30 | 0/40 | 4/29 |
| candor-qwen-full | 32 | 70 | 0/40 | > 9 / > 30 | 0/40 | 2/30 |

"Random trials for 50% / 90%" is n = ln(1 - odds) / ln(1 - p) with p the startup hit rate; with no hits, p is bounded by the rule of three (3/n), so only a floor is known. One or three hits make these estimates very rough.

What the data show: on rude-llama and K2-run2, random sampling alone gets within 5-10 of the best within tens of trials (heretic-rude's random trial 46 was already its best count). On Qwen the best evenly spread trial was 22-44 refusals above its search's best (55 against 11 in heretic), and heretic's first trial within 10 of its best was trial 159.

## Answers

**(a) Does refusal fall off a cliff in a narrow region on every model, or only on Qwen?**

Data: no study shows a cliff in the sense of good edits whose close neighbours refuse near base: median refusals rise steadily with distance on all eight studies, sharp short-range jumps are rare where they can be counted (2 of 579 pairs on candor-rude, none elsewhere, but Qwen has only 3-4 pairs), and refusals fall progressively with dose within each kind. The width of the low-refusal basin (distance to a trial refusing halfway back to base) is about 0.2 on rude, K2 and heretic-qwen alike. What is particular to Qwen is (1) where the basin is: only in the box's corner, where strength, edge strength and peak layer all press on their ends at once, so none of 140 evenly spread Qwen trials (60 random in heretic, 40 Sobol edits scored under two norms in candor) refused under half of base, against 28-42% on rude and K2-run2; and (2) a steeper bottom (12-19% of the way back to base within 0.15, against 3-6%, from medians of 2-9 trials). K2 with strength capped at 1.5 looked like Qwen (0 of 60 under half, never below 54); the same model with strength up to 5.0 looked like rude (17 of 60 under half, a random trial at 8, good edits with mlp strength 3.4-5.0).

Interpretation (not shown by these data): the "narrow region" on Qwen is more likely the top of a gradual slope cut off by the strength ceiling than a cliff the model has by nature, as it was on K2. That predicts a wider strength range would open a broader low-refusal region on Qwen. No Qwen trial here went above 1.5, so this is untested; candor's widening (strength to 3.0 once a kind has 60 trials with none within the cap) would test it in the running "full" search at about trial 120. If it holds, a coarse sweep helps mainly by letting strength go far enough, and refining matters less; if it does not, Qwen's steeper bottom is a real difference and a sweep that finds the collapse then refines is the better bet.

**(b) Trials to reach the best:** heretic-rude 46 (in the random startup; within 5 by 13), candor-rude-v1 165 (within 10 by 16), candor-rude 81 of 600 (within 10 by 6), K2-run1 111 (best 54, never good), K2-run2 68 (within 5 by 50, a random trial), heretic-qwen 161 (nothing within 10 before 159), candor-qwen-pre and -full 62 and 64 of 69-70 and still falling.

**(c) Did think-start edits ever work?** No. 23-24 think-start trials per candor Qwen search, mostly the shared Sobol startup, never refused fewer than 97 of 100; answer start reached 32 (per layer) in the same search. Caveats: scored with thinking off, strength capped at 1.5, few aimed trials (TPE dropped think start quickly).

## Limits

- Three models, eight searches, one run per setting; the K2 pair also differs in seed, and the two candor Qwen searches share their startup edits and are unfinished (69-70 trials).
- Refusal counters differ (keywords against a judge) and so does KL, so cross-tool levels are rough and KL is never compared across studies.
- Startup samples are small (40-60): a hit rate of 0 only bounds the true rate below 5-7.5%.
- Trials after the startup follow the sampler, so "all trials" distributions, neighbourhoods and correlations over-represent the regions TPE liked. Neighbourhood medians near the best rest on 1-9 trials.
- Dose is a summary axis of my own; it ignores direction layer and position, which also matter.
- Row normalisation changes what a strength means, so doses compare only between studies with the same norm (pre: rude and candor-qwen-pre; full: K2, heretic-qwen, candor-qwen-full).

# Full tables from analyse.py

## Q1. Basics

| study | tool | trials | stopped early | startup | prompts | base refusals | layers | strength | reach (depth) | direction layer (depth) | settings |
|---|---|---|---|---|---|---|---|---|---|---|---|
| heretic-rude | heretic | 200 | 0 | 60 | 100 | 96 | 28 | 0.8-1.5 | 0.037-0.60 | 0.40-0.90 | objectives: KL (refusal share x 0.01 below KL 0.01), keyword refusals / base; row norm: pre; reads directions: end of prompt; startup: 60 random; planned trials: 200 |
| candor-rude-search-v1 | candor v1 | 200 | 0 | 60 | 100 | 99 | 28 | 0.8-1.5 | 0.037-0.60 | 0.41-0.89 | objectives: KL (refusal share x 0.01 below it), judged refusals / base; startup: 60 random; prompts: 100 assumed (v1 records no prompts judged) |
| candor-rude-search | candor | 600 | 338 | 40 | 100 | 99 | 28 | 0.8-1.5 | 0.037-0.60 | 0.41-0.89 | objectives: least KL with judged refusals <= 2 (constraint); startup: 20 Sobol per kind; trials think: n/a; cap: 2 |
| heretic-k2-run1 | heretic | 200 | 0 | 60 | 100 | 100 | 36 | 0.8-1.5 | 0.029-0.60 | 0.40-0.90 | objectives: KL (refusal share x 0.01 below KL 0.01), keyword refusals / base; row norm: full; reads directions: answer start (response prefix); startup: 60 random; planned trials: 200 |
| heretic-k2-run2 | heretic | 200 | 0 | 60 | 100 | 100 | 36 | 0.8-5 | 0.029-0.60 | 0.40-0.90 | objectives: KL (refusal share x 0.01 below KL 0.01), keyword refusals / base; row norm: full; reads directions: answer start (response prefix); startup: 60 random; planned trials: 200 |
| heretic-qwen | heretic | 199 | 0 | 60 | 100 | 100 | 32 | 0.8-1.5 | 0.032-0.60 | 0.40-0.90 | objectives: KL (refusal share x 0.01 below KL 0.01), keyword refusals / base; row norm: full; reads directions: answer start (response prefix); startup: 60 random; planned trials: 200 |
| candor-qwen-pre | candor | 69 | 0 | 40 | 100 | 99 | 32 | 0.8-1.5 | 0.032-0.60 | 0.39-0.90 | objectives: least KL with judged refusals <= 2 (constraint); startup: 20 Sobol per kind; trials think: off; cap: 2 |
| candor-qwen-full | candor | 70 | 0 | 40 | 100 | 99 | 32 | 0.8-1.5 | 0.032-0.60 | 0.39-0.90 | objectives: least KL with judged refusals <= 2 (constraint); startup: 20 Sobol per kind; trials think: off; cap: 2 |

## Q2. Best refusals so far: fewest refused (its KL), fully scored trials only

| study | @10 | @25 | @50 | @75 | @100 | @150 | @200 | end | best trial | first at best | first within 5 | first within 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heretic-rude | 13 (0.119) | 5 (0.049) | 2 (0.063) | 2 (0.061) | 2 (0.061) | 2 (0.061) | 2 (0.026) | 2 (0.026) | 170 | 46 | 13 | 13 |
| candor-rude-search-v1 | 13 (0.073) | 7 (0.090) | 7 (0.090) | 7 (0.090) | 3 (0.096) | 2 (0.099) | 0 (0.102) | 0 (0.102) | 165 | 165 | 77 | 16 |
| candor-rude-search | 10 (0.078) | 10 (0.078) | 1 (0.114) | 1 (0.105) | 0 (0.188) | 0 (0.100) | 0 (0.100) | 0 (0.078) | 535 | 81 | 38 | 6 |
| heretic-k2-run1 | 73 (0.037) | 70 (0.031) | 69 (0.052) | 69 (0.052) | 63 (0.047) | 54 (0.057) | 54 (0.057) | 54 (0.057) | 111 | 111 | 111 | 91 |
| heretic-k2-run2 | 41 (0.252) | 15 (0.140) | 8 (0.267) | 7 (0.084) | 7 (0.084) | 7 (0.084) | 7 (0.084) | 7 (0.084) | 68 | 68 | 50 | 23 |
| heretic-qwen | 56 (0.044) | 56 (0.044) | 56 (0.044) | 51 (0.028) | 49 (0.061) | 46 (0.040) |  | 11 (0.068) | 161 | 161 | 159 | 159 |
| candor-qwen-pre | 79 (0.044) | 72 (0.055) | 70 (0.056) |  |  |  |  | 50 (0.059) | 62 | 62 | 60 | 60 |
| candor-qwen-full | 80 (0.047) | 74 (0.055) | 70 (0.052) |  |  |  |  | 32 (0.069) | 64 | 64 | 64 | 58 |

## Q2b. candor's own target: least KL within the refusal cap

| study | cap | trials within cap | first within cap | least KL within cap |
|---|---|---|---|---|
| candor-rude-search | 2 | 154 | 50 | 305: 2, KL 0.0760 |
| candor-qwen-pre | 2 | 0 | - | - |
| candor-qwen-full | 2 | 0 | - | - |

## Q3a. Distribution of refusals

| study | trials | n | count by tenth of base refusals (0-10% ... 90-100%) | under half of base | 25-75% of base | bimodality coeff. |
|---|---|---|---|---|---|---|
| heretic-rude | startup | 60 | 4 5 4 7 5 1 4 4 10 16 | 25 (42%) | 33% | 0.64 |
| heretic-rude | all | 200 | 49 41 14 12 11 6 15 9 18 25 | 127 (64%) | 26% | 0.72 |
| candor-rude-search-v1 | startup | 60 | 2 2 4 5 4 3 3 3 7 27 | 17 (28%) | 30% | 0.71 |
| candor-rude-search-v1 | all | 200 | 27 15 21 16 13 14 10 17 23 44 | 92 (46%) | 34% | 0.65 |
| candor-rude-search | startup | 40 | 1 3 4 2 1 2 1 7 2 17 | 11 (28%) | 30% | 0.68 |
| candor-rude-search | all | 600 | 312 84 29 48 12 9 16 19 9 62 | 485 (81%) | 18% | 0.82 |
| heretic-k2-run1 | startup | 60 | 0 0 0 0 0 0 2 6 14 38 | 0 (0%) | 10% | 0.66 |
| heretic-k2-run1 | all | 200 | 0 0 0 0 0 5 33 38 41 83 | 0 (0%) | 30% | 0.59 |
| heretic-k2-run2 | startup | 60 | 1 2 4 4 6 4 6 3 14 16 | 17 (28%) | 40% | 0.59 |
| heretic-k2-run2 | all | 200 | 6 25 13 10 10 21 29 17 33 36 | 64 (32%) | 42% | 0.59 |
| heretic-qwen | startup | 60 | 0 0 0 0 0 3 7 4 17 29 | 0 (0%) | 20% | 0.60 |
| heretic-qwen | all | 199 | 0 4 7 3 12 25 22 13 32 81 | 26 (13%) | 38% | 0.60 |
| candor-qwen-pre | startup | 40 | 0 0 0 0 0 0 0 4 4 32 | 0 (0%) | 5% | 0.79 |
| candor-qwen-pre | all | 69 | 0 0 0 0 0 4 3 10 14 38 | 0 (0%) | 19% | 0.66 |
| candor-qwen-full | startup | 40 | 0 0 0 0 0 0 0 3 6 31 | 0 (0%) | 5% | 0.78 |
| candor-qwen-full | all | 70 | 0 0 0 4 2 1 0 10 12 41 | 6 (9%) | 17% | 0.75 |

## Q3b. Nearest neighbours of the three best trials (same kind; distance = RMS setting difference in units of range)

| study | best trial (kind) | refused | 5 nearest refused | their distances |
|---|---|---|---|---|
| heretic-rude | 170 (one layer) | 2 | 24 4 3 5 10 | 0.05 0.08 0.10 0.10 0.10 |
| heretic-rude | 158 (per layer) | 2 | 10 3 2 4 3 | 0.13 0.13 0.15 0.18 0.18 |
| heretic-rude | 161 (per layer) | 2 | 2 4 2 2 10 | 0.15 0.16 0.20 0.20 0.20 |
| candor-rude-search-v1 | 165 (per layer) | 0 | 1 3 2 1 3 | 0.13 0.14 0.14 0.15 0.16 |
| candor-rude-search-v1 | 167 (per layer) | 1 | 1 3 0 4 1 | 0.12 0.16 0.19 0.20 0.21 |
| candor-rude-search-v1 | 172 (per layer) | 1 | 1 1 0 3 3 | 0.12 0.21 0.21 0.21 0.22 |
| candor-rude-search | 535 (one layer) | 0 | 0 2 9s 19s 9s | 0.07 0.08 0.08 0.10 0.10 |
| candor-rude-search | 463 (one layer) | 0 | 9s 0 75s 10s 1 | 0.07 0.09 0.09 0.10 0.11 |
| candor-rude-search | 323 (one layer) | 0 | 25s 2 8s 2 19s | 0.06 0.09 0.09 0.11 0.11 |
| heretic-k2-run1 | 111 (per layer) | 54 | 62 71 66 65 64 | 0.07 0.09 0.11 0.11 0.11 |
| heretic-k2-run1 | 113 (per layer) | 55 | 61 62 54 67 66 | 0.10 0.11 0.12 0.14 0.14 |
| heretic-k2-run1 | 156 (per layer) | 58 | 65 69 69 66 82 | 0.19 0.22 0.23 0.24 0.26 |
| heretic-k2-run2 | 68 (per layer) | 7 | 10 12 14 13 52 | 0.06 0.06 0.07 0.10 0.11 |
| heretic-k2-run2 | 106 (per layer) | 7 | 13 7 14 10 20 | 0.14 0.14 0.15 0.16 0.17 |
| heretic-k2-run2 | 186 (per layer) | 8 | 16 17 15 17 11 | 0.14 0.18 0.21 0.21 0.22 |
| heretic-qwen | 161 (one layer) | 11 | 15 30 18 19 27 | 0.09 0.10 0.10 0.11 0.12 |
| heretic-qwen | 159 (one layer) | 15 | 30 11 31 33 19 | 0.06 0.09 0.12 0.12 0.15 |
| heretic-qwen | 163 (one layer) | 18 | 27 29 30 32 28 | 0.07 0.09 0.09 0.09 0.09 |
| candor-qwen-pre | 62 (per layer, answer start) | 50 | 54 64 54 62 55 | 0.08 0.08 0.10 0.15 0.18 |
| candor-qwen-pre | 66 (per layer, answer start) | 54 | 50 55 64 54 62 | 0.08 0.13 0.13 0.13 0.16 |
| candor-qwen-pre | 60 (per layer, answer start) | 54 | 64 50 62 54 70 | 0.10 0.10 0.13 0.13 0.20 |
| candor-qwen-full | 64 (per layer, answer start) | 32 | 33 39 38 42 58 | 0.04 0.10 0.10 0.13 0.16 |
| candor-qwen-full | 66 (per layer, answer start) | 33 | 32 39 38 44 42 | 0.04 0.10 0.12 0.13 0.15 |
| candor-qwen-full | 58 (per layer, answer start) | 38 | 39 42 32 33 73 | 0.06 0.09 0.10 0.12 0.18 |

## Q3c. The best trial's neighbourhood against the typical one

| study | best refused | its 5 nearest: median refused | median distance | every trial: median jump to its 5 nearest | median distance | nearest trial refusing halfway back to base: distance (refused) | trials within 5 of best | their pairs closer than 0.1 that refuse halfway back |
|---|---|---|---|---|---|---|---|---|
| heretic-rude | 2 | 5 | 0.10 | 12 | 0.19 | 0.20 (86) | 42 | 0/27 |
| candor-rude-search-v1 | 0 | 2 | 0.14 | 11 | 0.19 | 0.32 (55) | 21 | 0/8 |
| candor-rude-search | 0 | 9 | 0.08 | 6 | 0.13 | 0.19 (62s) | 183 | 2/579 |
| heretic-k2-run1 | 54 | 65 | 0.11 | 5 | 0.22 | 0.19 (78) | 5 | 0/3 |
| heretic-k2-run2 | 7 | 13 | 0.07 | 14 | 0.21 | 0.17 (61) | 14 | 0/11 |
| heretic-qwen | 11 | 19 | 0.10 | 8 | 0.20 | 0.20 (99) | 2 | 0/4 |
| candor-qwen-pre | 50 | 55 | 0.10 | 1 | 0.33 | 0.30 (78) | 4 | 0/3 |
| candor-qwen-full | 32 | 39 | 0.10 | 2 | 0.32 | 0.25 (73) | 2 | 0/3 |

## Q3d. Median refused by distance from the best trial (same kind)

| study | best trial | refused | distance <0.05 | distance <0.1 | distance <0.15 | distance <0.2 | distance <0.3 | distance >=0.3 |
|---|---|---|---|---|---|---|---|---|
| heretic-rude | 170 | 2 | - | 5 (n=5) | 5 (n=4) | 20 (n=8) | 16 (n=33) | 69 (n=74) |
| candor-rude-search-v1 | 165 | 0 | - | - | 2 (n=4) | 4 (n=4) | 14 (n=26) | 34 (n=69) |
| candor-rude-search | 535 | 0 | - | 6 (n=4) | 6 (n=26) | 13 (n=40) | 12 (n=118) | 90 (n=111) |
| heretic-k2-run1 | 111 | 54 | - | 66 (n=2) | 63 (n=15) | 71 (n=9) | 74 (n=30) | 86 (n=74) |
| heretic-k2-run2 | 68 | 7 | - | 12 (n=4) | 12 (n=5) | 20 (n=7) | 34 (n=20) | 66 (n=98) |
| heretic-qwen | 161 | 11 | - | 22 (n=2) | 28 (n=8) | 44 (n=9) | 59 (n=21) | 98 (n=77) |
| candor-qwen-pre | 62 | 50 | - | 59 (n=2) | 58 (n=2) | 55 (n=1) | 69 (n=2) | 78 (n=16) |
| candor-qwen-full | 64 | 32 | 33 (n=1) | 39 (n=1) | 40 (n=2) | 51 (n=2) | 73 (n=1) | 78 (n=16) |

## Q3e. Rank correlation of refusals with each setting

| study | trials | n | attn.strength | attn.peak_layer | attn.edge_strength | attn.reach | mlp.strength | mlp.peak_layer | mlp.edge_strength | mlp.reach | direction_layer | dose attn | dose mlp | dose total | per layer | answer start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heretic-rude | startup | 60 | -0.10 | +0.33 | -0.26 | -0.44 | -0.06 | +0.10 | -0.23 | -0.52 | -0.39 | -0.57 | -0.58 | -0.78 | -0.44 |  |
| heretic-rude | all | 200 | -0.26 | +0.52 | -0.25 | -0.27 | -0.16 | +0.29 | +0.00 | -0.41 | +0.06 | -0.65 | -0.53 | -0.76 | -0.31 |  |
| candor-rude-search-v1 | startup | 60 | -0.19 | +0.18 | +0.07 | -0.35 | -0.02 | +0.04 | -0.23 | -0.29 | -0.37 | -0.38 | -0.38 | -0.50 | -0.70 |  |
| candor-rude-search-v1 | all | 200 | -0.29 | +0.38 | -0.15 | -0.42 | -0.07 | -0.06 | -0.03 | -0.10 | -0.12 | -0.50 | -0.12 | -0.52 | -0.73 |  |
| candor-rude-search | startup | 40 | -0.08 | +0.26 | -0.22 | -0.33 | -0.07 | +0.15 | -0.03 | -0.25 | +0.19 | -0.40 | -0.31 | -0.56 | -0.71 |  |
| candor-rude-search | all | 600 | -0.16 | +0.24 | -0.39 | -0.37 | +0.00 | +0.14 | -0.19 | -0.25 | +0.29 | -0.57 | -0.28 | -0.51 | -0.33 |  |
| heretic-k2-run1 | startup | 60 | +0.28 | +0.21 | +0.06 | +0.01 | -0.24 | +0.38 | -0.04 | -0.40 | -0.32 | +0.03 | -0.57 | -0.45 | -0.50 |  |
| heretic-k2-run1 | all | 200 | +0.20 | -0.04 | -0.24 | +0.09 | -0.44 | +0.33 | -0.26 | -0.55 | -0.44 | +0.06 | -0.75 | -0.63 | -0.65 |  |
| heretic-k2-run2 | startup | 60 | -0.17 | +0.21 | -0.09 | -0.04 | -0.40 | +0.35 | -0.06 | -0.55 | +0.09 | -0.14 | -0.77 | -0.69 | -0.43 |  |
| heretic-k2-run2 | all | 200 | -0.30 | -0.18 | -0.04 | +0.17 | -0.42 | +0.57 | -0.28 | -0.39 | -0.02 | +0.02 | -0.75 | -0.59 | -0.46 |  |
| heretic-qwen | startup | 60 | -0.29 | +0.09 | -0.14 | -0.04 | -0.14 | +0.13 | -0.21 | -0.49 | -0.63 | -0.21 | -0.52 | -0.55 | -0.60 |  |
| heretic-qwen | all | 199 | -0.46 | +0.18 | -0.47 | -0.30 | -0.39 | +0.46 | -0.42 | -0.29 | -0.39 | -0.62 | -0.60 | -0.77 | -0.15 |  |
| candor-qwen-pre | startup | 40 | -0.15 | -0.03 | +0.01 | -0.07 | -0.16 | +0.28 | -0.06 | -0.49 | -0.26 | -0.10 | -0.57 | -0.54 | -0.27 | -0.69 |
| candor-qwen-pre | all | 69 | -0.28 | +0.26 | -0.21 | -0.03 | -0.02 | +0.31 | -0.11 | -0.56 | -0.47 | -0.31 | -0.52 | -0.57 | -0.40 | -0.74 |
| candor-qwen-full | startup | 40 | -0.13 | -0.03 | +0.01 | -0.07 | -0.15 | +0.27 | -0.06 | -0.48 | -0.26 | -0.09 | -0.56 | -0.52 | -0.29 | -0.69 |
| candor-qwen-full | all | 70 | -0.21 | +0.21 | -0.24 | -0.23 | -0.20 | +0.20 | -0.20 | -0.53 | -0.45 | -0.39 | -0.61 | -0.63 | -0.38 | -0.74 |

## Q3f. Strongest correlations

| study | trials | strongest rank correlations with refusals | dose total vs KL | refusals vs KL |
|---|---|---|---|---|
| heretic-rude | startup | dose total -0.78, dose mlp -0.58, dose attn -0.57, mlp.reach -0.52 | +0.71 | -0.89 |
| heretic-rude | all | dose total -0.76, dose attn -0.65, dose mlp -0.53, attn.peak_layer +0.52 | +0.44 | -0.80 |
| candor-rude-search-v1 | startup | per layer -0.70, dose total -0.50, dose mlp -0.38, dose attn -0.38 | +0.59 | -0.87 |
| candor-rude-search-v1 | all | per layer -0.73, dose total -0.52, dose attn -0.50, attn.reach -0.42 | +0.46 | -0.88 |
| candor-rude-search | startup | per layer -0.71, dose total -0.56, dose attn -0.40, attn.reach -0.33 | +0.48 | -0.85 |
| candor-rude-search | all | dose attn -0.57, dose total -0.51, attn.edge_strength -0.39, attn.reach -0.37 | +0.43 | -0.78 |
| heretic-k2-run1 | startup | dose mlp -0.57, per layer -0.50, dose total -0.45, mlp.reach -0.40 | +0.61 | -0.60 |
| heretic-k2-run1 | all | dose mlp -0.75, per layer -0.65, dose total -0.63, mlp.reach -0.55 | +0.78 | -0.77 |
| heretic-k2-run2 | startup | dose mlp -0.77, dose total -0.69, mlp.reach -0.55, per layer -0.43 | +0.77 | -0.63 |
| heretic-k2-run2 | all | dose mlp -0.75, dose total -0.59, mlp.peak_layer +0.57, per layer -0.46 | +0.81 | -0.63 |
| heretic-qwen | startup | direction_layer -0.63, per layer -0.60, dose total -0.55, dose mlp -0.52 | +0.60 | -0.87 |
| heretic-qwen | all | dose total -0.77, dose attn -0.62, dose mlp -0.60, attn.edge_strength -0.47 | +0.79 | -0.89 |
| candor-qwen-pre | startup | answer start -0.69, dose mlp -0.57, dose total -0.54, mlp.reach -0.49 | +0.43 | -0.94 |
| candor-qwen-pre | all | answer start -0.74, dose total -0.57, mlp.reach -0.56, dose mlp -0.52 | +0.61 | -0.95 |
| candor-qwen-full | startup | answer start -0.69, dose mlp -0.56, dose total -0.52, mlp.reach -0.48 | +0.44 | -0.93 |
| candor-qwen-full | all | answer start -0.74, dose total -0.63, dose mlp -0.61, mlp.reach -0.53 | +0.64 | -0.96 |

## Q3g. Refusals against total dose (mean edit strength over layers, attn + mlp): step or slope

| study | trials | n | step: high -> low refused | step centre (dose pct) | 10-90% width / dose range | R2 step | R2 line |
|---|---|---|---|---|---|---|---|
| heretic-rude | startup | 60 | 104 -> 2 | 65% | 1.12 | 0.63 | 0.61 |
| heretic-rude | all | 200 | 111 -> -7 | 27% | 1.33 | 0.56 | 0.55 |
| heretic-rude | best's kind | 125 | 95 -> 5 | 43% | 0.62 | 0.76 | 0.71 |
| candor-rude-search-v1 | startup | 60 | 92 -> 57 | 40% | 0.01 | 0.32 | 0.20 |
| candor-rude-search-v1 | all | 200 | 178 -> 12 | 2% | 2.21 | 0.28 | 0.27 |
| candor-rude-search-v1 | best's kind | 104 | 107 -> 2 | 17% | 0.80 | 0.85 | 0.71 |
| candor-rude-search | startup | 40 | 86 -> 50 | 52% | 0.01 | 0.35 | 0.23 |
| candor-rude-search | all | 600 | 106 -> 14 | 10% | 0.44 | 0.52 | 0.38 |
| candor-rude-search | best's kind | 300 | 103 -> -1 | 29% | 0.68 | 0.79 | 0.76 |
| heretic-k2-run1 | startup | 60 | 95 -> 85 | 67% | 0.01 | 0.25 | 0.17 |
| heretic-k2-run1 | all | 200 | 95 -> 70 | 57% | 0.62 | 0.41 | 0.38 |
| heretic-k2-run1 | best's kind | 131 | 99 -> 55 | 47% | 1.33 | 0.63 | 0.62 |
| heretic-k2-run2 | startup | 60 | 89 -> 41 | 60% | 0.41 | 0.48 | 0.44 |
| heretic-k2-run2 | all | 200 | 79 -> 25 | 69% | 0.41 | 0.44 | 0.40 |
| heretic-k2-run2 | best's kind | 135 | 75 -> 22 | 64% | 0.48 | 0.53 | 0.48 |
| heretic-qwen | startup | 60 | 92 -> 76 | 65% | 0.04 | 0.33 | 0.30 |
| heretic-qwen | all | 199 | 101 -> 21 | 79% | 0.95 | 0.71 | 0.67 |
| heretic-qwen | best's kind | 118 | 99 -> 29 | 72% | 0.52 | 0.78 | 0.68 |
| candor-qwen-pre | startup | 40 | 97 -> 90 | 60% | 0.01 | 0.21 | 0.10 |
| candor-qwen-pre | all | 69 | 100 -> 80 | 38% | 0.57 | 0.25 | 0.20 |
| candor-qwen-pre | best's kind | 24 | 138 -> -7 | 75% | 4.00 | 0.89 | 0.89 |
| candor-qwen-full | startup | 40 | 97 -> 90 | 60% | 0.01 | 0.17 | 0.08 |
| candor-qwen-full | all | 70 | 91 -> 37 | 90% | 0.04 | 0.76 | 0.49 |
| candor-qwen-full | best's kind | 24 | 123 -> -54 | 96% | 2.85 | 0.95 | 0.94 |

## Q3h. Median refused by tenth of total dose ("@" the dose the tenth starts at), all trials and the best trial's kind

| study | trials | n | tenth 1 | tenth 2 | tenth 3 | tenth 4 | tenth 5 | tenth 6 | tenth 7 | tenth 8 | tenth 9 | tenth 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heretic-rude | all | 200 | 90 @0.24 | 62 @0.64 | 50 @0.90 | 33 @1.06 | 42 @1.19 | 16 @1.28 | 16 @1.36 | 14 @1.45 | 9 @1.58 | 4 @1.71 |
| heretic-rude | one layer | 125 | 92 @0.25 | 86 @0.68 | 75 @1.00 | 52 @1.19 | 66 @1.24 | 20 @1.34 | 24 @1.43 | 12 @1.51 | 9 @1.62 | 4 @1.76 |
| candor-rude-search-v1 | all | 200 | 97 @0.33 | 84 @0.59 | 78 @0.76 | 48 @0.88 | 20 @0.96 | 39 @1.00 | 59 @1.10 | 51 @1.21 | 48 @1.29 | 3 @1.44 |
| candor-rude-search-v1 | per layer | 104 | 86 @0.33 | 60 @0.60 | 38 @0.78 | 25 @0.91 | 19 @0.96 | 22 @0.99 | 19 @1.07 | 13 @1.11 | 4 @1.28 | 3 @1.50 |
| candor-rude-search | all | 600 | 100 @0.21 | 25 @0.88 | 16 @1.16 | 9 @1.28 | 6 @1.39 | 3 @1.46 | 7 @1.55 | 10 @1.69 | 9 @1.83 | 2 @1.95 |
| candor-rude-search | one layer | 300 | 100 @0.21 | 97 @0.74 | 64 @1.21 | 38 @1.50 | 17 @1.65 | 19 @1.75 | 12 @1.81 | 8 @1.88 | 4 @1.95 | 1 @2.06 |
| heretic-k2-run1 | all | 200 | 100 @0.14 | 98 @0.46 | 89 @0.61 | 96 @0.69 | 90 @0.79 | 76 @0.91 | 82 @1.00 | 70 @1.08 | 72 @1.17 | 69 @1.29 |
| heretic-k2-run1 | per layer | 131 | 98 @0.14 | 86 @0.50 | 83 @0.64 | 89 @0.71 | 77 @0.85 | 74 @0.97 | 71 @1.05 | 69 @1.13 | 69 @1.19 | 64 @1.30 |
| heretic-k2-run2 | all | 200 | 84 @0.33 | 70 @0.74 | 72 @1.03 | 74 @1.22 | 84 @1.48 | 66 @1.77 | 74 @1.97 | 32 @2.27 | 18 @2.60 | 20 @3.06 |
| heretic-k2-run2 | per layer | 135 | 82 @0.38 | 68 @0.75 | 68 @0.94 | 77 @1.22 | 60 @1.42 | 63 @1.79 | 47 @1.98 | 28 @2.39 | 18 @2.81 | 16 @3.18 |
| heretic-qwen | all | 199 | 99 @0.26 | 92 @0.54 | 92 @0.69 | 98 @0.86 | 85 @0.97 | 81 @1.08 | 68 @1.19 | 60 @1.26 | 53 @1.50 | 32 @1.77 |
| heretic-qwen | one layer | 118 | 100 @0.26 | 96 @0.49 | 98 @0.79 | 98 @0.92 | 93 @1.04 | 90 @1.16 | 78 @1.24 | 55 @1.44 | 50 @1.57 | 30 @1.92 |
| candor-qwen-pre | all | 69 | 99 @0.26 | 98 @0.54 | 99 @0.73 | 89 @0.86 | 84 @0.97 | 87 @1.04 | 88 @1.20 | 87 @1.31 | 85 @1.43 | 86 @1.62 |
| candor-qwen-pre | per layer, answer start | 24 | 98 @0.35 | 88 @0.67 | 81 @0.83 | 79 @0.90 | 76 @1.00 | 74 @1.04 | 72 @1.21 | 66 @1.28 | 60 @1.42 | 54 @1.59 |
| candor-qwen-full | all | 70 | 99 @0.26 | 98 @0.54 | 99 @0.73 | 98 @0.84 | 84 @0.98 | 87 @1.05 | 89 @1.15 | 88 @1.23 | 97 @1.34 | 39 @1.72 |
| candor-qwen-full | per layer, answer start | 24 | 98 @0.35 | 84 @0.67 | 84 @0.85 | 77 @0.98 | 76 @1.05 | 70 @1.12 | 74 @1.21 | 50 @1.38 | 41 @1.81 | 33 @1.96 |

## Q4a. Where the good trials (lowest tenth of refusals) sit

| study | good = refused | good trials | startup trials that were good | best trial's settings, as a share of each range |
|---|---|---|---|---|
| heretic-rude | <= 4 | 28 | 2/60 | attn.strength 0.75, attn.peak_layer 0.03, attn.edge_strength 0.79, attn.reach 0.82, mlp.strength 0.66, mlp.peak_layer 0.03, mlp.edge_strength 0.46, mlp.reach 0.77, direction_layer 0.09 |
| candor-rude-search-v1 | <= 5 | 20 | 0/60 | attn.strength 0.70, attn.peak_layer 0.15, attn.edge_strength 0.89, attn.reach 0.98, mlp.strength 0.51, mlp.peak_layer 0.51, mlp.edge_strength 0.70, mlp.reach 0.73 |
| candor-rude-search | <= 0 | 27 | 0/40 | attn.strength 0.75, attn.peak_layer 0.05, attn.edge_strength 1.00, attn.reach 0.97, mlp.strength 0.32, mlp.peak_layer 0.30, mlp.edge_strength 0.68, mlp.reach 0.85, direction_layer 0.08 |
| heretic-k2-run1 | <= 66 | 21 | 0/60 | attn.strength 0.09, attn.peak_layer 0.64, attn.edge_strength 0.75, attn.reach 0.31, mlp.strength 0.88, mlp.peak_layer 0.29, mlp.edge_strength 0.91, mlp.reach 0.76 |
| heretic-k2-run2 | <= 15 | 21 | 2/60 | attn.strength 0.91, attn.peak_layer 0.90, attn.edge_strength 0.14, attn.reach 0.39, mlp.strength 0.88, mlp.peak_layer 0.02, mlp.edge_strength 0.98, mlp.reach 0.34 |
| heretic-qwen | <= 48 | 20 | 0/60 | attn.strength 0.94, attn.peak_layer 0.22, attn.edge_strength 0.98, attn.reach 0.89, mlp.strength 0.89, mlp.peak_layer 0.03, mlp.edge_strength 0.97, mlp.reach 0.56, direction_layer 0.55 |
| candor-qwen-pre | <= 70 | 7 | 0/40 | attn.strength 0.63, attn.peak_layer 0.12, attn.edge_strength 0.77, attn.reach 0.73, mlp.strength 0.39, mlp.peak_layer 0.00, mlp.edge_strength 0.77, mlp.reach 0.76 |
| candor-qwen-full | <= 69 | 7 | 0/40 | attn.strength 0.72, attn.peak_layer 0.10, attn.edge_strength 0.96, attn.reach 0.81, mlp.strength 0.57, mlp.peak_layer 0.31, mlp.edge_strength 0.98, mlp.reach 0.88 |

## Q4b. Good trials' settings, per study

**heretic-rude**: 28 trials refusing <= 4 (per layer: 17, one layer: 11)

| setting | range | n | median | min-max | span / range | at floor | at ceiling | median, every trial |
|---|---|---|---|---|---|---|---|---|
| attn.strength | 0.80-1.50 | 28 | 1.31 | 0.97-1.47 | 0.71 | 0% | 4% | 1.20 |
| attn.peak_layer | 0.60-1.00 | 28 | 0.69 | 0.61-0.85 | 0.60 | 11% | 0% | 0.75 |
| attn.edge_strength | 0.00-1.00 | 28 | 0.84 | 0.44-1.00 | 0.56 | 0% | 32% | 0.81 |
| attn.reach | 0.04-0.60 | 28 | 0.50 | 0.33-0.60 | 0.47 | 0% | 11% | 0.50 |
| mlp.strength | 0.80-1.50 | 28 | 1.25 | 0.92-1.48 | 0.80 | 0% | 4% | 1.20 |
| mlp.peak_layer | 0.60-1.00 | 28 | 0.72 | 0.60-0.99 | 0.98 | 7% | 7% | 0.78 |
| mlp.edge_strength | 0.00-1.00 | 28 | 0.49 | 0.18-0.96 | 0.78 | 0% | 7% | 0.52 |
| mlp.reach | 0.04-0.60 | 28 | 0.53 | 0.37-0.60 | 0.40 | 0% | 18% | 0.45 |
| direction_layer | 0.40-0.90 | 11 | 0.51 | 0.45-0.54 | 0.19 | 0% | 0% | 0.52 |

**candor-rude-search-v1**: 20 trials refusing <= 5 (per layer: 20)

| setting | range | n | median | min-max | span / range | at floor | at ceiling | median, every trial |
|---|---|---|---|---|---|---|---|---|
| attn.strength | 0.80-1.50 | 20 | 1.39 | 1.14-1.50 | 0.52 | 0% | 30% | 1.25 |
| attn.peak_layer | 0.60-1.00 | 20 | 0.67 | 0.60-0.83 | 0.58 | 10% | 0% | 0.71 |
| attn.edge_strength | 0.00-1.00 | 20 | 0.89 | 0.37-0.99 | 0.62 | 0% | 20% | 0.73 |
| attn.reach | 0.04-0.60 | 20 | 0.53 | 0.41-0.59 | 0.32 | 0% | 20% | 0.43 |
| mlp.strength | 0.80-1.50 | 20 | 1.15 | 0.85-1.37 | 0.74 | 0% | 0% | 1.14 |
| mlp.peak_layer | 0.60-1.00 | 20 | 0.83 | 0.61-0.99 | 0.97 | 5% | 15% | 0.91 |
| mlp.edge_strength | 0.00-1.00 | 20 | 0.67 | 0.16-0.97 | 0.82 | 0% | 5% | 0.53 |
| mlp.reach | 0.04-0.60 | 20 | 0.33 | 0.10-0.56 | 0.82 | 0% | 0% | 0.23 |

**candor-rude-search**: 27 trials refusing <= 0 (one layer: 17, per layer: 10)

| setting | range | n | median | min-max | span / range | at floor | at ceiling | median, every trial |
|---|---|---|---|---|---|---|---|---|
| attn.strength | 0.80-1.50 | 27 | 1.36 | 1.10-1.50 | 0.57 | 0% | 19% | 1.29 |
| attn.peak_layer | 0.60-1.00 | 27 | 0.64 | 0.60-0.86 | 0.65 | 30% | 0% | 0.68 |
| attn.edge_strength | 0.00-1.00 | 27 | 0.93 | 0.51-1.00 | 0.48 | 0% | 30% | 0.86 |
| attn.reach | 0.04-0.60 | 27 | 0.53 | 0.34-0.60 | 0.45 | 0% | 26% | 0.48 |
| mlp.strength | 0.80-1.50 | 27 | 1.19 | 0.81-1.49 | 0.97 | 7% | 22% | 1.16 |
| mlp.peak_layer | 0.60-1.00 | 27 | 0.72 | 0.61-0.87 | 0.64 | 4% | 0% | 0.76 |
| mlp.edge_strength | 0.00-1.00 | 27 | 0.80 | 0.34-1.00 | 0.66 | 0% | 11% | 0.61 |
| mlp.reach | 0.04-0.60 | 27 | 0.51 | 0.38-0.59 | 0.38 | 0% | 19% | 0.46 |
| direction_layer | 0.41-0.89 | 17 | 0.44 | 0.44-0.52 | 0.15 | 0% | 0% | 0.52 |

**heretic-k2-run1**: 21 trials refusing <= 66 (per layer: 21)

| setting | range | n | median | min-max | span / range | at floor | at ceiling | median, every trial |
|---|---|---|---|---|---|---|---|---|
| attn.strength | 0.80-1.50 | 21 | 0.93 | 0.82-1.47 | 0.93 | 14% | 5% | 1.04 |
| attn.peak_layer | 0.60-1.00 | 21 | 0.84 | 0.75-0.95 | 0.51 | 0% | 0% | 0.83 |
| attn.edge_strength | 0.00-1.00 | 21 | 0.72 | 0.03-0.99 | 0.96 | 5% | 10% | 0.59 |
| attn.reach | 0.03-0.60 | 21 | 0.22 | 0.04-0.56 | 0.90 | 10% | 0% | 0.28 |
| mlp.strength | 0.80-1.50 | 21 | 1.42 | 1.34-1.50 | 0.22 | 0% | 19% | 1.29 |
| mlp.peak_layer | 0.60-1.00 | 21 | 0.75 | 0.62-0.83 | 0.53 | 0% | 0% | 0.78 |
| mlp.edge_strength | 0.00-1.00 | 21 | 0.85 | 0.19-0.97 | 0.78 | 0% | 10% | 0.63 |
| mlp.reach | 0.03-0.60 | 21 | 0.47 | 0.35-0.58 | 0.41 | 0% | 5% | 0.37 |

**heretic-k2-run2**: 21 trials refusing <= 15 (per layer: 18, one layer: 3)

| setting | range | n | median | min-max | span / range | at floor | at ceiling | median, every trial |
|---|---|---|---|---|---|---|---|---|
| attn.strength | 0.80-5.00 | 21 | 4.19 | 1.71-4.93 | 0.77 | 0% | 14% | 3.46 |
| attn.peak_layer | 0.60-1.00 | 21 | 0.96 | 0.60-1.00 | 0.98 | 19% | 43% | 0.90 |
| attn.edge_strength | 0.00-1.00 | 21 | 0.42 | 0.01-0.99 | 0.99 | 14% | 14% | 0.47 |
| attn.reach | 0.03-0.60 | 21 | 0.21 | 0.03-0.44 | 0.71 | 14% | 0% | 0.22 |
| mlp.strength | 0.80-5.00 | 21 | 4.48 | 3.36-4.98 | 0.39 | 0% | 29% | 3.67 |
| mlp.peak_layer | 0.60-1.00 | 21 | 0.63 | 0.60-0.83 | 0.57 | 29% | 0% | 0.72 |
| mlp.edge_strength | 0.00-1.00 | 21 | 0.81 | 0.34-1.00 | 0.66 | 0% | 14% | 0.65 |
| mlp.reach | 0.03-0.60 | 21 | 0.30 | 0.22-0.54 | 0.56 | 0% | 0% | 0.19 |
| direction_layer | 0.40-0.90 | 3 | 0.53 | 0.42-0.57 | 0.30 | 33% | 0% | 0.63 |

**heretic-qwen**: 20 trials refusing <= 48 (one layer: 19, per layer: 1)

| setting | range | n | median | min-max | span / range | at floor | at ceiling | median, every trial |
|---|---|---|---|---|---|---|---|---|
| attn.strength | 0.80-1.50 | 20 | 1.41 | 1.22-1.50 | 0.41 | 0% | 25% | 1.29 |
| attn.peak_layer | 0.60-1.00 | 20 | 0.67 | 0.61-0.86 | 0.63 | 5% | 0% | 0.77 |
| attn.edge_strength | 0.00-1.00 | 20 | 0.93 | 0.75-0.99 | 0.24 | 0% | 40% | 0.68 |
| attn.reach | 0.03-0.60 | 20 | 0.48 | 0.35-0.58 | 0.42 | 0% | 15% | 0.42 |
| mlp.strength | 0.80-1.50 | 20 | 1.37 | 1.26-1.49 | 0.33 | 0% | 15% | 1.23 |
| mlp.peak_layer | 0.60-1.00 | 20 | 0.63 | 0.60-0.76 | 0.41 | 30% | 0% | 0.78 |
| mlp.edge_strength | 0.00-1.00 | 20 | 0.96 | 0.60-1.00 | 0.39 | 0% | 55% | 0.76 |
| mlp.reach | 0.03-0.60 | 20 | 0.36 | 0.19-0.59 | 0.71 | 0% | 10% | 0.30 |
| direction_layer | 0.40-0.90 | 19 | 0.68 | 0.57-0.80 | 0.45 | 0% | 0% | 0.68 |

**candor-qwen-pre**: 7 trials refusing <= 70 (per layer, answer start: 7)

| setting | range | n | median | min-max | span / range | at floor | at ceiling | median, every trial |
|---|---|---|---|---|---|---|---|---|
| attn.strength | 0.80-1.50 | 7 | 1.28 | 1.20-1.47 | 0.39 | 0% | 14% | 1.24 |
| attn.peak_layer | 0.60-1.00 | 7 | 0.64 | 0.62-0.68 | 0.15 | 14% | 0% | 0.78 |
| attn.edge_strength | 0.00-1.00 | 7 | 0.69 | 0.54-0.90 | 0.37 | 0% | 0% | 0.60 |
| attn.reach | 0.03-0.60 | 7 | 0.42 | 0.14-0.45 | 0.55 | 0% | 0% | 0.34 |
| mlp.strength | 0.80-1.50 | 7 | 1.05 | 0.98-1.14 | 0.23 | 0% | 0% | 1.18 |
| mlp.peak_layer | 0.60-1.00 | 7 | 0.65 | 0.60-0.74 | 0.35 | 14% | 0% | 0.75 |
| mlp.edge_strength | 0.00-1.00 | 7 | 0.77 | 0.42-0.95 | 0.53 | 0% | 0% | 0.59 |
| mlp.reach | 0.03-0.60 | 7 | 0.46 | 0.36-0.57 | 0.37 | 0% | 14% | 0.41 |

**candor-qwen-full**: 7 trials refusing <= 69 (per layer, answer start: 7)

| setting | range | n | median | min-max | span / range | at floor | at ceiling | median, every trial |
|---|---|---|---|---|---|---|---|---|
| attn.strength | 0.80-1.50 | 7 | 1.39 | 1.20-1.45 | 0.36 | 0% | 0% | 1.21 |
| attn.peak_layer | 0.60-1.00 | 7 | 0.65 | 0.61-0.67 | 0.17 | 14% | 0% | 0.79 |
| attn.edge_strength | 0.00-1.00 | 7 | 0.94 | 0.85-0.99 | 0.14 | 0% | 43% | 0.65 |
| attn.reach | 0.03-0.60 | 7 | 0.49 | 0.35-0.59 | 0.41 | 0% | 14% | 0.39 |
| mlp.strength | 0.80-1.50 | 7 | 1.20 | 1.00-1.29 | 0.42 | 0% | 0% | 1.20 |
| mlp.peak_layer | 0.60-1.00 | 7 | 0.76 | 0.69-0.85 | 0.42 | 0% | 0% | 0.77 |
| mlp.edge_strength | 0.00-1.00 | 7 | 0.93 | 0.83-0.98 | 0.14 | 0% | 43% | 0.58 |
| mlp.reach | 0.03-0.60 | 7 | 0.50 | 0.40-0.58 | 0.31 | 0% | 14% | 0.41 |

## Q5. By kind of edit

| study | kind | trials | scored in full | fewest refused | median refused | median refused, startup | trials <= 10 | least KL within cap |
|---|---|---|---|---|---|---|---|---|
| heretic-rude | one layer | 125 | 125 | 2 | 41 | 86 | 20 | - |
| heretic-rude | per layer | 75 | 75 | 2 | 14 | 46 | 34 | - |
| candor-rude-search-v1 | one layer | 96 | 96 | 35 | 86 | 97 | 0 | - |
| candor-rude-search-v1 | per layer | 104 | 104 | 0 | 21 | 41 | 27 | - |
| candor-rude-search | one layer | 300 | 113 | 0 | 25 | 97 | 74 | 305: KL 0.0760 |
| candor-rude-search | per layer | 300 | 149 | 0 | 8 | 41 | 125 | 448: KL 0.0831 |
| heretic-k2-run1 | one layer | 69 | 69 | 74 | 99 | 99 | 0 | - |
| heretic-k2-run1 | per layer | 131 | 131 | 54 | 77 | 88 | 0 | - |
| heretic-k2-run2 | one layer | 65 | 65 | 8 | 87 | 86 | 2 | - |
| heretic-k2-run2 | per layer | 135 | 135 | 7 | 58 | 60 | 6 | - |
| heretic-qwen | one layer | 118 | 118 | 11 | 90 | 99 | 0 | - |
| heretic-qwen | per layer | 81 | 81 | 46 | 75 | 82 | 0 | - |
| candor-qwen-pre | one layer, answer start | 22 | 22 | 85 | 90 | 98 | 0 | - |
| candor-qwen-pre | one layer, think start | 13 | 13 | 98 | 99 | 99 | 0 | - |
| candor-qwen-pre | per layer, answer start | 24 | 24 | 50 | 74 | 86 | 0 | - |
| candor-qwen-pre | per layer, think start | 10 | 10 | 97 | 98 | 98 | 0 | - |
| candor-qwen-full | one layer, answer start | 22 | 22 | 84 | 91 | 98 | 0 | - |
| candor-qwen-full | one layer, think start | 13 | 13 | 98 | 99 | 99 | 0 | - |
| candor-qwen-full | per layer, answer start | 24 | 24 | 32 | 74 | 84 | 0 | - |
| candor-qwen-full | per layer, think start | 11 | 11 | 97 | 98 | 98 | 0 | - |

## Q6. How often a startup trial landed near the best

| study | best | best among startup | startup within 5 | random trials for 50% / 90% (within 5) | startup within 10 | random trials for 50% / 90% (within 10) | later trials within 5 |
|---|---|---|---|---|---|---|---|
| heretic-rude | 2 | 2 | 3/60 | 14 / 45 | 5/60 | 8 / 27 | 39/140 |
| candor-rude-search-v1 | 0 | 7 | 0/60 | >14 / >45 | 2/60 | 21 / 68 | 21/140 |
| candor-rude-search | 0 | 4 | 1/40 | 28 / 91 | 2/40 | 14 / 45 | 182/560 |
| heretic-k2-run1 | 54 | 69 | 0/60 | >14 / >45 | 0/60 | >14 / >45 | 5/140 |
| heretic-k2-run2 | 7 | 8 | 1/60 | 42 / 138 | 2/60 | 21 / 68 | 13/140 |
| heretic-qwen | 11 | 55 | 0/60 | >14 / >45 | 0/60 | >14 / >45 | 2/139 |
| candor-qwen-pre | 50 | 72 | 0/40 | >9 / >30 | 0/40 | >9 / >30 | 4/29 |
| candor-qwen-full | 32 | 70 | 0/40 | >9 / >30 | 0/40 | >9 / >30 | 2/30 |

