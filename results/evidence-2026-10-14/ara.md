# ARA: what it is, whether candor needs it, and how we missed it

Research only, 2026-10-14; nothing built. Three reports beside this one hold the sources:
`ara-what.md` (heretic's code, read at master 66a952b), `ara-evidence.md` (every reported result)
and `ara-missed.md` (the timelines). Claims below come from those, with their citations; my
inferences are marked.

## What it is

ARA, Arbitrary-Rank Ablation, is a new edit by heretic's maintainer, Philipp Weidmann. It has no
paper; a write-up was promised in March and again on 2026-10-03 (#473).

- **No refusal direction.** Instead it records what each layer's output matrices (`o_proj`, on
  Qwen3.5 also `linear_attn.out_proj`, and `down_proj`) produce for 400 harmful and 400 harmless
  prompts, at the last prompt token.
- **Each trial fits a rank-50 correction** to every such matrix in a band of layers, by a few steps
  of L-BFGS, with three goals: keep harmless outputs where they were, pull each harmful output
  towards its nearest harmless ones, and push it away from where it started ("overcorrection").
- **The search tunes 6 numbers:** the first and last layer, the weight of each goal, and how many
  neighbours count.
- **It ships as a fixed weight change** (merged weights or a LoRA adapter), so it fits candor's
  export to a GGUF, and needs nothing at inference.
- **Status:** default on heretic's master since PR #446 (2026-10-01), with trials cut from 200 to
  100. In no release: the latest is still 1.4.0 (2026-06-14). The maintainer wrote at the merge
  that it "will have to prove itself in manual testing" to stay the default in 2.0.
- **Known problems:** some trials run away to KL above 9, understood by nobody yet (#211, #332); the
  fit aims at a row-normalised matrix but ships the plain one, acknowledged and unfixed; and two
  bugs where the restored trial wasn't the one scored, fixed 2026-10-04 and 10-06 (#476, #478).

## How well it works

**On heretic's own measure (keywords, thinking off, first-token KL) ARA wins, moderately.**
Same-person comparisons show large gaps (gpt-oss-20b 3 against 49-74 refused of 100; Gemma-4-31B 8
against 63; Qwen3.8-27B 9 against 33; Qwen3.5-4B 4 against more than 50), at low KL. Each is one
run per method, which the maintainer himself says "tells you basically nothing" (#332). Against
"full" (the edit candor now uses) on Qwen3.5-4B: one win and one tie.

**As the model is used (a judge, thinking on) there is no evidence it removes more refusals, and
what exists leans against it.** Five independent benchmarks (DreamFast, jenerallee78) found ARA
models among the best at keeping KL and capability low, but never the best at removing refusals:
76-96% compliance against 85-100% for the best directional edits. None was controlled.

**On the model you plan to use next, Qwen3.8-27B** (DreamFast, a judge, thinking on, 400 HarmBench
prompts): single-pass ARA complied with 57.5%, the lowest of 12 variants, though its card claims
0/100; a second ARA pass 71.3%; a careful directional edit 82.2%; "full" 70.5%. ARA kept maths best
(GSM8K 91.5% against the base's 91.2%) and looped least.

**On Qwen3.5-9B:** no judge-scored ARA result exists. A directional heretic edit by another user
(trohrbaugh, heretic 1.2.0) refused 6/100 by keywords, and 0/400 when DreamFast judged it, losing
GSM8K 87.6% to 86.0%. So heretic's usual method can work very well on this model; the stall at
70/100 (#404) isn't the method's limit (my inference: more likely settings, such as strength).

**Keywords with thinking off flatter thinking models badly, whatever the method:** Qwen3.5-35B-A3B
3-5 refused by keywords against 33+ by a judge; Qwen3.8-27B 0 against 30.5%; Gemma-4-26B 2 against
31.4%. This is exactly what candor exists to avoid, now shown by others on the models you'll use.

## Do we need it?

Not now (my judgement from the above):
- The one thing candor needs on Qwen, fewer refusals as used, is where ARA's evidence is weakest.
  Its strength is keeping the model intact, which matters, but directional edits already reach 0/400
  judged on Qwen3.5-9B.
- The best-supported lever is still the strength range (lead 1 in STATUS), tested by this run.
- It would be a new kind of edit with known instabilities, no write-up, and an undecided future in
  heretic 2.0.

Worth revisiting if the directional edit stalls as used, or if maths and thinking length turn out to
be what separates candidates (ARA preserved those best on Qwen3.8-27B).

## Could candor build it better?

Only inference, untested. If ARA is ever worth it, candor's existing parts address its weak points:
scoring by a judge with thinking on (the keyword gap above); fitting and scoring where the answer
starts after the model's own thinking rather than heretic's closed-thinking prefix (which may even
come out malformed on Qwen3.5: render it before comparing anything); scoring the shipped file
(heretic #490: exported models differ from the trial scored); and a KL cap to reject runaway
trials. The open question nobody has answered, ARA against directional edits in a controlled,
as-used test, is one candor is built to answer, after Qwen.

## How we missed it

We didn't miss it entirely. On 2026-10-13 our research cited PR #211 and issue #404 but filed ARA as
"unreleased", a fallback; we read the closed PR without seeing it was superseded by #446, merged
on 2026-10-01.
- We tracked heretic's release (1.4.0, pinned everywhere), not its master.
- "Research" meant papers; ARA exists only in PRs, issues and code.
- "heretic frozen, no new features" was meant for our kit but froze our view of heretic too.
- Fair: until it merged, ARA was an experimental branch; but #404 named our exact model and was one
  search away.

**Fix** (simple, no automation): at each phase start, and before any decision rule names another
tool as the reference, check heretic's master (commit and default edit), the PRs merged since the
last check, and the issues on the model family we're about to work on, reading cited PRs to their
last comment. Record it as one replaceable "Tools checked DATE" line in STATUS. The commands are in
`ara-missed.md`.

## The no-prompts rule

All three agents reported reading small amounts of prompt or answer text quoted in GitHub threads,
model cards and Reddit comments while gathering this; none of it is in these reports, and every
copy was deleted. The rule in memory now covers public threads, and future research reads them only
through scripts that print numbers, code and technical sentences.
