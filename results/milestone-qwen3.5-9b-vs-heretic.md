# Milestone: candor beats heretic on Qwen3.5-9B, thinking on (2026-10-15)

Under phase 3's decision rule, signed off before the search ran, both of candor's finalists pass and
beat heretic's Trial 161 (heretic's own save). All three were exported to Q4_K_M and measured the
same way, as the model is used: llama-server, thinking on to a budget of 8192, Qwen's sampling, on
prompts no search saw.

| Held-out | Unedited | heretic Trial 161 | candor trial 205 | candor trial 175 |
| --- | --- | --- | --- | --- |
| Harmful refused, judge (/288) | 279 | 28 | 16 | **0** (p 7e-09) |
| Harmless refused, judge (/300) | 5 | 1 | 1 | 2 |
| KL from the unedited model | | 0.073 | **0.058** | **0.059** |
| Maths correct (/1319) | 1183 | 1162 | 1179 | 1181 |
| Maths thinking reaching the budget (/1319) | 184 | 297 | **174** (p 7e-18) | **187** (p 3e-14) |
| Harmful thinking, median tokens | 996 | 2,973 | 2,050.5 | 1,641 |

- Trial 175 removes every refusal (28 to 0) and keeps maths at the unedited model's level; Trial
  161's maths loss came from thinking that ran into the budget, which candor's edits don't cause.
- Trial 175 uses one direction, read at layer 15 where the thinking starts. The search behind it
  scored 545 trials with thinking off and only its best 6 as used. Details in
  `phase-3-qwen3.5-9b.md`.
- Not yet compared with the best public Qwen3.5-9B edit measured the same way.
