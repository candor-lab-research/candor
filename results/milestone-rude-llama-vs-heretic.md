# Milestone: candor beats heretic on rude-llama (2026-10-12, shipped 2026-10-13)

candor's own search, given no heretic recipe, found an edit that beats heretic's best (Trial 170) on
every count of phase 2's decision rule, which was signed off before any search ran. Both edits were
measured the same way: built by candor, in llama-server, on prompts neither search saw.

As scored, the edit applied as adapters to the Q4_K_M model:

| Held-out | heretic Trial 170 | candor trial 535 | Unedited |
| --- | --- | --- | --- |
| Harmful refused, judge (/288) | 6 | **5** | 277 |
| KL from the unedited model | 0.1011 | **0.0920** (9% less) | |
| Maths correct (/1319) | 977 | **996** (p 0.053) | 980 |
| Harmless refused, judge (/300) | 2 | 2 | 7 |
| Harmful refused, keywords (/288) | 25 | **16** (p 0.035) | 275 |

As shipped, each merged into the weights and exported the same way (Q4_K_M, edited tensors at Q8_0):

| Held-out | heretic Trial 170 | candor trial 535 |
| --- | --- | --- |
| Harmful refused, judge (/288) | 7 | **6** |
| KL from the unedited model | 0.1186 | **0.1132** (4.5% less) |
| Maths correct (/1319) | 986 | 982 |
| Harmless refused, judge (/300) | 2 | 3 |

- As scored, the clear wins are KL (9% less) and keyword refusals; the maths gain is just short of
  significance, and one fewer refusal is within chance.
- As shipped, 535 passes the rule against Trial 170 shipped the same way, but narrowly: the two files
  are nearly the same model (KL 0.0014 between them) and no difference between them is significant.
- 535 is Trial 170's recipe refined: the same direction layer (12), nearly the same attention edit,
  and a gentler MLP edit. It took 600 trials in 7 h 21 min on the Mac. Details are in
  `phase-2-rude-llama.md`.
