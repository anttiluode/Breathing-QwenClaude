# Results: run 1 (Qwen3-8B, 4-bit, 29 Sep 2026)

`python bench.py --load-4bit` on the 60 items in `data/tot_items.jsonl`, insert design
(clean = 3 true clues, corrupt = the same 3 plus one false clue). The raw output is in
`results/qwen3-8b-4bit_run1.jsonl`. Re-score it with `python analyze.py results/qwen3-8b-4bit_run1.jsonl`.
It took 63 minutes on Antti's Windows machine. The thinking-mode arm was not run.

## The verdict

The pre-registered rule killed both special claims:

- **Breathing (the iterative trust loop) is killed.** It beat plain sampling-and-checking
  (`fixed`), but not simple leave-one-clue-out (`dropout`). Its difference from `dropout`
  is −0.017 lenient and 0.000 strict, both with intervals that span zero.
- **Attention precision is not shown.** Turning trust down inside attention did no better
  than deleting the clue from the text: −0.017, with an interval spanning zero.

What survives is simpler: **ask again with each clue left out in turn, then pick the
answer that leaves the least unexplained residue.** On corrupted clues that goes from
52% (one shot) to 75% (strict), with no cost on clean items.

## Accuracy

Strict means the answer must be the target or a listed alias. So "Bilateral striatum" or
"Hodge Laplacian" is not counted as correct. Seconds per item are for the corrupt
condition.

| arm | clean | corrupt, lenient | corrupt, strict | generations/item | s/item |
|---|---|---|---|---|---|
| one_shot | 0.95 | 0.52 | 0.52 | 1 | 0.5 |
| fixed (sample + check) | 0.95 | 0.62 | 0.60 | 34 | 5.6 |
| dropout (leave one clue out + check) | 0.93 | 0.77 | 0.75 | 31 | 12.4 |
| breathe_text | 0.93 | 0.77 | **0.77** | 35 | 14.3 |
| breathe_attn | 0.95 | 0.75 | 0.75 | 34 | **8.7** |

All five arms answer clean items at 93–95%, so none of this machinery costs anything when
the clues are true.

`breathe_attn` is the fastest of the three that work. All its "let go of clue k" settings
share one prompt and run as one batch. That is an engineering win, not a scientific one.

## What they answered on corrupted clues

| arm | clean hit | blend | obeyed the false letter | other miss |
|---|---|---|---|---|
| one_shot | 31 | 0 | **25** | 4 |
| fixed | 36 | 1 | 20 | 3 |
| dropout | 45 | 1 | 9 | 5 |
| breathe_text | 46 | 0 | 10 | 4 |
| breathe_attn | 45 | 0 | 13 | 2 |

- **Obeying the lie is the dominant failure.** One-shot Qwen answered with a word
  starting with the false letter in 25 of 60 items: Basal Ganglia, Amygdala, Dendrite,
  Enzyme, Kepler, Cortisol, Gamma waves, Pineal gland, and so on. Leaving clues out cuts
  that to 9–13.
- **Blends exist but are rare.** "Je déjà vu" (false J, `fixed`) and "Hodge Laplacian"
  (false H, `dropout`) keep the right noun and bolt a false-letter word onto it. That is
  what "Bilateral striatum" did in the chat demo. In the benchmark itself, striatum came
  out clean ("Striatum"), and the loop set the B-clue's trust down to 0.32.
- **Some misses are defensible.** I judged these by hand, and they are not counted above.
  "Soroban" or "Suanpan" for abacus, "Omnipresent" for ubiquitous and "Biological clock"
  for circadian rhythm all fit every true clue *and* the false letter. In those items the
  true clues simply don't rule out the synonym, so following the letter is reasonable.
  Pompeii → Herculaneum is nearly the same case, since both were buried by Vesuvius in
  79 AD. These are weaknesses in the dataset, not in the method.

## The one thing the loop adds: it knows when it is right

`dropout` gives an answer. The loop gives an answer plus which clue it stopped trusting.
That flag turns out to be well calibrated:

| breathe_attn, corrupt condition | items | answer correct |
|---|---|---|
| named the false clue | 26 | **0.92** |
| named no clue | 29 | 0.72 |
| named a TRUE clue | 5 | **0.00** |

`breathe_text` looks the same: 28 items at 0.89, 28 at 0.71 and 4 at 0.25. False alarms on
clean items: 3%.

So when it says "I set aside *I think it starts with B*", it is right about nine times in
ten. When it blames a clue that was actually true, the answer is wrong. That is useful to
a user even though it doesn't raise accuracy.

It isn't proven that the loop is needed for this. The same flag could probably be read
off `dropout` for free: whichever clue's removal produced the winning answer. That is
the next thing to test.

## Where it lost to dropout (strict, paired)

- `breathe_attn` beat dropout on laplacian and schadenfreude, and lost on allotrope and
  insulin.
- `breathe_text` beat dropout on laplacian, schadenfreude and vaccine, and lost on insulin
  and adrenaline.

In every loss, the trust loop turned against a *true* clue first. For insulin, the
distrusted clue was "made by beta cells in the pancreas", and the loop answered Glucose.
Once a true clue loses trust, the loop's own attention stops listening to it. The loop
can amplify its first mistake, which a one-round method can't.

## Honest caveats

- **One run, one model, one seed, 60 items.** The standard errors are about ±0.06.
- **The insert design changes two things at once:** the clue count (3 → 4) and the
  corruption. Sol pointed this out, and it's fair. `bench.py --design replace` now
  compares 3 true + a **true** 4th against 3 true + the **false** 4th, in the same slot.
  For letter items the true 4th is the correct first letter, so it also tests whether a
  true letter clue is kept.
- **"Breathing" here means generation temperature plus clue suppression.** It is *not*
  an oscillating attention temperature. Sol also pointed this out, and it's correct. The
  belief sharpness β was fixed at 6, so the temperature-breathing idea from the
  conversation hasn't been tested.
- **About pre-registration:** the kill rule was in the README of the first version
  delivered, before any real run, and its wording never changed. The code changed once
  between the first chat demo and this benchmark: residue became relative to each clue's
  best fit, and the checker question was reworded, after the demo showed the checker
  rejecting a true clue for every candidate. The GitHub history can't show that order,
  because the repo was uploaded after the runs. The conversation it was built in can.

## Next: residue_search

Sol pointed out that Sihti's residue is what a purification step *removed*, kept intact.
My loop's "residue" was a fit deficit used to *suppress* clues. Every loss to dropout above
comes from that conflation. `residue_search` keeps every clue at full weight and feeds the
unexplained clues back into the next search. Its kill rule is in the README, written
before it has run.
