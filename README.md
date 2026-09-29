# Breathing Qwen

**Can a frozen language model recover the right answer when one remembered cue is confidently wrong?**

Breathing Qwen is a small inference-time experiment around frozen **Qwen3-8B**. It studies a common failure mode: several clues point toward the right concept, but one vivid false cue — often a wrong first letter — pulls generation toward a plausible but incorrect answer.

No training is performed and the model weights are unchanged. The repository compares several ways of searching the same frozen model under contradictory evidence, from ordinary one-shot generation to leave-one-clue-out search, iterative trust updates, attention steering, and a newer residue-driven search that preserves unexplained evidence instead of suppressing it.

> **Current result:** the complicated parts did not win. On a 60-item Qwen3-8B benchmark, strict accuracy on corrupted clues rose from **52% for one-shot generation to 75% for simple leave-one-clue-out search plus checking**. Three more elaborate methods (an iterative trust loop, attention steering, and a residue-driven search) were each tested against that simple control under a rule stated before the run, and none beat it. Full receipts and caveats are in **[RESULTS.md](RESULTS.md)**.

## The problem

Consider a tip-of-the-tongue query such as:

```text
a brain structure involved in interval timing and habit learning;
the main input nucleus of the basal ganglia;
loses much of its dopamine input in Parkinson's disease;
I think it starts with B
```

The first three clues point to **striatum**. The last clue is false. In the benchmark, plain Qwen3-8B answered **Basal Ganglia** on this corrupted version while counterfactual search recovered **Striatum**.

The motivating question is not whether the model knows the fact in its parameters. It often does. The question is whether a single contradictory cue can block access to that knowledge under one conditioning state, and whether nearby counterfactual queries can recover it.

## What survived

The strongest result so far is also the simplest:

1. Ask the model with all clues.
2. Ask again with clue 1 left out.
3. Ask again with clue 2 left out.
4. Continue for each clue.
5. Check the resulting candidates against the clues.
6. Prefer the candidate that leaves the least unexplained evidence.

This `dropout` / leave-one-clue-out control substantially reduced the model's tendency to obey a false first-letter cue.

### Run 1: insert design

Run 1 used 60 items. The clean condition had three true clues. The corrupt condition added one false clue.

| arm | clean | corrupt, strict |
|---|---:|---:|
| `one_shot` | 0.95 | 0.52 |
| `fixed` — sample + check | 0.95 | 0.60 |
| `dropout` — leave one clue out + check | 0.93 | **0.75** |
| `breathe_text` — iterative trust via text deletion | 0.93 | 0.77 |
| `breathe_attn` — iterative trust via attention bias | 0.95 | 0.75 |

The predeclared rule required `breathe_attn` to beat both compute-matched controls. It did not. The trust loop was therefore **killed by its own criterion**. Attention precision was also not shown: attenuating a clue inside attention did no better than deleting it from the prompt.

The dominant one-shot failure was simple obedience to the false cue. On 25 of 60 corrupted items, one-shot Qwen answered with a word beginning with the false letter. Leave-one-out search reduced that to 9–13 cases depending on the arm.

See **[RESULTS.md](RESULTS.md)** for paired intervals, per-item outcomes, raw-result paths, and failure analysis.

## Run 2: replacement design

Run 1 changed both clue truth and clue count. Run 2 fixed that confound.

The base condition became:

```text
3 true clues + 1 true fourth clue
```

and the corrupt condition became:

```text
the same 3 true clues + 1 false fourth clue in the same slot
```

For wrong-letter items, the true fourth clue is the correct first letter. This tests whether the methods merely learn to distrust letter clues in general.

Results:

| arm | true fourth clue | false fourth clue, strict |
|---|---:|---:|
| `one_shot` | 0.97 | 0.52 |
| `fixed` | 0.97 | 0.58 |
| `dropout` | 0.98 | 0.75 |
| `breathe_text` | 0.98 | 0.77 |
| `breathe_attn` | **1.00** | 0.77 |

Swapping one true clue for one false clue cost plain Qwen **45 percentage points**. Leave-one-out search recovered roughly half of that loss. The trust loop still did not beat the simpler control, and true letter clues were generally retained.

Run 2 is **not** an independent second sample of the corrupted items: it used the same corrupted prompts, slots, model, and seed. It mainly validates the cleaner base condition and shows that the result was not caused merely by adding a fourth clue.

## Run 3: residue search

Run 3 repeated the replacement design with one additional arm, `residue_search` (described below). The six earlier arms reproduced run 2 almost exactly.

| arm | true fourth clue | false fourth clue, strict | generations per item | seconds per item |
|---|---:|---:|---:|---:|
| `dropout` | 0.97 | **0.75** | 30.5 | 11.0 |
| `breathe_attn` | 0.98 | 0.75 | 33.1 | 7.7 |
| `residue_search` | 0.97 | 0.72 | **12.9** | **3.8** |

`residue_search` did not beat leave-one-out search (difference −0.033, paired 95% interval −0.133 to +0.050), so it fails its predeclared rule.

The main cause was its stopping rule. It was allowed the same generation budget as the other search arms but used only about 40% of it, because it stopped as soon as its best answer stayed the same for two rounds. In 13 of the 60 corrupted items it stopped while its best answer still failed to explain one or more *true* clues — for example answering *Geoffrey Hinton* for a John Hopfield item and *Naloxone* for a placebo item.

Two observations are worth keeping:

- **It fails on different items than leave-one-out.** It fixed three items where leave-one-out produced near-misses (*Suzhou compass*, *Serendip*, *Autovaccine*), while losing others. Counting an item as solved if either method solved it gives 48 of 60, compared with 45 for leave-one-out alone. This is an upper bound, not a result, but it suggests testing a combination.
- **It was about three times cheaper** per item than leave-one-out, for a three-point lower accuracy.

The raw output is in `results/qwen3-8b-4bit_run3_residue.jsonl`.

## The original breathing loop

The original mechanism maintained one trust value per clue.

Each cycle had two stages:

- **Open / inhale:** temporarily reduce one clue's authority and sample alternative answers.
- **Commit / exhale:** generate a greedy answer under the current clue trusts.

Every candidate was then checked against every clue with a Yes/No compatibility query. A clue that disagreed with the current candidate pool accumulated residue and lost trust through a robust Cauchy-style update.

In attention mode, clue trust is injected into Qwen's attention logits as

```text
gain * log(trust_j)
```

for tokens belonging to clue `j`. A clue with trust 1 is unchanged; lower trust reduces its influence without changing model weights or prompt text.

This mechanism worked mechanically, but the benchmark showed that its iterative trust update did not improve accuracy over one-round leave-one-out search. In its clearest failures, the loop distrusted a **true** clue first and then amplified its own mistake.

That negative result is kept rather than tuned away.

## Residue search: unexplained is not the same as wrong

The first two runs exposed a conceptual problem in the trust loop:

> A clue that the current hypothesis fails to explain is not necessarily a false clue.

`residue_search` is a post-run-1 experiment built around that distinction. It never lowers clue trust. Instead:

1. Generate candidate answers while keeping all clues available.
2. Identify which clues the current best answer leaves unexplained.
3. Feed those unexplained clues back into the next search as explicit pressure for a better candidate.
4. Preserve the residue instead of suppressing the evidence that produced it.

Only after search settles can a still-unexplained clue be reported as a likely bad cue.

Its predeclared rule is simple: `residue_search` survives only if it beats `dropout` on corrupted items with a paired 95% interval above zero while losing no more than 5 points on the base condition.

**Result:** it did not survive. See *Run 3* above. The tested version stopped searching too early. The idea of letting unexplained evidence drive further search, run with its full budget or combined with leave-one-out candidates, has not been tested yet.

## A note on the name

The repository began from an intuition about alternating broad search and sharp commitment — a system that “breathes” rather than making one irreversible retrieval decision.

The current benchmark does **not** test a globally oscillating attention temperature. The belief sharpness parameter is fixed, while the implemented loop varies generation sampling and clue authority. The name remains as project history; the stronger temperature-breathing hypothesis is still untested here.

## Reproduce the experiments

Install dependencies:

```bash
pip install -r requirements.txt
```

For a 4-bit Qwen3-8B run, install `bitsandbytes` as well:

```bash
python -m pip install -U bitsandbytes
```

Interactive demo:

```bash
python chat.py --demo --load-4bit
python chat.py --load-4bit
```

Original insert-design benchmark:

```bash
python bench.py --load-4bit
```

Cleaner replacement design:

```bash
python bench.py --load-4bit --design replace --out results_replace.jsonl
```

Strictly re-score a saved run:

```bash
python analyze.py results/qwen3-8b-4bit_run1.jsonl
```

Raw outputs of the published runs:

| file | design | arms |
|---|---|---|
| `results/qwen3-8b-4bit_run1.jsonl` | insert | five arms |
| `results/qwen3-8b-4bit_run2_replace.jsonl` | replace | five arms |
| `results/qwen3-8b-4bit_run3_residue.jsonl` | replace | six arms, including `residue_search` |

The optional Qwen3 thinking-mode control is available with `--think`, but it was not run in the published first two experiments.

Qwen3-8B in bf16 requires substantially more VRAM than the 4-bit path. Smaller Qwen3 models can also be selected with `--model`, but no cross-size result is claimed in this repository yet.

## Experimental arms

| arm | purpose |
|---|---|
| `one_shot` | ordinary greedy answer |
| `fixed` | repeated sampling + the same candidate checker, with all clues always active |
| `dropout` | leave each clue out in turn, then check/select candidates |
| `breathe_text` | iterative trust loop; low-trust clues are removed from the text |
| `breathe_attn` | iterative trust loop; low trust becomes an attention-logit bias |
| `residue_search` | preserve all clues; use currently unexplained clues to drive the next search |
| `think` | optional Qwen3 thinking-mode baseline |

The controls are important. The project is specifically designed to distinguish a complicated iterative mechanism from simpler sampling, leave-one-out search, and verification.

## Motivation

The project was motivated by a real tip-of-the-tongue failure: most semantic details of a concept were available, but one remembered first-letter cue was wrong. That kind of query is useful experimentally because the model may already contain the target knowledge while the false cue changes which part of that knowledge becomes accessible.

The project therefore focuses on **retrieval under contradictory evidence**, not general question answering.

## Related ideas and claim boundary

Several ingredients are established techniques rather than new inventions:

- Robust reweighting such as Huber/Cauchy-style IRLS is longstanding.
- Self-consistency and sample-and-select methods are established inference techniques.
- Leave-one-feature / leave-one-cue-out analysis is a standard robustness idea.
- Attention steering on selected spans has prior work, including methods such as PASTA.
- Candidate verification with a language model is common.
- Tip-of-the-tongue blocking, spreading activation, and resonance/reset ideas have long histories in cognitive science.

What this repository contributes is a controlled comparison of these ingredients on a deliberately corrupted-cue retrieval task, together with negative results that rule out some initially attractive mechanisms.

Supported by the current runs:

- one false clue can sharply reduce one-shot retrieval accuracy on this benchmark;
- leave-one-clue-out search recovers a substantial fraction of that loss;
- the tested iterative trust loop does not beat the simpler leave-one-out control;
- the tested attention-bias implementation does not beat text deletion;
- the loop's “distrusted clue” flag is informative, although it has not yet been shown to require the loop;
- the tested `residue_search` does not beat leave-one-out search, mainly because it stops too early.

Not supported yet:

- that rhythmic or oscillating attention temperature improves retrieval;
- that `residue_search` beats leave-one-out search (run 3: it did not);
- that combining leave-one-out candidates with residue-driven follow-up helps (not yet tested);
- that smaller models with extra search can replace larger models;
- that the mechanism explains biological memory or cortical rhythms.

## Known limitations

- The benchmark is small: one model, one main seed, 60 items.
- Most corruptions are wrong first-letter cues; false-detail and wrong-category cases are fewer.
- The checker is itself Qwen3-8B, so checker errors can affect search.
- Some targets have defensible synonyms not covered by the answer key.
- The first insert-design run changed clue count as well as clue truth; run 2 addresses this with the replacement design.
- The three runs share the same corrupted prompts and seed, so they are not independent samples of the corrupted condition.
- The historical preregistration sequence is documented in `RESULTS.md`; the repository was uploaded after some early interactive development, so GitHub timestamps alone do not establish the full ordering.

## Repository layout

```text
breathe/engine.py        retrieval loops, controls and trace logic
breathe/qwen_backend.py  Qwen3 generation, attention bias and clue compatibility checks
chat.py                  interactive demo
bench.py                 benchmark runner (--design insert | replace)
analyze.py               strict re-scoring and failure analysis
RESULTS.md               detailed experimental record and verdicts
results/                 raw outputs from real runs
data/tot_items.jsonl     60 benchmark items and corrupted/true fourth clues
tests/                   mechanics, mock-world logic and end-to-end wiring
```

CPU tests:

```bash
pip install pytest tokenizers
python -m pytest -q tests
```

## License

MIT
