# Breathing Qwen

A frozen Qwen3-8B wrapped in a loop that recalls the way you found *striatum*:
hold everything you know, let go of one clue at a time, and slowly stop trusting
the clue that keeps disagreeing with the rest. (I remembered a part of brain that 
deals with time starts with b. But that was false memory. I then asked ai's 
which part of brain deals with time similar to m septum. They were not able to 
give me the answer. (Claude, gemini, chatgpt). This tiny model was able to 
remember it) 

No training. No change to the weights. The only new part is outside the model:
one trust number per clue, fed back into Qwen's attention.

> **Status: run on Qwen3-8B, and the pre-registered test killed the loop.** Leaving each
> clue out in turn and checking the answers against the clues lifts accuracy on corrupted
> clues from 52% to 75%, with no cost on clean items. The iterative trust loop and the
> attention precision added nothing beyond that. What the loop does add is a
> well-calibrated flag: when it names a clue as the false one, it is right about nine
> times in ten. Full numbers are in **[RESULTS.md](RESULTS.md)**.

## What it does

You give it clues, separated by `;`:

```
the part of the brain that deals with time; it has to do with dopamine and habits;
we talked about it together with the medial septum; I think it starts with b
```

It breathes over them several times. Each breath has two phases:

- **Inhale (let go).** For each clue in turn, it drops that one clue's authority
  almost to zero and samples a few answers at high temperature. This shows what
  the other clues point to when any single clue is set aside, so an answer
  blocked by a wrong clue can surface.
- **Exhale (commit).** It gives one greedy answer with every clue at its current
  trust.

After each breath it checks every answer against every clue: it asks Qwen
"is this clue true of this answer?" and reads the Yes/No probabilities. The part
of a clue that an answer fails to explain is that clue's **residue**. The answer
with the least trusted residue wins. A clue that keeps leaving residue under the
winning answer loses trust. It is never deleted, and it can win trust back.

It stops when the answer has held for two breaths and no clue's trust is still
moving. Then it answers in its own words. For comparison, plain Qwen answers the
same message with no loop.

## How it talks

The chat prints each breath: what came up on the inhale, what it committed to,
the belief over candidates, and a trust bar per clue. Then it gives a normal
reply. The settled result is passed to Qwen as a note, and Qwen phrases it with
the same attention precision still applied. When a clue was set aside, it is
told to say so. The reply should sound like: *you said it starts with B, but
everything else points to the striatum, so I set the B aside.*

On the real striatum demo, Qwen3-8B found "Bilateral striatum". That is a blend: it kept
the right noun and bolted on a B-word to satisfy the false clue, so no clue lost trust.
The demo also prints the checker's fit table, so you can see exactly where each Yes and
No came from. The trace below is from the **mock world** in `tests/test_engine_mock.py`,
where the "model" is a hand-written table:

```
── breath 1 ────────────────────────────────────────────
  inhale (letting go of one clue at a time): biorhythm · striatum · brainstem · brodmann area
  exhale: basal ganglia
  belief:
    ███████████· 0.94  striatum
    █··········· 0.06  basal ganglia
  trust in each clue:
    ███████████· 0.95  involved in interval timing               residue 0.15
    ███████████· 0.88  main input nucleus of the basal ganglia   residue 0.23
    ████████████ 1.00  rich in dopamine receptors                residue 0.09
    ███████····· 0.61  I think it starts with B                  residue 0.75
...
── breath 5 ────────────────────────────────────────────
  exhale: striatum
    ████████████ 1.00  striatum
    ██·········· 0.15  I think it starts with B                  residue 0.99
```

## Run it

```bash
pip install -r requirements.txt        # plus bitsandbytes for --load-4bit
python chat.py --demo                  # the striatum morning, once
python chat.py                         # talk to it
python bench.py                        # the kill test (see below)
```

Qwen3-8B in bf16 needs about 17 GB of VRAM. On a 12 GB card, use `--load-4bit`.
`--model Qwen/Qwen3-4B` or `Qwen/Qwen3-1.7B` work too, and the small-model
question ("can a smaller model that breathes reach what a bigger one gets in one
shot?") is a good second run.

Chat commands: `/trace` (show or hide the breaths), `/plain` (show or hide plain
Qwen), `/gain X` (attention trust gain; 0 switches to text mode), `/quit`.

## How trust reaches the model

`breathe/qwen_backend.py` registers an attention function, `trust_sdpa`. It adds

    gain × log(trust_j)

to the attention logits of every token belonging to clue *j*, in every layer and
head. A clue at trust 1 is untouched. A clue at trust 0.1 is attended to roughly
10× less. The prompt and the weights stay exactly the same. With no trust set,
the function hands straight back to stock SDPA: `tests/test_mechanics.py` checks
that the output is bit-identical to the unmodified model in that case. It also
checks that a very large negative bias equals masking those tokens, and that
cached decoding matches a full forward pass under bias.

All the "let go of clue k" settings share the same prompt tokens and differ only
in their per-row bias, so one inhale is a single batched call.

## The kill test

`bench.py` runs 60 tip-of-the-tongue items (`data/tot_items.jsonl`). Each item
has three true clues and one corrupted clue: 49 wrong first letters, 9 false
details and 2 wrong categories. Every item runs twice, **clean** (true clues
only) and **corrupt** (the bad clue inserted at a random position). Six arms
share one frozen model:

| arm | what it is |
|---|---|
| `one_shot` | greedy answer |
| `think` | Qwen3 thinking mode, 768-token budget (`--think`, slow) |
| `fixed` | self-consistency: same number of samples as `breathe_attn`, same residue-based selection, every clue always at full trust |
| `dropout` | leave-one-clue-out sampling by deleting clues from the text, same selection, compute-matched, no trust loop |
| `breathe_text` | the full loop, with trust applied by deleting low-trust clues |
| `breathe_attn` | the full loop, with trust applied inside attention |
| `residue_search` | added after run 1 (Sol's Sihti-style idea): no clue ever loses trust; the clues the current best answer leaves unexplained are shown back to the model to drive the next search; compute-matched |

**The rule, written before any real run** (see RESULTS.md on what the history can and
cannot show):

- **Breathing survives** only if `breathe_attn` beats **both** `fixed` and
  `dropout` on corrupted items, with a paired 95% bootstrap interval above zero,
  and costs no more than 5 points against `one_shot` on clean items.
  Otherwise, the loop is decoration: sampling plus checking already does the work.
- **Attention precision survives** only if `breathe_attn − breathe_text` is
  above zero under the same test. Otherwise, deleting the clue is just as good.

**Run 1 result: both killed.** `breathe_attn − dropout` = −0.017 [−0.067, +0.033] on
corrupted clues. Details are in [RESULTS.md](RESULTS.md).

Options added after run 1:

```bash
python bench.py --load-4bit --design replace      # 3 true + TRUE 4th  vs  3 true + FALSE 4th
python analyze.py results/qwen3-8b-4bit_run1.jsonl   # strict re-score: clean / blend / obeyed / other
```

**`residue_search` and its rule.** Run 1 lost to dropout wherever the trust loop decided
a *true* clue didn't fit its current guess and stopped listening to it (insulin →
Glucose). The lesson: *unexplained is not the same as wrong*. So `residue_search` never
lowers a clue's weight. Each round it takes the best answer, lists the clues that answer
leaves unexplained, and asks again with "this guess does not explain: …" added to the
full clue list. It stops when nothing is left unexplained or the best answer holds. Only
then is the clue still left over reported as the likely false one: after the answer, not
before.

Rule, written before its first run: it survives only if `residue_search − dropout` on
corrupted clues has a paired 95% interval above zero, and it costs no more than 5 points
on base items. bench.py prints this verdict too.

bench.py now prints strict accuracy next to the lenient, pre-registered one. Strict means
"Bilateral striatum" is no longer counted as striatum.

The bench also reports how often the loop flagged the corrupted clue, and how
often it flagged a clue on clean items (false alarms). A rough runtime guess is
under an hour on a 24 GB GPU without `--think`. That is not measured.

## What is not new

Please say this before anyone else does:

- **Iteratively reweighted least squares** and robust weights (Cauchy/Huber)
  are decades old; the trust update is one of them.
- **Self-consistency** (sample many, pick the consistent answer; Wang et al.
  2022) is the `fixed` arm.
- **Attention steering on chosen spans** already exists: PASTA (Zhang et al.
  2023, "Tell Your Model Where to Attend") re-weights attention on
  user-specified text. What differs here is only that the weights are set by
  residue, in a loop.
- Leave-one-cue-out prompting and answer verification are both common.
- The cognitive picture (tip-of-the-tongue blockers, incubation, spreading
  activation, Grossberg's resonance and reset) is established psychology.

What this repo adds is the specific combination: residue-driven trust fed back
into attention, cycled with a let-go/commit rhythm, and a kill test designed so
that the combination has to beat each of its parts.

## Known limits

- It is built for answers that are a short name or term, not open conversation.
- Qwen checks each clue with a Yes/No. Clues it cannot verify (for example "we
  talked about it together") will leave residue and lose trust even when true.
- Two clues that contradict each other can accuse each other. The loop then
  relies on the remaining clues to break the tie, and with only three clues
  that tie-break can be weak.
- Residue is bounded (at most 1 per clue) on purpose. Summing log-probabilities
  would let one flatly violated false clue outvote every true one
  (`test_log_scoring_lets_one_cue_veto`). The price is that a true and very
  specific clue gets no more than one vote either.

## Files

```
breathe/engine.py        the loop, both control arms, trace printing (no torch)
breathe/qwen_backend.py  trust_sdpa attention hook, Qwen3 generation, Yes/No fit checks
chat.py                  interactive chat and --demo
bench.py                 the kill test (--design insert | replace)
analyze.py               strict re-scoring of a results file
RESULTS.md               run 1: numbers, failures, verdict
results/                 raw results from real runs
data/tot_items.jsonl     60 items, true clues + one corrupted clue each
tests/                   mechanics on a tiny random Qwen3, logic in a mock world, end-to-end wiring
```

Tests: `pip install pytest tokenizers && python -m pytest -q tests` (CPU, about
10 seconds, no downloads).
