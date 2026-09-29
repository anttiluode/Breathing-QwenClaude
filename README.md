# Breathing Qwen

A frozen Qwen3-8B wrapped in a loop that recalls the way you found *striatum*:
hold everything you know, let go of one clue at a time, and slowly stop trusting
the clue that keeps disagreeing with the rest.

No training. No change to the weights. The only new part is outside the model:
one trust number per clue, fed back into Qwen's attention.

> **Status: built and wired, not yet tested on Qwen3-8B.** Everything below was
> checked on a tiny random Qwen3 and in a hand-built mock world (19 tests pass).
> Whether it helps a real model is exactly what `bench.py` is for, and it can
> come back negative.

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

## How it will talk

The chat prints each breath: what came up on the inhale, what it committed to,
the belief over candidates, and a trust bar per clue. Then it gives a normal
reply. The settled result is passed to Qwen as a note, and Qwen phrases it with
the same attention precision still applied. When a clue was set aside, it is
told to say so. The reply should sound like: *you said it starts with B, but
everything else points to the striatum, so I set the B aside.*

That is a design intent, not a recorded output; nothing here has run on Qwen3-8B
yet. The trace format below is real output from the **mock world** in
`tests/test_engine_mock.py`, where the "model" is a hand-written table:

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

**The rule, fixed before any run:**

- **Breathing survives** only if `breathe_attn` beats **both** `fixed` and
  `dropout` on corrupted items, with a paired 95% bootstrap interval above zero,
  and costs no more than 5 points against `one_shot` on clean items.
  Otherwise, the loop is decoration: sampling plus checking already does the work.
- **Attention precision survives** only if `breathe_attn − breathe_text` is
  above zero under the same test. Otherwise, deleting the clue is just as good.

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
bench.py                 the kill test
data/tot_items.jsonl     60 items, true clues + one corrupted clue each
tests/                   mechanics on a tiny random Qwen3, logic in a mock world, end-to-end wiring
```

Tests: `pip install pytest tokenizers && python -m pytest -q tests` (CPU, about
10 seconds, no downloads).
