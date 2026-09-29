"""
The breathing retrieval loop — backend-agnostic.

State that persists across beats (the resident object):
    cues      the pieces of the question, fixed for the whole search
    trust     one weight per cue, in (0, 1]; starts at 1
    pool      every candidate answer proposed so far

One beat (one breath):
    inhale    let go of one cue at a time: for each cue k, sample a few
              candidates at HIGH temperature with cue k's authority dropped to
              the floor (and every other cue at its current trust).  This is the
              "relaxing" — it shows what the rest of the clues point at when any
              single clue is set aside, so a blocked answer can surface.
    exhale    one greedy candidate at temperature 0 under the current trust.
              Commit.
    residue   for every (cue, candidate) pair, ask the model whether the cue
              fits the candidate -> fit f in [0,1]; residue r = (clue's best fit
              in the pool) - f.
    select    belief over candidates = softmax(-beta * sum_j trust_j * r_jc):
              the candidate that leaves the least trusted residue wins.
              Residue is bounded (<= 1 per cue), so no single cue can veto.
    re-trust  each cue's residue under that belief, R_j = sum_c p(c) r_jc, and
              trust moves toward a Cauchy weight 1 / (1 + (R_j / kappa)^2),
              then is rescaled so the most trusted cue has trust 1 (precision
              is relative: if every cue fits badly, none is singled out).
              Iteratively reweighted least squares with a robust weight: a cue
              that keeps disagreeing with the settled picture loses authority,
              but it is never deleted and can win trust back.

Two ways to apply trust to generation:
    attention  (use_attention=True)  trust is added as log-trust to the
               attention logits of the cue's tokens (see qwen_backend.py).
               Dropping a cue = pushing its trust to the floor.
    text       (use_attention=False) dropping a cue deletes it from the prompt;
               the exhale uses only cues with trust >= text_keep.

Stop when the same top candidate has held for `settle_beats` beats with belief
>= `settle_p` AND no cue's trust moved more than `settle_move` on the last beat
(while a cue is still losing authority the search is not finished), or after
`beats` beats.

The backend must provide:
    propose(cues, trust, n, temperature) -> list[str]     trust=None: stock model
    compat(cues, candidates)             -> np.ndarray [len(cues), len(candidates)] fits in [0,1]
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, asdict

import numpy as np


# ----------------------------------------------------------------------------- text utils

_ARTICLES = re.compile(r"^(the|a|an)\s+")


def normalize(ans: str) -> str:
    """Lowercase, strip punctuation/quotes/articles, collapse spaces."""
    s = ans.strip().lower()
    s = s.replace("’", "'").replace("é", "e").replace("è", "e").replace("à", "a")
    s = re.sub(r"[\"“”`*_]", "", s)
    s = re.sub(r"[^\w\s'\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _ARTICLES.sub("", s)
    return s.strip(" -'")


def clean_answer(text: str, max_words: int = 7) -> str:
    """First line of a generation, minus 'Answer:' style prefixes, capped in length."""
    line = text.strip().split("\n")[0]
    line = re.sub(r"^(answer|term|it is|it's|probably|maybe|i think it'?s|you mean)\s*[:\-]?\s*", "",
                  line, flags=re.I)
    line = line.strip().strip("\"'“”*.").strip()
    return " ".join(line.split()[:max_words])


def split_cues(message: str) -> list[str]:
    """Split a free-text message into cues on newlines, ';' and sentence ends."""
    parts = re.split(r"[;\n]+|(?<=[.!?])\s+", message)
    cues = [p.strip(" .,-") for p in parts]
    return [c for c in cues if len(c) >= 2]


# ----------------------------------------------------------------------------- config / trace

@dataclass
class BreathConfig:
    beats: int = 5              # maximum breaths
    per_drop: int = 2           # inhale samples per let-go cue
    temp_open: float = 1.0      # inhale temperature (mixing)
    beta: float = 6.0           # belief sharpness over candidates
    kappa: float = 0.35         # residue scale in the Cauchy trust weight
    smooth: float = 0.5         # how far trust moves toward its target each beat
    trust_floor: float = 0.05   # trust never goes below this
    drop_trust: float = 0.02    # authority of the cue being let go on an inhale
    use_attention: bool = True  # False = text mode (delete cues instead of attenuating them)
    text_keep: float = 0.5      # text mode: exhale keeps cues with at least this trust
    settle_beats: int = 2       # stop once the top candidate held this many beats...
    settle_p: float = 0.5       # ...with at least this belief...
    settle_move: float = 0.05   # ...and no cue's trust moved more than this
    pool_max: int = 16          # keep at most this many candidates (lowest score dropped)


@dataclass
class Beat:
    t: int
    inhale: list[str]
    exhale: str
    top: str
    belief: dict[str, float]
    trust: list[float]
    residue: list[float]


@dataclass
class BreathResult:
    answer: str
    belief: float
    cues: list[str]
    trust: list[float]
    beats: list[Beat] = field(default_factory=list)
    settled: bool = False
    seconds: float = 0.0
    generations: int = 0
    fit_queries: int = 0
    fit_table: dict = field(default_factory=dict)   # candidate -> fit per clue (last pool)

    @property
    def distrusted(self) -> tuple[int, str, float] | None:
        """Most-distrusted cue, if any cue lost meaningful trust."""
        if not self.trust:
            return None
        j = int(np.argmin(self.trust))
        if self.trust[j] >= 0.6:
            return None
        return j, self.cues[j], self.trust[j]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["distrusted"] = self.distrusted
        return d


# ----------------------------------------------------------------------------- scoring

class _FitCache:
    """Caches cue x candidate fits so each pair is asked once."""

    def __init__(self, backend, cues):
        self.backend, self.cues = backend, cues
        self.fit: dict[str, np.ndarray] = {}
        self.queries = 0

    def matrix(self, cands: list[str]) -> np.ndarray:
        missing = [c for c in cands if c not in self.fit]
        if missing:
            F = np.asarray(self.backend.compat(self.cues, missing), dtype=np.float64)
            self.queries += F.size
            for k, c in enumerate(missing):
                self.fit[c] = F[:, k]
        return np.stack([self.fit[c] for c in cands], axis=1) if cands else np.zeros((len(self.cues), 0))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def belief_and_residue(F: np.ndarray, trust: np.ndarray, beta: float):
    """Belief over candidates and each cue's residue under it.

    A candidate's score is minus its trust-weighted residue, -sum_j w_j (1 - f_jc).
    Bounded residue means no single cue can veto a candidate.  The product-of-
    probabilities alternative (sum of log f) assumes every cue is true — exactly the
    assumption this machine exists to drop — and under it one flatly violated false
    cue outweighs all the true ones together.

    Residue is RELATIVE to the clue's best fit in the pool: r_jc = max_c' f_jc' - f_jc.
    A clue the checker rejects for every candidate (a strict or unverifiable clue, e.g.
    "the part of the brain that deals with time", or "we talked about it") is then
    uninformative rather than wrong, and keeps its trust.  A clue is only accused when
    it prefers some candidates over the ones the other clues settle on."""
    F = np.clip(F, 0.0, 1.0)
    Rm = F.max(1, keepdims=True) - F                    # [J, C]
    score = -(trust[:, None] * Rm).sum(0)               # [C]
    belief = _softmax(beta * score)
    R = (Rm * belief[None, :]).sum(1)                   # [J]
    return score, belief, R


# ----------------------------------------------------------------------------- the loop

def _inhale(backend, cues, trust, cfg: BreathConfig) -> list[str]:
    J = len(cues)
    if J == 1:
        att = trust if cfg.use_attention else None
        return backend.propose(cues, att, n=cfg.per_drop, temperature=cfg.temp_open)
    if cfg.use_attention:
        rows = []
        for k in range(J):
            w = trust.copy()
            w[k] = cfg.drop_trust
            rows.append(w)
        if hasattr(backend, "propose_multi"):   # all let-go settings in one batch
            return [a for grp in backend.propose_multi(cues, rows, cfg.per_drop, cfg.temp_open) for a in grp]
        return [a for w in rows for a in backend.propose(cues, w, n=cfg.per_drop, temperature=cfg.temp_open)]
    out = []
    for k in range(J):
        out += backend.propose([c for i, c in enumerate(cues) if i != k], None,
                               n=cfg.per_drop, temperature=cfg.temp_open)
    return out


def _exhale(backend, cues, trust, cfg: BreathConfig) -> str:
    if cfg.use_attention:
        return backend.propose(cues, trust, n=1, temperature=0.0)[0]
    kept = [c for c, w in zip(cues, trust) if w >= cfg.text_keep] or list(cues)
    return backend.propose(kept, None, n=1, temperature=0.0)[0]


def breathe(backend, cues: list[str], cfg: BreathConfig | None = None, on_beat=None) -> BreathResult:
    cfg = cfg or BreathConfig()
    t0 = time.time()
    J = len(cues)
    trust = np.ones(J)
    fits = _FitCache(backend, cues)
    pool: dict[str, str] = {}          # normalized -> display form
    beats: list[Beat] = []
    gens, held, last_top, settled = 0, 0, None, False

    for t in range(cfg.beats):
        inhale = _inhale(backend, cues, trust, cfg)
        exhale = _exhale(backend, cues, trust, cfg)
        gens += len(inhale) + 1
        for a in inhale + [exhale]:
            k = normalize(a)
            if k and k not in pool:
                pool[k] = a.strip()

        keys = list(pool)
        F = fits.matrix(keys)
        score, belief, R = belief_and_residue(F, trust, cfg.beta)

        moved = 0.0
        if J > 1:  # nothing to compare a lone cue against
            target = 1.0 / (1.0 + (R / cfg.kappa) ** 2)
            new = (1 - cfg.smooth) * trust + cfg.smooth * target
            new = np.clip(new / new.max(), cfg.trust_floor, 1.0)   # trust is relative: best cue = 1
            moved = float(np.abs(new - trust).max())
            trust = new

        if len(keys) > cfg.pool_max:  # keep the best, and this beat's commit
            keep = set(np.argsort(-score)[: cfg.pool_max - 1].tolist())
            ek = normalize(exhale)
            if ek in pool:
                keep.add(keys.index(ek))
            pool = {keys[i]: pool[keys[i]] for i in sorted(keep)}
            keys = list(pool)
            F = fits.matrix(keys)

        _, belief, _ = belief_and_residue(F, trust, cfg.beta)   # belief after re-trust
        order = np.argsort(-belief)
        top = keys[order[0]]
        beat = Beat(t=t + 1, inhale=inhale, exhale=exhale, top=pool[top],
                    belief={pool[keys[i]]: float(belief[i]) for i in order[:6]},
                    trust=[float(x) for x in trust], residue=[float(x) for x in R])
        beats.append(beat)
        if on_beat:
            on_beat(beat, cues)

        held = held + 1 if top == last_top else 1
        last_top = top
        if held >= cfg.settle_beats and belief[order[0]] >= cfg.settle_p and moved < cfg.settle_move:
            settled = True
            break

    final = beats[-1]
    return BreathResult(answer=final.top, belief=max(final.belief.values()) if final.belief else 0.0,
                        cues=list(cues), trust=[float(x) for x in trust], beats=beats, settled=settled,
                        seconds=time.time() - t0, generations=gens, fit_queries=fits.queries,
                        fit_table={pool[k]: [round(float(x), 3) for x in fits.fit[k]] for k in pool})


# ----------------------------------------------------------------------------- control arms

def _select(backend, cues, samples, beta):
    pool: dict[str, str] = {}
    for a in samples:
        k = normalize(a)
        if k and k not in pool:
            pool[k] = a.strip()
    keys = list(pool)
    fits = _FitCache(backend, cues)
    if not keys:
        return "", 0.0, fits.queries
    F = fits.matrix(keys)
    _, belief, _ = belief_and_residue(F, np.ones(len(cues)), beta)
    i = int(np.argmax(belief))
    return pool[keys[i]], float(belief[i]), fits.queries


def fixed_repeat(backend, cues, n_total: int, temperature: float = 1.0, beta: float = 6.0) -> BreathResult:
    """Control: same number of samples, all cues always at full authority, same residue-based
    selection, no breathing and no trust.  (Self-consistency + verification.)"""
    t0 = time.time()
    samples = backend.propose(cues, None, n=n_total - 1, temperature=temperature)
    samples += backend.propose(cues, None, n=1, temperature=0.0)
    ans, p, q = _select(backend, cues, samples, beta)
    return BreathResult(answer=ans, belief=p, cues=list(cues), trust=[1.0] * len(cues),
                        seconds=time.time() - t0, generations=n_total, fit_queries=q)


def dropout_once(backend, cues, per_drop: int = 2, temperature: float = 1.0, beta: float = 6.0,
                 rounds: int = 1) -> BreathResult:
    """Control: leave-one-cue-out sampling (by deleting each cue in turn from the text),
    `rounds` times, then the same residue-based selection with uniform trust.  No trust
    loop.  If breathing cannot beat this, the loop is decoration."""
    t0 = time.time()
    samples = []
    for _ in range(rounds):
        for k in range(len(cues)):
            sub = [c for i, c in enumerate(cues) if i != k] or list(cues)
            samples += backend.propose(sub, None, n=per_drop, temperature=temperature)
    samples += backend.propose(cues, None, n=1, temperature=0.0)
    ans, p, q = _select(backend, cues, samples, beta)
    return BreathResult(answer=ans, belief=p, cues=list(cues), trust=[1.0] * len(cues),
                        seconds=time.time() - t0, generations=len(samples), fit_queries=q)


# ----------------------------------------------------------------------------- display

def bar(x: float, width: int = 12) -> str:
    n = int(round(max(0.0, min(1.0, x)) * width))
    return "█" * n + "·" * (width - n)


def format_beat(beat: Beat, cues: list[str], width: int = 40) -> str:
    lines = [f"── breath {beat.t} " + "─" * 44]
    lines.append("  inhale (letting go of one clue at a time): " + " · ".join(dict.fromkeys(beat.inhale)))
    lines.append(f"  exhale: {beat.exhale}")
    lines.append("  belief:")
    for name, p in beat.belief.items():
        lines.append(f"    {bar(p)} {p:4.2f}  {name}")
    lines.append("  trust in each clue:")
    for c, w, r in zip(cues, beat.trust, beat.residue):
        short = (c[: width - 1] + "…") if len(c) > width else c
        lines.append(f"    {bar(w)} {w:4.2f}  {short:<{width}}  residue {r:4.2f}")
    return "\n".join(lines)
