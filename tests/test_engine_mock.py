"""The loop's logic in a hand-built world (no model).  This checks the bookkeeping,
not whether a real LLM behaves this way — that is what bench.py is for."""
import numpy as np

from breathe.engine import (BreathConfig, belief_and_residue, breathe, dropout_once,
                            fixed_repeat, normalize)

# fit table: is cue true of candidate?
TABLE = {
    #                 timing  basal-ganglia-input  dopamine  starts-with-B
    "basal ganglia": (0.7,    0.15,                0.8,      0.99),
    "brainstem":     (0.3,    0.02,                0.2,      0.99),
    "biorhythm":     (0.6,    0.01,                0.1,      0.99),
    "brodmann area": (0.2,    0.01,                0.05,     0.99),
    "striatum":      (0.9,    0.97,                0.95,     0.01),
    "cerebellum":    (0.85,   0.03,                0.2,      0.02),
}
CUES = ["involved in interval timing", "main input nucleus of the basal ganglia",
        "rich in dopamine receptors", "I think it starts with B"]
LIAR = CUES[3]


class MockWorld:
    """Proposals depend on how much authority the B-cue has (via trust in attention mode,
    or by being present in the text).  While it has authority, B-words dominate."""

    def __init__(self, seed=0, table=TABLE, base=0.05):
        self.rng = np.random.default_rng(seed)
        self.table = table
        self.base = base

    def _wb(self, cues, trust):
        if LIAR not in cues:
            return 0.0
        return 1.0 if trust is None else float(trust[cues.index(LIAR)])

    def propose(self, cues, trust, n, temperature):
        wb = self._wb(cues, trust)
        bwords = ["basal ganglia", "brainstem", "biorhythm", "brodmann area"]
        if temperature <= 0:
            return ["basal ganglia" if wb > 0.5 else "striatum"]
        out = []
        for _ in range(n):
            if self.rng.random() < self.base + 0.8 * (1 - wb):
                out.append(self.rng.choice(["striatum", "cerebellum"], p=[0.8, 0.2]))
            else:
                out.append(self.rng.choice(bwords))
        return out

    def compat(self, cues, cands):
        F = np.zeros((len(cues), len(cands)))
        for c, name in enumerate(cands):
            row = self.table[normalize(name)]
            F[:, c] = [row[CUES.index(q)] for q in cues]
        return F


def test_lying_cue_loses_trust_and_answer_settles():
    for use_att in (True, False):
        for seed in range(6):
            r = breathe(MockWorld(seed=seed), CUES, BreathConfig(use_attention=use_att))
            assert normalize(r.answer) == "striatum", (use_att, seed, r.answer)
            assert r.distrusted is not None and r.distrusted[0] == 3, (use_att, seed, r.trust)
            assert min(r.trust[:3]) > 0.6


def test_truthful_letter_cue_is_kept():
    """Same letter-style cue, but true (target 'basal ganglia'): it must not be rejected
    just for being a letter cue."""
    table = dict(TABLE)
    table["basal ganglia"] = (0.8, 0.9, 0.9, 0.99)
    table["striatum"] = (0.9, 0.5, 0.95, 0.01)
    for seed in range(4):
        r = breathe(MockWorld(seed=seed, table=table), CUES, BreathConfig())
        assert normalize(r.answer) == "basal ganglia"
        assert r.trust[3] > 0.6 and r.distrusted is None


def test_log_scoring_lets_one_cue_veto():
    """Why residue is bounded: with sum-of-log-fit scoring the single false B-cue outvotes
    three true cues; with bounded residue the three true cues win."""
    F = np.array([TABLE["basal ganglia"], TABLE["striatum"]]).T   # [cues, cands]
    logscore = np.log(np.clip(F, 1e-3, 1)).sum(0)
    assert np.argmax(logscore) == 0                                 # basal ganglia
    _, belief, _ = belief_and_residue(F, np.ones(4), beta=6.0)
    assert np.argmax(belief) == 1                                   # striatum


def test_controls_have_no_trust_update():
    r = fixed_repeat(MockWorld(seed=3), CUES, n_total=20)
    assert r.trust == [1.0] * 4 and r.generations == 20
    d = dropout_once(MockWorld(seed=3), CUES, per_drop=2)
    assert d.trust == [1.0] * 4 and d.generations == 9


def test_single_cue_does_not_crash():
    r = breathe(MockWorld(seed=4), [CUES[0]], BreathConfig(beats=3))
    assert r.trust == [1.0]


def test_clue_the_checker_rejects_everywhere_is_not_accused():
    """First real Qwen3-8B run: the checker said No to 'the part of the brain that deals
    with time' for EVERY candidate, and absolute residue then distrusted a true clue.
    With residue relative to the clue's best fit, a clue that fits nothing is
    uninformative, not guilty."""
    F = np.array([[0.02, 0.02, 0.02],     # strict clue: No for everything
                  [0.9, 0.2, 0.1],        # true clue
                  [0.1, 0.99, 0.99]])     # false letter clue
    _, belief, R = belief_and_residue(F, np.ones(3), beta=6.0)
    assert R[0] < 1e-9 and R[1] > 0 and R[2] > 0


def test_strict_outcomes_separate_blends():
    from analyze import outcome
    item = {"target": "laplacian", "aliases": ["laplacian", "laplace operator"],
            "corrupt": {"clue": "I think it starts with H"}}
    assert outcome("Laplacian", item) == "clean"
    assert outcome("the Laplace operator", item) == "clean"
    assert outcome("Hodge Laplacian", item) == "blend"
    assert outcome("Hessian", item) == "obeyed"
    assert outcome("Gradient", item) == "other"


def test_replace_design_keeps_structure():
    import json, random
    from bench import make_conditions
    items = [json.loads(l) for l in open("data/tot_items.jsonl", encoding="utf-8")]
    assert all(it.get("true_clue") for it in items)
    c = make_conditions(items[0], random.Random(0), "replace")
    (base, _), (cor, pos) = c["base"], c["corrupt"]
    assert len(base) == len(cor) == 4
    assert [x for k, x in enumerate(base) if k != pos] == [x for k, x in enumerate(cor) if k != pos]
    assert base[pos] == items[0]["true_clue"] and cor[pos] == items[0]["corrupt"]["clue"]


class ResidueWorld(MockWorld):
    """First guesses are B-words; asked what explains the leftover clues, it offers the
    candidates that fit those clues best."""

    def propose_residue(self, cues, guess, unexplained, tried, n, temperature):
        idx = [CUES.index(u) for u in unexplained]
        ranked = sorted(self.table, key=lambda c: -sum(self.table[c][i] for i in idx))
        fresh = [c for c in ranked if c not in {normalize(t) for t in tried}]
        return [fresh[k % 2] for k in range(n)]


def test_residue_search_keeps_every_clue_and_reports_the_leftover():
    from breathe.engine import residue_search
    r = residue_search(ResidueWorld(seed=0, base=0.0), CUES, n_total=30)
    assert normalize(r.answer) == "striatum"
    assert r.trust == [1.0] * 4                      # nothing was ever suppressed
    assert r.distrusted is not None and r.distrusted[0] == 3   # the B-clue is what is left over
    assert r.trail[0][0].lower() != "striatum"       # it did not start there


def test_residue_search_stops_when_everything_is_explained():
    from breathe.engine import residue_search
    table = dict(TABLE)
    table["basal ganglia"] = (0.8, 0.9, 0.9, 0.99)
    r = residue_search(ResidueWorld(seed=1, table=table, base=0.0), CUES, n_total=30)
    assert normalize(r.answer) == "basal ganglia" and r.distrusted is None
    assert r.generations < 30
