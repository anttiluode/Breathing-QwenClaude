"""End-to-end wiring on the tiny random model: every entry point runs and produces
well-formed output.  (A random model's answers are noise; only the plumbing is checked.)"""
import json
import sys

import numpy as np

import breathe.qwen_backend as qb
from breathe.engine import BreathConfig, breathe, dropout_once, fixed_repeat
from tests.tiny import tiny_tokenizer, tiny_model

TOK = tiny_tokenizer()
MODEL = tiny_model(TOK)
CUES = ["a brain structure involved in interval timing and habits",
        "the main input nucleus of the basal ganglia", "I think it starts with B"]


_REAL = qb.QwenBackend


def backend(**kw):
    return _REAL(model=MODEL, tokenizer=TOK, max_new_tokens=4, **kw)


def test_all_arms_run():
    be = backend()
    for att in (True, False):
        r = breathe(be, CUES, BreathConfig(beats=2, use_attention=att))
        assert 1 <= len(r.beats) <= 2 and len(r.trust) == 3
        assert all(0.05 <= w <= 1 for w in r.trust)
        assert r.generations == len(r.beats) * (3 * 2 + 1)
    f = fixed_repeat(be, CUES, n_total=5)
    d = dropout_once(be, CUES)
    assert f.generations == 5 and d.generations == 7
    assert isinstance(be.one_shot(CUES), str)
    assert isinstance(be.think(CUES, budget=6), str)
    assert qb.TRUST.bias is None


def test_reply_with_trust_and_note():
    be = backend()
    msg = "; ".join(CUES)
    p = be.raw_prompt(None, msg, CUES)
    assert all(p.cue_tokens)
    out = be.reply(msg, CUES, trust=[1.0, 1.0, 0.1], note="It settled on: striatum.", max_new=5)
    assert isinstance(out, str)


def test_chat_demo_runs(capsys):
    import chat
    chat.run(backend(), chat.DEMO, BreathConfig(beats=2), trace=True, plain=True)
    text = capsys.readouterr().out
    assert "breath 1" in text and "breathing qwen>" in text and "plain qwen>" in text


def test_bench_main_runs(tmp_path, monkeypatch):
    import bench
    items = tmp_path / "items.jsonl"
    with open("data/tot_items.jsonl", encoding="utf-8") as f:
        items.write_text("".join(f.readlines()[:2]), encoding="utf-8")
    out = tmp_path / "res.jsonl"
    monkeypatch.setattr(qb, "QwenBackend", lambda *a, **k: backend())
    monkeypatch.setattr(sys, "argv", ["bench.py", "--items", str(items), "--out", str(out),
                                      "--beats", "2", "--think", "--think-budget", "4"])
    bench.main()
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 * 2 * 6       # items x conditions x arms
    assert {r["arm"] for r in rows} == {"one_shot", "think", "fixed", "dropout", "breathe_text", "breathe_attn"}


def test_is_correct():
    from bench import is_correct
    assert is_correct("The striatum.", ["striatum"])
    assert is_correct("dorsal striatum", ["striatum"])
    assert is_correct("Epinephrine", ["adrenaline", "epinephrine"])
    assert not is_correct("basal ganglia", ["striatum"])
    assert not is_correct("", ["striatum"])
    assert is_correct("Déjà vu", ["deja vu"])
