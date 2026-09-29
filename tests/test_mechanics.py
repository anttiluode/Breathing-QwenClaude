"""Mechanical checks on a tiny random Qwen3.  Run:  python -m pytest -q tests"""
import numpy as np
import torch

from breathe.qwen_backend import QwenBackend, precision, TRUST
from tests.tiny import tiny_tokenizer, tiny_model

TOK = tiny_tokenizer()
CUES = ["a brain structure involved in interval timing and habits",
        "the main input nucleus of the basal ganglia",
        "I think it starts with B"]


def pair():
    stock = tiny_model(TOK, attn="sdpa")
    ours = tiny_model(TOK, attn="trust_sdpa")
    ours.load_state_dict(stock.state_dict())
    return stock, ours


def test_no_trust_is_the_stock_model():
    stock, ours = pair()
    ids = TOK("hello brain structure timing", return_tensors="pt")["input_ids"]
    with torch.no_grad(), precision(None):
        a = stock(input_ids=ids).logits
        b = ours(input_ids=ids).logits
    assert torch.equal(a, b)


def test_zero_bias_branch_matches_stock():
    """Forces our own SDPA branch (mask building, GQA repeat, transpose) with a zero bias."""
    stock, ours = pair()
    ids = TOK("hello brain structure timing and habits", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        a = stock(input_ids=ids).logits
        with precision(torch.zeros(ids.shape[1])):
            b = ours(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_huge_negative_bias_equals_masking_those_keys():
    stock, ours = pair()
    ids = TOK("the main input nucleus of the basal ganglia is here", return_tensors="pt")["input_ids"]
    P = ids.shape[1]
    span = list(range(2, 5))
    bias = torch.zeros(P)
    bias[span] = -1e4
    am = torch.ones(1, P, dtype=torch.long)
    am[0, span] = 0
    with torch.no_grad():
        masked = stock(input_ids=ids, attention_mask=am).logits[0, -1]
        with precision(bias):
            biased = ours(input_ids=ids).logits[0, -1]
    assert torch.allclose(masked, biased, atol=1e-4), (masked - biased).abs().max()


def test_bias_changes_output_and_restores():
    _, ours = pair()
    ids = TOK("the main input nucleus of the basal ganglia", return_tensors="pt")["input_ids"]
    bias = torch.zeros(ids.shape[1]); bias[1:4] = -3.0
    with torch.no_grad():
        a = ours(input_ids=ids).logits
        with precision(bias):
            b = ours(input_ids=ids).logits
        c = ours(input_ids=ids).logits
    assert not torch.allclose(a, b)
    assert torch.equal(a, c) and TRUST.bias is None


def test_incremental_decode_matches_full_forward_under_bias():
    _, ours = pair()
    be = QwenBackend(model=ours, tokenizer=TOK, attn_gain=1.0)
    p = be.recall_prompt(CUES)
    bias = be.bias_vector(p, [1.0, 1.0, 0.1])
    with precision(bias):
        gen = be._generate(p.ids, 1, 0.0, 6)[0]
        full = torch.cat([p.ids, torch.tensor([gen])], 1)
        with torch.no_grad():
            logits = ours(input_ids=full).logits[0]
    P = p.ids.shape[1]
    for k, t in enumerate(gen):
        assert int(logits[P - 1 + k].argmax()) == t


def test_cue_spans_cover_the_cue_text():
    be = QwenBackend(model=tiny_model(TOK), tokenizer=TOK)
    p = be.recall_prompt(CUES)
    ids = p.ids[0].tolist()
    for cue, toks in zip(CUES, p.cue_tokens):
        assert toks, cue
        assert cue in TOK.decode([ids[i] for i in toks]).strip()
    b = be.bias_vector(p, [1.0, 0.5, 0.1])
    assert b[p.cue_tokens[0]].abs().max() == 0
    assert torch.allclose(b[p.cue_tokens[2]], torch.tensor(np.log(0.1), dtype=b.dtype))
    others = set(range(p.ids.shape[1])) - set(sum(p.cue_tokens, []))
    assert b[sorted(others)].abs().max() == 0


def test_padded_fit_batch_equals_one_by_one():
    m = tiny_model(TOK, attn="trust_sdpa")
    cands = ["striatum", "brainstem", "cerebellum x y z w"]
    a = QwenBackend(model=m, tokenizer=TOK, fit_batch=32).compat(CUES, cands)
    b = QwenBackend(model=m, tokenizer=TOK, fit_batch=1).compat(CUES, cands)
    assert a.shape == (3, 3)
    assert np.allclose(a, b, atol=1e-4), np.abs(a - b).max()
    assert ((a > 0) & (a < 1)).all()


def test_sampling_is_seeded():
    m = tiny_model(TOK)
    a = QwenBackend(model=m, tokenizer=TOK, seed=3).propose(CUES, [1, 1, .2], n=4, temperature=1.1)
    b = QwenBackend(model=m, tokenizer=TOK, seed=3).propose(CUES, [1, 1, .2], n=4, temperature=1.1)
    assert a == b and len(a) == 4


def test_batched_let_go_rows_match_separate_calls():
    """propose_multi puts different trust rows in one batch; each row must equal its own
    separate call (greedy, so deterministic)."""
    be = QwenBackend(model=tiny_model(TOK), tokenizer=TOK, max_new_tokens=6)
    rows = [np.array([0.02, 1, 1]), np.array([1, 0.02, 1]), np.array([1, 1, 0.02]), None]
    multi = be.propose_multi(CUES, rows, n_each=2, temperature=0.0)
    for w, got in zip(rows, multi):
        assert got == be.propose(CUES, w, n=2, temperature=0.0)
    assert len({tuple(g) for g in multi}) > 1   # the bias rows really differ
