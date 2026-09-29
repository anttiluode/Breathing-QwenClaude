"""
The kill test.  60 tip-of-the-tongue items, each with three true clues and one
corrupted clue (mostly a wrong first letter, like the striatum morning).
Every item is run twice: CLEAN (true clues only) and CORRUPT (true clues plus the
corrupted one, inserted at a random position).

Arms (all use the same frozen Qwen3):
    one_shot        greedy answer, no tricks
    think           Qwen3 thinking mode, budgeted                    (--think; slow)
    fixed           self-consistency + the same residue-based selection,
                    compute-matched to breathe_attn, every cue at full authority
    dropout         one leave-one-cue-out sampling round by deleting cues from the
                    text, then the same selection; compute-matched; no trust loop
    breathe_text    the breathing loop, trust applied by deleting low-trust cues
    breathe_attn    the breathing loop, trust applied inside attention

Pre-registered rule (README):
    breathing earns its place only if breathe_attn beats BOTH fixed and dropout on
    CORRUPT items with a paired 95% bootstrap interval above zero, and loses no more
    than 5 points to one_shot on CLEAN items.  Attention precision earns its place only
    if breathe_attn - breathe_text is above zero with the same test.

    python bench.py --model Qwen/Qwen3-8B            (bf16, ~17 GB VRAM)
    python bench.py --model Qwen/Qwen3-8B --load-4bit
    python bench.py --limit 10                       (quick look)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

import numpy as np

from breathe.engine import BreathConfig, breathe, dropout_once, fixed_repeat, normalize


def is_correct(answer: str, aliases: list[str]) -> bool:
    a = normalize(answer)
    if not a:
        return False
    aw = a.split()
    for al in aliases:
        n = normalize(al)
        if a == n:
            return True
        nw = n.split()
        if len(aw) <= len(nw) + 2:
            for i in range(len(aw) - len(nw) + 1):
                if aw[i:i + len(nw)] == nw:
                    return True
    return False


def make_conditions(item, rng):
    clean = list(item["clues"])
    pos = rng.randrange(len(clean) + 1)
    corrupt = clean[:pos] + [item["corrupt"]["clue"]] + clean[pos:]
    return {"clean": (clean, None), "corrupt": (corrupt, pos)}


def paired_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0):
    d = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    try:  # the trust bars use block characters; Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--items", default="data/tot_items.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--think", action="store_true", help="also run the thinking-mode baseline")
    ap.add_argument("--think-budget", type=int, default=768)
    ap.add_argument("--gain", type=float, default=1.0, help="attention trust gain")
    ap.add_argument("--beats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()

    from breathe.qwen_backend import QwenBackend
    be = QwenBackend(args.model, load_4bit=args.load_4bit, load_8bit=args.load_8bit,
                     attn_gain=args.gain, seed=args.seed)

    items = [json.loads(l) for l in open(args.items, encoding="utf-8")]
    if args.limit:
        items = items[: args.limit]
    rng = random.Random(args.seed)
    arms = ["one_shot"] + (["think"] if args.think else []) + ["fixed", "dropout", "breathe_text", "breathe_attn"]
    cfg = BreathConfig(beats=args.beats)
    rows = []
    t_start = time.time()

    with open(args.out, "w", encoding="utf-8") as out:
        for n_item, item in enumerate(items, 1):
            conds = make_conditions(item, rng)
            for cond, (cues, cpos) in conds.items():
                res = {}
                r = breathe(be, cues, BreathConfig(**{**cfg.__dict__, "use_attention": True}))
                res["breathe_attn"] = r
                gens = r.generations
                res["breathe_text"] = breathe(be, cues, BreathConfig(**{**cfg.__dict__, "use_attention": False}))
                res["fixed"] = fixed_repeat(be, cues, n_total=gens, temperature=cfg.temp_open, beta=cfg.beta)
                rounds = max(1, round(gens / (len(cues) * cfg.per_drop + 1)))
                res["dropout"] = dropout_once(be, cues, per_drop=cfg.per_drop, temperature=cfg.temp_open,
                                              beta=cfg.beta, rounds=rounds)
                t0 = time.time()
                one = be.one_shot(cues)
                res["one_shot"] = type(r)(answer=one, belief=1.0, cues=cues, trust=[1.0] * len(cues),
                                          seconds=time.time() - t0, generations=1)
                if args.think:
                    t0 = time.time()
                    th = be.think(cues, budget=args.think_budget)
                    res["think"] = type(r)(answer=th, belief=1.0, cues=cues, trust=[1.0] * len(cues),
                                           seconds=time.time() - t0, generations=1)
                for arm in arms:
                    x = res[arm]
                    dis = x.distrusted
                    row = {"id": item["id"], "target": item["target"], "condition": cond, "arm": arm,
                           "corrupt_type": item["corrupt"]["type"], "corrupt_pos": cpos,
                           "answer": x.answer, "correct": is_correct(x.answer, item["aliases"]),
                           "trust": x.trust, "distrusted": dis[0] if dis else None,
                           "generations": x.generations, "fit_queries": x.fit_queries,
                           "beats": len(x.beats), "seconds": round(x.seconds, 2)}
                    rows.append(row)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
            last = [r for r in rows if r["id"] == item["id"] and r["condition"] == "corrupt"]
            show = "  ".join(f"{r['arm']}={'✓' if r['correct'] else '✗'}" for r in last)
            print(f"[{n_item}/{len(items)}] {item['target']:<24} corrupt: {show}   "
                  f"({time.time() - t_start:.0f}s)", flush=True)

    summarize(rows, arms)


def summarize(rows, arms):
    print("\n=== accuracy (fraction correct, ± standard error) ===")
    ids = sorted({r["id"] for r in rows})
    table = {}
    for cond in ("clean", "corrupt"):
        for arm in arms:
            v = np.array([next(r["correct"] for r in rows if r["id"] == i and r["condition"] == cond
                               and r["arm"] == arm) for i in ids], dtype=float)
            table[(cond, arm)] = v
    print(f"{'arm':<14}{'clean':>16}{'corrupt':>16}{'gens/item':>12}")
    for arm in arms:
        g = np.mean([r["generations"] for r in rows if r["arm"] == arm and r["condition"] == "corrupt"])
        cells = []
        for cond in ("clean", "corrupt"):
            v = table[(cond, arm)]
            cells.append(f"{v.mean():.2f} ± {v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:.2f}")
        print(f"{arm:<14}{cells[0]:>16}{cells[1]:>16}{g:>12.1f}")

    print("\n=== which clue did the loop distrust? (corrupt condition) ===")
    for arm in ("breathe_text", "breathe_attn"):
        rs = [r for r in rows if r["arm"] == arm and r["condition"] == "corrupt"]
        hit = np.mean([r["distrusted"] == r["corrupt_pos"] for r in rs])
        none = np.mean([r["distrusted"] is None for r in rs])
        fa = np.mean([r["distrusted"] is not None for r in rows if r["arm"] == arm and r["condition"] == "clean"])
        print(f"{arm:<14} flagged the corrupted clue: {hit:.2f}   flagged nothing: {none:.2f}   "
              f"false alarm on clean items: {fa:.2f}")

    print("\n=== by corruption type (corrupt condition) ===")
    types = sorted({r["corrupt_type"] for r in rows})
    for t in types:
        line = f"{t:<16}"
        for arm in arms:
            v = [r["correct"] for r in rows if r["arm"] == arm and r["condition"] == "corrupt" and r["corrupt_type"] == t]
            line += f"  {arm}={np.mean(v):.2f}"
        print(line + f"  (n={len(v)})")

    print("\n=== pre-registered comparisons (paired bootstrap, 95%) ===")
    A = table[("corrupt", "breathe_attn")]
    verdicts = []
    for other in ("fixed", "dropout", "breathe_text", "one_shot") + (("think",) if "think" in arms else ()):
        m, lo, hi = paired_ci(A, table[("corrupt", other)])
        print(f"corrupt: breathe_attn - {other:<12} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
        verdicts.append((other, lo > 0))
    m, lo, hi = paired_ci(table[("clean", "breathe_attn")], table[("clean", "one_shot")])
    print(f"clean:   breathe_attn - one_shot     {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    clean_ok = m >= -0.05
    v = dict(verdicts)
    print("\n=== verdict ===")
    if v["fixed"] and v["dropout"] and clean_ok:
        print("BREATHING SURVIVES: beats both compute-matched controls on corrupted clues.")
    else:
        why = [k for k in ("fixed", "dropout") if not v[k]]
        print("BREATHING KILLED" + (f": does not beat {', '.join(why)}" if why else "") +
              ("" if clean_ok else "; costs more than 5 points on clean items") + ".")
    print("ATTENTION PRECISION " + ("SURVIVES (beats text deletion)." if v["breathe_text"]
                                    else "NOT SHOWN: no better than deleting cues from the text."))


if __name__ == "__main__":
    main()
