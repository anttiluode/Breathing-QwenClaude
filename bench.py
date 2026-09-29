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
from collections import Counter
import json
import random
import sys
import time

import numpy as np

from analyze import outcome
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


def make_conditions(item, rng, design="insert"):
    """insert : clean = 3 true clues;          corrupt = 3 true + the false clue
       replace: base  = 3 true + a TRUE 4th;    corrupt = 3 true + the false 4th, same slot
    'replace' keeps the clue count and structure identical, so only the corruption differs
    (for letter items the true 4th is the correct first letter: can the loop keep a true
    letter clue while rejecting a false one?)."""
    clean = list(item["clues"])
    pos = rng.randrange(len(clean) + 1)
    corrupt = clean[:pos] + [item["corrupt"]["clue"]] + clean[pos:]
    if design == "replace":
        true4 = clean[:pos] + [item["true_clue"]] + clean[pos:]
        return {"base": (true4, None), "corrupt": (corrupt, pos)}
    return {"base": (clean, None), "corrupt": (corrupt, pos)}


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
    ap.add_argument("--design", choices=["insert", "replace"], default="insert",
                    help="insert: clean=3 true clues (run 1); replace: base=3 true + a true 4th")
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
            conds = make_conditions(item, rng, args.design)
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
                    row = {"id": item["id"], "target": item["target"], "design": args.design,
                           "condition": cond if cond == "corrupt" else ("clean" if args.design == "insert" else "true4"),
                           "arm": arm, "outcome": outcome(x.answer, item),
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
    ids = sorted({r["id"] for r in rows})
    base = next(c for c in ("clean", "true4") if any(r["condition"] == c for r in rows))
    get = {(r["id"], r["condition"], r["arm"]): r for r in rows}

    def col(cond, arm, strict):
        return np.array([(get[(i, cond, arm)]["outcome"] == "clean") if strict else get[(i, cond, arm)]["correct"]
                         for i in ids], dtype=float)

    for strict in (False, True):
        name = "STRICT (answer must be the target, no blends)" if strict else "lenient, as pre-registered"
        print(f"\n=== accuracy, {name} (± standard error) ===")
        print(f"{'arm':<14}{base:>16}{'corrupt':>16}{'gens/item':>12}{'s/item':>9}")
        for arm in arms:
            cells = []
            for cond in (base, "corrupt"):
                v = col(cond, arm, strict)
                cells.append(f"{v.mean():.2f} ± {v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:.2f}")
            g = np.mean([get[(i, "corrupt", arm)]["generations"] for i in ids])
            sec = np.mean([get[(i, "corrupt", arm)]["seconds"] for i in ids])
            print(f"{arm:<14}{cells[0]:>16}{cells[1]:>16}{g:>12.1f}{sec:>9.1f}")

    print("\n=== corrupted clue: what was answered (clean / blend / obeyed false letter / other) ===")
    for arm in arms:
        c = Counter(get[(i, "corrupt", arm)]["outcome"] for i in ids)
        print(f"{arm:<14}{c['clean']:>4}{c['blend']:>6}{c['obeyed']:>6}{c['other']:>6}")

    print("\n=== the distrust flag (corrupt condition) ===")
    for arm in ("breathe_text", "breathe_attn"):
        groups = {"named the false clue": [], "named nothing": [], "named a TRUE clue": []}
        for i in ids:
            r = get[(i, "corrupt", arm)]
            key = ("named the false clue" if r["distrusted"] == r["corrupt_pos"]
                   else "named nothing" if r["distrusted"] is None else "named a TRUE clue")
            groups[key].append(r["outcome"] == "clean")
        line = "   ".join(f"{k} {len(v)} (right {np.mean(v):.2f})" for k, v in groups.items() if v)
        fa = np.mean([get[(i, base, arm)]["distrusted"] is not None for i in ids])
        print(f"{arm:<14}{line}   false alarms on {base}: {fa:.2f}")

    print("\n=== by corruption type (corrupt, strict) ===")
    for t in sorted({r["corrupt_type"] for r in rows}):
        tid = [i for i in ids if get[(i, "corrupt", arms[0])]["corrupt_type"] == t]
        line = f"{t:<16}" + "".join(
            f"  {arm}={np.mean([get[(i, 'corrupt', arm)]['outcome'] == 'clean' for i in tid]):.2f}" for arm in arms)
        print(line + f"  (n={len(tid)})")

    for strict in (False, True):
        print(f"\n=== pre-registered comparisons, {'strict' if strict else 'lenient'} (paired bootstrap, 95%) ===")
        A = col("corrupt", "breathe_attn", strict)
        v = {}
        for other in ("fixed", "dropout", "breathe_text", "one_shot") + (("think",) if "think" in arms else ()):
            m, lo, hi = paired_ci(A, col("corrupt", other, strict))
            print(f"corrupt: breathe_attn - {other:<12} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
            v[other] = lo > 0
        m, lo, hi = paired_ci(col(base, "breathe_attn", strict), col(base, "one_shot", strict))
        print(f"{base + ':':<9}breathe_attn - one_shot     {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
        base_ok = m >= -0.05
        if v["fixed"] and v["dropout"] and base_ok:
            print("-> BREATHING SURVIVES: beats both compute-matched controls on corrupted clues.")
        else:
            why = [k for k in ("fixed", "dropout") if not v[k]]
            print("-> BREATHING KILLED" + (f": does not beat {', '.join(why)}" if why else "") +
                  ("" if base_ok else f"; costs more than 5 points on {base} items") + ".")
        print("-> ATTENTION PRECISION " + ("SURVIVES (beats text deletion)." if v["breathe_text"]
                                          else "NOT SHOWN: no better than deleting cues from the text."))


if __name__ == "__main__":
    main()
