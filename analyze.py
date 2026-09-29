"""
Re-score a bench.py results file more strictly than bench.py does.

    python analyze.py results/qwen3-8b-4bit_run1.jsonl

bench.py counts an answer correct if it contains the target within two extra words, so
"Bilateral striatum" counts as striatum.  That hides exactly the failure this experiment
is about: an answer that keeps the right noun but bends to satisfy the false clue.
This script splits every corrupted-clue answer into four outcomes:

    clean     the target itself (or a listed alias)
    blend     target plus extra words, where an extra word starts with the false letter
              ("Hodge Laplacian" for a wrong "starts with H")
    obeyed    wrong, and starts with the false letter ("Basal ganglia" for striatum)
    other     wrong in some other way

It also reports how often the loop's distrust flag named the corrupted clue, and how
accurate the answer was in each flag case.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from breathe.engine import normalize

ARMS = ["one_shot", "think", "fixed", "dropout", "breathe_text", "breathe_attn", "residue_search"]

# Hand judgement (mine, not the model's): wrong-by-the-answer-key answers that are
# still defensible because every TRUE clue also fits them.  Reported separately, never
# folded into the headline numbers.
DEFENSIBLE = {
    "hippocampus": {"ammon's horn"},
    "abacus": {"soroban", "suanpan"},
    "ubiquitous": {"omnipresent"},
    "circadian rhythm": {"biological clock"},
}


def false_letter(item):
    m = re.search(r"starts with ([A-Z])", item["corrupt"]["clue"])
    return m.group(1).lower() if m else None


def outcome(answer, item):
    a = normalize(answer)
    aliases = {normalize(x) for x in item["aliases"]}
    L = false_letter(item)
    if a in aliases:
        return "clean"
    words = a.split()
    for al in aliases:
        n = al.split()
        for i in range(len(words) - len(n) + 1):
            if words[i:i + len(n)] == n:
                extra = words[:i] + words[i + len(n):]
                if L and any(w.startswith(L) for w in extra):
                    return "blend"
                return "clean"          # e.g. "Ada Byron Lovelace", "the striatum"
    if L and a.startswith(L):
        return "obeyed"
    return "other"


def main(path, items_path="data/tot_items.jsonl"):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    items = {json.loads(l)["id"]: json.loads(l) for l in open(items_path, encoding="utf-8")}
    arms = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    ids = sorted({r["id"] for r in rows})
    get = {(r["id"], r["condition"], r["arm"]): r for r in rows}

    print(f"{len(ids)} items\n")
    print("=== corrupted clue: what each arm answered ===")
    print(f"{'arm':<14}{'clean':>7}{'blend':>7}{'obeyed':>8}{'other':>7}   strict acc   clean-condition acc   s/item")
    for arm in arms:
        c = Counter(outcome(get[(i, 'corrupt', arm)]["answer"], items[i]) for i in ids)
        clean_cond = np.mean([outcome(get[(i, 'clean', arm)]["answer"], items[i]) in ("clean",) for i in ids])
        secs = np.mean([get[(i, 'corrupt', arm)]["seconds"] for i in ids])
        print(f"{arm:<14}{c['clean']:>7}{c['blend']:>7}{c['obeyed']:>8}{c['other']:>7}   "
              f"{c['clean'] / len(ids):10.2f}   {clean_cond:19.2f}   {secs:6.1f}")

    print("\n=== defensible alternatives (hand-judged; not counted above) ===")
    for arm in arms:
        d = [items[i]["target"] + " -> " + get[(i, 'corrupt', arm)]["answer"] for i in ids
             if normalize(get[(i, 'corrupt', arm)]["answer"]) in DEFENSIBLE.get(items[i]["target"], set())]
        print(f"{arm:<14}{len(d):>3}  " + "; ".join(d))

    print("\n=== paired against dropout (corrupt, strict) ===")
    for arm in arms:
        if arm == "dropout":
            continue
        a = [outcome(get[(i, 'corrupt', arm)]["answer"], items[i]) == "clean" for i in ids]
        b = [outcome(get[(i, 'corrupt', 'dropout')]["answer"], items[i]) == "clean" for i in ids]
        wins = [items[i]["target"] for i, x, y in zip(ids, a, b) if x and not y]
        losses = [items[i]["target"] for i, x, y in zip(ids, a, b) if y and not x]
        print(f"{arm:<14} wins {len(wins):>2} {wins}\n{'':<14} loses {len(losses):>2} {losses}")

    print("\n=== the distrust flag (corrupt condition) ===")
    for arm in ("breathe_text", "breathe_attn", "residue_search"):
        if arm not in arms:
            continue
        groups = defaultdict(list)
        for i in ids:
            r = get[(i, 'corrupt', arm)]
            key = ("named the false clue" if r["distrusted"] == r["corrupt_pos"]
                   else "named nothing" if r["distrusted"] is None else "named a TRUE clue")
            groups[key].append(outcome(r["answer"], items[i]) == "clean")
        for key in ("named the false clue", "named nothing", "named a TRUE clue"):
            v = groups.get(key, [])
            if v:
                print(f"{arm:<14}{key:<22} n={len(v):>2}   answer correct {np.mean(v):.2f}")
        fa = np.mean([get[(i, 'clean', arm)]["distrusted"] is not None for i in ids])
        print(f"{arm:<14}false alarms on clean items: {fa:.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
