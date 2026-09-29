"""
Talk to the breathing Qwen.

Every message is split into clues (on ';', new lines and sentence ends).  The loop
breathes over them, shows each breath, then answers in its own words — and, for
comparison, plain Qwen answers the same message with no loop.

    python chat.py                        Qwen/Qwen3-8B in bf16 (~17 GB VRAM)
    python chat.py --load-4bit            fits a 12 GB card
    python chat.py --demo                 runs the striatum morning once and exits

Commands:  /quit   /trace  (show/hide breaths)   /plain  (show/hide plain Qwen)
           /gain X (attention trust gain; 0 = text-mode loop)
"""
from __future__ import annotations

import argparse
import sys

from breathe.engine import BreathConfig, breathe, format_beat, split_cues

DEMO = ("the part of the brain that deals with time; it has to do with dopamine and habits; "
        "we talked about it together with the medial septum; I think it starts with b")


def note_for(result) -> str:
    parts = [f"It settled on: {result.answer} (confidence {result.belief:.2f})."]
    dis = result.distrusted
    if dis:
        j, cue, w = dis
        kept = [c for i, c in enumerate(result.cues) if i != j and result.trust[i] >= 0.6]
        parts.append(f"It found this clue inconsistent with the others and set it aside: "
                     f"\"{cue}\" (trust {w:.2f}).")
        if kept:
            parts.append("The clues it relied on: " + "; ".join(f"\"{c}\"" for c in kept) + ".")
    else:
        parts.append("All the clues fitted together.")
    if not result.settled:
        parts.append("It did not fully settle, so hedge a little.")
    return " ".join(parts)


def run(be, message: str, cfg: BreathConfig, trace: bool, plain: bool):
    cues = split_cues(message)
    if len(cues) < 2:
        print("  (one clue only — separate clues with ';' so the loop has something to weigh)")
    on_beat = (lambda b, c: print(format_beat(b, c))) if trace else None
    r = breathe(be, cues, cfg, on_beat=on_beat)
    if trace and r.fit_table:
        print("\n  checker's fit table (does clue j fit the candidate?):")
        print("    " + " ".join(f"c{j + 1:<5}" for j in range(len(cues))) + " candidate")
        for cand, fs in sorted(r.fit_table.items(), key=lambda kv: -sum(kv[1])):
            print("    " + " ".join(f"{f:5.2f} " for f in fs) + f" {cand}")
        for j, c in enumerate(cues):
            print(f"    c{j + 1} = {c}")
    trust = r.trust if cfg.use_attention else None
    print(f"\nbreathing qwen> {be.reply(message, cues, trust=trust, note=note_for(r))}")
    print(f"                 [{len(r.beats)} breaths, {r.generations} samples, {r.fit_queries} checks, "
          f"{r.seconds:.1f}s{', settled' if r.settled else ', not settled'}]")
    if plain:
        print(f"\nplain qwen>      {be.reply(message, cues)}")
    print()


def main():
    try:  # the trust bars use block characters; Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--beats", type=int, default=5)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from breathe.qwen_backend import QwenBackend
    print(f"loading {args.model} ...")
    be = QwenBackend(args.model, load_4bit=args.load_4bit, load_8bit=args.load_8bit,
                     attn_gain=args.gain, seed=args.seed)
    cfg = BreathConfig(beats=args.beats, use_attention=args.gain > 0)
    trace, plain = True, True

    if args.demo:
        print(f"\nyou> {DEMO}\n")
        run(be, DEMO, cfg, trace, plain)
        return

    print("separate clues with ';'.  /quit /trace /plain /gain X\n")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/trace":
            trace = not trace; print(f"  trace {'on' if trace else 'off'}"); continue
        if msg == "/plain":
            plain = not plain; print(f"  plain qwen {'on' if plain else 'off'}"); continue
        if msg.startswith("/gain"):
            try:
                be.attn_gain = float(msg.split()[1])
                cfg.use_attention = be.attn_gain > 0
                print(f"  attention gain {be.attn_gain}")
            except (IndexError, ValueError):
                print("  usage: /gain 1.0")
            continue
        run(be, msg, cfg, trace, plain)


if __name__ == "__main__":
    main()
