#!/usr/bin/env python3
"""Vary the format-SFT think block so Stage 0 does not burn one fixed string.

The shipped SFT corpus uses an identical <think> body on all 35k rows
("Let me work through this step by step and determine the answer."). Training
on one constant string teaches the model that the think block is a fixed
placeholder emitted before the answer — a strong prior that RLVR then has to
undo, which is counterproductive for a project about the geometry of reasoning.

This script replaces that constant with a small rotating set of neutral,
CONTENT-FREE scaffolds, chosen deterministically per row, and leaves the
<answer> untouched. It does NOT inject reasoning content (which would be
teacher distillation). The intent is only to remove the single-string prior so
RLVR is free to fill the think block.

Open design choice (see the review): the alternative is to drop the think
scaffold from SFT entirely and let RLVR learn when/whether to think. Use
--mode strip for that variant.

Usage (from 0.peft/):
  python3 diversify_think.py --in dataset/train.jsonl --out dataset/train_varied.jsonl
  python3 diversify_think.py --in dataset/train.jsonl --out dataset/train_nothink.jsonl --mode strip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.S)

# Neutral, content-free scaffolds: no problem-specific reasoning in any of them.
_SCAFFOLDS = [
    "Let me work through this.",
    "Working through the steps.",
    "Let me reason about this carefully.",
    "Let me think it through.",
    "Considering the problem.",
    "Let me work it out.",
    "Reasoning through this now.",
    "Let me take this step by step.",
]


def _pick(seed: str) -> str:
    h = int(hashlib.sha1(seed.encode()).hexdigest(), 16)
    return _SCAFFOLDS[h % len(_SCAFFOLDS)]


def _completion_key(row: dict) -> str:
    for k in ("completion", "formatted_output", "response"):
        if k in row:
            return k
    return "completion"


def rewrite(row: dict, mode: str) -> dict:
    key = _completion_key(row)
    comp = str(row.get(key, ""))
    ans = _ANSWER_RE.search(comp)
    answer_block = ans.group(0) if ans else comp
    seed = str(row.get("instance_id") or row.get("prompt", "")) + comp[:32]
    if mode == "strip":
        comp_new = answer_block
    else:
        comp_new = f"<think>\n{_pick(seed)}\n</think>\n{answer_block}"
    out = dict(row)
    out[key] = comp_new
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["vary", "strip"], default="vary")
    args = ap.parse_args()

    n = 0
    seen_think: set[str] = set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.inp.open(encoding="utf-8") as fi, args.out.open("w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            new = rewrite(row, args.mode)
            m = _THINK_RE.search(str(new.get(_completion_key(new), "")))
            if m:
                seen_think.add(m.group(0))
            fo.write(json.dumps(new, ensure_ascii=False) + "\n")
            n += 1
    print(f"rewrote {n} rows -> {args.out} (mode={args.mode}); "
          f"distinct think blocks now: {len(seen_think) if args.mode == 'vary' else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
