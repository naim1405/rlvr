#!/usr/bin/env python3
"""Corpus QA for the P0-P6 Gym JSONL, run before any training.

Three checks the transfer matrix depends on:

  1. Gold round-trip. Every row's own gold answer, placed in <answer>..</answer>,
     must score 1.0 under that row's verifier. A gold that does not verify is a
     train/verify format mismatch (the kind that silently zeroes correct model
     answers); the per-family round-trip rate should be 100%.
  2. Duplicates. Prompts are hashed (whitespace-normalized) within each file and
     across files, so a small combinatorial family that exhausts its parameter
     space — or a train/eval leak — is visible. instance_id is NOT used (it is a
     random id, not a content hash).
  3. Label balance. Yes/No answer skew per family (a strong prior is exploitable).

Usage (from nemotron_rlvr/):
  python3 scripts/validate_corpus.py --file train=data/train-split.jsonl --file val=data/val-split.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "resources_servers/nemotron_verifier"))


def _prompt(row: dict) -> str:
    if isinstance(row.get("prompt"), str):
        return row["prompt"]
    for m in reversed((row.get("responses_create_params") or {}).get("input") or []):
        if isinstance(m, dict) and isinstance(m.get("content"), str):
            return m["content"]
    return ""


def _phash(p: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", p).strip().encode()).hexdigest()


def _gold_text(row: dict) -> str:
    gt = row.get("expected_answer", row.get("ground_truth", ""))
    if isinstance(gt, (list, tuple)):
        return json.dumps(gt)
    return str(gt)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", action="append", default=[], metavar="NAME=PATH", required=True)
    ap.add_argument("--max-gold-check", type=int, default=0,
                    help="cap rows per file for the gold round-trip (0 = all)")
    args = ap.parse_args()

    from score import extra_env_info_from_mapping, score_response

    files = {}
    for spec in args.file:
        name, _, path = spec.partition("=")
        files[name.strip()] = Path(path)

    all_hashes: dict[str, list[str]] = defaultdict(list)  # phash -> [file:idx]
    ok = True

    for name, path in files.items():
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        print(f"\n=== {name}: {len(rows)} rows ({path}) ===")

        # 2. duplicates within this file
        dup = Counter(_phash(_prompt(r)) for r in rows)
        within = sum(c - 1 for c in dup.values() if c > 1)
        print(f"duplicate prompts within file: {within} extra copies "
              f"({sum(1 for c in dup.values() if c > 1)} prompts repeated)")
        for r in rows:
            all_hashes[_phash(_prompt(r))].append(name)

        # 1. gold round-trip + 3. label balance
        fam_tot: Counter = Counter()
        fam_ok: Counter = Counter()
        yn: dict[str, Counter] = defaultdict(Counter)
        check = rows if not args.max_gold_check else rows[: args.max_gold_check]
        for r in check:
            fam = str(r.get("family") or r.get("task_name") or "?").upper()[:2]
            extra = extra_env_info_from_mapping(r)
            gold = _gold_text(r)
            resp = f"<think>x</think><answer>{gold}</answer>"
            reward = score_response(resp, extra)
            fam_tot[fam] += 1
            fam_ok[fam] += int(reward == 1.0)
            g = gold.strip().lower()
            if g in ("yes", "no", "true", "false"):
                yn[fam][g] += 1
        print("gold round-trip rate (should be 100%):")
        for fam in sorted(fam_tot):
            rate = 100 * fam_ok[fam] / fam_tot[fam]
            flag = "" if rate > 99.0 else "   <-- FORMAT MISMATCH"
            if rate <= 99.0:
                ok = False
            print(f"   {fam}: {rate:5.1f}%  ({fam_ok[fam]}/{fam_tot[fam]}){flag}")
        for fam in sorted(yn):
            tot = sum(yn[fam].values())
            if tot:
                frac = {k: f"{100*v/tot:.0f}%" for k, v in yn[fam].items()}
                print(f"   {fam} yes/no balance: {frac}")

    # 2b. cross-file leaks
    cross = {h: set(f) for h, f in all_hashes.items() if len(set(f)) > 1}
    if cross:
        print(f"\nCROSS-FILE duplicate prompts (potential leak): {len(cross)}")
        ok = False
    else:
        print("\nno cross-file duplicate prompts")

    print("\nRESULT:", "PASS" if ok else "ISSUES FOUND (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
