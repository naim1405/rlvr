#!/usr/bin/env python3
"""Split a Gym train JSONL into the seven transfer-matrix training conditions.

Produces one single-family file per reasoning family (P1..P6) and one balanced
mixed file, each holding the SAME number of problems so the single-family runs
and the mixed run are a fair comparison (training volume is held constant; only
the family composition varies). P0 is never a training condition.

  R1..R6  train_P1.jsonl .. train_P6.jsonl   (N rows, one family each)
  R-Mix   train_mix.jsonl                    (N rows, N/6 per family)

Usage (from nemotron_rlvr/):
  python3 scripts/split_by_family.py --in data/train-split.jsonl --out-dir data/conditions
  python3 scripts/split_by_family.py --in data/train-split.jsonl --out-dir data/conditions --size 10000
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

FAMILIES = ["P1", "P2", "P3", "P4", "P5", "P6"]


def _family(row: dict) -> str:
    fam = row.get("family") or row.get("task_name") or ""
    fam = str(fam).strip().upper()
    if fam.startswith("P") and fam[1:2].isdigit():
        return fam[:2]
    # task_name like p3_logic -> P3
    m = {"P0": "P0", "P1": "P1", "P2": "P2", "P3": "P3", "P4": "P4", "P5": "P5", "P6": "P6"}
    for k in m:
        if k.lower() in str(fam).lower():
            return k
    return fam


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--size", type=int, default=0,
                    help="rows per condition; 0 = use the largest size every family can supply")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_fam: dict[str, list[dict]] = defaultdict(list)
    with args.inp.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            fam = _family(row)
            if fam in FAMILIES:
                by_fam[fam].append(row)

    for fam in FAMILIES:
        rng.shuffle(by_fam[fam])
    avail = {fam: len(by_fam[fam]) for fam in FAMILIES}
    print("available per family:", avail)
    if any(avail[f] == 0 for f in FAMILIES):
        raise SystemExit(f"Some family has no rows: {avail}")

    # Single-family size N: either the requested --size, or the largest N such
    # that the mix can also draw N/6 from every family (so all seven files are N).
    per_fam_cap = min(avail.values())
    if args.size:
        n = args.size
        if n > min(avail.values()):
            print(f"WARN: --size {n} exceeds the smallest family ({min(avail.values())}); "
                  "single-family files will be capped to what each family has.")
    else:
        n = per_fam_cap

    mix_per_fam = n // len(FAMILIES)

    # Single-family files.
    for fam in FAMILIES:
        rows = by_fam[fam][: min(n, avail[fam])]
        _write(args.out_dir / f"train_{fam}.jsonl", rows)
        print(f"train_{fam}.jsonl: {len(rows)} rows")

    # Balanced mix: mix_per_fam from each family, drawn from the tail so it does
    # not fully overlap the single-family head (reduces duplicate instances).
    mix: list[dict] = []
    for fam in FAMILIES:
        pool = by_fam[fam]
        take = pool[-mix_per_fam:] if len(pool) >= mix_per_fam else pool
        mix.extend(take)
    rng.shuffle(mix)
    _write(args.out_dir / "train_mix.jsonl", mix)
    print(f"train_mix.jsonl: {len(mix)} rows ({mix_per_fam}/family)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
