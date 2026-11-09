#!/usr/bin/env python3
"""Assemble the 6x6 reasoning-transfer matrix from evaluation JSONL.

Each cell T[i,j] = accuracy(model trained on family Pi, eval family Pj)
                 - accuracy(post-SFT baseline, eval family Pj),

measured on the SAME evaluation set and paired by instance_id, with a bootstrap
95% CI over eval instances. The diagonal is in-domain gain; off-diagonal is
transfer. A P0 column is reported as the factual-stability check, and an R-Mix
row as the multi-family baseline.

Inputs are the per-model eval files written by scripts/evaluate_model.py (one
row per generation, carrying `family`, `instance_id`, `reward`). Average over
samples first if a file has multiple samples per instance.

Usage (from nemotron_rlvr/):
  python3 scripts/build_transfer_matrix.py \
      --baseline eval/sft_baseline.jsonl \
      --condition P1=eval/trained_P1.jsonl --condition P2=eval/trained_P2.jsonl \
      ... --condition P6=eval/trained_P6.jsonl --condition MIX=eval/trained_mix.jsonl \
      --out-json eval/transfer_matrix.json --out-markdown eval/transfer_matrix.md
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

EVAL_FAMILIES = ["P1", "P2", "P3", "P4", "P5", "P6"]
TRAIN_ROWS = ["P1", "P2", "P3", "P4", "P5", "P6", "MIX"]


def _family(row: dict) -> str:
    fam = str(row.get("family") or row.get("task_name") or "").strip().upper()
    return fam[:2] if fam[:2] in ("P0", "P1", "P2", "P3", "P4", "P5", "P6") else fam


def load_rewards(path: Path) -> dict[str, dict]:
    """instance_id -> {family, reward(mean over samples)}."""
    acc: dict[str, list[float]] = defaultdict(list)
    fam: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            iid = str(r.get("instance_id") or r.get("row_index"))
            acc[iid].append(float(r.get("reward", 0.0)))
            fam[iid] = _family(r)
    return {iid: {"family": fam[iid], "reward": sum(v) / len(v)} for iid, v in acc.items()}


def paired_delta(base: dict, cond: dict, family: str) -> list[float]:
    """Per-instance (cond - base) reward deltas for one eval family."""
    out = []
    for iid, b in base.items():
        if b["family"] != family or iid not in cond:
            continue
        out.append(cond[iid]["reward"] - b["reward"])
    return out


def boot_ci(deltas: list[float], n_boot: int, rng: random.Random) -> tuple[float, float, float]:
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    mean = sum(deltas) / len(deltas)
    boots = []
    k = len(deltas)
    for _ in range(n_boot):
        s = sum(deltas[rng.randrange(k)] for _ in range(k)) / k
        boots.append(s)
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[min(len(boots) - 1, int(0.975 * len(boots)))]
    return mean, lo, hi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--condition", action="append", default=[], metavar="NAME=PATH",
                    help="e.g. P1=eval/trained_P1.jsonl ; repeat for each condition")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--out-markdown", type=Path)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = load_rewards(args.baseline)
    conds = {}
    for spec in args.condition:
        name, _, path = spec.partition("=")
        conds[name.strip().upper()] = load_rewards(Path(path))

    # Baseline per-family accuracy (incl. P0 for the stability check).
    base_acc = {}
    for fam in EVAL_FAMILIES + ["P0"]:
        vals = [b["reward"] for b in base.values() if b["family"] == fam]
        base_acc[fam] = sum(vals) / len(vals) if vals else float("nan")

    matrix = {}   # train_row -> {eval_fam -> {delta, ci_lo, ci_hi, n}}
    for row in TRAIN_ROWS:
        if row not in conds:
            continue
        cond = conds[row]
        matrix[row] = {}
        for fam in EVAL_FAMILIES + ["P0"]:
            deltas = paired_delta(base, cond, fam)
            mean, lo, hi = boot_ci(deltas, args.bootstrap, rng)
            matrix[row][fam] = {"delta": mean, "ci_lo": lo, "ci_hi": hi, "n": len(deltas)}

    report = {"baseline_accuracy": base_acc, "matrix": matrix,
              "note": "delta = trained_acc - sft_baseline_acc, paired by instance_id; "
                      "95% bootstrap CI; diagonal = in-domain, off-diagonal = transfer; "
                      "P0 column = factual-stability check."}
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2))

    # Markdown heatmap-style table (significant cells marked with *).
    lines = ["# Reasoning-transfer matrix (delta over SFT baseline)", "",
             "Rows = training family, columns = eval family. `*` = 95% CI excludes 0.",
             "Diagonal = in-domain gain; off-diagonal = transfer; P0 = stability.", "",
             "| train\\eval | " + " | ".join(EVAL_FAMILIES) + " | P0 |",
             "|---|" + "---|" * (len(EVAL_FAMILIES) + 1)]
    for row in TRAIN_ROWS:
        if row not in matrix:
            continue
        cells = []
        for fam in EVAL_FAMILIES + ["P0"]:
            c = matrix[row][fam]
            sig = "*" if (c["ci_lo"] > 0 or c["ci_hi"] < 0) else ""
            cells.append(f"{c['delta']:+.1f}{sig}" if c["n"] else "-")
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    lines += ["", "Baseline accuracy (%): " +
              ", ".join(f"{f} {100*base_acc[f]:.0f}" for f in EVAL_FAMILIES + ["P0"]
                        if base_acc[f] == base_acc[f])]
    md = "\n".join(lines) + "\n"
    if args.out_markdown:
        args.out_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.out_markdown.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
