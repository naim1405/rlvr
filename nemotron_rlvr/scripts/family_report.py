#!/usr/bin/env python3
"""Per-family reward report from NeMo RL's per-step JSONL logs.

NeMo RL writes ``logs/val_data_step{N}.jsonl`` (one row per validation sample,
in dataset order) and ``logs/train_data_step{N}.jsonl`` (one row per rollout).
Neither carries the task family, so this script re-derives it by matching the
logged user prompt back to the dataset JSONL that train.py loaded (after the
same family interleaving train_utils.py applies).

Usage (from nemotron_rlvr/):
    python3 scripts/family_report.py                       # val, all steps found
    python3 scripts/family_report.py --split train         # train rollouts
    python3 scripts/family_report.py --logs logs --val data/val-split.jsonl

Output: one table per step with accuracy per family, plus the overall number
NeMo RL prints. Works without torch/omegaconf.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def _family_key(example: dict) -> str:
    fam = example.get("family") or example.get("task_name") or "unknown"
    return str(fam).strip().upper()


def interleave_by_family(examples: list[dict]) -> list[dict]:
    # Mirror of train_utils.interleave_by_family (kept local: no torch import).
    buckets: dict[str, list[dict]] = {}
    for ex in examples:
        buckets.setdefault(_family_key(ex), []).append(ex)
    if len(buckets) <= 1:
        return examples
    order = sorted(buckets)
    out: list[dict] = []
    longest = max(len(b) for b in buckets.values())
    for i in range(longest):
        for fam in order:
            bucket = buckets[fam]
            if i < len(bucket):
                out.append(bucket[i])
    return out


def _prompt_of(example: dict) -> str:
    rcp = example.get("responses_create_params") or {}
    for msg in rcp.get("input") or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content") or "")
    return str(example.get("prompt") or "")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def load_dataset(path: Path, interleave: bool) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return interleave_by_family(rows) if interleave else rows


def logged_user_prompt(row: dict) -> str | None:
    content = row.get("content")
    if isinstance(content, list):  # validation: list of {role, content}
        for msg in content:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content") or "")
        return None
    if isinstance(content, str):  # train: flattened chat text
        return content
    return None


def step_of(path: str) -> int:
    m = re.search(r"step(\d+)\.jsonl$", path)
    return int(m.group(1)) if m else -1


def report(split: str, logs_dir: Path, dataset: list[dict], min_step: int) -> None:
    files = sorted(glob.glob(str(logs_dir / f"{split}_data_step*.jsonl")), key=step_of)
    files = [f for f in files if step_of(f) >= min_step]
    if not files:
        print(f"No {split}_data_step*.jsonl under {logs_dir}")
        return

    by_prompt: dict[str, str] = {}
    for ex in dataset:
        by_prompt.setdefault(_norm(_prompt_of(ex)), _family_key(ex))
    families = sorted(set(by_prompt.values()))

    print(f"\n{split.upper()} — per-family mean reward (n)  [logs: {logs_dir}]")
    header = f"{'step':>6} | " + " | ".join(f"{f:>12}" for f in families) + f" | {'overall':>12}"
    print(header)
    print("-" * len(header))

    for f in files:
        step = step_of(f)
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        unmatched = 0
        with open(f, "r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        for idx, row in enumerate(rows):
            reward = row.get("rewards")
            if isinstance(reward, list):
                reward = reward[0] if reward else None
            if reward is None:
                continue
            fam = None
            prompt = logged_user_prompt(row)
            if prompt is not None:
                key = _norm(prompt)
                fam = by_prompt.get(key)
                if fam is None and isinstance(row.get("content"), str):
                    # Train logs are flattened chat text; find which dataset prompt it contains.
                    for p, fm in by_prompt.items():
                        if p and p in key:
                            fam = fm
                            break
            if fam is None and split == "val" and idx < len(dataset):
                fam = _family_key(dataset[idx])  # val rows are in dataset order
            if fam is None:
                unmatched += 1
                fam = "unknown"
            sums[fam] += float(reward)
            counts[fam] += 1
        total_n = sum(counts.values())
        total = sum(sums.values()) / total_n if total_n else float("nan")
        cells = []
        for fam in families:
            n = counts.get(fam, 0)
            cells.append(f"{(sums[fam] / n):.2f} ({n:>3})" if n else f"{'-':>12}")
        line = f"{step:>6} | " + " | ".join(f"{c:>12}" for c in cells) + f" | {total:.3f} ({total_n:>3})"
        if unmatched:
            line += f"   [{unmatched} unmatched]"
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default="logs", help="NeMo RL logger.log_dir")
    ap.add_argument("--split", choices=["val", "train", "both"], default="val")
    ap.add_argument("--val", default="data/val-split.jsonl")
    ap.add_argument("--train", default="data/train-split.jsonl")
    ap.add_argument("--no-interleave", action="store_true", help="dataset was loaded with data.interleave_families=false")
    ap.add_argument("--min-step", type=int, default=0)
    args = ap.parse_args()

    os.chdir(HERE.parent)
    logs_dir = Path(args.logs)
    interleave = not args.no_interleave
    if args.split in ("val", "both"):
        report("val", logs_dir, load_dataset(Path(args.val), interleave), args.min_step)
    if args.split in ("train", "both"):
        report("train", logs_dir, load_dataset(Path(args.train), interleave), args.min_step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
