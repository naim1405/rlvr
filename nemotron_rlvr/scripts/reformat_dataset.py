#!/usr/bin/env python3
"""Dataset reformatting script for checkpoints/new-dataset/corpus (P0-P6).

Reads the multi-family JSONL corpus directory (p0/ ... p6/) using:
  - train.jsonl for training
  - test_ood.jsonl for validation/test
Maintains original split boundaries without re-splitting.
"""

import json
from pathlib import Path

CORPUS_DIR = Path(__file__).parent.parent.parent / "checkpoints" / "new-dataset" / "corpus"
if not CORPUS_DIR.exists():
    CORPUS_DIR = Path(__file__).parent.parent / "checkpoints" / "new-dataset" / "corpus"

DATA_DIR = Path(__file__).parent.parent / "data"

FAMILY_MAP = {
    "P0": "p0_retrieval",
    "P1": "p1_arithmetic",
    "P2": "p2_symbolic",
    "P3": "p3_logic",
    "P4": "p4_planning",
    "P5": "p5_pattern",
    "P6": "p6_constraint",
}


def load_family_records(family_folder: Path, filename: str) -> list[dict]:
    filepath = family_folder / filename
    if not filepath.exists():
        print(f"Warning: File {filepath} not found!")
        return []

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            family = item.get("family", family_folder.name.upper())
            task_name = FAMILY_MAP.get(family, f"{family.lower()}_task")

            rec = {
                "prompt": item.get("prompt", ""),
                "expected_answer": item.get("ground_truth", ""),
                "task_name": task_name,
                "instance_id": item.get("instance_id", ""),
                "family": family,
                "verifier": item.get("verifier", ""),
                "verifier_meta": item.get("verifier_meta", {}),
            }
            records.append(rec)
    return records


def main():
    if not CORPUS_DIR.exists():
        print(f"ERROR: Corpus directory not found at {CORPUS_DIR}")
        return

    print(f"Processing corpus from: {CORPUS_DIR}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_train = []
    all_val = []

    for folder_name in sorted(FAMILY_MAP.keys()):
        family_dir = CORPUS_DIR / folder_name.lower()
        if not family_dir.exists():
            print(f"Warning: Folder {family_dir} does not exist, skipping...")
            continue

        train_recs = load_family_records(family_dir, "train.jsonl")
        val_recs = load_family_records(family_dir, "test_ood.jsonl")

        all_train.extend(train_recs)
        all_val.extend(val_recs)

        print(f"Family {folder_name} ({FAMILY_MAP[folder_name]}): {len(train_recs)} train (from train.jsonl), {len(val_recs)} val (from test_ood.jsonl).")

    train_path = DATA_DIR / "train-split.jsonl"
    val_path = DATA_DIR / "val-split.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for rec in all_train:
            f.write(json.dumps(rec) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for rec in all_val:
            f.write(json.dumps(rec) + "\n")

    print("\n--- Dataset Processing Summary ---")
    print(f"Total Train Records Written: {len(all_train)} -> {train_path}")
    print(f"Total Val Records Written:   {len(all_val)} -> {val_path}")


if __name__ == "__main__":
    main()
