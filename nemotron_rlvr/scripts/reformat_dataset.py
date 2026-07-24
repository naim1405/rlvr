#!/usr/bin/env python3
"""Script to convert JSON dataset into split JSONL files for NeMo-RL / NeMo-Gym."""

import json
from pathlib import Path
import random

SOURCE_JSON = Path(__file__).parent.parent.parent / "phase2_format_dataset.json"
if not SOURCE_JSON.exists():
    SOURCE_JSON = Path(__file__).parent.parent / "phase2_format_dataset.json"

DATA_DIR = Path(__file__).parent.parent / "data"

CATEGORY_TO_TASK = {
    "math_word_problem": "p1_arithmetic",
    "word_problem": "p1_arithmetic",
    "arithmetic": "p1_arithmetic",
    "symbolic": "p2_symbolic",
    "symbolic_transformation": "p2_symbolic",
    "logic": "p3_logic",
    "deductive_logic": "p3_logic",
    "science_reasoning": "p0_retrieval",
    "factual_retrieval": "p0_retrieval",
    "procedural": "p4_planning",
    "planning": "p4_planning",
    "pattern_completion": "p5_pattern",
    "pattern": "p5_pattern",
    "constraint_satisfaction": "p6_constraint",
    "constraint": "p6_constraint",
}


def main():
    if not SOURCE_JSON.exists():
        print(f"Source JSON file not found at {SOURCE_JSON}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(SOURCE_JSON, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"Loaded {len(items)} raw dataset items.")

    formatted_items = []
    for idx, item in enumerate(items):
        cat = item.get("category", "").lower()
        task_name = CATEGORY_TO_TASK.get(cat, "p1_arithmetic")

        record = {
            "prompt": item.get("question", ""),
            "expected_answer": item.get("answer", ""),
            "task_name": task_name,
            "idx": idx,
            "category": cat,
            "source": item.get("source", "unknown"),
        }

        if "constraints" in item:
            record["constraints"] = item["constraints"]
        if "expected_steps" in item:
            record["expected_steps"] = item["expected_steps"]

        formatted_items.append(record)

    random.seed(42)
    random.shuffle(formatted_items)

    val_count = min(100, max(1, int(len(formatted_items) * 0.1)))
    val_items = formatted_items[:val_count]
    train_items = formatted_items[val_count:]

    train_path = DATA_DIR / "train-split.jsonl"
    val_path = DATA_DIR / "val-split.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_items:
            f.write(json.dumps(item) + "\n")

    print(f"Successfully generated train dataset: {train_path} ({len(train_items)} rows)")
    print(f"Successfully generated val dataset:   {val_path} ({len(val_items)} rows)")


if __name__ == "__main__":
    main()
