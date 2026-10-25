#!/usr/bin/env python3
"""Flatten the P0–P6 corpus into NeMo Gym JSONL for GRPO.

Output schema (one object per line):

    {
      "responses_create_params": {
        "input": [{"role": "user", "content": "<prompt>"}]
      },
      "expected_answer": ...,
      "task_name": "p0_retrieval",
      "instance_id": "...",
      "family": "P0",
      "verifier": "p0_retrieval",
      "verifier_meta": {...},
      "prompt": "<prompt>",
      "agent_ref": {
        "type": "responses_api_agents",
        "name": "nemotron_verifier_simple_agent"
      }
    }

The `prompt` field is kept so train_utils.py can fall back if the Gym converter
is missing. Gym itself reads `responses_create_params` + extra fields.

Usage:
    python3 scripts/reformat_dataset.py
    python3 scripts/reformat_dataset.py --corpus /path/to/dataset --out-dir data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional


FAMILY_TO_TASK = {
    "P0": "p0_retrieval",
    "P1": "p1_arithmetic",
    "P2": "p2_symbolic",
    "P3": "p3_logic",
    "P4": "p4_planning",
    "P5": "p5_pattern",
    "P6": "p6_constraint",
    "p0_retrieval": "p0_retrieval",
    "p1_arithmetic": "p1_arithmetic",
    "p2_symbolic": "p2_symbolic",
    "p3_logic": "p3_logic",
    "p4_planning": "p4_planning",
    "p5_pattern": "p5_pattern",
    "p6_constraint": "p6_constraint",
}

AGENT_REF = {
    "type": "responses_api_agents",
    "name": "nemotron_verifier_simple_agent",
}


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _first_existing(record: dict, *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def extract_prompt(record: dict) -> str:
    prompt = _first_existing(record, "prompt", "question", "input")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    rcp = record.get("responses_create_params") or {}
    messages = rcp.get("input") if isinstance(rcp, dict) else None
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict) and part.get("text"):
                        texts.append(str(part["text"]))
                    elif isinstance(part, str):
                        texts.append(part)
                if texts:
                    return "".join(texts)
    raise KeyError(f"Record missing prompt: keys={list(record.keys())}")


def extract_expected_answer(record: dict) -> Any:
    answer = _first_existing(
        record,
        "expected_answer",
        "ground_truth",
        "answer",
        "target",
        "completion",
    )
    if answer is not None:
        return answer
    raise KeyError(f"Record missing expected_answer: keys={list(record.keys())}")


def extract_task_name(record: dict, default_family: Optional[str] = None) -> str:
    raw = _first_existing(record, "task_name", "family", "verifier")
    if raw is None:
        raw = default_family
    if raw is None:
        raise KeyError(f"Record missing task_name/family: keys={list(record.keys())}")
    key = str(raw).strip()
    if key in FAMILY_TO_TASK:
        return FAMILY_TO_TASK[key]
    upper = key.upper()
    if upper in FAMILY_TO_TASK:
        return FAMILY_TO_TASK[upper]
    lowered = key.lower().replace("-", "_")
    if lowered in FAMILY_TO_TASK:
        return FAMILY_TO_TASK[lowered]
    for family, task in FAMILY_TO_TASK.items():
        if family.lower() in lowered or task in lowered:
            return task
    raise KeyError(f"Unknown family/task_name {raw!r}")


def extract_family(task_name: str, record: dict) -> str:
    raw = record.get("family")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().upper() if len(raw.strip()) <= 2 else raw.strip()
    prefix = task_name.split("_", 1)[0]  # p0
    if prefix.startswith("p") and prefix[1:].isdigit():
        return f"P{prefix[1:]}"
    return task_name


def extract_verifier_meta(record: dict) -> dict:
    meta = record.get("verifier_meta")
    if isinstance(meta, dict):
        return meta
    nested = record.get("extra_env_info")
    if isinstance(nested, dict) and isinstance(nested.get("verifier_meta"), dict):
        return nested["verifier_meta"]
    return {}


def extract_instance_id(record: dict, fallback: str) -> str:
    value = _first_existing(record, "instance_id", "id")
    if value is None:
        return fallback
    return str(value)


def to_gym_record(record: dict, default_family: Optional[str] = None, line_no: int = 0) -> dict:
    task_name = extract_task_name(record, default_family=default_family)
    prompt = extract_prompt(record)
    expected = extract_expected_answer(record)
    family = extract_family(task_name, record)
    instance_id = extract_instance_id(record, fallback=f"{task_name}:{line_no}")
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": prompt}],
        },
        "expected_answer": expected,
        "task_name": task_name,
        "instance_id": instance_id,
        "family": family,
        "verifier": task_name,
        "verifier_meta": extract_verifier_meta(record),
        "prompt": prompt,
        "agent_ref": dict(AGENT_REF),
    }


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc


def family_from_filename(path: Path) -> Optional[str]:
    name = path.stem.lower()
    for family, task in FAMILY_TO_TASK.items():
        if family.lower() in name or task in name:
            return family if family.startswith("P") else family
    parent = path.parent.name.lower()
    for family in ("p0", "p1", "p2", "p3", "p4", "p5", "p6"):
        if parent.startswith(family) or parent == family:
            return family.upper()
    return None


def collect_source_files(corpus_dir: Path, split: str) -> list[Path]:
    files: list[Path] = []
    patterns = [
        f"{split}.jsonl",
        f"{split}-split.jsonl",
        f"{split}_split.jsonl",
    ]
    if corpus_dir.is_file():
        return [corpus_dir]
    for path in sorted(corpus_dir.rglob("*.jsonl")):
        lowered = path.name.lower()
        if any(lowered == pat or lowered.endswith(pat) for pat in patterns):
            files.append(path)
            continue
        # family shards: p0/train.jsonl, dataset/p1/train.jsonl
        if path.stem.lower() == split:
            files.append(path)
    return files


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def reformat(corpus_dir: Path, out_dir: Path, train_split: str, val_split: str) -> None:
    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_dir}. Datasets are not in git — "
            "point --corpus at the local P0–P6 JSONL tree."
        )

    split_map = {"train": train_split, "val": val_split}
    for out_name, src_split in split_map.items():
        sources = collect_source_files(corpus_dir, src_split)
        if not sources:
            raise FileNotFoundError(
                f"No JSONL for split {src_split!r} under {corpus_dir}. "
                f"Expected files named {src_split}.jsonl (optionally under p0/…/p6/)."
            )
        rows: list[dict] = []
        for src in sources:
            default_family = family_from_filename(src)
            for line_no, record in iter_jsonl(src):
                rows.append(
                    to_gym_record(record, default_family=default_family, line_no=line_no)
                )
        dest = out_dir / f"{out_name}-split.jsonl"
        write_jsonl(dest, rows)
        print(f"Wrote {len(rows)} rows from {len(sources)} file(s) -> {dest}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("dataset"),
        help="Directory (or single JSONL) containing per-family shards",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for train-split.jsonl and val-split.jsonl",
    )
    parser.add_argument("--train-split", default="train", help="Source split name for training")
    parser.add_argument(
        "--val-split",
        default="test_id",
        help="Source split name for validation (corpus uses test_id / TEST_ID)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reformat(args.corpus, args.out_dir, args.train_split, args.val_split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
