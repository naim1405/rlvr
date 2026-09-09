"""JSONL → AllTaskProcessedDataset for Gym GRPO."""

from __future__ import annotations

import json
import sys
from itertools import chain, repeat
from pathlib import Path

import torch


def _user_text_from_example(example: dict) -> str:
    prompt = example.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    rcp = example.get("responses_create_params") or {}
    messages = rcp.get("input") if isinstance(rcp, dict) else None
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("text"):
                        parts.append(str(item["text"]))
                    elif isinstance(item, str):
                        parts.append(item)
                if parts:
                    return "".join(parts)
    return ""


def _looks_like_gym_example(example: dict) -> bool:
    rcp = example.get("responses_create_params")
    agent = example.get("agent_ref")
    return (
        isinstance(rcp, dict)
        and isinstance(rcp.get("input"), list)
        and isinstance(agent, dict)
        and agent.get("name")
    )


def _to_gym_example(example: dict, idx: int) -> dict:
    if _looks_like_gym_example(example):
        return example
    scripts_dir = Path(__file__).resolve().parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from reformat_dataset import to_gym_record

    return to_gym_record(example, line_no=idx)


def load_nemo_gym_dataset(jsonl_path: str, tokenizer, num_repeats: int | None = None):
    """Load Gym JSONL and wrap each row as a NeMo-RL AllTaskProcessedDataset."""
    from nemo_rl.data.datasets import AllTaskProcessedDataset
    from nemo_rl.data.interfaces import DatumSpec

    try:
        from nemo_rl.environments.nemo_gym import nemo_gym_example_to_nemo_rl_datum_spec
    except ImportError:

        def nemo_gym_example_to_nemo_rl_datum_spec(nemo_gym_example: dict, idx: int) -> DatumSpec:
            content = _user_text_from_example(nemo_gym_example)
            return DatumSpec(
                message_log=[
                    {
                        "role": "user",
                        "content": content,
                        "token_ids": torch.tensor([]),
                    }
                ],
                length=0,
                extra_env_info=nemo_gym_example,
                loss_multiplier=1.0,
                idx=idx,
                task_name="nemo_gym",
                stop_strings=None,
                token_ids=[],
            )

    with open(jsonl_path, "r", encoding="utf-8") as handle:
        examples = [json.loads(line) for line in handle if line.strip()]

    converted = [_to_gym_example(ex, idx) for idx, ex in enumerate(examples)]
    n_converted = sum(1 for old, new in zip(examples, converted) if old is not new)
    examples = converted
    print(f"Loaded dataset from {jsonl_path}: {len(examples)} records")
    if n_converted:
        print(
            f"[train_utils] converted {n_converted}/{len(examples)} records to Gym schema "
            "(responses_create_params + agent_ref). On-disk JSONL is unchanged."
        )

    if num_repeats:
        examples = list(chain.from_iterable(repeat(ex, num_repeats) for ex in examples))

    datum_specs = [
        nemo_gym_example_to_nemo_rl_datum_spec(ex, idx) for idx, ex in enumerate(examples)
    ]

    def passthrough_processor(datum_dict, *args, **kwargs):
        return datum_dict

    return AllTaskProcessedDataset(
        datum_specs,
        tokenizer,
        None,
        passthrough_processor,
    )


def prepare_nemo_gym_dataset(config, tokenizer, jsonl_path: str):
    repeats = None
    try:
        from omegaconf import OmegaConf

        repeats = OmegaConf.select(config, "data.dataset_num_repeats")
    except Exception:
        data_cfg = config.get("data") if hasattr(config, "get") else None
        if data_cfg is not None and hasattr(data_cfg, "get"):
            repeats = data_cfg.get("dataset_num_repeats")
    return load_nemo_gym_dataset(jsonl_path, tokenizer, num_repeats=repeats)
