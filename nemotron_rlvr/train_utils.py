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


def _family_key(example: dict) -> str:
    fam = example.get("family") or example.get("task_name") or "unknown"
    return str(fam).strip().upper()


def interleave_by_family(examples: list[dict]) -> list[dict]:
    """Round-robin rows across families, preserving order within a family.

    reformat_dataset.py concatenates the corpus family by family (P0, P1, ...).
    NeMo RL's validation loader is unshuffled and `validate()` only consumes
    `max_val_samples // val_batch_size` batches, so on a capped run the
    validation slice was 100% P0. Interleaving makes every prefix of the
    dataset family-balanced without changing the row contents.
    """
    buckets: dict[str, list[dict]] = {}
    for ex in examples:
        buckets.setdefault(_family_key(ex), []).append(ex)
    if len(buckets) <= 1:
        return examples
    order = sorted(buckets)  # deterministic: P0, P1, ..., P6
    out: list[dict] = []
    longest = max(len(b) for b in buckets.values())
    for i in range(longest):
        for fam in order:
            bucket = buckets[fam]
            if i < len(bucket):
                out.append(bucket[i])
    return out


def load_nemo_gym_dataset(
    jsonl_path: str,
    tokenizer,
    num_repeats: int | None = None,
    interleave_families: bool = False,
):
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

    if interleave_families:
        examples = interleave_by_family(examples)
        print(
            "[train_utils] interleaved rows round-robin across families so any "
            "prefix (e.g. a capped validation slice) covers every family"
        )

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


def _select(config, dotted: str, default=None):
    try:
        from omegaconf import OmegaConf

        value = OmegaConf.select(config, dotted)
        return default if value is None else value
    except Exception:
        node = config
        for part in dotted.split("."):
            if node is None or not hasattr(node, "get"):
                return default
            node = node.get(part)
        return default if node is None else node


def prepare_nemo_gym_dataset(config, tokenizer, jsonl_path: str):
    repeats = _select(config, "data.dataset_num_repeats")
    # Default on: the train loader shuffles anyway, and the (unshuffled)
    # validation loader needs it to see more than the first family.
    interleave = bool(_select(config, "data.interleave_families", True))
    return load_nemo_gym_dataset(
        jsonl_path, tokenizer, num_repeats=repeats, interleave_families=interleave
    )
