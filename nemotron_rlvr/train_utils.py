"""Helper dataset loader for RLVR training."""

import json
from itertools import chain, repeat
import torch


def load_nemo_gym_dataset(jsonl_path: str, tokenizer, num_repeats: int | None = None):
    """Load JSONL data and convert to NeMo-RL compatible AllTaskProcessedDataset."""
    from nemo_rl.data.datasets import AllTaskProcessedDataset
    from nemo_rl.data.interfaces import DatumSpec

    try:
        from nemo_rl.environments.nemo_gym import nemo_gym_example_to_nemo_rl_datum_spec
    except ImportError:
        def nemo_gym_example_to_nemo_rl_datum_spec(nemo_gym_example: dict, idx: int) -> DatumSpec:
            return DatumSpec(
                message_log=[{"role": "user", "content": nemo_gym_example.get("prompt", ""), "token_ids": torch.tensor([])}],
                length=0,
                extra_env_info=nemo_gym_example,
                loss_multiplier=1.0,
                idx=idx,
                task_name="nemo_gym",
                stop_strings=None,
                token_ids=[],
            )

    with open(jsonl_path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded dataset from {jsonl_path}: {len(examples)} records")

    if num_repeats:
        examples = list(chain.from_iterable(repeat(ex, num_repeats) for ex in examples))

    datum_specs = [
        nemo_gym_example_to_nemo_rl_datum_spec(ex, idx)
        for idx, ex in enumerate(examples)
    ]

    def passthrough_processor(datum_dict, *args, **kwargs):
        return datum_dict

    return AllTaskProcessedDataset(
        datum_specs,
        tokenizer,
        None,
        passthrough_processor,
    )
