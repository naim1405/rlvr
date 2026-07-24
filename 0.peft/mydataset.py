"""Qwen PEFT SFT dataset for NeMo AutoModel using checkpoints/dataset/train.jsonl."""

from collections.abc import Iterable
from pathlib import Path
import json
import random
from typing import List

from torch.utils.data import Dataset


class QwenPEFTDataset(Dataset):
    """Qwen chat-format SFT dataset for NeMo AutoModel.

    Loads JSONL files (e.g., /tmp/checkpoints/dataset/train.jsonl) containing:
      - prompt: User question / task input
      - completion: Assistant response with <think>...</think><answer>...</answer>

    Optionally supports fallback key names:
      - question / input -> user prompt
      - formatted_output / response -> assistant completion

    Trains on pure User -> Assistant turns without system prompt by default,
    making XML formatting an unconditional response style for RLVR inference.

    Prompt tokens are masked with -100 so loss is applied only to assistant tokens.
    """

    DEFAULT_SYSTEM_PROMPT = """You must always answer using the following format:

<think>
...
</think>
<answer>
...
</answer>
"""

    def __init__(
        self,
        tokenizer,
        path: str | Path,
        max_length: int = 2048,
        split: str = "train",
        validation_fraction: float = 0.02,
        seed: int = 1111,
        include_system_prompt: bool = False,
        system_prompt: str | None = None,
        system_prompt_fraction: float = 1.0,
        num_samples_limit: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.path = Path(path)
        self.max_length = max_length
        self.split = split.lower()
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.include_system_prompt = include_system_prompt
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.system_prompt_fraction = system_prompt_fraction
        self.num_samples_limit = num_samples_limit

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.path}")

        # Load raw JSONL file
        raw_rows = self._read_jsonl(self.path)

        # Train / Validation splitting
        if self.validation_fraction > 0 and self.split in {"train", "validation", "val"}:
            rng = random.Random(f"{self.seed}:{self.path.name}")
            indices = list(range(len(raw_rows)))
            rng.shuffle(indices)
            val_count = max(1, int(len(indices) * self.validation_fraction))
            val_indices = set(indices[:val_count])

            if self.split in {"validation", "val"}:
                self.data = [row for i, row in enumerate(raw_rows) if i in val_indices]
            else:
                self.data = [row for i, row in enumerate(raw_rows) if i not in val_indices]
        else:
            self.data = raw_rows

        if self.num_samples_limit is not None:
            self.data = self.data[: self.num_samples_limit]

        if not self.data:
            raise ValueError(f"No samples loaded from path={str(self.path)!r} for split={self.split!r}")

    def __len__(self):
        return len(self.data)

    def _read_jsonl(self, path: Path) -> list[dict]:
        rows = []
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(path.glob("*.jsonl"))
        else:
            raise ValueError(f"Invalid path: {path}")

        for p in files:
            with p.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in {p} at line {line_no}: {e}") from e
        return rows

    def _extract_user_prompt(self, sample: dict) -> str:
        if "prompt" in sample:
            return str(sample["prompt"])
        if "question" in sample:
            return str(sample["question"])
        if "input" in sample:
            return str(sample["input"])
        raise KeyError(f"Sample missing prompt/question key: {list(sample.keys())}")

    def _extract_assistant_completion(self, sample: dict) -> str:
        if "completion" in sample:
            return str(sample["completion"])
        if "formatted_output" in sample:
            return str(sample["formatted_output"])
        if "response" in sample:
            return str(sample["response"])
        if "ground_truth" in sample:
            gt = sample["ground_truth"]
            return f"<think>\nWorking through this now.\n</think>\n<answer>\n{gt}\n</answer>"
        raise KeyError(f"Sample missing completion/formatted_output key: {list(sample.keys())}")

    def _use_system_prompt_for(self, idx: int) -> bool:
        if not self.include_system_prompt:
            return False
        if self.system_prompt_fraction >= 1.0:
            return True
        if self.system_prompt_fraction <= 0.0:
            return False

        rng = random.Random((self.seed, idx))
        return rng.random() < self.system_prompt_fraction

    def __getitem__(self, idx: int):
        sample = self.data[idx]

        user_prompt = self._extract_user_prompt(sample)
        assistant_completion = self._extract_assistant_completion(sample)

        messages = []
        if self._use_system_prompt_for(idx):
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": assistant_completion})

        prompt_messages = messages[:-1]

        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length + 1,
            add_special_tokens=False,
        )

        prompt = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length + 1,
            add_special_tokens=False,
        )

        full_ids = full["input_ids"]
        full_attention = full.get("attention_mask", [1] * len(full_ids))

        if len(full_ids) < 2:
            return {
                "input_ids": full_ids,
                "attention_mask": full_attention,
                "labels": [-100] * len(full_ids),
            }

        input_ids = full_ids[:-1]
        attention_mask = full_attention[:-1]
        labels = full_ids[1:].copy()

        prompt_len = len(prompt["input_ids"])
        mask_until = min(max(prompt_len - 1, 0), len(labels))
        labels[:mask_until] = [-100] * mask_until

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# Backward-compatible aliases
QwenSFTDataset = QwenPEFTDataset
