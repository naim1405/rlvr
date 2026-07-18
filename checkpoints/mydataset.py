from collections.abc import Iterable
from pathlib import Path
import json
import random
from typing import List

from torch.utils.data import Dataset


class QwenSFTDataset(Dataset):
    """Qwen chat-format SFT dataset for NeMo AutoModel using the final SRP corpus.

    Expected dataset layout (inside the Docker container if you mount the
    `checkpoints/` directory to `/tmp/checkpoints`):

        /tmp/checkpoints/new-dataset/corpus/
            p0/
              train.jsonl
              test_id.jsonl
              test_ood.jsonl
              transfer.jsonl
              gen_ood.jsonl
            p1/
            ...
            p6/

    Each JSONL row is expected to contain at least:
      - prompt
      - ground_truth

    This dataset converts each row into chat messages:
      user      -> sample["prompt"]
      assistant -> a synthesized XML-formatted answer using ground_truth

    Because the final corpus does not include gold chain-of-thought, the
    assistant response uses a short fixed reasoning placeholder:

        <think>
        Working through this now.
        </think>
        <answer>
        ...canonical answer...
        </answer>

    The dataset returns shifted causal-LM examples:
      input_ids = full_ids[:-1]
      labels    = full_ids[1:]

    Prompt tokens (everything up to and including the assistant generation
    prefix) are masked with -100 so loss is applied only to assistant
    response tokens.
    """

    DEFAULT_SYSTEM_PROMPT = """You must always answer using the following format:

<think>
...
</think>
<answer>
...
</answer>
"""

    EXPLICIT_SPLIT_FILES = {
        "test_id": "test_id.jsonl",
        "test_ood": "test_ood.jsonl",
        "transfer": "transfer.jsonl",
        "gen_ood": "gen_ood.jsonl",
    }

    def __init__(
        self,
        tokenizer,
        root,
        max_length=2048,
        split="train",
        validation_fraction=0.02,
        seed=1111,
        families=None,
        include_system_prompt=False,
        system_prompt=None,
        system_prompt_fraction=1.0,
        num_samples_limit=None,
    ):
        self.tokenizer = tokenizer
        self.root = Path(root)
        self.max_length = max_length
        self.split = split
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.include_system_prompt = include_system_prompt
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.system_prompt_fraction = system_prompt_fraction
        self.num_samples_limit = num_samples_limit

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {self.root}")

        self.family_dirs = self._resolve_family_dirs(families)
        self.data = self._load_split_data(split)

        if self.num_samples_limit is not None:
            self.data = self.data[: self.num_samples_limit]

        if not self.data:
            raise ValueError(
                f"No samples loaded for split={split!r} from root={str(self.root)!r} "
                f"and families={[p.name for p in self.family_dirs]!r}"
            )

    def __len__(self):
        return len(self.data)

    def _resolve_family_dirs(self, families) -> List[Path]:
        all_family_dirs = sorted(
            [p for p in self.root.iterdir() if p.is_dir() and p.name.lower().startswith("p")],
            key=lambda p: p.name.lower(),
        )
        if not all_family_dirs:
            raise ValueError(f"No family directories found under {self.root}")

        if families is None:
            return all_family_dirs

        if isinstance(families, str):
            requested = [x.strip().lower() for x in families.split(",") if x.strip()]
        elif isinstance(families, Iterable):
            requested = [str(x).strip().lower() for x in families if str(x).strip()]
        else:
            raise TypeError("families must be None, a comma-separated string, or an iterable")

        family_map = {p.name.lower(): p for p in all_family_dirs}
        missing = [name for name in requested if name not in family_map]
        if missing:
            raise ValueError(
                f"Requested family directories not found under {self.root}: {missing}. "
                f"Available: {sorted(family_map)}"
            )
        return [family_map[name] for name in requested]

    def _read_jsonl(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Expected dataset file not found: {path}")

        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {path} at line {line_no}: {e}") from e
        return rows

    def _split_train_validation(self, rows, family_name):
        if self.validation_fraction <= 0:
            return rows, []
        if not (0 < self.validation_fraction < 1):
            raise ValueError("validation_fraction must be in the range (0, 1)")

        indices = list(range(len(rows)))
        rng = random.Random(f"{self.seed}:{family_name}")
        rng.shuffle(indices)

        val_count = max(1, int(len(indices) * self.validation_fraction))
        val_indices = set(indices[:val_count])

        train_rows = [row for i, row in enumerate(rows) if i not in val_indices]
        val_rows = [row for i, row in enumerate(rows) if i in val_indices]
        return train_rows, val_rows

    def _load_split_data(self, split):
        split = split.lower()
        data = []

        if split in {"train", "validation", "val"}:
            if split in {"validation", "val"} and self.validation_fraction <= 0:
                raise ValueError("validation split requested but validation_fraction <= 0")

            for family_dir in self.family_dirs:
                rows = self._read_jsonl(family_dir / "train.jsonl")
                train_rows, val_rows = self._split_train_validation(rows, family_dir.name)
                if split == "train":
                    data.extend(train_rows)
                else:
                    data.extend(val_rows)
            return data

        if split in self.EXPLICIT_SPLIT_FILES:
            split_file = self.EXPLICIT_SPLIT_FILES[split]
            for family_dir in self.family_dirs:
                data.extend(self._read_jsonl(family_dir / split_file))
            return data

        raise ValueError(
            f"Unsupported split: {split!r}. Supported splits: train, validation, val, "
            f"{', '.join(sorted(self.EXPLICIT_SPLIT_FILES))}"
        )

    def _stringify_answer_value(self, value):
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        return json.dumps(value, ensure_ascii=False)

    def _canonical_answer(self, sample):
        if "ground_truth" not in sample:
            raise KeyError("Sample is missing required key 'ground_truth'")

        ground_truth = sample["ground_truth"]

        if isinstance(ground_truth, str):
            return ground_truth

        if isinstance(ground_truth, (int, float, bool)):
            return str(ground_truth)

        if isinstance(ground_truth, list):
            if len(ground_truth) == 0:
                raise ValueError(f"Empty ground_truth list in sample: {sample.get('instance_id', '<unknown>')}")
            if len(ground_truth) == 1:
                return self._stringify_answer_value(ground_truth[0])

            # Some samples accept multiple valid answers (for example, naming any one
            # valid span). Pick one deterministically so training is reproducible.
            seed_value = int(sample.get("seed", 0)) if str(sample.get("seed", 0)).isdigit() else 0
            idx = seed_value % len(ground_truth)
            return self._stringify_answer_value(ground_truth[idx])

        raise TypeError(
            f"Unsupported ground_truth type {type(ground_truth).__name__} in sample "
            f"{sample.get('instance_id', '<unknown>')}"
        )

    def _build_assistant_response(self, sample):
        answer = self._canonical_answer(sample)
        return f"<think>\nWorking through this now.\n</think>\n<answer>\n{answer}\n</answer>"

    def _use_system_prompt_for(self, sample):
        if not self.include_system_prompt:
            return False
        if self.system_prompt_fraction >= 1.0:
            return True
        if self.system_prompt_fraction <= 0.0:
            return False

        key = sample.get("instance_id", sample.get("seed", "0"))
        rng = random.Random(f"{self.seed}:{key}")
        return rng.random() < self.system_prompt_fraction

    def __getitem__(self, idx):
        sample = self.data[idx]

        if "prompt" not in sample:
            raise KeyError("Sample is missing required key 'prompt'")

        messages = []
        if self._use_system_prompt_for(sample):
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": sample["prompt"]})
        messages.append({"role": "assistant", "content": self._build_assistant_response(sample)})

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

        # The chat template already contains the required Qwen special tokens.
        # add_special_tokens=False avoids adding extra BOS/EOS outside the chat template.
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

        # labels[j] is full_ids[j + 1]. If prompt_len is P, the first assistant
        # content token is full_ids[P], predicted at label index P - 1.
        prompt_len = len(prompt["input_ids"])
        mask_until = min(max(prompt_len - 1, 0), len(labels))
        labels[:mask_until] = [-100] * mask_until

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# Backward-compatible alias if you prefer a more explicit class name later.
QwenSRPCorpusDataset = QwenSFTDataset
