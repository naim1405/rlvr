from pathlib import Path
import json
from torch.utils.data import Dataset

SYSTEM_PROMPT = """You must always answer using the following format:

<think>
...
</think>
<answer>
...
</answer>
"""


class QwenSFTDataset(Dataset):
    """Qwen chat-format SFT dataset for NeMo AutoModel.

    Expected JSON format: a list of dicts with at least:
      - question
      - formatted_output

    The dataset returns shifted causal-LM examples:
      input_ids = full_ids[:-1]
      labels    = full_ids[1:]

    System/user/prompt tokens are masked with -100 so loss is applied only to
    assistant response tokens.
    """

    def __init__(
        self,
        tokenizer,
        path,
        max_length=2048,
        split="train",
        validation_fraction=0.1,
        seed=1111,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        data = json.loads(Path(path).read_text(encoding="utf-8"))

        # Deterministic train/validation split from a single JSON file.
        # This avoids validating on the exact same samples used for training.
        if split in {"train", "validation", "val"} and validation_fraction > 0:
            import random

            rng = random.Random(seed)
            indices = list(range(len(data)))
            rng.shuffle(indices)
            val_count = max(1, int(len(indices) * validation_fraction))
            val_indices = set(indices[:val_count])

            if split in {"validation", "val"}:
                data = [x for i, x in enumerate(data) if i in val_indices]
            else:
                data = [x for i, x in enumerate(data) if i not in val_indices]

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["formatted_output"]},
        ]

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
        # add_special_tokens=False avoids NeMoAutoTokenizer appending extra BOS/EOS
        # outside the chat template.
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

        # labels[j] is full_ids[j + 1].  If prompt_len is P, the first
        # assistant-content token is full_ids[P], predicted at label index P-1.
        prompt_len = len(prompt["input_ids"])
        mask_until = min(max(prompt_len - 1, 0), len(labels))
        labels[:mask_until] = [-100] * mask_until

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
