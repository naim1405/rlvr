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

    def __init__(
        self,
        tokenizer,
        path,
        max_length=2048,
    ):
        self.data = json.loads(Path(path).read_text())
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": sample["question"],
            },
            {
                "role": "assistant",
                "content": sample["formatted_output"],
            },
        ]

        prompt = messages[:-1]

        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_text = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

        full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
        )

        prompt = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
        )

        labels = full["input_ids"].copy()

        prompt_len = len(prompt["input_ids"])

        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }