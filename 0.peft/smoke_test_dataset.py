import argparse
from pathlib import Path
from transformers import AutoTokenizer

from mydataset import QwenPEFTDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--path", default="/home/ezio/Projects/rlhf/rlvr/checkpoints/dataset/train.jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--include-system-prompt", action="store_true")
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    print(f"Loading dataset from {args.path}...")
    ds = QwenPEFTDataset(
        tokenizer=tokenizer,
        path=args.path,
        split=args.split,
        max_length=args.max_length,
        validation_fraction=args.validation_fraction,
        include_system_prompt=args.include_system_prompt,
    )

    print(f"Loaded {len(ds)} samples for split='{args.split}'")
    print()

    for i in range(min(args.num_examples, len(ds))):
        raw = ds.data[i]
        processed = ds[i]

        print(f"===== EXAMPLE {i} =====")
        print("instance_id:", raw.get("instance_id"))
        print("family:", raw.get("family"))
        print("user prompt preview:")
        print(ds._extract_user_prompt(raw)[:400])
        print()
        print("assistant completion preview:")
        print(ds._extract_assistant_completion(raw)[:400])
        print()
        print("tensor lengths:")
        print({
            "input_ids": len(processed["input_ids"]),
            "attention_mask": len(processed["attention_mask"]),
            "labels": len(processed["labels"]),
            "supervised_tokens": sum(1 for x in processed["labels"] if x != -100),
        })
        print()

    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
