
import argparse
from pprint import pprint

from transformers import AutoTokenizer

from mydataset import QwenSFTDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--root", default="/tmp/checkpoints/new-dataset/corpus")
    parser.add_argument("--split", default="train")
    parser.add_argument("--families", nargs="*", default=["p0", "p1"])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--include-system-prompt", action="store_true")
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    print("Loading dataset...")
    ds = QwenSFTDataset(
        tokenizer=tokenizer,
        root=args.root,
        split=args.split,
        families=args.families,
        max_length=args.max_length,
        validation_fraction=args.validation_fraction,
        include_system_prompt=args.include_system_prompt,
    )

    print(f"Loaded {len(ds)} samples")
    print(f"Families: {args.families}")
    print(f"Split: {args.split}")
    print()

    for i in range(min(args.num_examples, len(ds))):
        raw = ds.data[i]
        processed = ds[i]

        print(f"===== EXAMPLE {i} =====")
        print("instance_id:", raw.get("instance_id"))
        print("family:", raw.get("family"))
        print("generator:", raw.get("generator"))
        print("verifier:", raw.get("verifier"))
        print("prompt preview:")
        print(raw["prompt"][:700])
        print()
        print("canonical answer used for training:")
        print(ds._canonical_answer(raw))
        print()
        print("assistant target:")
        print(ds._build_assistant_response(raw))
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
