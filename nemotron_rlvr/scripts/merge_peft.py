#!/usr/bin/env python3
"""Merge a LoRA PEFT adapter into base model weights."""

import argparse
from pathlib import Path
import sys


def merge_lora(base_model_path: str, lora_adapter_path: str, output_path: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading Base Model: {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA Adapter: {lora_adapter_path}...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    print("Merging adapter weights into base model...")
    merged_model = model.merge_and_unload()

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged standalone checkpoint to {output_path}...")
    merged_model.save_pretrained(out_dir)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(out_dir)

    print("Merge successfully completed!")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA PEFT checkpoint into base model")
    parser.add_argument("--base", type=str, required=True, help="Path to base model checkpoint")
    parser.add_argument("--adapter", type=str, required=True, help="Path to PEFT adapter checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output directory for merged model")

    args = parser.parse_args()
    merge_lora(args.base, args.adapter, args.output)


if __name__ == "__main__":
    main()
