import argparse
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer_block(text: str) -> str | None:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", default="/tmp/checkpoints/checkpoints")
    parser.add_argument(
        "--question",
        default=(
            "Evaluate the following expression:\n\n"
            "7 - 1\n\n"
            "Answer with the exact value only — an integer or an exact fraction in lowest "
            "terms (p/q), never a decimal approximation.\n\n"
            "Solve the problem. Put your reasoning inside <think> </think> and your final "
            "answer inside <answer> </answer>."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # You can pass either the checkpoint root or the epoch_x_step_y/model directory.
    adapter_path = args.adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    messages = [
        {"role": "user", "content": args.question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if args.temperature and args.temperature > 0:
        generation_kwargs.update(
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        generation_kwargs.update(do_sample=False)

    with torch.no_grad():
        output = model.generate(**inputs, **generation_kwargs)

    generated_ids = output[0][inputs["input_ids"].shape[-1] :]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("=== RAW OUTPUT ===")
    print(decoded)

    extracted = extract_answer_block(decoded)
    if extracted is not None:
        print("\n=== EXTRACTED <answer> ===")
        print(extracted)


if __name__ == "__main__":
    main()
