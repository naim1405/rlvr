"""Manual checkpoint tester for the 0.peft LoRA model (no system prompt).

Loads the base model + trained LoRA adapter, generates answers for test
questions, and checks each output for the required format:

    <think> ... </think> <answer> ... </answer>

Updated from main:checkpoints/infer_no_system.py with:
  - auto-discovery of the latest checkpoint (no hardcoded epoch_*_step_* path)
  - batch testing: --question, --questions-file, or random --dataset samples
  - automatic FORMAT PASS/FAIL verdict per sample + summary
  - --show-expected to compare generation vs training completion side by side
  - explicit enable_thinking=True (== Qwen3 template default, matches training)

Usage (inside the container, from /tmp/0.peft):
  python infer.py                                        # auto adapter + 5 dataset samples
  python infer.py --question "What is 12*13?"            # single custom question
  python infer.py --num-samples 10 --show-expected       # 10 samples with expected outputs
  python infer.py --adapter /tmp/0.peft/checkpoints-smoke/epoch_0_step_200/model
  python infer.py --questions-file my_questions.txt      # one question per line
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Candidate checkpoint roots (first existing one wins when --adapter is absent).
DEFAULT_CHECKPOINT_DIRS = [
    "/tmp/0.peft/checkpoints",
    "/tmp/0.peft/checkpoints-smoke",
]
DEFAULT_DATASET = "/tmp/0.peft/dataset/train.jsonl"
DEFAULT_QUESTION = "A train travels at 98 miles per hour. How far does it travel in 30 hours?"

# Lenient on whitespace: the chat template inserts an extra newline between
# </think> and <answer> at training time, so allow \s* everywhere.
FORMAT_RE = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)


def find_latest_adapter(checkpoint_dir: Path) -> Path:
    """Pick the newest epoch_*_step_*/model (or epoch_*/model) under checkpoint_dir."""
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"Checkpoint dir not found: {checkpoint_dir}")
    cands = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
    if not cands:
        raise SystemExit(f"No epoch_* checkpoint found under {checkpoint_dir}")

    def sort_key(d: Path):
        m = re.fullmatch(r"epoch_(\d+)(?:_step_(\d+))?", d.name)
        if m:
            return (0, int(m.group(1)), int(m.group(2) or 0))
        return (1, 0, d.stat().st_mtime)  # unknown layout: fall back to mtime

    latest = sorted(cands, key=sort_key)[-1]
    model_dir = latest / "model"
    return model_dir if model_dir.is_dir() else latest


def resolve_adapter(args) -> Path:
    if args.adapter:
        p = Path(args.adapter)
        if not p.is_dir():
            raise SystemExit(f"Adapter dir not found: {p}")
        return p
    search = [Path(args.checkpoint_dir)] if args.checkpoint_dir else [Path(d) for d in DEFAULT_CHECKPOINT_DIRS]
    for d in search:
        try:
            return find_latest_adapter(d)
        except SystemExit:
            continue
    raise SystemExit(
        "No adapter found. Pass --adapter <.../model> or --checkpoint-dir <...>. "
        f"Searched: {', '.join(str(d) for d in search)}"
    )


def check_format(text: str) -> tuple[bool, str]:
    """Return (passed, detail). Order + presence of all four tags."""
    tags = ["<think>", "</think>", "<answer>", "</answer>"]
    low = text.lower()
    pos = [low.find(t) for t in tags]
    missing = [t for t, p in zip(tags, pos) if p == -1]
    if missing:
        return False, f"missing {', '.join(missing)}"
    if not (pos[0] < pos[1] < pos[2] < pos[3]):
        return False, "tags out of order"
    if not FORMAT_RE.search(text):
        return False, "tags present but pattern mismatch"
    think_body = text[pos[0] + 7 : pos[1]].strip()
    answer_body = text[pos[2] + 8 : pos[3]].strip()
    if not think_body or not answer_body:
        return False, "empty <think> or <answer> body"
    return True, "ok"


def load_questions_file(path: Path) -> list[dict]:
    """Plain text (one question per line) or JSONL with prompt/completion keys."""
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "prompt" in obj:
                    items.append({"prompt": str(obj["prompt"]),
                                  "expected": str(obj.get("completion", ""))})
                    continue
            except json.JSONDecodeError:
                pass
        items.append({"prompt": line, "expected": ""})
    if not items:
        raise SystemExit(f"No questions found in {path}")
    return items


def sample_dataset(path: Path, n: int, seed: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rng = random.Random(seed)
    return [{"prompt": str(r["prompt"]), "expected": str(r.get("completion", "")),
             "instance_id": r.get("instance_id", ""), "family": r.get("family", "")}
            for r in rng.sample(rows, min(n, len(rows)))]


def main():
    parser = argparse.ArgumentParser(description="Test a 0.peft LoRA checkpoint manually.")
    parser.add_argument("--base", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", default=None, help="Direct path to .../model adapter dir")
    parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint root to auto-pick latest from")
    parser.add_argument("--question", default=None, help="Single question to test")
    parser.add_argument("--questions-file", default=None, help="Text/JSONL file, one question per line")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="train.jsonl to sample from")
    parser.add_argument("--num-samples", type=int, default=5, help="Dataset samples when no --question/--questions-file")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed for --dataset")
    parser.add_argument("--show-expected", action="store_true", help="Print training completion alongside generation")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true", help="Sample instead of greedy decode")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-prompt-chars", type=int, default=2000, help="Truncate long prompts in display")
    args = parser.parse_args()

    adapter_path = resolve_adapter(args)
    print(f"Base:    {args.base}")
    print(f"Adapter: {adapter_path}")
    if not (adapter_path / "adapter_config.json").exists():
        print("WARNING: no adapter_config.json here — is this the right dir?", file=sys.stderr)

    # Collect test items.
    if args.question:
        items = [{"prompt": args.question, "expected": ""}]
    elif args.questions_file:
        items = load_questions_file(Path(args.questions_file))
    elif Path(args.dataset).is_file():
        items = sample_dataset(Path(args.dataset), args.num_samples, args.seed)
        print(f"Sampled {len(items)} row(s) from {args.dataset} (seed={args.seed})")
    else:
        items = [{"prompt": DEFAULT_QUESTION, "expected": ""}]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:  {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    try:
        target = model.device
    except AttributeError:  # sharded model has no single .device
        target = next(model.parameters()).device

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens,
                      pad_token_id=tokenizer.eos_token_id,
                      eos_token_id=tokenizer.eos_token_id)
    if args.do_sample:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    else:
        gen_kwargs.update(do_sample=False)

    passed = 0
    for i, item in enumerate(items):
        # No system prompt: matches training (include_system_prompt: false),
        # where the XML format is an unconditional response style.
        # enable_thinking=True == Qwen3 template default == what training used.
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(target)
        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        ok, detail = check_format(generated)
        passed += ok
        tag = " ".join(f"{k}={item[k]}" for k in ("instance_id", "family") if item.get(k))
        print(f"\n===== SAMPLE {i} {'[' + tag + ']' if tag else ''} "
              f"FORMAT: {'PASS' if ok else 'FAIL (' + detail + ')'} =====")
        p = item["prompt"]
        print("PROMPT:")
        print(p if len(p) <= args.max_prompt_chars else p[:args.max_prompt_chars] + f"\n... [truncated {len(p)} chars]")
        print("\nGENERATED:")
        print(generated.strip() or "[EMPTY]")
        if args.show_expected and item.get("expected"):
            print("\nEXPECTED (training completion):")
            print(item["expected"])

    print(f"\n==== SUMMARY: {passed}/{len(items)} passed format check ====")
    if passed < len(items):
        print("FAILs mean the model didn't emit <think>...</think><answer>...</answer> — "
              "check undertraining, wrong adapter path, or truncated generation (raise --max-new-tokens).")


if __name__ == "__main__":
    main()
