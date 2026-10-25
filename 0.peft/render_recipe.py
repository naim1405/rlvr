#!/usr/bin/env python3
"""Render a concrete NeMo AutoModel PEFT recipe for one run.

The committed ``recipe.yaml`` is the template/default for Qwen3-0.6B. This
script copies it and rewrites only the run-specific fields:

* base model id (``model.pretrained_model_name_or_path``),
* container paths for the dataset class, ``train.jsonl`` and checkpoints,
* batch sizes / sequence length / LR / epochs, including conservative
  per-model-size micro-batch presets, and
* an optional smoke-test preset (subset of rows, frequent checkpoints).

The rendered YAML is written to a runs directory and passed to ``automodel``.
Keeping the rendered file means every checkpoint can be traced to the exact
configuration that produced it, and switching to a larger dense Qwen model
does not require hand-editing the committed template.

Examples:
    python3 render_recipe.py --model Qwen/Qwen3-0.6B \
        --out runs/recipes/smoke.yaml --smoke

    python3 render_recipe.py --model Qwen/Qwen3-8B \
        --out runs/recipes/train-8b.yaml \
        --checkpoint-dir /workspace/rlvr/0.peft/checkpoints-8b
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only without PyYAML
    raise SystemExit(
        "PyYAML is required to render a recipe. Run this inside the "
        "NeMo AutoModel container, or `pip install pyyaml`."
    ) from exc


HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "recipe.yaml"
DEFAULT_CONTAINER_PEFT_DIR = "/workspace/rlvr/0.peft"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

# Conservative starting micro-batch sizes for ONE 80 GB GPU, sequence length
# 2048, bf16 LoRA, and no activation checkpointing in the base recipe.
# The global batch is deliberately left at the template value; AutoModel
# reaches it with gradient accumulation. These are starting points to
# validate with the smoke job, not guarantees for every GPU/driver stack.
PRESET_LOCAL_BATCH: dict[str, int] = {
    "0.5b": 4,
    "0.6b": 4,
    "1.7b": 4,
    "4b": 2,
    "8b": 1,
    "14b": 1,
    "32b": 1,
}

PRESET_NAMES = ("auto", "custom", *PRESET_LOCAL_BATCH.keys())


def model_from_args_or_env(explicit: str | None) -> str:
    return explicit or os.environ.get("MODEL_ID") or os.environ.get("PEFT_BASE_MODEL") or DEFAULT_MODEL


def infer_preset(model_id: str) -> str | None:
    """Return a preset key such as ``8b`` from ``Qwen/Qwen3-8B``."""
    name = Path(str(model_id)).name.lower().replace("_", "-")
    # Sizes with a decimal point must be checked before bare integer sizes.
    for key in ("0.6b", "0.5b", "1.7b", "14b", "32b", "4b", "8b"):
        if re.search(rf"(?<![0-9.]){re.escape(key)}(?![0-9])", name):
            return key
    return None


def _require_positive(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"--{name} must be > 0 (got {value})")


def render_recipe(
    base_config: dict[str, Any],
    *,
    container_peft_dir: str = DEFAULT_CONTAINER_PEFT_DIR,
    model: str = DEFAULT_MODEL,
    data_path: str | None = None,
    checkpoint_dir: str | None = None,
    preset: str = "auto",
    smoke: bool = False,
    global_batch_size: int | None = None,
    local_batch_size: int | None = None,
    num_epochs: int | None = None,
    max_length: int | None = None,
    lr: float | None = None,
    ckpt_every_steps: int | None = None,
    val_every_steps: int | None = None,
    num_samples_limit: int | None = None,
    val_num_samples_limit: int | None = None,
    group_by_length: bool | None = None,
    use_triton: bool | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Return a modified deep-enough copy of ``base_config``.

    The input mapping is not mutated (a YAML round-trip is used to copy it).
    """
    cfg: dict[str, Any] = yaml.safe_load(yaml.safe_dump(base_config, sort_keys=False)) or {}

    peft_dir = str(container_peft_dir).rstrip("/")
    if not peft_dir.startswith("/"):
        raise ValueError("--container-peft-dir must be an absolute path inside the container")

    resolved_preset = infer_preset(model) if preset == "auto" else (None if preset == "custom" else preset)
    if preset not in PRESET_NAMES:
        raise ValueError(f"unknown preset {preset!r}; choose one of {PRESET_NAMES}")

    if resolved_preset == "32b":
        print(
            "[render_recipe] WARNING: a dense 32B model in bf16 is ~64 GB of "
            "weights alone. With this recipe (no activation checkpointing, "
            "no sharding) it is not expected to fit one 80 GB GPU once "
            "activations are included. Treat this render as experimental; "
            "see CLUSTER_RUNBOOK.md.",
            file=sys.stderr,
        )

    # --- model ---------------------------------------------------------
    cfg.setdefault("model", {})["pretrained_model_name_or_path"] = str(model)

    # --- paths ---------------------------------------------------------
    target = f"{peft_dir}/mydataset.py:QwenPEFTDataset"
    resolved_data = str(data_path or f"{peft_dir}/dataset/train.jsonl")
    for section_name in ("dataset", "validation_dataset"):
        section = cfg.setdefault(section_name, {})
        section["_target_"] = target
        section["path"] = resolved_data

    default_ckpt_name = "checkpoints-smoke" if smoke else "checkpoints"
    cfg.setdefault("checkpoint", {})["checkpoint_dir"] = str(
        checkpoint_dir or f"{peft_dir}/{default_ckpt_name}"
    )

    # --- smoke defaults (only where the caller did not override) -------
    if smoke:
        num_epochs = 1 if num_epochs is None else num_epochs
        max_length = 1024 if max_length is None else max_length
        global_batch_size = 16 if global_batch_size is None else global_batch_size
        ckpt_every_steps = 10 if ckpt_every_steps is None else ckpt_every_steps
        val_every_steps = 10 if val_every_steps is None else val_every_steps
        num_samples_limit = 512 if num_samples_limit is None else num_samples_limit
        val_num_samples_limit = 128 if val_num_samples_limit is None else val_num_samples_limit

    # --- scheduling / memory knobs --------------------------------------
    scheduler = cfg.setdefault("step_scheduler", {})
    if local_batch_size is None and resolved_preset is not None:
        local_batch_size = PRESET_LOCAL_BATCH[resolved_preset]

    for key, value in (
        ("global_batch_size", global_batch_size),
        ("local_batch_size", local_batch_size),
        ("num_epochs", num_epochs),
        ("ckpt_every_steps", ckpt_every_steps),
        ("val_every_steps", val_every_steps),
    ):
        if value is not None:
            _require_positive(key.replace("_", "-"), int(value))
            scheduler[key] = int(value)

    if max_length is not None:
        _require_positive("max-length", int(max_length))
        cfg["dataset"]["max_length"] = int(max_length)
        cfg["validation_dataset"]["max_length"] = int(max_length)

    if num_samples_limit is not None:
        _require_positive("num-samples-limit", int(num_samples_limit))
        cfg["dataset"]["num_samples_limit"] = int(num_samples_limit)
    elif not smoke:
        # A full run should never inherit a sample cap from a template edit.
        cfg["dataset"].pop("num_samples_limit", None)

    if val_num_samples_limit is not None:
        _require_positive("val-num-samples-limit", int(val_num_samples_limit))
        cfg["validation_dataset"]["num_samples_limit"] = int(val_num_samples_limit)

    if lr is not None:
        if float(lr) <= 0:
            raise ValueError("--lr must be > 0")
        cfg.setdefault("optimizer", {})["lr"] = float(lr)

    if group_by_length is not None:
        cfg.setdefault("dataloader", {})["group_by_length"] = bool(group_by_length)

    if use_triton is not None:
        # Triton LoRA kernels are faster but keep extra workspace; disabling
        # them can be the difference between OOM and running on a 4 GB GPU.
        cfg.setdefault("peft", {})["use_triton"] = bool(use_triton)

    if seed is not None:
        cfg.setdefault("rng", {})["seed"] = int(seed)

    effective_global = int(scheduler.get("global_batch_size", 0))
    effective_local = int(scheduler.get("local_batch_size", 0))
    if effective_global <= 0 or effective_local <= 0:
        raise ValueError("global_batch_size and local_batch_size must be positive after rendering")
    if effective_global % effective_local != 0:
        raise ValueError(
            "global_batch_size must be divisible by local_batch_size x GPUs; "
            f"for one GPU got global={effective_global}, local={effective_local} "
            f"(remainder {effective_global % effective_local})."
        )

    return cfg


def _bool_arg(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Template recipe YAML")
    parser.add_argument("--out", type=Path, required=True, help="Rendered recipe YAML to write")
    parser.add_argument(
        "--container-peft-dir",
        default=os.environ.get("PEFT_DIR", DEFAULT_CONTAINER_PEFT_DIR),
        help="Absolute path of 0.peft *inside* the container",
    )
    parser.add_argument("--model", default=None, help="HF model id or local path (default: $MODEL_ID or Qwen/Qwen3-0.6B)")
    parser.add_argument("--data", default=None, help="train.jsonl path inside the container")
    parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint dir inside the container")
    parser.add_argument("--preset", choices=PRESET_NAMES, default="auto")
    parser.add_argument("--smoke", action="store_true", help="Small, fast pipeline-verification preset")
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--local-batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ckpt-every-steps", type=int, default=None)
    parser.add_argument("--val-every-steps", type=int, default=None)
    parser.add_argument("--num-samples-limit", type=int, default=None)
    parser.add_argument("--val-num-samples-limit", type=int, default=None)
    parser.add_argument("--group-by-length", type=_bool_arg, default=None)
    parser.add_argument("--use-triton", type=_bool_arg, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.base.is_file():
        raise SystemExit(f"Template recipe not found: {args.base}")

    with args.base.open("r", encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle) or {}

    model = model_from_args_or_env(args.model)
    try:
        rendered = render_recipe(
            base_cfg,
            container_peft_dir=args.container_peft_dir,
            model=model,
            data_path=args.data,
            checkpoint_dir=args.checkpoint_dir,
            preset=args.preset,
            smoke=args.smoke,
            global_batch_size=args.global_batch_size,
            local_batch_size=args.local_batch_size,
            num_epochs=args.num_epochs,
            max_length=args.max_length,
            lr=args.lr,
            ckpt_every_steps=args.ckpt_every_steps,
            val_every_steps=args.val_every_steps,
            num_samples_limit=args.num_samples_limit,
            val_num_samples_limit=args.val_num_samples_limit,
            group_by_length=args.group_by_length,
            use_triton=args.use_triton,
            seed=args.seed,
        )
    except ValueError as exc:
        raise SystemExit(f"[render_recipe] {exc}") from exc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Rendered by 0.peft/render_recipe.py - do not edit by hand.\n"
        f"# Template: {args.base}\n"
        f"# Model: {model}\n"
        f"# Smoke preset: {args.smoke}\n\n"
    )
    args.out.write_text(header + yaml.safe_dump(rendered, sort_keys=False, width=120), encoding="utf-8")

    scheduler = rendered.get("step_scheduler", {})
    print(f"[render_recipe] wrote {args.out}")
    print(f"[render_recipe] model={model}")
    print(
        "[render_recipe] "
        f"global_batch_size={scheduler.get('global_batch_size')} "
        f"local_batch_size={scheduler.get('local_batch_size')} "
        f"num_epochs={scheduler.get('num_epochs')} "
        f"max_length={rendered.get('dataset', {}).get('max_length')}"
    )
    print(f"[render_recipe] checkpoint_dir={rendered.get('checkpoint', {}).get('checkpoint_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
