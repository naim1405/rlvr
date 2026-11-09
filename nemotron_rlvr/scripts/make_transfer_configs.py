#!/usr/bin/env python3
"""Render one GRPO config per transfer-matrix condition from a base config.

For each condition (P1..P6 single-family, plus MIX) and each seed, writes a
config that points at that condition's training file and a distinct checkpoint
dir / run name, leaving every other knob identical. Prints the launch plan.

Usage (from nemotron_rlvr/):
  python3 scripts/make_transfer_configs.py \
      --base configs/cluster_5xA100.yaml \
      --conditions-dir data/conditions \
      --out-dir configs/transfer --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

CONDITIONS = {
    "P1": "train_P1.jsonl",
    "P2": "train_P2.jsonl",
    "P3": "train_P3.jsonl",
    "P4": "train_P4.jsonl",
    "P5": "train_P5.jsonl",
    "P6": "train_P6.jsonl",
    "MIX": "train_mix.jsonl",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--conditions-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("configs/transfer"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    base = yaml.safe_load(args.base.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan: list[str] = []

    for cond, fname in CONDITIONS.items():
        for seed in args.seeds:
            cfg = copy.deepcopy(base)
            train_path = str(args.conditions_dir / fname)
            cfg.setdefault("data", {}).setdefault("train", {})["data_path"] = train_path
            tag = f"{cond.lower()}_seed{seed}"
            cfg.setdefault("checkpointing", {})["checkpoint_dir"] = f"checkpoints/transfer/{tag}"
            cfg.setdefault("grpo", {})["seed"] = seed
            cfg.setdefault("logger", {}).setdefault("wandb", {})["name"] = f"rlvr-transfer-{tag}"
            out = args.out_dir / f"{tag}.yaml"
            out.write_text(yaml.safe_dump(cfg, sort_keys=False))
            plan.append(f"python3 train.py --config {out}")

    plan_path = args.out_dir / "LAUNCH_PLAN.sh"
    plan_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(plan) + "\n")
    print(f"Wrote {len(plan)} configs to {args.out_dir}")
    print(f"Launch plan: {plan_path}")
    print("\n".join(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
