#!/usr/bin/env python3
"""CPU-only tests for render_recipe.py (no torch, GPU, or container needed).

Run from the repo root:
    python3 0.peft/test_render_recipe.py
or:
    cd 0.peft && python3 -m unittest test_render_recipe -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import yaml  # noqa: E402

from render_recipe import infer_preset, render_recipe  # noqa: E402


def base_cfg() -> dict:
    with (HERE / "recipe.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class PresetInferenceTests(unittest.TestCase):
    def test_model_ids(self):
        self.assertEqual(infer_preset("Qwen/Qwen3-0.6B"), "0.6b")
        self.assertEqual(infer_preset("Qwen/Qwen3-4B"), "4b")
        self.assertEqual(infer_preset("Qwen/Qwen3-8B"), "8b")
        self.assertEqual(infer_preset("Qwen/Qwen3-14B"), "14b")
        self.assertEqual(infer_preset("Qwen/Qwen3-32B"), "32b")
        self.assertIsNone(infer_preset("Qwen/Qwen3-30B-A3B"))


class RenderRecipeTests(unittest.TestCase):
    def test_paths_and_model_are_rewritten(self):
        cfg = render_recipe(
            base_cfg(),
            container_peft_dir="/workspace/rlvr/0.peft",
            model="Qwen/Qwen3-0.6B",
        )
        self.assertEqual(cfg["model"]["pretrained_model_name_or_path"], "Qwen/Qwen3-0.6B")
        self.assertEqual(
            cfg["dataset"]["_target_"],
            "/workspace/rlvr/0.peft/mydataset.py:QwenPEFTDataset",
        )
        self.assertEqual(cfg["dataset"]["path"], "/workspace/rlvr/0.peft/dataset/train.jsonl")
        self.assertEqual(
            cfg["validation_dataset"]["path"],
            "/workspace/rlvr/0.peft/dataset/train.jsonl",
        )
        self.assertEqual(cfg["checkpoint"]["checkpoint_dir"], "/workspace/rlvr/0.peft/checkpoints")
        self.assertNotIn("num_samples_limit", cfg["dataset"])

    def test_larger_model_lowers_local_batch_but_keeps_global(self):
        cfg = render_recipe(base_cfg(), model="Qwen/Qwen3-8B")
        self.assertEqual(cfg["step_scheduler"]["local_batch_size"], 1)
        self.assertEqual(cfg["step_scheduler"]["global_batch_size"], 32)

    def test_explicit_values_win_over_preset(self):
        cfg = render_recipe(
            base_cfg(),
            model="Qwen/Qwen3-8B",
            local_batch_size=2,
            global_batch_size=32,
            max_length=1536,
            lr=1.0e-4,
            group_by_length=False,
        )
        self.assertEqual(cfg["step_scheduler"]["local_batch_size"], 2)
        self.assertEqual(cfg["dataset"]["max_length"], 1536)
        self.assertEqual(cfg["optimizer"]["lr"], 1.0e-4)
        self.assertFalse(cfg["dataloader"]["group_by_length"])

    def test_smoke_preset(self):
        cfg = render_recipe(base_cfg(), model="Qwen/Qwen3-0.6B", smoke=True)
        self.assertEqual(cfg["step_scheduler"]["num_epochs"], 1)
        self.assertEqual(cfg["step_scheduler"]["global_batch_size"], 16)
        self.assertEqual(cfg["step_scheduler"]["ckpt_every_steps"], 10)
        self.assertEqual(cfg["dataset"]["num_samples_limit"], 512)
        self.assertEqual(cfg["validation_dataset"]["num_samples_limit"], 128)
        self.assertEqual(cfg["dataset"]["max_length"], 1024)
        self.assertTrue(cfg["checkpoint"]["checkpoint_dir"].endswith("/checkpoints-smoke"))

    def test_indivisible_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            render_recipe(
                base_cfg(),
                model="Qwen/Qwen3-0.6B",
                global_batch_size=30,
                local_batch_size=4,
            )

    def test_input_config_is_not_mutated(self):
        original = base_cfg()
        original_model = original["model"]["pretrained_model_name_or_path"]
        render_recipe(original, model="Qwen/Qwen3-8B")
        self.assertEqual(original["model"]["pretrained_model_name_or_path"], original_model)


if __name__ == "__main__":
    unittest.main()
