#!/usr/bin/env python3
"""Smoke tests for Gym JSONL conversion and server install (no Gym package)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
SCRIPTS = HERE.parent
NEMOTRON_RLVR = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from reformat_dataset import to_gym_record  # noqa: E402
from install_gym_server import gym_config_paths, install_nemotron_verifier  # noqa: E402


class ReformatTests(unittest.TestCase):
    def test_gym_schema(self):
        rec = to_gym_record(
            {
                "prompt": "Capital of France?",
                "expected_answer": "Paris",
                "family": "P0",
                "instance_id": "srp:abc",
                "verifier_meta": {"normalize": "default"},
            }
        )
        self.assertEqual(
            rec["responses_create_params"]["input"],
            [{"role": "user", "content": "Capital of France?"}],
        )
        self.assertEqual(rec["task_name"], "p0_retrieval")
        self.assertEqual(rec["family"], "P0")
        self.assertEqual(rec["expected_answer"], "Paris")
        self.assertEqual(
            rec["agent_ref"],
            {"type": "responses_api_agents", "name": "nemotron_verifier_simple_agent"},
        )


class InstallTests(unittest.TestCase):
    def test_copies_pyproject_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            gym_root = Path(tmp) / "Gym"
            (gym_root / "resources_servers").mkdir(parents=True)
            dest = install_nemotron_verifier(gym_root=gym_root, project_root=NEMOTRON_RLVR)
            self.assertTrue((dest / "pyproject.toml").is_file())
            self.assertTrue((dest / "app.py").is_file())
            self.assertTrue((dest / "score.py").is_file())
            self.assertTrue((dest / "configs" / "nemotron_verifier.yaml").is_file())
            self.assertTrue((dest / "verifiers" / "p0_retrieval.py").is_file())
            self.assertFalse((dest / "requirements.txt").exists())
            paths = gym_config_paths(gym_root, NEMOTRON_RLVR)
            self.assertEqual(len(paths), 2)
            self.assertTrue(paths[1].endswith("nemotron_verifier.yaml"))


if __name__ == "__main__":
    unittest.main()
