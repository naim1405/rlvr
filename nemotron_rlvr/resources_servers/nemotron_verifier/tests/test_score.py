#!/usr/bin/env python3
"""Gym-free tests for P0/P4/P5/P6 routing (no sympy/z3 required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
SERVER_DIR = HERE.parents[1]
NEMOTRON_RLVR = HERE.parents[3]
sys.path.insert(0, str(NEMOTRON_RLVR))
sys.path.insert(0, str(SERVER_DIR))

from score import (  # noqa: E402
    canonicalize_task_name,
    extract_assistant_text,
    extra_env_info_from_mapping,
    score_response,
)


def _answered(text: str) -> str:
    return f"<think>\nworking\n</think>\n<answer>\n{text}\n</answer>"


class CanonicalizeTests(unittest.TestCase):
    def test_family_p0(self):
        self.assertEqual(canonicalize_task_name("P0"), "p0_retrieval")

    def test_task_name(self):
        self.assertEqual(canonicalize_task_name("p4_planning"), "p4_planning")

    def test_legacy_entrypoint(self):
        self.assertEqual(
            canonicalize_task_name("verifiers.p5_pattern:PatternVerifier"),
            "p5_pattern",
        )


class ExtractTests(unittest.TestCase):
    def test_gym_output_text(self):
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": _answered("Paris")}],
                }
            ]
        }
        self.assertIn("Paris", extract_assistant_text(response))


class ScoreTests(unittest.TestCase):
    def test_p0_match(self):
        extra = extra_env_info_from_mapping(
            {
                "task_name": "p0_retrieval",
                "expected_answer": "Paris",
                "verifier_meta": {},
            }
        )
        self.assertEqual(score_response(_answered("Paris"), extra), 1.0)

    def test_p0_mismatch(self):
        extra = extra_env_info_from_mapping(
            {"family": "P0", "expected_answer": "Paris", "verifier_meta": {}}
        )
        self.assertEqual(score_response(_answered("London"), extra), 0.0)

    def test_p5_sequence(self):
        extra = extra_env_info_from_mapping(
            {
                "task_name": "p5_pattern",
                "expected_answer": "8",
                "verifier_meta": {},
            }
        )
        self.assertEqual(score_response(_answered("8"), extra), 1.0)

    def test_missing_task_is_zero(self):
        extra = extra_env_info_from_mapping({"expected_answer": "x"})
        self.assertEqual(score_response(_answered("x"), extra), 0.0)


if __name__ == "__main__":
    unittest.main()
