#!/usr/bin/env python3
"""Regression tests for the verifier correctness fixes.

Each test states a property the reward must have for the transfer matrix to be
sound. Run from resources_servers/nemotron_verifier/ or via the repo test
runner. P2/P3 cases need sympy / z3; they skip if those are absent.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
NEMOTRON_RLVR = HERE.parents[3]
SERVER_DIR = HERE.parents[1]
sys.path.insert(0, str(NEMOTRON_RLVR))
sys.path.insert(0, str(SERVER_DIR))

from score import extra_env_info_from_mapping, score_response  # noqa: E402


def A(x: str) -> str:
    return f"<think>reasoning</think>\n<answer>{x}</answer>"


def score(task, expected, answer, meta=None, prompt=""):
    extra = extra_env_info_from_mapping(
        {"task_name": task, "expected_answer": expected, "verifier_meta": meta or {}, "prompt": prompt}
    )
    return score_response(A(answer), extra)


def have(mod) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


class P1Arithmetic(unittest.TestCase):
    def test_correct_value_scores_one(self):
        self.assertEqual(score("p1_arithmetic", "386", "386"), 1.0)

    def test_reduced_fraction_gold_round_trips(self):
        self.assertEqual(score("p1_arithmetic", "-3945162121/5", "-3945162121/5"), 1.0)

    def test_wrong_value_scores_zero(self):
        self.assertEqual(score("p1_arithmetic", "386", "387"), 0.0)

    def test_unevaluated_expression_is_rejected(self):
        # Equivalent to 386 but not carried out -> not a valid answer.
        self.assertEqual(score("p1_arithmetic", "386", "17*23 - 5"), 0.0)

    def test_decimal_rejected_when_exact_required(self):
        self.assertEqual(score("p1_arithmetic", "7/2", "3.5"), 0.0)


@unittest.skipUnless(have("sympy"), "sympy not installed")
class P2Symbolic(unittest.TestCase):
    FACTOR = "Factor the following polynomial completely:\n\nx**2 + x - 12\n\nAnswer with the transformed expression only."
    PF = "Decompose into partial fractions:\n\n(x**2 - 11*x + 15*y)/(x**2 + x - 132)\n\nAnswer with the transformed expression only."
    PF_GOLD = "15*y/(23*(x - 11)) + 1 - 3*(5*y + 92)/(23*(x + 12))"

    def test_factor_correct(self):
        self.assertEqual(score("p2_symbolic", "(x - 3)*(x + 4)", "(x - 3)*(x + 4)", {"symbols": ["x"]}, self.FACTOR), 1.0)

    def test_factor_echo_of_input_rejected(self):
        self.assertEqual(score("p2_symbolic", "(x - 3)*(x + 4)", "x**2 + x - 12", {"symbols": ["x"]}, self.FACTOR), 0.0)

    def test_partial_fractions_correct(self):
        self.assertEqual(score("p2_symbolic", self.PF_GOLD, self.PF_GOLD, {"symbols": ["x", "y"]}, self.PF), 1.0)

    def test_partial_fractions_echo_rejected(self):
        echo = "(x**2 - 11*x + 15*y)/(x**2 + x - 132)"
        self.assertEqual(score("p2_symbolic", self.PF_GOLD, echo, {"symbols": ["x", "y"]}, self.PF), 0.0)


@unittest.skipUnless(have("z3"), "z3-solver not installed")
class P3Logic(unittest.TestCase):
    ENT = {"kind": "entailment", "premises": [["implies", "p", "q"], "p"], "query": "q"}
    KK = {"kind": "knights_knaves", "people": ["alice", "bob"],
          "statements": {"alice": ["not", "bob"], "bob": ["not", "alice"]}}

    def test_clean_yes(self):
        self.assertEqual(score("p3_logic", "yes", "yes", self.ENT), 1.0)

    def test_ambiguous_answer_rejected(self):
        self.assertEqual(score("p3_logic", "yes", "yes... actually no", self.ENT), 0.0)

    def test_kk_accepts_any_valid_solution(self):
        self.assertEqual(score("p3_logic", "", '{"alice":"knight","bob":"knave"}', self.KK), 1.0)
        self.assertEqual(score("p3_logic", "", '{"alice":"knave","bob":"knight"}', self.KK), 1.0)

    def test_kk_rejects_invalid(self):
        self.assertEqual(score("p3_logic", "", '{"alice":"knight","bob":"knight"}', self.KK), 0.0)


class P5Pattern(unittest.TestCase):
    def test_single_value_bracket_insensitive(self):
        for gold in ("[10]", "10"):
            for ans in ("10", "[10]"):
                self.assertEqual(score("p5_pattern", gold, ans, {"kind": "int"}), 1.0, (gold, ans))

    def test_wrong_value(self):
        self.assertEqual(score("p5_pattern", "10", "11", {"kind": "int"}), 0.0)


class P0Retrieval(unittest.TestCase):
    SPANS = ["Nenglex Region", "Pepoth Zone", "Drethzhiax Sector", "Bialnoml Sector"]

    def test_single_span_accepted(self):
        self.assertEqual(score("p0_retrieval", self.SPANS, "Pepoth Zone"), 1.0)

    def test_full_list_accepted(self):
        import json
        self.assertEqual(score("p0_retrieval", self.SPANS, json.dumps(self.SPANS)), 1.0)

    def test_wrong_span_rejected(self):
        self.assertEqual(score("p0_retrieval", self.SPANS, "Atlantis"), 0.0)

    def test_scalar_gold_exact(self):
        self.assertEqual(score("p0_retrieval", "VM-3137", "VM-3137"), 1.0)


if __name__ == "__main__":
    unittest.main()
