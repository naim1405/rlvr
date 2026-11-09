"""P1 (Arithmetic) verifier.

The task asks for the *value* of an expression, and the prompt requires the
answer as "an integer or an exact fraction in lowest terms (p/q), never a
decimal approximation." So the reward must check two things, not one:

  1. the answer equals the gold value, and
  2. the answer is a reduced literal (an integer or `p/q`), i.e. the arithmetic
     was actually carried out.

Checking equivalence alone is unsound here: the prompt's own expression is, by
definition, equivalent to its value, so an answer that merely restates the
input (or any other unevaluated but equivalent expression) would score 1.0
without any computation. Requiring the reduced-literal surface form closes
that path while still accepting every genuinely-evaluated answer.

P1.4 (expression verification) is a yes/no sub-task and is scored as such.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional

from verifiers._util import extract_answer

# An integer or a fraction literal: optional sign, digits, optional /digits.
# Nothing else (no operators, parentheses, letters, decimals, exponents).
_LITERAL_RE = re.compile(r"^[+-]?\d+(?:/\d+)?$")

_YES = {"yes", "y", "true", "t", "equal", "equals", "correct", "1"}
_NO = {"no", "n", "false", "f", "not equal", "incorrect", "0"}


def _as_fraction(text: str) -> Optional[Fraction]:
    """Parse a bare integer/fraction literal into a Fraction, else None."""
    s = text.strip()
    if not _LITERAL_RE.match(s):
        return None
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def _verify_yes_no(predicted: str, ground_truth: str) -> bool:
    p = predicted.strip().lower().rstrip(".!,")
    gt = ground_truth.strip().lower()
    head = re.split(r"[\s,.:;]+", p, 1)[0] if p else ""
    pred_yes = p in _YES or head in _YES
    pred_no = p in _NO or head in _NO
    if pred_yes == pred_no:  # neither, or ambiguously both
        return False
    gt_yes = gt in _YES
    return pred_yes if gt_yes else pred_no


class ArithmeticVerifier:
    """Verifier for P1: value must equal the gold AND be a reduced literal."""

    def __init__(self, reward_params: dict = None):
        reward_params = reward_params or {}

    def verify(self, model_response: str, extra_env_info: dict) -> float:
        ans = extract_answer(model_response)
        if ans is None:
            return 0.0

        meta = extra_env_info.get("verifier_meta") or {}
        subcat = str(extra_env_info.get("subcategory", "")) + str(meta.get("kind", ""))
        gt_text = str(extra_env_info.get("expected_answer", "")).strip()

        # P1.4 verification sub-task: yes/no.
        if meta.get("mode") == "yes_no" or "verification" in subcat.lower() or gt_text.lower() in (_YES | _NO):
            return 1.0 if _verify_yes_no(ans, gt_text) else 0.0

        pred = _as_fraction(ans)
        gt = _as_fraction(gt_text)
        if pred is None:
            # Answer is not a reduced literal (e.g. an unevaluated expression,
            # a decimal, or prose). The task requires an exact reduced value.
            return 0.0
        if gt is None:
            # Gold is not a plain literal; fall back to a strict string compare
            # so a malformed corpus row cannot silently accept arbitrary input.
            return 1.0 if ans.strip() == gt_text else 0.0
        # Fraction equality is exact and already in lowest terms on both sides.
        return 1.0 if pred == gt else 0.0
