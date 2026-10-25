"""P1 (Arithmetic) verifier based on srp.verify.sympy_eq."""

from __future__ import annotations

import re
from typing import Any
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from verifiers._util import TimeoutExceeded, extract_answer, run_with_timeout

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def _parse(text: str, symbols: list[str]) -> Any:
    local = {s: sympy.Symbol(s) for s in symbols}
    return parse_expr(
        text,
        local_dict=local,
        transformations=_TRANSFORMS,
        evaluate=True,
    )


class ArithmeticVerifier:
    """Verifier for P1: Arithmetic tasks using SymPy equivalence and numeric tolerance."""

    def __init__(self, reward_params: dict = None):
        reward_params = reward_params or {}
        self.default_atol = float(reward_params.get("tolerance", 1e-5))

    def verify(self, model_response: str, extra_env_info: dict) -> float:
        ans = extract_answer(model_response)
        if ans is None:
            return 0.0

        meta = extra_env_info.get("verifier_meta") or {}
        mode = meta.get("mode", "equiv")
        symbols = list(meta.get("symbols", []))
        timeout_s = float(meta.get("timeout_s", 3.0))
        gt_text = str(extra_env_info.get("expected_answer", ""))

        def _judge() -> bool:
            try:
                pred = _parse(ans, symbols)
                gt = _parse(gt_text, symbols)
            except Exception:
                # Fallback to direct numeric extract
                try:
                    num_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", ans)
                    if num_match:
                        return abs(float(num_match.group(0)) - float(gt_text)) <= self.default_atol
                except Exception:
                    pass
                return False

            if mode == "numeric":
                atol = float(meta.get("atol", self.default_atol))
                try:
                    diff = abs(complex(sympy.N(pred)) - complex(sympy.N(gt)))
                    return diff <= atol
                except Exception:
                    return False

            try:
                return bool(sympy.simplify(pred - gt) == 0)
            except Exception:
                return False

        try:
            ok = run_with_timeout(_judge, timeout_s)
            return 1.0 if ok else 0.0
        except TimeoutExceeded:
            return 0.0
        except Exception:
            return 0.0
