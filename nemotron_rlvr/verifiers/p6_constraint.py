"""P6 (Constraint Satisfaction) verifier based on srp.verify.cpsat."""

from __future__ import annotations

import json
import re
from typing import Any

from verifiers._util import extract_answer


def _parse_assignment(text: str, variables: list[str]) -> dict[str, Any] | None:
    text = text.strip()
    out: dict[str, Any] = {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            out = {str(k): v for k, v in obj.items()}
    except (json.JSONDecodeError, ValueError):
        for line in re.split(r"[\n;,]+", text):
            m = re.match(r"\s*([A-Za-z_]\w*)\s*[:=]\s*(-?\d+|[A-Za-z]\w*)\s*$", line.strip())
            if m:
                v = m.group(2)
                out[m.group(1)] = int(v) if re.fullmatch(r"-?\d+", v) else v
    if set(out) != set(variables):
        return None
    for k, v in list(out.items()):
        if isinstance(v, str) and re.fullmatch(r"-?\d+", v):
            out[k] = int(v)
    return out


def _term(t: Any, asn: dict[str, Any]) -> Any:
    if isinstance(t, list) and len(t) == 2 and t[0] == "const":
        return t[1]
    return asn[t]


def _check_constraint(c: dict, asn: dict[str, Any]) -> bool:
    kind = c["type"]
    if kind == "all_different":
        vals = [asn[v] for v in c["vars"]]
        return len(set(vals)) == len(vals)
    if kind == "eq":
        return _term(c["a"], asn) == _term(c["b"], asn)
    if kind == "neq":
        return _term(c["a"], asn) != _term(c["b"], asn)
    if kind == "linear":
        lhs = sum(coef * asn[v] for coef, v in c["terms"])
        op, rhs = c["op"], c["rhs"]
        return {
            "<=": lhs <= rhs, ">=": lhs >= rhs, "==": lhs == rhs,
            "<": lhs < rhs, ">": lhs > rhs,
        }[op]
    if kind == "relation":
        tup = tuple(asn[v] for v in c["vars"])
        return list(tup) in [list(a) for a in c["allowed"]]
    if kind == "forbidden":
        tup = tuple(asn[v] for v in c["vars"])
        return list(tup) not in [list(f) for f in c["forbidden"]]
    raise ValueError(f"unknown constraint type {kind!r}")


class ConstraintVerifier:
    """Verifier for P6: Constraint Satisfaction tasks."""

    def __init__(self, reward_params: dict = None):
        reward_params = reward_params or {}

    def verify(self, model_response: str, extra_env_info: dict) -> float:
        ans = extract_answer(model_response)
        if ans is None:
            return 0.0

        meta = extra_env_info.get("verifier_meta") or {}
        variables = meta.get("variables")
        domains = meta.get("domains")
        constraints = meta.get("constraints")

        if not variables or not domains or constraints is None:
            # Fallback to ground truth string match
            gt = str(extra_env_info.get("expected_answer", "")).strip().lower()
            return 1.0 if ans.strip().lower() == gt else 0.0

        variables = list(variables)
        asn = _parse_assignment(ans, variables)
        if asn is None:
            return 0.0

        for v in variables:
            if asn[v] not in domains[v]:
                return 0.0

        for c in constraints:
            try:
                if not _check_constraint(c, asn):
                    return 0.0
            except Exception:
                return 0.0

        return 1.0
