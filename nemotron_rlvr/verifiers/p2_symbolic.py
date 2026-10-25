"""P2 (Symbolic Transformation) verifier based on srp.verify.sympy_eq."""

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

_EMPTY_SET_TOKENS = {
    "{}", "{ }", "[]", "()", "∅", "emptyset", "empty set", "none",
    "no solution", "no solutions", "nosolution", "the empty set",
}


def _parse(text: str, symbols: list[str]) -> Any:
    local = {s: sympy.Symbol(s) for s in symbols}
    return parse_expr(
        text,
        local_dict=local,
        transformations=_TRANSFORMS,
        evaluate=True,
    )


def _split_top_level(s: str) -> list[str]:
    out, depth, buf = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return [p.strip() for p in out if p.strip()]


def _parse_set(text: str, symbols: list[str]) -> list[Any]:
    t = text.strip()
    if t.lower() in _EMPTY_SET_TOKENS or t == "":
        return []
    m = re.match(r"^\s*[A-Za-z_]\w*\s*(?:∈|\bin\b)\s*(.+)$", t)
    if m:
        t = m.group(1).strip()
    if len(t) >= 2 and t[0] in "([{" and t[-1] in ")]}":
        t = t[1:-1].strip()
    if t == "" or t.lower() in _EMPTY_SET_TOKENS:
        return []
    pieces = _split_top_level(t)
    vals: list[Any] = []
    for p in pieces:
        if "=" in p:
            p = p.split("=")[-1].strip()
        if p:
            vals.append(_parse(p, symbols))
    return vals


def _equiv(a: Any, b: Any) -> bool:
    try:
        return sympy.simplify(a - b) == 0
    except Exception:
        return False


def _set_equal(pred: list[Any], gt: list[Any]) -> bool:
    def dedupe(xs: list[Any]) -> list[Any]:
        u: list[Any] = []
        for x in xs:
            if not any(_equiv(x, y) for y in u):
                u.append(x)
        return u

    P, G = dedupe(pred), dedupe(gt)
    if len(P) != len(G):
        return False
    remaining = list(G)
    for p in P:
        for i, g in enumerate(remaining):
            if _equiv(p, g):
                remaining.pop(i)
                break
        else:
            return False
    return not remaining


def _is_fully_factored(pred: Any) -> bool:
    expr = sympy.together(pred)
    _c, true_facs = sympy.factor_list(sympy.expand(expr))
    true_count = sum(m for _f, m in true_facs)
    if true_count == 0:
        return False

    got_count = 0
    for a in sympy.Mul.make_args(expr):
        base, exp = a.as_base_exp()
        if not base.free_symbols:
            continue
        sub = sympy.factor_list(base)[1]
        if sum(m for _f, m in sub) != 1:
            return False
        e = int(exp) if getattr(exp, "is_Integer", False) else 1
        got_count += e
    return got_count == true_count


class SymbolicVerifier:
    """Verifier for P2: Symbolic Transformation tasks using SymPy."""

    def __init__(self, reward_params: dict = None):
        reward_params = reward_params or {}

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
                return False

            if mode == "numeric":
                atol = float(meta.get("atol", 1e-9))
                try:
                    diff = abs(complex(sympy.N(pred)) - complex(sympy.N(gt)))
                    return diff <= atol
                except Exception:
                    return False

            try:
                equiv = bool(sympy.simplify(pred - gt) == 0)
            except Exception:
                return False

            if not equiv:
                return False

            if mode == "factored":
                try:
                    return _is_fully_factored(pred)
                except Exception:
                    return False
            return True

        def _judge_set() -> bool:
            try:
                pred_items = _parse_set(ans, symbols)
                gt_raw = extra_env_info.get("expected_answer")
                if isinstance(gt_raw, (list, tuple)):
                    gt_items = [_parse(str(x), symbols) for x in gt_raw]
                else:
                    gt_items = _parse_set(str(gt_raw), symbols)
                return _set_equal(pred_items, gt_items)
            except Exception:
                return False

        try:
            chosen = _judge_set if mode == "set" else _judge
            ok = run_with_timeout(chosen, timeout_s)
            return 1.0 if ok else 0.0
        except TimeoutExceeded:
            return 0.0
        except Exception:
            return 0.0
