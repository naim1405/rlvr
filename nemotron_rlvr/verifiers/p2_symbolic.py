"""P2 (Symbolic Transformation) verifier.

P2 tasks ask for a *transformed* form of a given expression (expand, factor,
simplify, partial-fraction, log/exp rewrite, trig rewrite, solve for roots).
Mathematical equivalence to the gold is necessary but not sufficient: the
input the task hands the model is itself equivalent to the gold, so an answer
that merely echoes the input would score 1.0 without performing the transform.

This verifier therefore gates on three things:
  1. equivalence to the gold (or set-equality for solution sets);
  2. the answer is not a verbatim restatement of the task's input expression
     (for genuine transforms, where gold != input); and
  3. for modes with a required normal form (expand / factor), the answer is in
     that form.
"""

from __future__ import annotations

import re
from typing import Any, Optional

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

# Parse untrusted model text with SymPy's public namespace only and no Python
# builtins, so an answer string cannot reach __import__/open/eval through the
# parser. (Process isolation + rlimits in score.py are the second layer.)
_SAFE_GLOBALS = {k: v for k, v in vars(sympy).items() if not k.startswith("_")}
_SAFE_GLOBALS["__builtins__"] = {}


def _parse(text: str, symbols: list[str]) -> Any:
    local = {s: sympy.Symbol(s) for s in symbols}
    return parse_expr(
        text,
        local_dict=local,
        global_dict=_SAFE_GLOBALS,
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


def _is_expanded(pred: Any) -> bool:
    try:
        return sympy.expand(pred) == pred
    except Exception:
        return False


def _same_form(a: Any, b: Any) -> bool:
    try:
        return sympy.srepr(a) == sympy.srepr(b)
    except Exception:
        return False


def _required_form(mode: str, prompt: str) -> str:
    """The canonical form the answer must be in, from mode or the task verb."""
    if mode not in (None, "", "equiv"):
        return mode
    verb = prompt.strip().split("\n", 1)[0].lower() if prompt else ""
    if "factor" in verb:
        return "factored"
    if "expand" in verb:
        return "expanded"
    if "partial fraction" in verb or "decompose" in verb:
        return "partial_fractions"
    if "simplif" in verb or "rewrite" in verb:
        return "simplify"
    return "equiv"


def _form_ok(pred: Any, inp: Any, gt: Any, prompt: str, mode: str) -> bool:
    """True when `pred` is in the form the task demands (not just equivalent).

    A transform that only restates the input in its original form is rejected:
    the two are mathematically equal, so equivalence cannot tell them apart —
    the required *form* does.
    """
    kind = _required_form(mode, prompt)
    if inp is not None and _same_form(pred, inp):
        return False  # verbatim echo of the input
    try:
        if kind == "factored":
            return _is_fully_factored(pred)
        if kind == "expanded":
            return _is_expanded(pred)
        if kind == "partial_fractions":
            # A decomposition is a sum of ≥2 terms; the single-fraction input is not.
            return len(sympy.Add.make_args(pred)) >= 2
        if kind == "simplify":
            if inp is not None and sympy.count_ops(pred) >= sympy.count_ops(inp):
                return False  # no simpler than the input
            return sympy.count_ops(pred) <= sympy.count_ops(gt) + 1
    except Exception:
        return False
    return True  # unknown transform: equivalence + not-an-echo is enough


def _extract_input_expr(extra_env_info: dict, symbols: list[str]) -> Optional[Any]:
    """Recover the task's input expression so a verbatim echo can be rejected.

    Prefers an explicit `verifier_meta.input_expr`; otherwise lifts the single
    expression block out of the prompt (the lines between the instruction line
    and the "Answer with ..." line).
    """
    meta = extra_env_info.get("verifier_meta") or {}
    raw = meta.get("input_expr")
    if not raw:
        prompt = str(extra_env_info.get("prompt", ""))
        if not prompt:
            return None
        lines = [ln.strip() for ln in prompt.splitlines()]
        body: list[str] = []
        for ln in lines[1:]:  # skip the first instruction line
            low = ln.lower()
            if not ln:
                if body:  # stop at the blank line after the expression block
                    break
                continue
            if low.startswith(("answer", "solve the problem", "put your reasoning", "give the", "options")):
                break
            body.append(ln)
        raw = " ".join(body).strip()
    if not raw:
        return None
    try:
        return _parse(raw, symbols)
    except Exception:
        return None


class SymbolicVerifier:
    """Verifier for P2: equivalent, in the required form, and not an echo."""

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

            if not _equiv(pred, gt):
                return False

            inp = _extract_input_expr(extra_env_info, symbols)
            prompt = str(extra_env_info.get("prompt", ""))
            return _form_ok(pred, inp, gt, prompt, mode)

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
