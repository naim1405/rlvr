"""P3 (Deductive Logic) verifier based on srp.verify.z3_logic."""

from __future__ import annotations

import itertools
import json
import re
from typing import Any
import z3

from verifiers._util import TimeoutExceeded, extract_answer, run_with_timeout

_YES = {"yes", "y", "true", "t", "entailed", "entails", "valid", "1", "satisfiable", "sat"}
_NO = {"no", "n", "false", "f", "not entailed", "invalid", "0", "unsatisfiable", "unsat"}


def _build(
    formula: Any,
    env: dict[str, z3.BoolRef],
    universe: list[str] | None = None,
    subst: dict[str, str] | None = None,
) -> z3.BoolRef:
    if isinstance(formula, bool):
        return z3.BoolVal(formula)
    if isinstance(formula, str):
        if formula not in env:
            env[formula] = z3.Bool(formula)
        return env[formula]
    if not isinstance(formula, list) or not formula:
        raise ValueError(f"malformed formula node: {formula!r}")
    op = formula[0]

    if op == "pred":
        name = formula[1]
        resolved = [(subst[a] if subst and a in subst else a) for a in formula[2:]]
        key = f"{name}({','.join(map(str, resolved))})"
        if key not in env:
            env[key] = z3.Bool(key)
        return env[key]

    if op == "eq":
        a = subst[formula[1]] if subst and formula[1] in subst else formula[1]
        b = subst[formula[2]] if subst and formula[2] in subst else formula[2]
        return z3.BoolVal(str(a) == str(b))

    if op in ("forall", "exists"):
        if universe is None:
            raise ValueError(f"{op!r} requires a finite 'universe' in verifier_meta")
        bound = list(formula[1])
        body = formula[2]
        clauses = []
        for combo in itertools.product(universe, repeat=len(bound)):
            ext = dict(subst or {})
            ext.update(dict(zip(bound, combo)))
            clauses.append(_build(body, env, universe, ext))
        return z3.And(*clauses) if op == "forall" else z3.Or(*clauses)

    args = [_build(a, env, universe, subst) for a in formula[1:]]
    if op == "not":
        return z3.Not(args[0])
    if op == "and":
        return z3.And(*args)
    if op == "or":
        return z3.Or(*args)
    if op == "implies":
        return z3.Implies(args[0], args[1])
    if op == "iff":
        return args[0] == args[1]
    if op == "xor":
        return z3.Xor(args[0], args[1])
    raise ValueError(f"unknown operator {op!r}")


def _parse_kk_answer(text: str, people: list[str]) -> dict[str, str] | None:
    text = text.strip()
    out: dict[str, str] = {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k, v in obj.items():
                out[str(k)] = str(v).strip().lower()
    except (json.JSONDecodeError, ValueError):
        for line in re.split(r"[\n;,]+", text):
            m = re.match(r"\s*([A-Za-z_][\w'-]*)\s*[:=\-]\s*(knights?|knaves?)\s*$", line.strip(), re.IGNORECASE)
            if m:
                out[m.group(1)] = m.group(2).lower()
    if set(out) != set(people):
        return None
    norm: dict[str, str] = {}
    for p, role in out.items():
        r = "knight" if role.startswith("knight") else ("knave" if role.startswith("knave") else None)
        if r is None:
            return None
        norm[p] = r
    return norm


def _yesno(text: str) -> bool | None:
    t = text.strip().lower().strip(".'\"")
    if t in _YES:
        return True
    if t in _NO:
        return False
    head = re.split(r"[\s,.:;]+", t, 1)[0]
    if head in _YES:
        return True
    if head in _NO:
        return False
    return None


class LogicVerifier:
    """Verifier for P3: Deductive Logic tasks using Z3 theorem prover."""

    def __init__(self, reward_params: dict = None):
        reward_params = reward_params or {}

    def verify(self, model_response: str, extra_env_info: dict) -> float:
        ans = extract_answer(model_response)
        if ans is None:
            return 0.0
        meta = extra_env_info.get("verifier_meta") or {}
        kind = meta.get("kind")
        timeout_s = float(meta.get("timeout_s", 3.0))

        def _judge() -> bool:
            if kind == "knights_knaves":
                people = list(meta["people"])
                env: dict[str, z3.BoolRef] = {p: z3.Bool(p) for p in people}
                s = z3.Solver()
                for p in people:
                    stmt = _build(meta["statements"][p], env)
                    s.add(env[p] == stmt)
                if s.check() != z3.sat:
                    return False
                m = s.model()
                sol = {p: ("knight" if z3.is_true(m.eval(env[p], True)) else "knave") for p in people}
                parsed = _parse_kk_answer(ans, people)
                return parsed == sol

            if kind == "entailment":
                env: dict[str, z3.BoolRef] = {}
                universe = meta.get("universe")
                premises = [_build(p, env, universe) for p in meta.get("premises", [])]
                query = _build(meta["query"], env, universe)
                s = z3.Solver()
                s.add(*premises)
                s.add(z3.Not(query))
                entailed = s.check() == z3.unsat
                got = _yesno(ans)
                return got == entailed if got is not None else False

            if kind == "sat":
                env: dict[str, z3.BoolRef] = {}
                f = _build(meta["formula"], env, meta.get("universe"))
                s = z3.Solver()
                s.add(f)
                satisfiable = s.check() == z3.sat
                got = _yesno(ans)
                return got == satisfiable if got is not None else False

            # Fallback string comparison with ground truth if kind is unspecified
            gt = str(extra_env_info.get("expected_answer", "")).strip().lower()
            return ans.strip().lower() == gt

        try:
            ok = run_with_timeout(_judge, timeout_s)
            return 1.0 if ok else 0.0
        except Exception:
            return 0.0
