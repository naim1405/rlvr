"""P5 (Pattern Recognition) verifier based on srp.verify.seq_match."""

from __future__ import annotations

import json
import re
from typing import Any

from verifiers._util import extract_answer


def _tokenize(text: str, kind: str) -> list[str] | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            toks = [str(x).strip() for x in obj]
            return _coerce(toks, kind)
    except (json.JSONDecodeError, ValueError):
        pass
    parts = [p for p in re.split(r"[\s,;]+", text) if p != ""]
    return _coerce(parts, kind)


def _coerce(toks: list[str], kind: str) -> list[str] | None:
    if kind == "int":
        out = []
        for t in toks:
            t = t.strip()
            if re.fullmatch(r"[+-]?\d+", t):
                out.append(str(int(t)))
            else:
                return None
        return out
    return [t.strip() for t in toks]


def _as_list(gt: Any, kind: str) -> list[str]:
    if isinstance(gt, (list, tuple)):
        seq = [str(x).strip() for x in gt]
    else:
        seq = [p for p in re.split(r"[\s,;]+", str(gt)) if p != ""]
    return _coerce(seq, kind) or seq


class PatternVerifier:
    """Verifier for P5: Pattern Recognition tasks."""

    def __init__(self, reward_params: dict = None):
        reward_params = reward_params or {}

    def verify(self, model_response: str, extra_env_info: dict) -> float:
        ans = extract_answer(model_response)
        if ans is None:
            return 0.0

        meta = extra_env_info.get("verifier_meta") or {}
        match = meta.get("match", "exact")
        kind = meta.get("kind", "int")

        gt_src = extra_env_info.get("expected_answer", "")
        expected = _as_list(gt_src, kind)

        got = _tokenize(ans, kind)
        if got is None:
            # Fallback simple string match if non-integer sequence
            gt_str = str(gt_src).strip().lower()
            return 1.0 if ans.strip().lower() == gt_str else 0.0

        if match == "prefix":
            ok = got[: len(expected)] == expected and len(got) >= len(expected)
        else:
            ok = got == expected

        return 1.0 if ok else 0.0
