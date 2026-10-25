"""P0 (Factual Retrieval) verifier based on srp.verify.string_match."""

from __future__ import annotations

import re
from verifiers._util import extract_answer


def _norm(s: str, mode: str) -> str:
    s = s.strip()
    if mode == "strict":
        return s
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\n\r\"'.,;:!?()[]{}")
    return s


class RetrievalVerifier:
    """Verifier for P0: Factual Retrieval tasks."""

    def __init__(self, reward_params: dict = None):
        reward_params = reward_params or {}
        self.default_mode = reward_params.get("normalize", "default")

    def verify(self, model_response: str, extra_env_info: dict) -> float:
        ans = extract_answer(model_response)
        if ans is None:
            return 0.0

        meta = extra_env_info.get("verifier_meta") or {}
        mode = meta.get("normalize", self.default_mode)

        acceptable: list[str] = []
        gt = extra_env_info.get("expected_answer", "")
        if isinstance(gt, (list, tuple)):
            acceptable.extend(str(x) for x in gt)
        else:
            acceptable.append(str(gt))
        acceptable.extend(str(a) for a in meta.get("aliases", []))

        na = _norm(ans, mode)
        for cand in acceptable:
            if na == _norm(cand, mode):
                return 1.0
        return 0.0
