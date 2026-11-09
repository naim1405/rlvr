"""P0 (Factual Retrieval) verifier.

Gold answers are either a single span or a list of spans. Some generators ask
for one span ("Name one area served by…") while storing the full list as gold;
others expect the whole set. To keep a correct answer from being silently
scored 0 under that inconsistency, this verifier accepts, by default, EITHER a
single valid span (membership) OR the complete set (set-equality). A generator
can pin the intended behavior with `verifier_meta.span_match`:

  "any" → any one listed span is correct (membership)
  "all" → the full set is required (set-equality)
  unset → accept either (default)
"""

from __future__ import annotations

import json
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


def _split_items(text: str) -> list[str]:
    """Parse an answer into candidate spans (JSON list, or comma/newline split)."""
    t = text.strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, (list, tuple)):
            return [str(x) for x in obj]
    except (json.JSONDecodeError, ValueError):
        pass
    t = t.strip("[](){}")
    return [p for p in re.split(r"[,\n;]+", t) if p.strip()]


class RetrievalVerifier:
    """Verifier for P0: single-span membership or full-set equality."""

    def __init__(self, reward_params: dict = None):
        reward_params = reward_params or {}
        self.default_mode = reward_params.get("normalize", "default")

    def verify(self, model_response: str, extra_env_info: dict) -> float:
        ans = extract_answer(model_response)
        if ans is None:
            return 0.0

        meta = extra_env_info.get("verifier_meta") or {}
        mode = meta.get("normalize", self.default_mode)
        span_match = meta.get("span_match", "either")

        gt = extra_env_info.get("expected_answer", "")
        if isinstance(gt, (list, tuple)):
            gold_items = [str(x) for x in gt]
        else:
            gold_items = [str(gt)]
        aliases = [str(a) for a in meta.get("aliases", [])]

        na = _norm(ans, mode)
        gold_norm = {_norm(g, mode) for g in gold_items}
        alias_norm = {_norm(a, mode) for a in aliases}

        # Membership: the whole answer, or any parsed item, is a listed span.
        ans_items = {_norm(x, mode) for x in _split_items(ans)}
        membership = (
            na in gold_norm
            or na in alias_norm
            or bool(ans_items & gold_norm)
        )
        # Set-equality: the answer names exactly the full gold set.
        set_equal = bool(gold_norm) and ans_items == gold_norm

        if span_match == "any":
            return 1.0 if membership else 0.0
        if span_match == "all":
            return 1.0 if set_equal else 0.0
        return 1.0 if (membership or set_equal) else 0.0
