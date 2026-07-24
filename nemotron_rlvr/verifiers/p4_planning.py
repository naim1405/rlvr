"""P4 (Procedural / Planning) verifier based on srp.verify.pddl_plan."""

from __future__ import annotations

import json
import re
from typing import Any

from verifiers._util import extract_answer


def _parse_plan(text: str) -> list[str] | None:
    text = text.strip()
    if text == "" or text.lower() in {"[]", "(empty)", "empty", "noop", "no-op"}:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj]
    except (json.JSONDecodeError, ValueError):
        pass
    raw = re.split(r"[\n,;]+|->|=>|→", text)
    plan: list[str] = []
    for tok in raw:
        t = tok.strip()
        t = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", t).strip()
        if t:
            plan.append(t)
    return plan or None


class PlanningVerifier:
    """Verifier for P4: Procedural Reasoning / Planning tasks using STRIPS PDDL simulation."""

    def __init__(self, reward_params: dict = None):
        reward_params = reward_params or {}

    def verify(self, model_response: str, extra_env_info: dict) -> float:
        ans = extract_answer(model_response)
        if ans is None:
            return 0.0

        meta: dict[str, Any] = extra_env_info.get("verifier_meta") or {}
        actions = meta.get("actions", {})
        state: set[str] = set(meta.get("init", []))
        goal: set[str] = set(meta.get("goal", []))
        goal_neg: set[str] = set(meta.get("goal_neg", []))

        plan = _parse_plan(ans)
        if plan is None:
            # Fallback check against ground truth string if not parseable as PDDL plan
            gt = str(extra_env_info.get("expected_answer", "")).strip()
            return 1.0 if ans.strip() == gt else 0.0

        if not actions and not state and not goal:
            gt = str(extra_env_info.get("expected_answer", "")).strip()
            return 1.0 if ans.strip() == gt else 0.0

        for step, act_name in enumerate(plan):
            if act_name not in actions:
                return 0.0
            a = actions[act_name]
            pre = set(a.get("pre", []))
            if not pre.issubset(state):
                return 0.0
            state -= set(a.get("del", []))
            state |= set(a.get("add", []))

        ok = goal.issubset(state) and goal_neg.isdisjoint(state)
        return 1.0 if ok else 0.0
