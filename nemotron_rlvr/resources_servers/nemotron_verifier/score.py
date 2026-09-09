"""Gym-free P0–P6 scoring used by the NeMo Gym resources server.

The FastAPI wrapper in app.py extracts assistant text from a Gym rollout and
passes it here. Keeping this module free of nemo_gym imports lets us unit-test
routing without a Gym checkout.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

_LOG = logging.getLogger("nemotron_verifier")

# Z3 (P3) can SIGSEGV. A thread timeout cannot contain that — it kills Gym
# and training hangs on "Collecting rollouts". Score P3 in a child process.
_ISOLATED_TASKS = frozenset({"p3_logic"})
_CHILD_ENV = "NEMOTRON_VERIFIER_CHILD"

_CHILD_BOOTSTRAP = (
    "import json,os,sys;"
    "from score import score_response_impl as _impl;"
    "p=json.load(sys.stdin);"
    "r=_impl(p['text'], p['extra'], p.get('params') or {});"
    "json.dump(float(r), sys.stdout)"
)


TASK_ALIASES: dict[str, str] = {
    "p0": "p0_retrieval",
    "p0_retrieval": "p0_retrieval",
    "retrieval": "p0_retrieval",
    "p1": "p1_arithmetic",
    "p1_arithmetic": "p1_arithmetic",
    "arithmetic": "p1_arithmetic",
    "p2": "p2_symbolic",
    "p2_symbolic": "p2_symbolic",
    "symbolic": "p2_symbolic",
    "p3": "p3_logic",
    "p3_logic": "p3_logic",
    "logic": "p3_logic",
    "p4": "p4_planning",
    "p4_planning": "p4_planning",
    "planning": "p4_planning",
    "p5": "p5_pattern",
    "p5_pattern": "p5_pattern",
    "pattern": "p5_pattern",
    "p6": "p6_constraint",
    "p6_constraint": "p6_constraint",
    "constraint": "p6_constraint",
}

# Uppercase family tags from the raw corpus (P0 … P6).
for _family_idx, _task in enumerate(
    (
        "p0_retrieval",
        "p1_arithmetic",
        "p2_symbolic",
        "p3_logic",
        "p4_planning",
        "p5_pattern",
        "p6_constraint",
    )
):
    TASK_ALIASES[f"P{_family_idx}"] = _task


def canonicalize_task_name(*candidates: Any) -> str:
    """Map task_name / family / verifier strings onto p0_retrieval … p6_constraint."""
    for raw in candidates:
        if raw is None:
            continue
        key = str(raw).strip()
        if not key:
            continue
        if key in TASK_ALIASES:
            return TASK_ALIASES[key]
        lowered = key.lower().replace("-", "_").replace(" ", "_")
        if lowered in TASK_ALIASES:
            return TASK_ALIASES[lowered]
        # "verifiers.p1_arithmetic:ArithmeticVerifier" and similar leftovers.
        for alias, task in TASK_ALIASES.items():
            if alias in lowered:
                return task
    raise ValueError(
        f"Cannot resolve P0–P6 task from candidates={candidates!r}. "
        "Set task_name (e.g. p0_retrieval) or family (e.g. P0) on the example."
    )


def _verifier_class(task: str):
    if task == "p0_retrieval":
        from verifiers.p0_retrieval import RetrievalVerifier

        return RetrievalVerifier
    if task == "p1_arithmetic":
        from verifiers.p1_arithmetic import ArithmeticVerifier

        return ArithmeticVerifier
    if task == "p2_symbolic":
        from verifiers.p2_symbolic import SymbolicVerifier

        return SymbolicVerifier
    if task == "p3_logic":
        from verifiers.p3_logic import LogicVerifier

        return LogicVerifier
    if task == "p4_planning":
        from verifiers.p4_planning import PlanningVerifier

        return PlanningVerifier
    if task == "p5_pattern":
        from verifiers.p5_pattern import PatternVerifier

        return PatternVerifier
    if task == "p6_constraint":
        from verifiers.p6_constraint import ConstraintVerifier

        return ConstraintVerifier
    raise ValueError(f"Unknown task {task!r}")


def extra_env_info_from_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """Build the dict the Python verifiers expect from a Gym example / verify body."""
    meta = data.get("verifier_meta")
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        meta = dict(meta)
    return {
        "expected_answer": data.get("expected_answer", data.get("ground_truth", "")),
        "verifier_meta": meta,
        "task_name": data.get("task_name", ""),
        "family": data.get("family", ""),
        "verifier": data.get("verifier", ""),
        "instance_id": data.get("instance_id", ""),
        "prompt": data.get("prompt", ""),
    }


def _content_text(content: Any) -> list[str]:
    texts: list[str] = []
    if content is None:
        return texts
    if isinstance(content, str):
        texts.append(content)
        return texts
    if isinstance(content, list):
        for item in content:
            texts.extend(_content_text(item))
        return texts
    if isinstance(content, dict):
        ctype = content.get("type")
        if ctype in (None, "output_text", "text", "input_text"):
            if content.get("text"):
                texts.append(str(content["text"]))
        elif content.get("content"):
            texts.extend(_content_text(content["content"]))
        return texts
    text = getattr(content, "text", None)
    if text:
        texts.append(str(text))
        return texts
    nested = getattr(content, "content", None)
    if nested is not None and nested is not content:
        texts.extend(_content_text(nested))
    return texts


def extract_assistant_text(response: Any) -> str:
    """Pull concatenated assistant text out of a Gym response object or dict."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response

    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")

    if not output:
        # Some dumps store the whole message on `text`.
        if isinstance(response, dict) and response.get("text"):
            return str(response["text"])
        return ""

    chunks: list[str] = []
    for item in output:
        if isinstance(item, dict):
            itype = item.get("type")
            if item.get("generation_str"):
                chunks.append(str(item["generation_str"]))
                continue
            if itype in (None, "message", "output_text"):
                chunks.extend(_content_text(item.get("content", item.get("text"))))
            continue
        itype = getattr(item, "type", None)
        if itype in (None, "message", "output_text"):
            chunks.extend(_content_text(getattr(item, "content", None)))
            text = getattr(item, "text", None)
            if text and not chunks:
                chunks.append(str(text))
    return "".join(chunks)


def score_response_impl(
    model_response: str,
    extra_env_info: dict[str, Any],
    reward_params: Optional[dict[str, dict]] = None,
) -> float:
    """In-process P0–P6 scoring. Native crashes here can kill the interpreter."""
    try:
        task = canonicalize_task_name(
            extra_env_info.get("task_name"),
            extra_env_info.get("family"),
            extra_env_info.get("verifier"),
        )
    except Exception:
        return 0.0

    params: dict[str, Any] = {}
    if reward_params:
        params = dict(reward_params.get(task) or reward_params.get(task.split("_")[0], {}) or {})

    try:
        cls = _verifier_class(task)
        verifier = cls(reward_params=params)
        reward = verifier.verify(model_response, extra_env_info)
        return float(reward)
    except Exception:
        _LOG.exception(
            "verifier failed task=%s family=%s",
            extra_env_info.get("task_name"),
            extra_env_info.get("family"),
        )
        return 0.0


def _p3_timeout_s(extra_env_info: dict[str, Any]) -> float:
    meta = extra_env_info.get("verifier_meta") or {}
    try:
        timeout_s = float(meta.get("timeout_s", 3.0))
    except (TypeError, ValueError):
        timeout_s = 3.0
    # Child import + Z3; cap so a wedged solver cannot stall the step.
    return min(max(timeout_s, 1.0), 8.0) + 2.0


def _score_isolated(
    model_response: str,
    extra_env_info: dict[str, Any],
    reward_params: Optional[dict[str, dict]],
) -> float:
    server_dir = str(Path(__file__).resolve().parent)
    env = os.environ.copy()
    env[_CHILD_ENV] = "1"
    env["NEMOTRON_VERIFIER_DIR"] = server_dir
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = server_dir + (os.pathsep + pythonpath if pythonpath else "")
    try:
        payload = json.dumps(
            {
                "text": model_response,
                "extra": extra_env_info,
                "params": reward_params or {},
            },
            default=str,
        )
    except (TypeError, ValueError):
        return 0.0
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD_BOOTSTRAP],
            input=payload,
            capture_output=True,
            text=True,
            timeout=_p3_timeout_s(extra_env_info),
            env=env,
            cwd=server_dir,
        )
    except subprocess.TimeoutExpired:
        _LOG.warning("isolated P3 verifier timed out")
        return 0.0
    except Exception:
        _LOG.exception("isolated P3 verifier failed to start")
        return 0.0
    if proc.returncode != 0:
        err = (proc.stderr or "")[-400:]
        _LOG.warning("isolated P3 verifier died rc=%s stderr=%s", proc.returncode, err)
        return 0.0
    try:
        return float(json.loads(proc.stdout or "0"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0


def score_response(
    model_response: str,
    extra_env_info: dict[str, Any],
    reward_params: Optional[dict[str, dict]] = None,
) -> float:
    """Run the matching P0–P6 verifier. Returns 0.0/1.0 (never raises)."""
    if os.environ.get(_CHILD_ENV) == "1":
        return score_response_impl(model_response, extra_env_info, reward_params)
    try:
        task = canonicalize_task_name(
            extra_env_info.get("task_name"),
            extra_env_info.get("family"),
            extra_env_info.get("verifier"),
        )
    except Exception:
        return 0.0
    if task in _ISOLATED_TASKS:
        return _score_isolated(model_response, extra_env_info, reward_params)
    return score_response_impl(model_response, extra_env_info, reward_params)
