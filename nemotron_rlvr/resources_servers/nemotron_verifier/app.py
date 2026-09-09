"""NeMo Gym resources server that routes P0–P6 examples to in-process verifiers.

Gym looks up this package at:

    $GYM_ROOT/resources_servers/nemotron_verifier/{app.py,pyproject.toml}

`scripts/install_gym_server.py` (also invoked from train.py) copies this directory
and `nemotron_rlvr/verifiers/` into that location.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Optional


def _ensure_verifiers_on_path() -> None:
    here = Path(__file__).resolve().parent
    env_root = os.environ.get("NEMOTRON_RLVR_ROOT", "")
    candidates = [
        here,  # vendored copy: dest/verifiers/
        here.parent.parent,  # repo: nemotron_rlvr/verifiers/
        Path(env_root) if env_root else None,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if (candidate / "verifiers" / "p0_retrieval.py").is_file():
            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)
            return
    raise ImportError(
        "Cannot import P0–P6 verifiers. Expected verifiers/p0_retrieval.py next to "
        "this app, under nemotron_rlvr/, or at $NEMOTRON_RLVR_ROOT. "
        "From nemotron_rlvr/ run: python3 scripts/install_gym_server.py"
    )


_ensure_verifiers_on_path()

from score import (  # noqa: E402
    extract_assistant_text,
    extra_env_info_from_mapping,
    score_response,
)

from fastapi import FastAPI  # noqa: E402
from pydantic import Field  # noqa: E402

from nemo_gym.base_resources_server import (  # noqa: E402
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

try:
    from nemo_gym.base_resources_server import BaseRunRequest
except ImportError:  # older Gym: extra fields live only on the verify body
    BaseRunRequest = BaseVerifyRequest  # type: ignore[misc,assignment]


class NemotronVerifierServerConfig(BaseResourcesServerConfig):
    """Optional per-family constructor kwargs forwarded to the Python verifiers."""

    reward_params: dict[str, dict] = Field(default_factory=dict)


class NemotronRunRequest(BaseRunRequest):
    """Dataset fields that must survive Pydantic parsing into verify()."""

    expected_answer: Any = ""
    task_name: str = ""
    family: str = ""
    verifier: str = ""
    verifier_meta: dict[str, Any] = Field(default_factory=dict)
    instance_id: str = ""
    prompt: str = ""
    extra_env_info: dict[str, Any] = Field(default_factory=dict)


class NemotronVerifyRequest(NemotronRunRequest, BaseVerifyRequest):
    pass


class NemotronVerifyResponse(BaseVerifyResponse):
    task_name: str = ""
    family: str = ""
    instance_id: str = ""
    extracted_assistant_text: Optional[str] = None


class NemotronVerifierResourcesServer(SimpleResourcesServer):
    config: NemotronVerifierServerConfig

    def setup_webserver(self) -> FastAPI:
        # No tools — single-turn generation, then verify().
        return super().setup_webserver()

    async def verify(self, body: NemotronVerifyRequest) -> NemotronVerifyResponse:
        assistant_text = extract_assistant_text(body.response)
        nested = body.extra_env_info or {}

        def _pick(primary: Any, key: str, default: Any = "") -> Any:
            if primary not in (None, ""):
                return primary
            value = nested.get(key, default)
            return default if value in (None, "") else value

        extra = extra_env_info_from_mapping(
            {
                **nested,
                "expected_answer": _pick(body.expected_answer, "expected_answer"),
                "task_name": _pick(body.task_name, "task_name"),
                "family": _pick(body.family, "family"),
                "verifier": _pick(body.verifier, "verifier"),
                "verifier_meta": _pick(body.verifier_meta, "verifier_meta", {}) or {},
                "instance_id": _pick(body.instance_id, "instance_id"),
                "prompt": _pick(body.prompt, "prompt"),
            }
        )
        reward_params = getattr(self.config, "reward_params", None) or {}
        reward = await asyncio.to_thread(
            score_response, assistant_text, extra, reward_params
        )
        payload = body.model_dump() if hasattr(body, "model_dump") else body.dict()
        return NemotronVerifyResponse(
            **payload,
            reward=float(reward),
            task_name=extra.get("task_name", "") or "",
            family=extra.get("family", "") or "",
            instance_id=extra.get("instance_id", "") or "",
            extracted_assistant_text=assistant_text,
        )


if __name__ == "__main__":
    NemotronVerifierResourcesServer.run_webserver()
