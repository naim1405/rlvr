"""verifiers._util — shared verifier helpers adapted from srp.verify._util."""

from __future__ import annotations

import re
import signal
import threading
from typing import Any, Callable, Optional

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(model_output: str) -> Optional[str]:
    """Return the content of the LAST <answer>...</answer> span, stripped.

    Returns None if no well-formed span exists. Strips surrounding
    Markdown code fences if present.
    """
    if not model_output:
        return None
    spans = _ANSWER_RE.findall(model_output)
    if not spans:
        return None
    ans = spans[-1].strip()
    fence = re.match(r"^```[a-zA-Z0-9]*\n(.*)\n```$", ans, re.DOTALL)
    if fence:
        ans = fence.group(1).strip()
    elif len(ans) >= 2 and ans[0] == "`" and ans[-1] == "`":
        ans = ans[1:-1].strip()
    return ans


class TimeoutExceeded(Exception):
    """Raised when a guarded call exceeds its wall-clock budget."""


def run_with_timeout(
    fn: Callable[[], Any],
    timeout_s: float,
) -> Any:
    """Run fn() with a hard wall-clock cap."""
    is_main = threading.current_thread() is threading.main_thread()
    if is_main and hasattr(signal, "SIGALRM"):
        def _handler(signum, frame):
            raise TimeoutExceeded(f"exceeded {timeout_s}s")

        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            return fn()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _target():
        try:
            result["v"] = fn()
        except BaseException as e:
            error["e"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutExceeded(f"exceeded {timeout_s}s")
    if "e" in error:
        raise error["e"]
    return result.get("v")
