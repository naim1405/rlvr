"""Verifiers package for P0–P6. Imports are lazy so missing sympy/z3 does not break P0."""

from __future__ import annotations

from typing import Any

__all__ = [
    "RetrievalVerifier",
    "ArithmeticVerifier",
    "SymbolicVerifier",
    "LogicVerifier",
    "PlanningVerifier",
    "PatternVerifier",
    "ConstraintVerifier",
]

_LAZY = {
    "RetrievalVerifier": ("verifiers.p0_retrieval", "RetrievalVerifier"),
    "ArithmeticVerifier": ("verifiers.p1_arithmetic", "ArithmeticVerifier"),
    "SymbolicVerifier": ("verifiers.p2_symbolic", "SymbolicVerifier"),
    "LogicVerifier": ("verifiers.p3_logic", "LogicVerifier"),
    "PlanningVerifier": ("verifiers.p4_planning", "PlanningVerifier"),
    "PatternVerifier": ("verifiers.p5_pattern", "PatternVerifier"),
    "ConstraintVerifier": ("verifiers.p6_constraint", "ConstraintVerifier"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value
