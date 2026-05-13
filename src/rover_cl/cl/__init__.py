"""Continual-learning methods for sequential RL training."""

from __future__ import annotations

from typing import Any

from .base import CLMethod
from .ewc import EwcCL
from .naive import NaiveCL
from .replay import ReplayCL


_REGISTRY: dict[str, type] = {
    "naive": NaiveCL,
    "replay": ReplayCL,
    "ewc": EwcCL,
}


def make_cl(name: str, **kwargs: Any) -> CLMethod:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown CL method '{name}'. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key](**kwargs)


__all__ = ["CLMethod", "EwcCL", "NaiveCL", "ReplayCL", "make_cl"]
