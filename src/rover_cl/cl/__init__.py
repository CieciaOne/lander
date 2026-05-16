"""Continual-learning methods for sequential RL training."""

from __future__ import annotations

from typing import Any

from .base import CLMethod
from .distill import DistillCL
from .ewc import EwcCL
from .hybrid import HybridEwcReplayCL
from .l2 import L2CL
from .mas import MasCL
from .naive import NaiveCL
from .replay import ReplayCL


_REGISTRY: dict[str, type] = {
    "naive": NaiveCL,        # no protection (control)
    "replay": ReplayCL,      # BC rehearsal on stored transitions
    "ewc": EwcCL,            # Fisher-weighted weight regularisation
    "hybrid": HybridEwcReplayCL,  # EWC + Replay together
    "l2": L2CL,              # uniform-weight L2 (baseline — Fisher = 1)
    "mas": MasCL,            # Memory Aware Synapses (alt importance)
    "distill": DistillCL,    # KL distillation from frozen teacher
}


def make_cl(name: str, **kwargs: Any) -> CLMethod:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown CL method '{name}'. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key](**kwargs)


__all__ = [
    "CLMethod", "DistillCL", "EwcCL", "HybridEwcReplayCL", "L2CL",
    "MasCL", "NaiveCL", "ReplayCL", "make_cl",
]
