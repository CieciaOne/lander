"""L2CL: naive parameter-drift penalty (uniform importance).

The dumb baseline in the CL comparison. Same penalty STRUCTURE as EWC —
pull parameters toward post-task snapshots after each new task's PPO
learn pass — but with UNIFORM per-parameter weight (every weight is
equally important) instead of EWC's Fisher-derived weighting.

Useful for showing whether EWC's Fisher matters. If L2 retains nearly
as well as EWC, the Fisher information isn't doing much. If L2 is
significantly worse, EWC's importance estimation is what's making the
difference.

Implementation: subclasses EwcCL and overrides `_collect_fisher` to
return ones instead of computing real Fisher. Reuses all the penalty-
step machinery.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch as th

from .ewc import EwcCL


class L2CL(EwcCL):
    name = "l2"

    def __init__(
        self,
        lam: float = 100.0,  # smaller default than EWC — uniform weight
                              # means the penalty value is much bigger per
                              # param, so the effective scale needs to drop
        ppo_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(lam=lam, ppo_kwargs=ppo_kwargs, **kwargs)

    def _collect_fisher(
        self, env: gym.Env, n_samples: int
    ) -> dict[str, th.Tensor]:
        """Return uniform-1 'importance' for every parameter — no env
        rollout needed, no actual Fisher information used."""
        assert self.model is not None
        return {
            n: th.ones_like(p)
            for n, p in self.model.policy.named_parameters()
            if p.requires_grad
        }
