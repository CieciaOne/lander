"""MasCL: Memory Aware Synapses (Aljundi et al., 2018).

Like EWC, MAS estimates per-parameter importance after each task and
penalises weight drift on important parameters during subsequent
training. The DIFFERENCE is what's used as the importance signal:

  * EWC: ``E[(d log_pi / d param)^2]`` — Fisher information of the
    policy. Big when the parameter strongly affects the likelihood of
    the action the policy took.
  * MAS: ``E[|d output / d param|]`` — sensitivity of the output to
    the parameter. Big when the parameter strongly affects the action.

MAS is generally simpler (no log-probability evaluation, just gradient
of the policy mean) and works well in unsupervised / RL settings where
the Fisher form is awkward. Aljundi et al. report MAS matching or
beating EWC on several CL benchmarks.

Implementation: subclasses EwcCL and only overrides `_collect_fisher`
to compute the MAS sensitivity instead of true Fisher. Reuses the
penalty machinery (which doesn't care whether the importance came
from Fisher or MAS — both are non-negative per-parameter weights).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch as th

from .ewc import EwcCL


class MasCL(EwcCL):
    name = "mas"

    def __init__(
        self,
        lam: float = 5000.0,    # MAS importance values tend to be smaller
                                  # than Fisher; ~5x EWC lam to match its
                                  # effective penalty scale. (Was 1000 —
                                  # inconsistent with this very comment.)
        fisher_sample_size: int = 512,
        ppo_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            lam=lam, fisher_sample_size=fisher_sample_size,
            ppo_kwargs=ppo_kwargs, **kwargs,
        )

    def _collect_fisher(
        self, env: gym.Env, n_samples: int
    ) -> dict[str, th.Tensor]:
        """Compute MAS importance: E[|d output / d param|] over rollouts.

        For continuous-action policies the "output" is the action mean.
        We sample obs, compute the policy's action mean, then backprop
        the L2 norm of the mean to get gradients wrt each parameter.
        The absolute value of those gradients, averaged over samples,
        is the MAS importance.
        """
        assert self.model is not None
        policy = self.model.policy
        device = policy.device

        importance = {
            n: th.zeros_like(p, device=device)
            for n, p in policy.named_parameters()
            if p.requires_grad
        }

        n_samples = max(1, int(n_samples))
        obs, _info = env.reset()
        collected = 0
        max_iters = n_samples * 10
        for _ in range(max_iters):
            if collected >= n_samples:
                break

            obs_t = th.as_tensor(
                np.asarray(obs, dtype=np.float32), dtype=th.float32, device=device
            ).unsqueeze(0)

            # Get the policy's action MEAN (deterministic output). We backprop
            # ||mean||^2 — its gradient norm with respect to a parameter is
            # the sensitivity of the output to that parameter.
            policy.zero_grad()
            distribution = policy.get_distribution(obs_t)
            mean_action = distribution.distribution.mean
            # squared L2 norm of the action mean. Gradient = 2 * mean * d_mean/dp,
            # so |grad| ∝ sensitivity.
            output_norm_sq = (mean_action ** 2).sum()
            output_norm_sq.backward()

            for n, p in policy.named_parameters():
                if p.grad is None or n not in importance:
                    continue
                importance[n] = importance[n] + p.grad.detach().abs()

            # Use the same action to step the env (with stochasticity, for
            # rollout diversity).
            action, _ = self.model.predict(obs, deterministic=False)
            step_out = env.step(action)
            if len(step_out) == 5:
                next_obs, _reward, terminated, truncated, _info = step_out
                done = bool(terminated or truncated)
            else:  # pragma: no cover
                next_obs, _reward, done, _info = step_out  # type: ignore[misc]

            collected += 1
            if done:
                obs, _info = env.reset()
            else:
                obs = next_obs

        policy.zero_grad()
        denom = float(max(collected, 1))
        for n in list(importance.keys()):
            importance[n] = (importance[n] / denom).detach()
        return importance
