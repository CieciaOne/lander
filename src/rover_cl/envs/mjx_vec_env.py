"""Stable-Baselines3 `VecEnv` adapter around `MjxNavEnv`.

`MjxNavEnv` runs an entire batch of rover-nav episodes in JAX (jit + vmap).
SB3 expects a `VecEnv` returning numpy arrays. This wrapper sits at the
boundary: it forwards `step(actions: ndarray) -> (obs, reward, done, info[])`
to the underlying jitted step and converts JAX arrays to numpy on the way
out.

Auto-reset is handled by the underlying `MjxNavEnv` (per-env), so per SB3
convention we surface the terminal obs in `info[i]["terminal_observation"]`
and the reset (post-reset) obs as the regular return.
"""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import jax
import jax.numpy as jp
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
)

from .nav_mjx import MjxNavEnv, MjxReward


class MjxVecEnv(VecEnv):
    """Wrap an `MjxNavEnv` to look like an SB3 VecEnv."""

    def __init__(
        self,
        terrain: str,
        n_envs: int = 64,
        seed: int = 0,
        max_steps: int = 500,
        reward_cfg: MjxReward | None = None,
        impl: str = "jax",
        **mjx_kwargs: Any,
    ):
        self._env = MjxNavEnv(
            terrain=terrain,
            n_envs=n_envs,
            seed=seed,
            max_steps=max_steps,
            reward_cfg=reward_cfg,
            impl=impl,
            **mjx_kwargs,
        )

        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._env.obs_dim,), dtype=np.float32,
        )
        super().__init__(
            num_envs=n_envs,
            observation_space=observation_space,
            action_space=action_space,
        )

        # Async-step plumbing.
        self._pending_actions: jp.ndarray | None = None
        # Per-env step counters tracked on the python side too — used to
        # write `episode` outcome dicts on done transitions for SB3 monitors.
        self._py_step_count = np.zeros(n_envs, dtype=np.int64)
        self._py_cum_reward = np.zeros(n_envs, dtype=np.float32)
        # Last obs cached for "terminal_observation" semantics on autoreset.
        self._last_obs_np: np.ndarray | None = None
        self._initial_seed = seed

    # ------------------------------------------------------------------ VecEnv API

    def reset(self) -> VecEnvObs:
        obs, _ = self._env.reset(seed=self._initial_seed)
        obs_np = np.asarray(obs)
        self._py_step_count[:] = 0
        self._py_cum_reward[:] = 0.0
        self._last_obs_np = obs_np
        return obs_np

    def step_async(self, actions: np.ndarray) -> None:
        # Cast once to JAX array; the jit cache picks up the device-resident
        # value on subsequent calls.
        self._pending_actions = jp.asarray(actions, dtype=jp.float32)

    def step_wait(self) -> VecEnvStepReturn:
        assert self._pending_actions is not None
        obs, reward, done, info_jax = self._env.step(self._pending_actions)
        self._pending_actions = None

        obs_np = np.asarray(obs)
        reward_np = np.asarray(reward).astype(np.float32)
        done_np = np.asarray(done).astype(bool)

        # Convert JAX info dict to per-env Python dicts. SB3 expects info as
        # a list of dicts (length = num_envs).
        info_keys = list(info_jax.keys())
        info_arrays = {k: np.asarray(info_jax[k]) for k in info_keys}

        self._py_step_count += 1
        self._py_cum_reward += reward_np

        infos: list[dict[str, Any]] = []
        for i in range(self.num_envs):
            inf: dict[str, Any] = {}
            for k in info_keys:
                v = info_arrays[k]
                inf[k] = v[i].item() if v.ndim == 1 else tuple(v[i].tolist())
            # SB3 convention on done transitions:
            #   - return the POST-reset obs (already the case — MjxNavEnv
            #     autoresets and returns the post-reset obs for done envs)
            #   - put the terminal obs in info["terminal_observation"]
            #   - put an "episode" dict for Monitor / EpisodeCounter
            if done_np[i]:
                inf["terminal_observation"] = (
                    self._last_obs_np[i].copy() if self._last_obs_np is not None
                    else obs_np[i].copy()
                )
                inf["episode"] = {
                    "r": float(self._py_cum_reward[i]),
                    "l": int(self._py_step_count[i]),
                    "is_success": bool(inf.get("is_success", False)),
                }
                self._py_step_count[i] = 0
                self._py_cum_reward[i] = 0.0
                # SB3 also expects TimeLimit.truncated in info for proper
                # advantage/return bootstrap (Gymnasium-style).
                inf["TimeLimit.truncated"] = bool(inf.get("truncated", False)) and not bool(inf.get("terminated", False))
            infos.append(inf)

        self._last_obs_np = obs_np
        return obs_np, reward_np, done_np, infos

    def close(self) -> None:
        # MJX env holds JAX device arrays; explicit cleanup is unnecessary,
        # JAX manages them via reference counting. Drop our references.
        self._env = None
        self._pending_actions = None
        self._last_obs_np = None

    # ---- the remaining VecEnv abstract methods --------------------------------
    # These are required by the SB3 API but rarely called by PPO. We return
    # empty / sensible defaults; raise on the ones that would silently
    # mislead callers if we faked them.

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        # Read-through to the wrapped env where possible.
        indices = self._get_indices(indices) if indices is not None else range(self.num_envs)
        if hasattr(self._env, attr_name):
            v = getattr(self._env, attr_name)
            return [v for _ in indices]
        raise AttributeError(f"MjxNavEnv has no attribute {attr_name!r}")

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        # The wrapped env is shared across all "envs" in the batch, so per-
        # index set is not supported. Only allow whole-batch writes.
        if indices is not None:
            raise NotImplementedError("MjxVecEnv: per-env set_attr is not supported")
        setattr(self._env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError(
            "MjxVecEnv runs a single batched JAX env; per-env method calls "
            "aren't meaningful. Use the wrapped MjxNavEnv directly via "
            "vec_env._env if you need this."
        )

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        indices = self._get_indices(indices) if indices is not None else range(self.num_envs)
        return [False for _ in indices]


def make_mjx_vec_env(
    terrain: str,
    n_envs: int = 64,
    seed: int = 0,
    max_steps: int = 500,
    impl: str = "jax",
    **kwargs: Any,
) -> MjxVecEnv:
    return MjxVecEnv(
        terrain=terrain,
        n_envs=n_envs,
        seed=seed,
        max_steps=max_steps,
        impl=impl,
        **kwargs,
    )
