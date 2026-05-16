"""Base class for continual-learning methods wrapping a single SB3 PPO model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


DEFAULT_PPO_KWARGS: dict[str, Any] = {
    # n_steps bumped 512 → 2048: longer rollouts give better advantage estimates
    # for the 1000+-step episodes where the goal_bonus arrives only at the end.
    "n_steps": 2048,
    "batch_size": 128,
    "learning_rate": 3e-4,
    # gamma 0.99 → 0.995: effective horizon ≈ 200 steps instead of 100, so the
    # +50 goal_bonus at step 500 backpropagates as +4 at step 0 instead of +0.3.
    "gamma": 0.995,
    "gae_lambda": 0.95,
    # net_arch [64,64] → [128,128]: more capacity to reason over 8-obstacle
    # bounding-box obs (38-D input) and combine with pose / velocity.
    "policy_kwargs": {"net_arch": [128, 128]},
    # ent_coef 0 → 0.01: nudges the policy to stay stochastic during training
    # so it keeps exploring alternatives instead of collapsing to "drive
    # forward and freeze" — that local optimum was eating the chart.
    "ent_coef": 0.01,
    # Force PPO's policy onto CPU. For our [128, 128] MLP, kernel-launch
    # overhead from running on GPU exceeds the compute saving, AND it would
    # add CPU↔GPU memcpys per step on the MJX backend (obs comes off the
    # JAX device as numpy → would have to copy to CUDA for the policy
    # forward, then action back to CPU for the env). Keeping the policy
    # on CPU means JAX uses the GPU exclusively and PyTorch uses the CPU
    # exclusively — no cross-device traffic. Also silences the SB3 warning
    # "PPO on GPU is primarily intended to run on CPU when not using a CNN".
    "device": "cpu",
    "verbose": 0,
}


@runtime_checkable
class CLMethod(Protocol):
    name: str

    def train_on(
        self,
        env: gym.Env,
        total_timesteps: int,
        task_id: str,
        log_dir: Path | None = None,
    ) -> None: ...

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, Any]: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "CLMethod": ...


class BaseCLMethod:
    """Shared machinery: a single PPO model kept across all train_on calls."""

    name: str = "base"

    def __init__(self, ppo_kwargs: dict[str, Any] | None = None) -> None:
        merged = {**DEFAULT_PPO_KWARGS, **(ppo_kwargs or {})}
        # policy_kwargs needs a deep-ish merge so callers can override net_arch only
        if ppo_kwargs and "policy_kwargs" in ppo_kwargs:
            merged["policy_kwargs"] = {
                **DEFAULT_PPO_KWARGS["policy_kwargs"],
                **ppo_kwargs["policy_kwargs"],
            }
        self._ppo_kwargs: dict[str, Any] = merged
        self.model: PPO | None = None
        self.seen_task_ids: list[str] = []

    def _ensure_model(self, env: gym.Env, log_dir: Path | None) -> None:
        # Lazy init keeps the same PPO weights across task boundaries (CL contract).
        if self.model is None:
            tb_log = str(log_dir) if log_dir is not None else None
            self.model = PPO(
                "MlpPolicy", env, tensorboard_log=tb_log, **self._ppo_kwargs
            )
        else:
            self.model.set_env(env)

    def train_on(
        self,
        env: gym.Env,
        total_timesteps: int,
        task_id: str,
        log_dir: Path | None = None,
        skip_post_train: bool = False,
        callback: Any | None = None,
        ent_coef: float | None = None,
    ) -> None:
        """Run PPO training on `env`. By default, the subclass should also call
        `self.post_train(env, task_id)` at the end (or call it itself). When
        `skip_post_train=True`, the subclass MUST skip that step; the Runner
        will call `post_train` separately on a single env after closing a
        (possibly multi-process) training VecEnv.

        `callback` is forwarded to `model.learn(callback=...)` so the Runner
        can attach an episode counter or any other SB3 callback.

        `ent_coef` overrides the PPO entropy coefficient for this phase
        only (subclasses should restore the prior value after `learn()`).
        """
        raise NotImplementedError

    def post_train(self, env: gym.Env, task_id: str) -> None:
        """Post-PPO data collection on a single env (Fisher / replay buffer
        / etc.). Default is a noop — Naive does nothing here. Subclasses
        override to do their thing."""
        return None

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, Any]:
        if self.model is None:
            raise RuntimeError("predict() called before any train_on()")
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("save() called before any train_on()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "BaseCLMethod":
        inst = cls()
        inst.model = PPO.load(str(Path(path)))
        return inst
