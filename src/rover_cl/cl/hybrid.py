"""HybridEwcReplayCL: EWC weight regularisation + per-task experience replay.

The strongest practical CL approach for our setting. Combines:

  * **Replay** (from `ReplayCL`): per-task FIFO buffers of recorded
    transitions, replayed via behaviour-cloning BEFORE each new task's PPO
    learn pass. Anchors the policy to past-task behaviours at the boundary.

  * **EWC** (from `EwcCL`): diagonal Fisher snapshot per task + penalty
    gradient integrated into every PPO gradient step via parameter hooks
    (see ewc.py). Constrains drift on parameters important to past tasks
    THROUGHOUT the new task's training, not just at the boundary.

The implementation reuses EwcCL's and ReplayCL's helper methods via
Python's unbound-method assignment — no new abstraction layer, just
function reuse. The helpers operate on `self`, accessing whichever
attribute set this class owns (we own both `fisher`/`theta_star` and
`buffers`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import torch as th

from .base import BaseCLMethod
from .ewc import EwcCL
from .replay import ReplayCL, Transition, _TaskBuffer

import numpy as np


class HybridEwcReplayCL(BaseCLMethod):
    name = "hybrid"

    def __init__(
        self,
        # EWC knobs (defaults same as EwcCL but slightly lower lam — replay
        # does some of the work, EWC doesn't have to push as hard).
        lam: float = 400.0,
        fisher_sample_size: int = 512,
        # Replay knobs.
        buffer_size_per_task: int = 1000,
        rehearsal_batch_size: int = 64,
        rehearsal_steps: int = 100,
        rehearsal_lr_scale: float = 0.5,
        ppo_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(ppo_kwargs=ppo_kwargs)
        # EWC state.
        self.lam = float(lam)
        self.fisher_sample_size = int(fisher_sample_size)
        self.fisher: dict[str, dict[str, th.Tensor]] = {}
        self.theta_star: dict[str, dict[str, th.Tensor]] = {}
        self.last_penalty_steps_run: int = 0
        self.last_penalty_value: float = 0.0
        # Replay state.
        self.buffer_size_per_task = int(buffer_size_per_task)
        self.rehearsal_batch_size = int(rehearsal_batch_size)
        self.rehearsal_steps = int(rehearsal_steps)
        self.rehearsal_lr_scale = float(rehearsal_lr_scale)
        self.buffers: dict[str, _TaskBuffer] = {}
        self.last_rehearsal_steps_run: int = 0

    # ---- helper methods reused from EwcCL and ReplayCL --------------------
    # These are plain function bodies that take self as their first arg.
    # Assigning them at class level lets us reuse the implementations
    # without inheriting either class (which would mess with __init__
    # chains and post_train semantics). Not a new abstraction — just
    # Python's normal way to share method implementations.
    _snapshot_params = EwcCL._snapshot_params
    _install_penalty_hooks = EwcCL._install_penalty_hooks
    _compute_penalty_value = EwcCL._compute_penalty_value
    _collect_fisher = EwcCL._collect_fisher
    _rehearse = ReplayCL._rehearse
    _sample_mixed_batch = ReplayCL._sample_mixed_batch
    _collect_into_buffer = ReplayCL._collect_into_buffer

    # ---- main entry ------------------------------------------------------

    def train_on(
        self,
        env: gym.Env,
        total_timesteps: int,
        task_id: str,
        log_dir: Path | None = None,
        skip_post_train: bool = False,
        callback=None,
        ent_coef: float | None = None,
    ) -> None:
        self._ensure_model(env, log_dir)
        assert self.model is not None

        past_ids = [tid for tid in self.buffers if tid != task_id]

        # 1) BEFORE PPO learn: replay rehearsal on past tasks.
        if past_ids:
            self._rehearse(past_ids)
        else:
            self.last_rehearsal_steps_run = 0

        # 2) PPO learning on the current task, with the EWC penalty gradient
        #    hooked into every backward pass (same mechanism as EwcCL).
        ewc_past_ids = [tid for tid in self.fisher if tid != task_id]
        handles = self._install_penalty_hooks(ewc_past_ids) if ewc_past_ids else []

        _ent_restore = None
        if ent_coef is not None:
            _ent_restore = float(self.model.ent_coef)
            self.model.ent_coef = float(ent_coef)
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                tb_log_name=f"hybrid_{task_id}",
                callback=callback,
            )
        finally:
            for h in handles:
                h.remove()
        if _ent_restore is not None:
            self.model.ent_coef = _ent_restore

        self.last_penalty_steps_run = (
            self._hook_backward_count if ewc_past_ids else 0
        )
        self.last_penalty_value = (
            self._compute_penalty_value(ewc_past_ids) if ewc_past_ids else 0.0
        )

        if task_id not in self.seen_task_ids:
            self.seen_task_ids.append(task_id)

        # 4) Post-train: collect Fisher AND replay buffer for this task.
        if not skip_post_train:
            self.post_train(env, task_id)

    def post_train(self, env: gym.Env, task_id: str) -> None:
        # Buffer collection uses deterministic policy; Fisher uses
        # stochastic policy. Both just READ env + policy — order doesn't
        # affect state.
        self._collect_into_buffer(env, task_id)
        self.fisher[task_id] = self._collect_fisher(env, self.fisher_sample_size)
        self.theta_star[task_id] = self._snapshot_params()

    # ====================================================================== persistence

    def save(self, path: Path) -> None:
        super().save(path)
        # Write BOTH EWC and Replay sidecars.
        ewc_path = Path(path).with_suffix(".ewc.npz")
        payload_ewc: dict[str, np.ndarray] = {}
        tids = list(self.fisher.keys())
        payload_ewc["__task_ids__"] = np.asarray(tids)
        for tid in tids:
            for n, t in self.fisher[tid].items():
                payload_ewc[f"fisher::{tid}::{n}"] = t.detach().cpu().numpy()
            for n, t in self.theta_star.get(tid, {}).items():
                payload_ewc[f"theta::{tid}::{n}"] = t.detach().cpu().numpy()
        payload_ewc["__meta__"] = np.asarray(
            [self.lam, self.fisher_sample_size], dtype=np.float64
        )
        np.savez(ewc_path, **payload_ewc)

        replay_path = Path(path).with_suffix(".replay.npz")
        flat_obs: list[np.ndarray] = []
        flat_act: list[np.ndarray] = []
        flat_rew: list[float] = []
        flat_next: list[np.ndarray] = []
        flat_done: list[bool] = []
        flat_tid: list[str] = []
        for tid, buf in self.buffers.items():
            for t in buf.items:
                flat_obs.append(t.obs)
                flat_act.append(t.action)
                flat_rew.append(t.reward)
                flat_next.append(t.next_obs)
                flat_done.append(t.done)
                flat_tid.append(tid)
        if flat_obs:
            np.savez(
                replay_path,
                obs=np.stack(flat_obs),
                action=np.stack(flat_act),
                reward=np.asarray(flat_rew, dtype=np.float32),
                next_obs=np.stack(flat_next),
                done=np.asarray(flat_done, dtype=bool),
                task_id=np.asarray(flat_tid),
            )

    @classmethod
    def load(cls, path: Path) -> "HybridEwcReplayCL":
        from stable_baselines3 import PPO

        inst = cls()
        inst.model = PPO.load(str(Path(path)))

        # EWC sidecar.
        ewc_path = Path(path).with_suffix(".ewc.npz")
        if ewc_path.exists():
            data = np.load(ewc_path, allow_pickle=False)
            if "__meta__" in data.files:
                meta = data["__meta__"]
                inst.lam = float(meta[0])
                inst.fisher_sample_size = int(meta[1])
            device = inst.model.policy.device
            for key in data.files:
                if key.startswith("fisher::"):
                    _, tid, name = key.split("::", 2)
                    inst.fisher.setdefault(tid, {})[name] = th.as_tensor(
                        data[key], device=device
                    )
                elif key.startswith("theta::"):
                    _, tid, name = key.split("::", 2)
                    inst.theta_star.setdefault(tid, {})[name] = th.as_tensor(
                        data[key], device=device
                    )

        # Replay sidecar.
        replay_path = Path(path).with_suffix(".replay.npz")
        if replay_path.exists():
            data = np.load(replay_path, allow_pickle=False)
            obs_arr = data["obs"]
            act_arr = data["action"]
            rew_arr = data["reward"]
            next_arr = data["next_obs"]
            done_arr = data["done"]
            tid_arr = data["task_id"]
            for i in range(len(obs_arr)):
                tid = str(tid_arr[i])
                buf = inst.buffers.setdefault(
                    tid, _TaskBuffer(capacity=inst.buffer_size_per_task)
                )
                buf.add(
                    Transition(
                        obs=obs_arr[i],
                        action=act_arr[i],
                        reward=float(rew_arr[i]),
                        next_obs=next_arr[i],
                        done=bool(done_arr[i]),
                        task_id=tid,
                    )
                )

        inst.seen_task_ids = list(
            set(list(inst.fisher.keys()) + list(inst.buffers.keys()))
        )
        return inst
