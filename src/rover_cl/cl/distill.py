"""DistillCL: policy distillation for continual learning.

After each task k, freeze a snapshot of the policy (a "teacher"). On
subsequent tasks, alongside the normal PPO learning, do dedicated
distillation steps that minimise KL divergence between the CURRENT
policy and each frozen teacher's action distribution on transitions
sampled from a stored replay buffer.

Comparison vs Replay:
  * Replay: maximise log_prob(stored_action | obs) under current
    policy. Behaviour cloning on point estimates. Doesn't preserve
    action *spread* — if the teacher was uncertain (high-variance
    Gaussian) and we BC on a single sample, we lose that.
  * Distill: minimise KL(student || teacher) on stored obs. The
    teacher is a frozen NETWORK, so we get the full distribution
    parameters (mean + std for continuous) — preserves uncertainty
    and is generally more sample-efficient.

Implementation: keep a small obs-only replay buffer per task (don't
need actions because we re-query the teacher), plus a state_dict
snapshot of the policy after each task to use as the teacher.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO

from .base import BaseCLMethod


class DistillCL(BaseCLMethod):
    name = "distill"

    def __init__(
        self,
        buffer_size_per_task: int = 1000,
        distill_batch_size: int = 64,
        distill_steps: int = 100,
        distill_lr: float = 1e-4,
        distill_kl_weight: float = 1.0,
        ppo_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(ppo_kwargs=ppo_kwargs)
        self.buffer_size_per_task = int(buffer_size_per_task)
        self.distill_batch_size = int(distill_batch_size)
        self.distill_steps = int(distill_steps)
        self.distill_lr = float(distill_lr)
        self.distill_kl_weight = float(distill_kl_weight)
        # Per-task observation buffers (we don't need stored actions; we
        # re-query the corresponding teacher at distill time).
        self.obs_buffers: dict[str, list[np.ndarray]] = {}
        # Per-task frozen teacher state_dicts (cpu tensors, no grad).
        self.teachers: dict[str, dict[str, th.Tensor]] = {}
        self.last_distill_steps_run: int = 0
        self.last_distill_kl: float = 0.0

    # ====================================================================== main

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

        # Distill from past teachers BEFORE the new task's PPO learn.
        # Same ordering as Replay: anchor → train → snapshot. Keeps the
        # final post-PPO checkpoint tuned for the current task.
        past_ids = [tid for tid in self.teachers if tid != task_id]
        if past_ids:
            self._distill(past_ids)
        else:
            self.last_distill_steps_run = 0

        _ent_restore = None
        if ent_coef is not None:
            _ent_restore = float(self.model.ent_coef)
            self.model.ent_coef = float(ent_coef)
        self.model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            tb_log_name=f"distill_{task_id}",
            callback=callback,
        )
        if _ent_restore is not None:
            self.model.ent_coef = _ent_restore

        if task_id not in self.seen_task_ids:
            self.seen_task_ids.append(task_id)

        if not skip_post_train:
            self.post_train(env, task_id)

    def post_train(self, env: gym.Env, task_id: str) -> None:
        self._collect_obs_buffer(env, task_id)
        self._snapshot_teacher(task_id)

    # ====================================================================== distill

    def _snapshot_teacher(self, task_id: str) -> None:
        """Freeze a copy of the current policy as the teacher for task_id."""
        assert self.model is not None
        self.teachers[task_id] = {
            n: p.detach().clone()
            for n, p in self.model.policy.named_parameters()
            if p.requires_grad
        }

    def _build_teacher_policy(self, teacher_state: dict[str, th.Tensor]):
        """Load a teacher's parameters into a temporary policy clone.

        We can't run the teacher inline against `self.model.policy`
        (that's the student). Trick: clone the policy via copy.deepcopy,
        overwrite its parameters with the teacher's snapshot, no_grad it.
        """
        import copy as _copy
        assert self.model is not None
        teacher = _copy.deepcopy(self.model.policy)
        device = self.model.policy.device
        with th.no_grad():
            for n, p in teacher.named_parameters():
                if n in teacher_state:
                    p.copy_(teacher_state[n].to(device))
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        return teacher

    def _distill(self, past_ids: list[str]) -> None:
        """Distill student toward each past teacher on its stored obs."""
        assert self.model is not None
        student = self.model.policy
        optimizer = student.optimizer
        device = student.device

        # Pre-build teacher policies (saves recreating them per step).
        teachers = {
            tid: self._build_teacher_policy(self.teachers[tid])
            for tid in past_ids
            if tid in self.teachers
        }
        if not teachers:
            self.last_distill_steps_run = 0
            return

        original_lrs = [g["lr"] for g in optimizer.param_groups]
        # Distill uses a smaller LR than the PPO update — pure KL pull-back
        # at full RL LR over-corrects.
        for g in optimizer.param_groups:
            g["lr"] = self.distill_lr

        steps_done = 0
        last_kl = 0.0
        try:
            for _ in range(self.distill_steps):
                # Sample a batch of obs from a randomly-chosen past task.
                tid = random.choice(list(teachers.keys()))
                obs_pool = self.obs_buffers.get(tid, [])
                if not obs_pool:
                    continue
                picks = random.choices(obs_pool, k=self.distill_batch_size)
                obs_t = th.as_tensor(
                    np.stack(picks).astype(np.float32),
                    dtype=th.float32, device=device,
                )

                teacher = teachers[tid]
                with th.no_grad():
                    teacher_dist = teacher.get_distribution(obs_t)
                student_dist = student.get_distribution(obs_t)

                # KL(student || teacher) — student is pulled to match teacher.
                # SB3's distributions expose the underlying torch.distributions,
                # which support kl_divergence between like types.
                kl = th.distributions.kl.kl_divergence(
                    student_dist.distribution, teacher_dist.distribution,
                )
                loss = self.distill_kl_weight * kl.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                steps_done += 1
                last_kl = float(kl.mean().item())
        finally:
            for g, lr in zip(optimizer.param_groups, original_lrs):
                g["lr"] = lr

        self.last_distill_steps_run = steps_done
        # Residual KL to the teacher(s) after the distill pass — if this
        # stays high (or equals the pre-distill value), the pass is a no-op
        # and distill_steps / distill_lr need raising.
        self.last_distill_kl = last_kl

    # ====================================================================== buffer

    def _collect_obs_buffer(self, env: gym.Env, task_id: str) -> None:
        assert self.model is not None
        target = self.buffer_size_per_task

        obs, _info = env.reset()
        collected: list[np.ndarray] = []
        max_iters = target * 10
        for _ in range(max_iters):
            if len(collected) >= target:
                break
            collected.append(np.asarray(obs, dtype=np.float32).copy())
            action, _ = self.model.predict(obs, deterministic=True)
            step_out = env.step(action)
            if len(step_out) == 5:
                next_obs, _reward, terminated, truncated, _info = step_out
                done = bool(terminated or truncated)
            else:  # pragma: no cover
                next_obs, _reward, done, _info = step_out  # type: ignore[misc]
            if done:
                obs, _info = env.reset()
            else:
                obs = next_obs

        self.obs_buffers[task_id] = collected

    # ====================================================================== persistence

    def save(self, path: Path) -> None:
        super().save(path)
        sidecar = Path(path).with_suffix(".distill.npz")
        payload: dict[str, np.ndarray] = {}
        tids = list(self.teachers.keys())
        payload["__task_ids__"] = np.asarray(tids)
        for tid in tids:
            for n, t in self.teachers[tid].items():
                payload[f"teacher::{tid}::{n}"] = t.detach().cpu().numpy()
            obs_pool = self.obs_buffers.get(tid, [])
            if obs_pool:
                payload[f"obs::{tid}"] = np.stack(obs_pool).astype(np.float32)
        payload["__meta__"] = np.asarray(
            [
                self.buffer_size_per_task,
                self.distill_batch_size,
                self.distill_steps,
                self.distill_lr,
                self.distill_kl_weight,
            ],
            dtype=np.float64,
        )
        np.savez(sidecar, **payload)

    @classmethod
    def load(cls, path: Path) -> "DistillCL":
        inst = cls()
        inst.model = PPO.load(str(Path(path)))
        sidecar = Path(path).with_suffix(".distill.npz")
        if sidecar.exists():
            data = np.load(sidecar, allow_pickle=False)
            if "__meta__" in data.files:
                meta = data["__meta__"]
                inst.buffer_size_per_task = int(meta[0])
                inst.distill_batch_size = int(meta[1])
                inst.distill_steps = int(meta[2])
                inst.distill_lr = float(meta[3])
                inst.distill_kl_weight = float(meta[4])
            device = inst.model.policy.device
            for key in data.files:
                if key.startswith("teacher::"):
                    _, tid, name = key.split("::", 2)
                    inst.teachers.setdefault(tid, {})[name] = th.as_tensor(
                        data[key], device=device
                    )
                elif key.startswith("obs::"):
                    _, tid = key.split("::", 1)
                    inst.obs_buffers[tid] = [
                        data[key][i] for i in range(len(data[key]))
                    ]
            inst.seen_task_ids = list(inst.teachers.keys())
        return inst
