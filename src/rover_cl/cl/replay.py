"""ReplayCL: keep a small per-task transition buffer and rehearse via BC after each task."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th

from .base import BaseCLMethod


@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    task_id: str


@dataclass
class _TaskBuffer:
    """Per-task FIFO with reservoir sampling once at capacity."""

    capacity: int
    items: list[Transition] = field(default_factory=list)
    seen: int = 0  # total transitions ever offered (for reservoir math)

    def add(self, t: Transition) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(t)
        else:
            # Reservoir sampling keeps the buffer an unbiased sample of the stream.
            j = random.randint(0, self.seen - 1)
            if j < self.capacity:
                self.items[j] = t

    def __len__(self) -> int:
        return len(self.items)


class ReplayCL(BaseCLMethod):
    name = "replay"

    def __init__(
        self,
        buffer_size_per_task: int = 1000,
        rehearsal_batch_size: int = 64,
        rehearsal_steps: int = 100,
        rehearsal_lr_scale: float = 0.5,
        ppo_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(ppo_kwargs=ppo_kwargs)
        self.buffer_size_per_task = buffer_size_per_task
        self.rehearsal_batch_size = rehearsal_batch_size
        self.rehearsal_steps = rehearsal_steps
        self.rehearsal_lr_scale = rehearsal_lr_scale
        self.buffers: dict[str, _TaskBuffer] = {}
        self.last_rehearsal_steps_run: int = 0  # for tests / introspection

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

        # Rehearse on past tasks BEFORE PPO learning on the new task. The
        # earlier ordering (rehearse-after) was an unforced bug for PPO:
        # `_rehearse` runs pure behaviour-cloning gradient steps on stored
        # past-task transitions, which moves the policy back toward past
        # behaviour — desirable as a *task-boundary anchor*, destructive
        # when applied *after* PPO has just finished tuning the policy for
        # the new task. Order now is anchor → train → collect, so the
        # checkpoint saved at end of phase is fully tuned for the current
        # task, and past-task retention is supplied by the next phase's
        # rehearsal pass (which will see this task in its buffer).
        past_ids = [tid for tid in self.buffers if tid != task_id]
        if past_ids:
            self._rehearse(past_ids)
        else:
            self.last_rehearsal_steps_run = 0

        _ent_restore = None
        if ent_coef is not None:
            _ent_restore = float(self.model.ent_coef)
            self.model.ent_coef = float(ent_coef)
        self.model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            tb_log_name=f"replay_{task_id}",
            callback=callback,
        )
        if _ent_restore is not None:
            self.model.ent_coef = _ent_restore

        if task_id not in self.seen_task_ids:
            self.seen_task_ids.append(task_id)

        if not skip_post_train:
            # Collect fresh transitions for THIS task using the same env that
            # PPO trained on. Runner sets skip_post_train=True when env is a
            # VecEnv (manual stepping doesn't generalize) and then calls
            # post_train separately with a single env.
            self.post_train(env, task_id)

    def post_train(self, env: gym.Env, task_id: str) -> None:
        self._collect_into_buffer(env, task_id)

    # ---------- internal helpers ----------

    def _rehearse(self, past_ids: list[str]) -> None:
        assert self.model is not None
        policy = self.model.policy
        optimizer = policy.optimizer
        device = policy.device

        # Temporarily scale LR for the rehearsal phase — pure BC at the full RL LR
        # tends to over-correct the policy mean and destabilize the value head.
        original_lrs = [g["lr"] for g in optimizer.param_groups]
        for g in optimizer.param_groups:
            g["lr"] = g["lr"] * self.rehearsal_lr_scale

        steps_done = 0
        try:
            for _ in range(self.rehearsal_steps):
                batch = self._sample_mixed_batch(past_ids, self.rehearsal_batch_size)
                if batch is None:
                    break
                obs_np, act_np = batch
                obs_t = th.as_tensor(obs_np, dtype=th.float32, device=device)
                act_t = th.as_tensor(act_np, dtype=th.float32, device=device)

                _values, log_prob, _entropy = policy.evaluate_actions(obs_t, act_t)
                loss = -log_prob.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                steps_done += 1
        finally:
            for g, lr in zip(optimizer.param_groups, original_lrs):
                g["lr"] = lr

        self.last_rehearsal_steps_run = steps_done

    def _sample_mixed_batch(
        self, past_ids: list[str], batch_size: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        pool: list[Transition] = []
        for tid in past_ids:
            pool.extend(self.buffers[tid].items)
        if not pool:
            return None
        picks = random.choices(pool, k=batch_size)
        obs = np.stack([p.obs for p in picks]).astype(np.float32)
        actions = np.stack([np.asarray(p.action) for p in picks]).astype(np.float32)
        return obs, actions

    def _collect_into_buffer(self, env: gym.Env, task_id: str) -> None:
        assert self.model is not None
        buf = self.buffers.setdefault(
            task_id, _TaskBuffer(capacity=self.buffer_size_per_task)
        )
        target = self.buffer_size_per_task

        obs, _info = env.reset()
        collected = 0
        # Hard cap on env steps in case episodes are very short and produce no progress.
        max_iters = target * 10
        for _ in range(max_iters):
            if collected >= target:
                break
            action, _ = self.model.predict(obs, deterministic=True)
            step_out = env.step(action)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, _info = step_out
                done = bool(terminated or truncated)
            else:  # pragma: no cover - gym<0.26 fallback
                next_obs, reward, done, _info = step_out  # type: ignore[misc]
                terminated = done
            buf.add(
                Transition(
                    obs=np.asarray(obs, dtype=np.float32).copy(),
                    action=np.asarray(action, dtype=np.float32).copy(),
                    reward=float(reward),
                    next_obs=np.asarray(next_obs, dtype=np.float32).copy(),
                    done=done,
                    task_id=task_id,
                )
            )
            collected += 1
            if done:
                obs, _info = env.reset()
            else:
                obs = next_obs

    # ---------- persistence ----------

    def save(self, path: Path) -> None:
        super().save(path)
        # Buffers aren't pickled into the SB3 zip; sidecar npz keeps things simple.
        sidecar = Path(path).with_suffix(".replay.npz")
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
                sidecar,
                obs=np.stack(flat_obs),
                action=np.stack(flat_act),
                reward=np.asarray(flat_rew, dtype=np.float32),
                next_obs=np.stack(flat_next),
                done=np.asarray(flat_done, dtype=bool),
                task_id=np.asarray(flat_tid),
            )

    @classmethod
    def load(cls, path: Path) -> "ReplayCL":
        inst = cls()
        from stable_baselines3 import PPO

        inst.model = PPO.load(str(Path(path)))
        sidecar = Path(path).with_suffix(".replay.npz")
        if sidecar.exists():
            data = np.load(sidecar, allow_pickle=False)
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
        return inst
