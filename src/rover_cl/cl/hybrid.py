"""HybridEwcReplayCL: EWC weight regularisation + per-task experience replay.

The strongest practical CL approach for our setting. Combines:

  * **Replay** (from `ReplayCL`): per-task FIFO buffers of recorded
    transitions, replayed via behaviour-cloning BEFORE each new task's PPO
    learn pass. Anchors the policy to past-task behaviours at the boundary.

  * **EWC** (from `EwcCL`): diagonal Fisher snapshot per task + dedicated
    penalty-only optimization steps AFTER PPO learning. Pulls weights
    back toward `theta_star` so the new PPO update doesn't drift too far
    on parameters important to past tasks.

The two methods are complementary:
  - Replay supplies the gradient signal "what did the policy used to do",
    via concrete (obs, action) pairs.
  - EWC supplies the weight-space prior "don't move these parameters far",
    via Fisher-weighted L2.

Empirically, hybrid usually retains 1.5-2x better than either alone on
multi-phase RL curricula.

Implementation note: rather than multiple inheritance, we keep one
HybridEwcReplayCL class that owns both pieces of state (Fisher + buffers)
and applies both at the right points in the train_on / post_train cycle.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th

from .base import BaseCLMethod
from .ewc import EwcCL
from .replay import ReplayCL, Transition, _TaskBuffer


class HybridEwcReplayCL(BaseCLMethod):
    name = "hybrid"

    def __init__(
        self,
        # EWC knobs (defaults same as EwcCL but slightly lower lam — replay
        # does some of the work, EWC doesn't have to push as hard).
        lam: float = 400.0,
        fisher_sample_size: int = 512,
        penalty_steps: int = 50,
        penalty_lr: float = 1e-3,
        penalty_grad_clip: float = 1.0,
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
        self.penalty_steps = int(penalty_steps)
        self.penalty_lr = float(penalty_lr)
        self.penalty_grad_clip = float(penalty_grad_clip)
        self.fisher: dict[str, dict[str, th.Tensor]] = {}
        self.theta_star: dict[str, dict[str, th.Tensor]] = {}
        self.last_penalty_steps_run: int = 0
        # Replay state.
        self.buffer_size_per_task = int(buffer_size_per_task)
        self.rehearsal_batch_size = int(rehearsal_batch_size)
        self.rehearsal_steps = int(rehearsal_steps)
        self.rehearsal_lr_scale = float(rehearsal_lr_scale)
        self.buffers: dict[str, _TaskBuffer] = {}
        self.last_rehearsal_steps_run: int = 0

    # ====================================================================== main entry

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
        #    Anchors the policy to past-task behaviour at the boundary.
        if past_ids:
            self._rehearse(past_ids)
        else:
            self.last_rehearsal_steps_run = 0

        # 2) PPO learning on the current task.
        _ent_restore = None
        if ent_coef is not None:
            _ent_restore = float(self.model.ent_coef)
            self.model.ent_coef = float(ent_coef)
        self.model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            tb_log_name=f"hybrid_{task_id}",
            callback=callback,
        )
        if _ent_restore is not None:
            self.model.ent_coef = _ent_restore

        # 3) AFTER PPO learn: EWC penalty pull-back on past tasks.
        #    Counteracts any drift the new PPO update introduced on
        #    parameters Fisher-flagged as important to old tasks.
        ewc_past_ids = [tid for tid in self.fisher if tid != task_id]
        if ewc_past_ids:
            self._apply_ewc_penalty(ewc_past_ids)
        else:
            self.last_penalty_steps_run = 0

        if task_id not in self.seen_task_ids:
            self.seen_task_ids.append(task_id)

        # 4) Post-train: collect Fisher AND replay buffer for this task.
        if not skip_post_train:
            self.post_train(env, task_id)

    def post_train(self, env: gym.Env, task_id: str) -> None:
        # Collect replay buffer first (uses deterministic policy), then
        # Fisher (uses stochastic policy). Order doesn't matter for state
        # consistency since both just READ the env + policy; the policy's
        # parameters aren't modified by either.
        self._collect_into_buffer(env, task_id)
        self.fisher[task_id] = self._collect_fisher(env, self.fisher_sample_size)
        self.theta_star[task_id] = self._snapshot_params()

    # ====================================================================== EWC bits

    def _snapshot_params(self) -> dict[str, th.Tensor]:
        assert self.model is not None
        return {
            n: p.detach().clone()
            for n, p in self.model.policy.named_parameters()
            if p.requires_grad
        }

    def _apply_ewc_penalty(self, past_ids: list[str]) -> None:
        # Same as EwcCL._apply_ewc_penalty.
        assert self.model is not None
        policy = self.model.policy
        params = [p for p in policy.parameters() if p.requires_grad]
        penalty_opt = th.optim.SGD(params, lr=self.penalty_lr)

        steps_done = 0
        for _ in range(self.penalty_steps):
            penalty_opt.zero_grad()
            penalty = th.zeros((), device=policy.device)
            for tid in past_ids:
                fisher_tid = self.fisher[tid]
                theta_tid = self.theta_star[tid]
                for n, p in policy.named_parameters():
                    if not p.requires_grad or n not in fisher_tid:
                        continue
                    f = fisher_tid[n]
                    ts = theta_tid[n]
                    penalty = penalty + (f * (p - ts) ** 2).sum()
            loss = 0.5 * self.lam * penalty
            loss.backward()
            if self.penalty_grad_clip > 0:
                th.nn.utils.clip_grad_norm_(params, self.penalty_grad_clip)
            penalty_opt.step()
            steps_done += 1
        self.last_penalty_steps_run = steps_done

    def _collect_fisher(
        self, env: gym.Env, n_samples: int
    ) -> dict[str, th.Tensor]:
        # Same as EwcCL._collect_fisher.
        assert self.model is not None
        policy = self.model.policy
        device = policy.device

        fisher = {
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
            action, _ = self.model.predict(obs, deterministic=False)
            obs_t = th.as_tensor(
                np.asarray(obs, dtype=np.float32), dtype=th.float32, device=device
            ).unsqueeze(0)
            act_t = th.as_tensor(
                np.asarray(action, dtype=np.float32), dtype=th.float32, device=device
            ).unsqueeze(0)

            policy.zero_grad()
            _values, log_prob, _entropy = policy.evaluate_actions(obs_t, act_t)
            loss = -log_prob.mean()
            loss.backward()

            for n, p in policy.named_parameters():
                if p.grad is None or n not in fisher:
                    continue
                fisher[n] = fisher[n] + p.grad.detach() ** 2

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
        for n in list(fisher.keys()):
            fisher[n] = (fisher[n] / denom).detach()
        return fisher

    # ====================================================================== Replay bits

    def _rehearse(self, past_ids: list[str]) -> None:
        # Same as ReplayCL._rehearse.
        assert self.model is not None
        policy = self.model.policy
        optimizer = policy.optimizer
        device = policy.device

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
        # Same as ReplayCL._collect_into_buffer.
        assert self.model is not None
        buf = self.buffers.setdefault(
            task_id, _TaskBuffer(capacity=self.buffer_size_per_task)
        )
        target = self.buffer_size_per_task

        obs, _info = env.reset()
        collected = 0
        max_iters = target * 10
        for _ in range(max_iters):
            if collected >= target:
                break
            action, _ = self.model.predict(obs, deterministic=True)
            step_out = env.step(action)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, _info = step_out
                done = bool(terminated or truncated)
            else:  # pragma: no cover
                next_obs, reward, done, _info = step_out  # type: ignore[misc]
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
            [self.lam, self.fisher_sample_size, self.penalty_steps], dtype=np.float64
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
                inst.penalty_steps = int(meta[2])
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
