"""EwcCL: Elastic Weight Consolidation on top of a shared PPO policy.

After each task k completes, we estimate the diagonal Fisher information of the
policy parameters by replaying transitions and accumulating squared gradients of
``-log_prob(action | obs)``. We also snapshot the post-training parameters
``theta_star``. When subsequent tasks (k+1, k+2, ...) train, an extra penalty

    L_ewc = (lam / 2) * sum_tasks sum_params fisher_task[p] * (p - theta_star_task[p])^2

is minimized via a small number of dedicated gradient steps on the same
optimizer AFTER `model.learn(...)`. This keeps the implementation independent of
SB3's internal PPO loss and matches the most common public EWC-for-PPO pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th

from .base import BaseCLMethod


class EwcCL(BaseCLMethod):
    name = "ewc"

    def __init__(
        self,
        lam: float = 1000.0,
        fisher_sample_size: int = 512,
        penalty_steps: int = 50,
        penalty_lr: float = 1e-3,
        penalty_grad_clip: float = 1.0,
        ppo_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(ppo_kwargs=ppo_kwargs)
        self.lam = float(lam)
        self.fisher_sample_size = int(fisher_sample_size)
        self.penalty_steps = int(penalty_steps)
        # We use a *fresh* SGD optimizer for the EWC penalty pass instead of
        # reusing the policy's Adam optimizer. Adam's running v_t is calibrated
        # for PPO-scale gradients (small); when the much larger penalty
        # gradients hit it, the adaptive LR overshoots and pushes params AWAY
        # from theta_star — empirically verified to flip the EWC direction.
        # SGD with a small fixed lr is well-behaved and matches what most
        # public EWC-for-PPO implementations do.
        self.penalty_lr = float(penalty_lr)
        self.penalty_grad_clip = float(penalty_grad_clip)
        # task_id -> { param_name -> Tensor } (both on policy device, no grad).
        self.fisher: dict[str, dict[str, th.Tensor]] = {}
        self.theta_star: dict[str, dict[str, th.Tensor]] = {}
        self.last_penalty_steps_run: int = 0

    # ---------- main entry ----------

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

        _ent_restore = None
        if ent_coef is not None:
            _ent_restore = float(self.model.ent_coef)
            self.model.ent_coef = float(ent_coef)
        self.model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            tb_log_name=f"ewc_{task_id}",
            callback=callback,
        )
        if _ent_restore is not None:
            self.model.ent_coef = _ent_restore

        # If we already have Fisher info for previous tasks, pull params back
        # toward their snapshots via N dedicated penalty-only gradient steps.
        past_ids = [tid for tid in self.fisher if tid != task_id]
        if past_ids:
            self._apply_ewc_penalty(past_ids)
        else:
            self.last_penalty_steps_run = 0

        if task_id not in self.seen_task_ids:
            self.seen_task_ids.append(task_id)

        if not skip_post_train:
            # Estimate Fisher on the same env PPO trained on. Runner sets
            # skip_post_train=True when env is a (Subproc)VecEnv and calls
            # post_train separately with a fresh single env.
            self.post_train(env, task_id)

    def post_train(self, env: gym.Env, task_id: str) -> None:
        self.fisher[task_id] = self._collect_fisher(env, self.fisher_sample_size)
        self.theta_star[task_id] = self._snapshot_params()

    # ---------- internal helpers ----------

    def _snapshot_params(self) -> dict[str, th.Tensor]:
        assert self.model is not None
        return {
            n: p.detach().clone()
            for n, p in self.model.policy.named_parameters()
            if p.requires_grad
        }

    def _apply_ewc_penalty(self, past_ids: list[str]) -> None:
        assert self.model is not None
        policy = self.model.policy

        # Fresh SGD optimizer just for this penalty pass — reusing the policy's
        # Adam was empirically wrong: Adam's running v_t (calibrated to small
        # PPO grads) overshoots when a much larger penalty gradient lands on
        # top, and the steps push params AWAY from theta_star instead of toward
        # it. Plain SGD with a small lr + grad clipping is well-behaved.
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
                    if not p.requires_grad:
                        continue
                    if n not in fisher_tid:
                        continue
                    f = fisher_tid[n]
                    ts = theta_tid[n]
                    penalty = penalty + (f * (p - ts) ** 2).sum()
            loss = 0.5 * self.lam * penalty
            # If lam == 0 the loss is identically 0; backward yields zero
            # gradients but we still take a step so last_penalty_steps_run
            # increments (test_lambda_zero_disables_protection relies on this).
            loss.backward()
            if self.penalty_grad_clip > 0:
                th.nn.utils.clip_grad_norm_(params, self.penalty_grad_clip)
            penalty_opt.step()
            steps_done += 1

        self.last_penalty_steps_run = steps_done

    def _collect_fisher(
        self, env: gym.Env, n_samples: int
    ) -> dict[str, th.Tensor]:
        """Roll out the deterministic policy and estimate diagonal Fisher.

        For each (obs, action) sample we backprop ``-log_prob`` and accumulate
        per-parameter squared gradients. Final result is averaged over samples.
        """
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
        # Hard cap iterations so a degenerate env can't loop forever.
        max_iters = n_samples * 10
        for _ in range(max_iters):
            if collected >= n_samples:
                break
            # Sample stochastically (deterministic=False) so the action is not
            # the policy's mean. The gradient of log_prob(mu | obs) wrt the
            # mean-producing weights is identically zero — using sampled
            # actions is the only way to get a meaningful Fisher diagonal.
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
            else:  # pragma: no cover - gym<0.26 fallback
                next_obs, _reward, done, _info = step_out  # type: ignore[misc]

            collected += 1
            if done:
                obs, _info = env.reset()
            else:
                obs = next_obs

        # Clear any lingering grads so subsequent PPO updates start clean.
        policy.zero_grad()

        denom = float(max(collected, 1))
        for n in list(fisher.keys()):
            fisher[n] = (fisher[n] / denom).detach()
        return fisher

    # ---------- persistence ----------

    def save(self, path: Path) -> None:
        super().save(path)
        sidecar = Path(path).with_suffix(".ewc.npz")
        payload: dict[str, np.ndarray] = {}
        # Encode task list, then per-task fisher / theta_star arrays under
        # mangled keys "fisher::<tid>::<param>" / "theta::<tid>::<param>".
        tids = list(self.fisher.keys())
        payload["__task_ids__"] = np.asarray(tids)
        for tid in tids:
            for n, t in self.fisher[tid].items():
                payload[f"fisher::{tid}::{n}"] = t.detach().cpu().numpy()
            for n, t in self.theta_star.get(tid, {}).items():
                payload[f"theta::{tid}::{n}"] = t.detach().cpu().numpy()
        payload["__meta__"] = np.asarray(
            [self.lam, self.fisher_sample_size, self.penalty_steps], dtype=np.float64
        )
        np.savez(sidecar, **payload)

    @classmethod
    def load(cls, path: Path) -> "EwcCL":
        from stable_baselines3 import PPO

        inst = cls()
        inst.model = PPO.load(str(Path(path)))
        sidecar = Path(path).with_suffix(".ewc.npz")
        if sidecar.exists():
            data = np.load(sidecar, allow_pickle=False)
            if "__meta__" in data.files:
                meta = data["__meta__"]
                inst.lam = float(meta[0])
                inst.fisher_sample_size = int(meta[1])
                inst.penalty_steps = int(meta[2])
            tids = (
                [str(t) for t in data["__task_ids__"]]
                if "__task_ids__" in data.files
                else []
            )
            device = inst.model.policy.device
            for tid in tids:
                inst.fisher[tid] = {}
                inst.theta_star[tid] = {}
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
            inst.seen_task_ids = list(inst.fisher.keys())
        return inst
