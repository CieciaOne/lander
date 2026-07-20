"""EwcCL: Elastic Weight Consolidation on top of a shared PPO policy.

After each task k completes, we estimate the diagonal Fisher information of the
policy parameters by replaying transitions and accumulating squared gradients of
``-log_prob(action | obs)``. We also snapshot the post-training parameters
``theta_star``. When subsequent tasks (k+1, k+2, ...) train, the penalty

    L_ewc = (lam / 2) * sum_tasks sum_params fisher_task[p] * (p - theta_star_task[p])^2

is integrated INTO the PPO loss for every gradient step, via per-parameter
gradient hooks: each hook adds the analytic penalty gradient
``lam * fisher * (p - theta_star)`` to the parameter's gradient during the
backward pass of PPO's loss. This is mathematically identical to adding the
penalty term to the loss itself, but requires no SB3 subclassing.

(Earlier versions applied the penalty as a separate SGD pull-back pass AFTER
`model.learn()` — a "periodic consolidation" variant. That let PPO drift
unconstrained for an entire phase before any correction, which is not the
EWC from the literature and empirically under-protected old tasks.)
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
        ppo_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(ppo_kwargs=ppo_kwargs)
        self.lam = float(lam)
        self.fisher_sample_size = int(fisher_sample_size)
        # task_id -> { param_name -> Tensor } (both on policy device, no grad).
        self.fisher: dict[str, dict[str, th.Tensor]] = {}
        self.theta_star: dict[str, dict[str, th.Tensor]] = {}
        # Diagnostics, refreshed per train_on call:
        #   last_penalty_steps_run — number of backward passes that included
        #     the penalty gradient (0 on the first task).
        #   last_penalty_value — the penalty term 0.5*lam*sum(F*(θ-θ*)^2)
        #     measured AFTER training, i.e. the residual drift cost.
        self.last_penalty_steps_run: int = 0
        self.last_penalty_value: float = 0.0

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

        # Install gradient hooks so the EWC penalty gradient rides along with
        # every PPO backward pass during learn(). Removed afterwards so the
        # Fisher-collection backward in post_train stays clean.
        past_ids = [tid for tid in self.fisher if tid != task_id]
        handles = self._install_penalty_hooks(past_ids) if past_ids else []

        _ent_restore = None
        if ent_coef is not None:
            _ent_restore = float(self.model.ent_coef)
            self.model.ent_coef = float(ent_coef)
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                tb_log_name=f"ewc_{task_id}",
                callback=callback,
            )
        finally:
            for h in handles:
                h.remove()
        if _ent_restore is not None:
            self.model.ent_coef = _ent_restore

        self.last_penalty_steps_run = self._hook_backward_count if past_ids else 0
        self.last_penalty_value = (
            self._compute_penalty_value(past_ids) if past_ids else 0.0
        )

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

    def _install_penalty_hooks(self, past_ids: list[str]) -> list:
        """Register per-parameter gradient hooks implementing the EWC penalty.

        For each policy parameter p protected by at least one past task, the
        hook adds ``lam * sum_t F_t * (p - theta*_t)`` to p's gradient during
        every backward pass. This is the analytic gradient of
        ``0.5 * lam * sum_t F_t * (p - theta*_t)^2`` — mathematically the same
        as adding the penalty term to PPO's loss, without subclassing SB3.

        The hooks also run during the rollout phase's occasional backward-free
        forward passes harmlessly (hooks only fire on backward). They MUST be
        removed before Fisher collection, or the penalty gradient would
        contaminate the squared-gradient Fisher estimate — `train_on` removes
        them in a finally block.

        Returns the list of hook handles; caller removes them after learn().
        """
        assert self.model is not None
        policy = self.model.policy
        handles = []
        self._hook_backward_count = 0
        first_param_hooked = True
        for n, p in policy.named_parameters():
            if not p.requires_grad:
                continue
            terms: list[tuple[th.Tensor, th.Tensor]] = []
            for tid in past_ids:
                f = self.fisher[tid].get(n)
                ts = self.theta_star[tid].get(n)
                if f is not None and ts is not None:
                    terms.append((f, ts))
            if not terms:
                continue

            count_this = first_param_hooked
            first_param_hooked = False

            def _make_hook(p_ref: th.Tensor, terms=terms, count=count_this):
                def _hook(grad: th.Tensor) -> th.Tensor:
                    if count:
                        # One designated hook counts backward passes so
                        # last_penalty_steps_run reflects how many gradient
                        # steps actually carried the penalty.
                        self._hook_backward_count += 1
                    with th.no_grad():
                        pen = th.zeros_like(grad)
                        for f, ts in terms:
                            pen += f * (p_ref - ts)
                        return grad + self.lam * pen
                return _hook

            handles.append(p.register_hook(_make_hook(p)))
        return handles

    def _compute_penalty_value(self, past_ids: list[str]) -> float:
        """Current value of 0.5*lam*sum(F*(θ-θ*)^2) — diagnostic only."""
        assert self.model is not None
        with th.no_grad():
            penalty = 0.0
            for tid in past_ids:
                fisher_tid = self.fisher[tid]
                theta_tid = self.theta_star[tid]
                for n, p in self.model.policy.named_parameters():
                    if not p.requires_grad or n not in fisher_tid:
                        continue
                    penalty += float(
                        (fisher_tid[n] * (p - theta_tid[n]) ** 2).sum().item()
                    )
        return 0.5 * self.lam * penalty

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
            [self.lam, self.fisher_sample_size], dtype=np.float64
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
