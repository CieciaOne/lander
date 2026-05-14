"""NaiveCL: vanilla fine-tuning baseline (the 'expected to forget' control)."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from .base import BaseCLMethod


class NaiveCL(BaseCLMethod):
    name = "naive"

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
        # reset_num_timesteps=False keeps the global step counter monotonically
        # increasing across tasks so tensorboard plots are continuous.
        self.model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            tb_log_name=f"naive_{task_id}",
            callback=callback,
        )
        if _ent_restore is not None:
            self.model.ent_coef = _ent_restore
        if task_id not in self.seen_task_ids:
            self.seen_task_ids.append(task_id)
        # NaiveCL has no post-training collection, so skip_post_train is a no-op here.
