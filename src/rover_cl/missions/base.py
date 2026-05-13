"""Mission framework: Task, Mission, Runner.

A `Task` describes one task in the sequence (env factory + training budget).
A `Mission` is an ordered list of tasks + a CL method choice.
The `Runner` loops over the mission, trains the CL method on each task in turn,
and evaluates retention on all seen tasks after every phase.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import gymnasium as gym

from rover_cl.cl import CLMethod, make_cl
from rover_cl.eval.metrics import EpisodeStats, evaluate_with_trajectories


EnvFactory = Callable[[int], gym.Env]


@dataclass
class Task:
    """One task in the sequence."""
    task_id: str
    env_factory: EnvFactory          # called with seed -> gym.Env
    train_timesteps: int = 20_000
    eval_episodes: int = 10
    eval_max_steps: int = 500


@dataclass
class Mission:
    """A continual-learning experiment definition."""
    name: str
    tasks: list[Task]
    cl_method: str = "naive"          # "naive" | "replay" | ...
    cl_kwargs: dict = field(default_factory=dict)
    seed: int = 0


@dataclass
class PhaseResult:
    phase: int
    after_training: str
    per_task: dict[str, EpisodeStats | None]
    # All durations are wall-clock seconds. `eval_seconds` aggregates
    # evaluations across all tasks seen at this phase. `post_train_seconds`
    # is non-zero only for EWC / Replay (Fisher / buffer collection).
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "after_training": self.after_training,
            "per_task": {
                k: (v.to_dict() if v is not None else None)
                for k, v in self.per_task.items()
            },
            "timings": dict(self.timings),
        }


@dataclass
class MissionResult:
    mission_name: str
    cl_method: str
    seed: int
    task_ids: list[str]
    evaluations: list[PhaseResult]
    # ISO-8601 UTC timestamps + total wall-clock seconds.
    started_at: str = ""
    ended_at: str = ""
    total_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "mission_name": self.mission_name,
            "cl_method": self.cl_method,
            "seed": self.seed,
            "task_ids": self.task_ids,
            "evaluations": [e.to_dict() for e in self.evaluations],
            "timings": {
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "total_seconds": self.total_seconds,
            },
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


class Runner:
    """Executes a Mission and returns a MissionResult."""

    def __init__(self, mission: Mission, results_dir: Path | None = None,
                 verbose: bool = True, n_envs: int = 1):
        self.mission = mission
        self.results_dir = Path(results_dir) if results_dir is not None else None
        self.verbose = verbose
        # n_envs > 1 uses SubprocVecEnv to collect PPO rollouts from N parallel
        # MuJoCo instances. On a Mac M3 (8 cores) 4 is a sweet spot — leaves
        # cores for OS / Python / the policy gradient step. EWC/Replay's
        # post-training collection still uses a single fresh env.
        self.n_envs = max(1, int(n_envs))
        # `_run_start_perf` is set when run() begins; _log uses it to print
        # elapsed-since-start alongside the local clock time.
        self._run_start_perf: float | None = None

    def _log(self, msg: str) -> None:
        if not self.verbose:
            return
        wall = time.strftime("%H:%M:%S")
        if self._run_start_perf is None:
            print(f"[{wall}] {msg}", flush=True)
        else:
            elapsed = time.perf_counter() - self._run_start_perf
            # MM:SS for the elapsed counter — easier to read than raw seconds
            # once you cross the 1-minute mark, which most phases do.
            mm, ss = divmod(int(elapsed), 60)
            print(f"[{wall}  +{mm:02d}:{ss:02d}] {msg}", flush=True)

    @staticmethod
    def _fmt_dur(seconds: float) -> str:
        """Format a wall-clock duration as `M m S.s s` (or `S.s s` < 60 s)."""
        if seconds < 60:
            return f"{seconds:.1f} s"
        mm, ss = divmod(seconds, 60)
        return f"{int(mm)} m {ss:.1f} s"

    def _tb_dir(self) -> Path | None:
        if self.results_dir is None:
            return None
        try:
            import tensorboard  # noqa: F401
        except ImportError:
            return None
        return self.results_dir / "tb"

    def run(self) -> MissionResult:
        cl: CLMethod = make_cl(self.mission.cl_method, **self.mission.cl_kwargs)
        task_ids = [t.task_id for t in self.mission.tasks]
        evaluations: list[PhaseResult] = []

        # Wall-clock start anchors all timestamps. `time.perf_counter()` is
        # the monotonic high-res clock we use for durations; `datetime.now`
        # gives the user-readable wall start/end stamps stored in results.json.
        self._run_start_perf = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._log(f"[Mission {self.mission.name}] starting "
                  f"({len(self.mission.tasks)} phases, "
                  f"cl_method={self.mission.cl_method}, "
                  f"n_envs={self.n_envs})")

        tb_dir = self._tb_dir()
        for phase, task in enumerate(self.mission.tasks):
            phase_start = time.perf_counter()
            self._log(f"\n[Phase {phase}/{len(self.mission.tasks)-1}] "
                      f"train on {task.task_id} "
                      f"({task.train_timesteps} timesteps, n_envs={self.n_envs})")

            using_vec = self.n_envs > 1
            if using_vec:
                from stable_baselines3.common.monitor import Monitor
                from stable_baselines3.common.vec_env import SubprocVecEnv

                def make_env_thunk(idx: int):
                    seed_i = self.mission.seed + phase * 1000 + idx

                    def _make():
                        return Monitor(task.env_factory(seed_i))

                    return _make

                train_env = SubprocVecEnv([make_env_thunk(i) for i in range(self.n_envs)])
            else:
                train_env = task.env_factory(self.mission.seed + phase)

            # --- training ----------------------------------------------------
            train_start = time.perf_counter()
            cl.train_on(
                env=train_env,
                total_timesteps=task.train_timesteps,
                task_id=task.task_id,
                log_dir=tb_dir,
                skip_post_train=using_vec,
            )
            train_seconds = time.perf_counter() - train_start
            self._log(f"  training done in {self._fmt_dur(train_seconds)}")

            try:
                train_env.close()
            except Exception:
                pass

            # --- post-train (Fisher / buffer) -------------------------------
            post_train_seconds = 0.0
            if using_vec:
                post_train_start = time.perf_counter()
                post_env = task.env_factory(self.mission.seed + phase * 1000 + 9999)
                try:
                    cl.post_train(post_env, task.task_id)
                finally:
                    try:
                        post_env.close()
                    except Exception:
                        pass
                post_train_seconds = time.perf_counter() - post_train_start
                if post_train_seconds >= 0.1:
                    self._log(f"  post-train ({self.mission.cl_method}) done in "
                              f"{self._fmt_dur(post_train_seconds)}")

            # --- evaluation -------------------------------------------------
            eval_start = time.perf_counter()
            per_task: dict[str, EpisodeStats | None] = {}
            for j, other in enumerate(self.mission.tasks):
                if j <= phase:
                    eval_env = other.env_factory(self.mission.seed + 1000 + j)
                    eval_seed_base = (self.mission.seed
                                      + 10000 * (phase + 1)
                                      + 100 * j)
                    task_eval_start = time.perf_counter()
                    stats, trajectories = evaluate_with_trajectories(
                        policy_predict_fn=lambda o: cl.predict(o, deterministic=True)[0],
                        env=eval_env,
                        n_episodes=other.eval_episodes,
                        max_steps=other.eval_max_steps,
                        seed_base=eval_seed_base,
                    )
                    task_eval_seconds = time.perf_counter() - task_eval_start
                    if self.results_dir is not None:
                        try:
                            from rover_cl.viz.plots import plot_run_report
                            terrain_spec = getattr(eval_env.unwrapped, "terrain", None)
                            if terrain_spec is not None:
                                report_path = (self.results_dir /
                                               f"report_phase_{phase}_after_"
                                               f"{task.task_id}_on_{other.task_id}.png")
                                plot_run_report(
                                    terrain=terrain_spec,
                                    trajectories=trajectories,
                                    out=report_path,
                                    title=f"{self.mission.name} | phase {phase} "
                                          f"(after {task.task_id}) | eval on {other.task_id}",
                                )
                                import matplotlib.pyplot as plt
                                plt.close("all")
                        except Exception as e:
                            self._log(f"  [warn] run report skipped for "
                                      f"{other.task_id}: {e!r}")
                    try:
                        eval_env.close()
                    except Exception:
                        pass
                    per_task[other.task_id] = stats
                    self._log(f"  eval on {other.task_id}: "
                              f"success_rate={stats.success_rate:.2f} "
                              f"mean_return={stats.mean_return:+.2f} "
                              f"({self._fmt_dur(task_eval_seconds)})")
                else:
                    per_task[other.task_id] = None
            eval_seconds = time.perf_counter() - eval_start

            phase_total_seconds = time.perf_counter() - phase_start
            phase_timings = {
                "train_seconds": round(train_seconds, 2),
                "post_train_seconds": round(post_train_seconds, 2),
                "eval_seconds": round(eval_seconds, 2),
                "total_seconds": round(phase_total_seconds, 2),
            }
            evaluations.append(PhaseResult(
                phase=phase, after_training=task.task_id,
                per_task=per_task, timings=phase_timings,
            ))
            self._log(f"  phase {phase} done · "
                      f"train {self._fmt_dur(train_seconds)} · "
                      f"eval {self._fmt_dur(eval_seconds)} · "
                      f"total {self._fmt_dur(phase_total_seconds)}")

            if self.results_dir is not None:
                ckpt = self.results_dir / f"ckpt_phase_{phase}_after_{task.task_id}.zip"
                cl.save(ckpt)

        total_seconds = time.perf_counter() - self._run_start_perf
        ended_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._log(f"\n[Mission {self.mission.name}] complete · "
                  f"total wall-clock {self._fmt_dur(total_seconds)}")

        result = MissionResult(
            mission_name=self.mission.name,
            cl_method=self.mission.cl_method,
            seed=self.mission.seed,
            task_ids=task_ids,
            evaluations=evaluations,
            started_at=started_at,
            ended_at=ended_at,
            total_seconds=round(total_seconds, 2),
        )
        if self.results_dir is not None:
            result.save(self.results_dir / "results.json")
        return result
