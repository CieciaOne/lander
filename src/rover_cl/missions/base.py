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
from stable_baselines3.common.callbacks import BaseCallback

from rover_cl.cl import CLMethod, make_cl
from rover_cl.eval.metrics import EpisodeStats, evaluate_with_trajectories


class EpisodeCounter(BaseCallback):
    """SB3 callback that counts completed episodes during a training phase.

    On each environment step, SB3 puts the per-env `infos` and `dones`
    arrays into `self.locals`. A done flag = one completed episode (works
    for both single-env and SubprocVecEnv setups). We also track the mean
    episode length across all completed episodes so the Runner can print
    "episodes=K (mean len=L steps)" in the per-phase summary.
    """

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.n_episodes: int = 0
        self.total_episode_steps: int = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is None:
            return True
        infos = self.locals.get("infos") or []
        for i, done in enumerate(dones):
            if not bool(done):
                continue
            self.n_episodes += 1
            # SB3's Monitor wrapper places `episode: {"r": ..., "l": ..., "t": ...}`
            # into the info dict on the terminal step. Use the reported length
            # when available; else fall back to a single step.
            info_i = infos[i] if i < len(infos) else {}
            ep = info_i.get("episode") if isinstance(info_i, dict) else None
            if isinstance(ep, dict) and "l" in ep:
                self.total_episode_steps += int(ep["l"])
        return True

    @property
    def mean_episode_length(self) -> float:
        if self.n_episodes == 0:
            return 0.0
        return self.total_episode_steps / self.n_episodes


EnvFactory = Callable[[int], gym.Env]


@dataclass
class Task:
    """One task in the sequence."""
    task_id: str
    env_factory: EnvFactory          # called with seed -> gym.Env
    train_timesteps: int = 20_000
    eval_episodes: int = 10
    eval_max_steps: int = 500
    # Per-phase PPO entropy-coefficient override. When set, the Runner
    # updates `cl.model.ent_coef` to this value before `model.learn()`.
    # Used by scenario_10 to raise exploration on the obstacle-heavy
    # phases (3-5) — those phases were getting stuck in the "freeze near
    # start" local minimum, which higher entropy helps escape. None means
    # "leave the default from PPO kwargs alone".
    ent_coef: float | None = None
    # Adaptive-advance gate. When `min_success_to_advance` is set, the
    # Runner keeps training the phase in `gate_check_interval`-step chunks
    # until either eval success on the current phase task crosses the
    # threshold OR the cumulative trained steps hit `train_timesteps *
    # max_budget_multiplier`. Off by default so existing scenarios are
    # unaffected; opt-in per-task. Used by scenario_11_robust_generalist.
    min_success_to_advance: float | None = None
    max_budget_multiplier: float = 2.0
    gate_check_interval: int = 50_000
    # 16 episodes → binomial std ≈ 0.125 at p=0.5; at the old default of 8
    # the gate advanced phases on ±18pp noise.
    gate_eval_episodes: int = 16
    # Interim eval cadence. When > 0, the Runner runs `interim_eval_episodes`
    # rollouts on the current phase task every `interim_eval_every` env
    # steps and stashes the per-checkpoint success rate in results.json
    # under `interim_eval`. 0 disables. Independent from the adaptive gate
    # (you can run interim eval without gating, or gate without separate
    # interim eval — though the gate produces interim eval as a side
    # effect).
    interim_eval_every: int = 0
    interim_eval_episodes: int = 5


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
                 verbose: bool = True, n_envs: int = 1,
                 backend: str = "cpu", mjx_impl: str = "jax"):
        self.mission = mission
        self.results_dir = Path(results_dir) if results_dir is not None else None
        self.verbose = verbose
        # n_envs > 1 uses SubprocVecEnv to collect PPO rollouts from N parallel
        # MuJoCo instances. On a Mac M3 (8 cores) 4 is a sweet spot — leaves
        # cores for OS / Python / the policy gradient step. EWC/Replay's
        # post-training collection still uses a single fresh env.
        self.n_envs = max(1, int(n_envs))
        # backend = "cpu" → SubprocVecEnv of native-MuJoCo envs (default).
        # backend = "mjx" → MjxVecEnv (JAX + mjx.put_model). On CUDA the GPU
        # backend runs N envs in parallel inside one process. `mjx_impl` is
        # forwarded to mjx.put_model — "jax" everywhere, "warp" only on a
        # Linux box with nvidia-warp installed and an Nvidia GPU.
        self.backend = backend
        self.mjx_impl = mjx_impl
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
        # Thread the mission seed into PPO itself (network init, action
        # sampling). Without this, PPO draws from the global torch RNG and
        # a re-run of the same (scenario, method, seed) triple produces a
        # different policy trajectory — not reproducible.
        cl_kwargs = dict(self.mission.cl_kwargs)
        ppo_kw = dict(cl_kwargs.get("ppo_kwargs") or {})
        ppo_kw.setdefault("seed", self.mission.seed)
        cl_kwargs["ppo_kwargs"] = ppo_kw
        cl: CLMethod = make_cl(self.mission.cl_method, **cl_kwargs)
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

            using_vec = self.n_envs > 1 or self.backend == "mjx"
            if self.backend == "mjx":
                # MJX: a single MjxVecEnv that batches N envs in JAX. The
                # `task.env_factory` is a closure built by the scenario that
                # returns a `RoverNavEnv(terrain=...)` — we peek at the
                # terrain name and rebuild an MjxVecEnv on top of it.
                terrain_name = task.task_id
                from rover_cl.envs.mjx_vec_env import MjxVecEnv
                train_env = MjxVecEnv(
                    terrain=terrain_name,
                    n_envs=max(self.n_envs, 1),
                    seed=self.mission.seed + phase,
                    max_steps=getattr(task, "max_steps", 500),
                    impl=self.mjx_impl,
                )
            elif using_vec:
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
            # Two opt-in features piggyback on this loop:
            #   * Interim eval. If `task.interim_eval_every > 0`, every N env
            #     steps we run a quick eval on this phase's own task and
            #     record (steps_trained, success_rate) into the results.json's
            #     `interim_eval` field. Lets us see mid-phase learning curves
            #     without waiting for the post-phase eval.
            #   * Adaptive gate. If `task.min_success_to_advance` is set, we
            #     train in `gate_check_interval`-step chunks and stop early
            #     once eval success crosses the threshold (saving budget on
            #     easy phases), or train up to `train_timesteps *
            #     max_budget_multiplier` (saving the harder ones).
            #
            # When neither feature is enabled (the default for all existing
            # scenarios), we fall through to a single `cl.train_on` call —
            # identical to the previous behaviour.
            episode_counter = EpisodeCounter()
            train_start = time.perf_counter()
            if task.ent_coef is not None:
                self._log(f"  ent_coef override: {task.ent_coef:.4f} for this phase")

            interim_history: list[dict[str, float]] = []

            use_chunked = (task.interim_eval_every > 0
                           or task.min_success_to_advance is not None)
            if not use_chunked:
                cl.train_on(
                    env=train_env,
                    total_timesteps=task.train_timesteps,
                    task_id=task.task_id,
                    log_dir=tb_dir,
                    skip_post_train=using_vec,
                    callback=episode_counter,
                    ent_coef=task.ent_coef,
                )
            else:
                base = task.train_timesteps
                max_total = int(base * task.max_budget_multiplier)
                if task.min_success_to_advance is not None:
                    chunk = int(task.gate_check_interval)
                elif task.interim_eval_every > 0:
                    chunk = int(task.interim_eval_every)
                else:
                    chunk = base
                trained = 0
                last_interim_at = 0
                # Train in `chunk` increments; check both stopping conditions
                # after each. The chunk is the LCM between the two cadences
                # in spirit (we use the smaller of the two so neither feature
                # misses a checkpoint).
                while trained < max_total:
                    this_chunk = min(chunk, max_total - trained)
                    cl.train_on(
                        env=train_env,
                        total_timesteps=this_chunk,
                        task_id=task.task_id,
                        log_dir=tb_dir,
                        skip_post_train=True,   # we'll do post-train once at end
                        callback=episode_counter,
                        ent_coef=task.ent_coef,
                    )
                    trained += this_chunk

                    # Interim eval (independent of the gate).
                    do_interim = (
                        task.interim_eval_every > 0
                        and trained - last_interim_at >= task.interim_eval_every
                    )
                    if do_interim or task.min_success_to_advance is not None:
                        # Run a small eval on this phase's task. Re-use the
                        # env_factory (single-env) so the gate-check is cheap
                        # and doesn't touch the SubprocVecEnv / MjxVecEnv that
                        # PPO is training on.
                        gate_env = task.env_factory(self.mission.seed
                                                    + phase * 1000
                                                    + 7777 + trained)
                        gate_n = (task.gate_eval_episodes
                                  if task.min_success_to_advance is not None
                                  else task.interim_eval_episodes)
                        gate_stats, _ = evaluate_with_trajectories(
                            policy_predict_fn=(lambda o: cl.predict(o, deterministic=True)[0]),
                            env=gate_env,
                            n_episodes=gate_n,
                            max_steps=task.eval_max_steps,
                            seed_base=trained,
                        )
                        try:
                            gate_env.close()
                        except Exception:
                            pass
                        interim_history.append({
                            "steps_trained_in_phase": int(trained),
                            "success_rate": float(gate_stats.success_rate),
                            "mean_return": float(gate_stats.mean_return),
                            "n_episodes": int(gate_n),
                        })
                        last_interim_at = trained
                        self._log(
                            f"  [interim @ {trained:,}/{max_total:,}] "
                            f"success_rate={gate_stats.success_rate:.2f} "
                            f"mean_return={gate_stats.mean_return:+.2f}"
                        )
                        # Gate check.
                        if (task.min_success_to_advance is not None
                                and gate_stats.success_rate >= task.min_success_to_advance
                                and trained >= base):
                            self._log(
                                f"  gate satisfied ({gate_stats.success_rate:.2f} "
                                f">= {task.min_success_to_advance:.2f}) — "
                                f"advancing early at {trained:,} of "
                                f"max {max_total:,} steps"
                            )
                            break

            train_seconds = time.perf_counter() - train_start
            n_train_episodes = episode_counter.n_episodes
            mean_ep_len = episode_counter.mean_episode_length
            self._log(
                f"  training done in {self._fmt_dur(train_seconds)} · "
                f"episodes={n_train_episodes} "
                f"(mean len={mean_ep_len:.0f} steps)"
            )
            # CL-method diagnostics: surface the penalty / rehearsal scale so
            # a mis-calibrated lam (penalty dominating or a no-op) is visible
            # in the log instead of silently corrupting the comparison.
            cl_diag: dict[str, float] = {}
            for attr in ("last_penalty_value", "last_penalty_steps_run",
                         "last_rehearsal_steps_run", "last_distill_kl"):
                val = getattr(cl, attr, None)
                if val is not None and val != 0:
                    cl_diag[attr] = float(val)
            if cl_diag:
                self._log("  cl diagnostics: "
                          + " ".join(f"{k}={v:.4g}" for k, v in cl_diag.items()))

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
                # Episode-level training stats, surfaced into results.json so
                # "how many agent runs happened in each phase" is recoverable
                # without re-parsing logs.
                "train_episodes": int(n_train_episodes),
                "train_mean_episode_steps": round(mean_ep_len, 1),
                # Interim-eval history (mid-phase checkpoints). Empty list
                # when the phase ran in single-call mode without interim
                # eval / adaptive gating. Each entry has steps_trained,
                # success_rate, mean_return, n_episodes.
                "interim_eval": interim_history,
                # CL-method scale diagnostics from the last train_on call of
                # this phase (penalty value, hook/rehearsal step counts).
                "cl_diagnostics": cl_diag,
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
