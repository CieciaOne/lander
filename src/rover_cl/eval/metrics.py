"""Evaluation metrics for continual learning experiments.

Provides:
- ``evaluate_policy`` for rolling out a deterministic policy on a Gymnasium env
  and collecting success / return / steps statistics.
- Continual-learning retention helpers operating on the Runner JSON schema:
  ``compute_retention_matrix``, ``compute_forgetting``, ``compute_avg_retention``.
- ``load_results`` for reading a Runner JSON file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class EpisodeTrajectory:
    """Per-step record of one evaluation episode, used for top-down reports."""

    positions: np.ndarray            # (T, 2) world-frame (x, y) per step
    yaws: np.ndarray                 # (T,) world-frame yaw (rad) per step
    contact_positions: np.ndarray    # (n_contact, 2) sampled at collision steps
    waypoint_hit_steps: list[int]    # step index at which wp_idx advanced
    success: bool
    tipped: bool
    truncated: bool
    steps: int
    final_distance_to_goal: float
    cumulative_reward: float
    # Per-episode obstacle layout (only meaningful for randomized terrains,
    # where it differs from reset to reset). Each row = (cx, cy, sx, sy).
    # Empty array when no obstacles or env didn't expose them.
    obstacle_layout: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 4), dtype=np.float32)
    )
    # Optional per-episode start / waypoints / goal snapshot. Lets the plot
    # show the actual rolled goal location instead of the template value.
    start_pos: tuple[float, float] | None = None
    goal_pos: tuple[float, float] | None = None
    waypoints: tuple[tuple[float, float], ...] = ()

    @property
    def outcome(self) -> str:
        if self.success:
            return "success"
        if self.tipped:
            return "tipped"
        return "timeout"


def rollout_with_trajectory(
    env: Any,
    policy_predict_fn: Callable[[Any], Any],
    max_steps: int,
    seed: int | None = None,
) -> EpisodeTrajectory:
    """Roll out one deterministic episode and record positions / contacts.

    Reads `pos_xy`, `yaw`, `collision`, `waypoint_index`, `is_success`,
    `tipped`, `distance_to_goal` from the env's info dict (see
    ``RoverNavEnv.step``). Truncation is whichever of `truncated=True` or the
    `max_steps` cap fires first. If `seed` is provided, it's passed to
    `env.reset(seed=...)` so callers can sweep eval seeds to get path diversity.
    """
    reset_out = env.reset(seed=seed) if seed is not None else env.reset()
    obs, _ = reset_out if isinstance(reset_out, tuple) else (reset_out, {})

    # Snapshot the obstacle layout right after reset — for randomized
    # terrains the model's geom_pos / geom_size have just been re-rolled
    # by `_apply_terrain_roll`, and they're the ground truth for what this
    # episode's rover actually faces. Filtering on `cz > -10` skips hidden
    # slots (positioned at HIDE_Z = -50 below the floor).
    unwrapped = getattr(env, "unwrapped", env)
    layout_rows: list[tuple[float, float, float, float]] = []
    start_pos = None
    goal_pos = None
    waypoints: tuple[tuple[float, float], ...] = ()
    model = getattr(unwrapped, "_model", None)
    terrain = getattr(unwrapped, "terrain", None)
    if model is not None and terrain is not None:
        try:
            import mujoco as _mj
            # Read data.geom_xpos so we work for both static-geom obstacles
            # (T1_blocked_arc) and mocap-body obstacles (RT_* randomized,
            # where model.geom_pos is body-local zero — the world position
            # lives in data.mocap_pos AND is propagated to data.geom_xpos).
            data = getattr(unwrapped, "_data", None)
            if data is None:
                raise RuntimeError("env has no _data")
            for i in range(len(terrain.obstacles)):
                gid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_GEOM, f"obs_{i}")
                if gid < 0:
                    continue
                cx, cy, cz = (float(v) for v in data.geom_xpos[gid])
                sx, sy, _sz = (float(v) for v in model.geom_size[gid])
                if cz > -10.0:  # not hidden
                    layout_rows.append((cx, cy, sx, sy))
            start_pos = tuple(terrain.start_pos)
            goal_pos = tuple(terrain.goal_pos)
            waypoints = tuple(terrain.waypoints)
        except Exception:
            pass

    positions: list[tuple[float, float]] = []
    yaws: list[float] = []
    contacts: list[tuple[float, float]] = []
    wp_hits: list[int] = []
    cum_return = 0.0
    success = False
    tipped = False
    truncated_flag = False
    last_dist_to_goal = float("nan")
    last_wp_idx = 0

    for step in range(1, max_steps + 1):
        action = policy_predict_fn(obs)
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated) or bool(truncated)
        else:  # pragma: no cover - legacy gym
            obs, reward, done, info = step_out  # type: ignore[misc]
            terminated, truncated = done, False
        cum_return += float(reward)

        pos = info.get("pos_xy")
        if pos is not None:
            positions.append((float(pos[0]), float(pos[1])))
            yaws.append(float(info.get("yaw", 0.0)))
            if info.get("collision"):
                contacts.append((float(pos[0]), float(pos[1])))
        if info.get("waypoint_index", last_wp_idx) != last_wp_idx:
            last_wp_idx = int(info["waypoint_index"])
            wp_hits.append(step)
        if "distance_to_goal" in info:
            last_dist_to_goal = float(info["distance_to_goal"])
        if info.get("is_success"):
            success = True
        if info.get("tipped"):
            tipped = True
        if bool(truncated):
            truncated_flag = True
        if done:
            break

    return EpisodeTrajectory(
        positions=np.asarray(positions, dtype=np.float32) if positions
        else np.zeros((0, 2), dtype=np.float32),
        yaws=np.asarray(yaws, dtype=np.float32) if yaws
        else np.zeros(0, dtype=np.float32),
        contact_positions=np.asarray(contacts, dtype=np.float32) if contacts
        else np.zeros((0, 2), dtype=np.float32),
        waypoint_hit_steps=wp_hits,
        success=success,
        tipped=tipped,
        truncated=truncated_flag and not success,
        steps=len(positions),
        final_distance_to_goal=last_dist_to_goal,
        cumulative_reward=cum_return,
        obstacle_layout=np.asarray(layout_rows, dtype=np.float32) if layout_rows
        else np.zeros((0, 4), dtype=np.float32),
        start_pos=start_pos,
        goal_pos=goal_pos,
        waypoints=waypoints,
    )


def evaluate_with_trajectories(
    policy_predict_fn: Callable[[Any], Any],
    env: Any,
    n_episodes: int,
    max_steps: int,
    seed_base: int | None = None,
) -> tuple["EpisodeStats", list[EpisodeTrajectory]]:
    """Evaluate + record trajectories. Aggregated stats match `evaluate_policy`.

    If `seed_base` is given, episode `i` resets the env with seed
    `seed_base + i` — diversifies trajectories across the 10 rollouts that
    would otherwise be identical with a deterministic policy + deterministic
    env. The seed is consumed by `RoverNavEnv.reset(seed=...)`, which routes
    through Gymnasium's superclass and re-randomizes the terrain RNG.
    """
    trajectories = [
        rollout_with_trajectory(
            env, policy_predict_fn, max_steps,
            seed=(seed_base + i) if seed_base is not None else None,
        )
        for i in range(n_episodes)
    ]
    successes = [t.success for t in trajectories]
    returns = [t.cumulative_reward for t in trajectories]
    steps_to_success = [t.steps for t in trajectories if t.success]
    stats = EpisodeStats(
        success_rate=float(np.mean(successes)) if successes else 0.0,
        mean_return=float(np.mean(returns)) if returns else 0.0,
        mean_steps_to_goal=(
            float(np.mean(steps_to_success)) if steps_to_success else None
        ),
        n_episodes=n_episodes,
    )
    return stats, trajectories


@dataclass
class EpisodeStats:
    """Aggregate statistics over a batch of evaluation episodes."""

    success_rate: float
    mean_return: float
    mean_steps_to_goal: float | None
    n_episodes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "success_rate": float(self.success_rate),
            "mean_return": float(self.mean_return),
            "mean_steps_to_goal": (
                None
                if self.mean_steps_to_goal is None
                else float(self.mean_steps_to_goal)
            ),
            "n_episodes": int(self.n_episodes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeStats":
        return cls(
            success_rate=float(data["success_rate"]),
            mean_return=float(data["mean_return"]),
            mean_steps_to_goal=(
                None
                if data.get("mean_steps_to_goal") is None
                else float(data["mean_steps_to_goal"])
            ),
            n_episodes=int(data["n_episodes"]),
        )


def evaluate_policy(
    policy_predict_fn: Callable[[Any], Any],
    env: Any,
    n_episodes: int = 20,
    max_steps: int = 500,
    success_info_key: str = "is_success",
) -> EpisodeStats:
    """Roll out a deterministic policy and aggregate statistics.

    Parameters
    ----------
    policy_predict_fn:
        Callable mapping an observation to an action.
    env:
        A Gymnasium-style env with ``reset()`` and ``step()``.
    n_episodes:
        Number of episodes to roll out.
    max_steps:
        Hard cap on environment steps per episode.
    success_info_key:
        Key in the ``info`` dict that reports success per step. If absent in
        every step of an episode, we fall back to ``final_reward > 0``.
    """
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")

    successes: list[bool] = []
    returns: list[float] = []
    steps_to_success: list[int] = []

    for _ in range(n_episodes):
        reset_out = env.reset()
        # Gymnasium ``reset`` returns ``(obs, info)``.
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, _info = reset_out
        else:  # pragma: no cover - legacy gym compatibility
            obs = reset_out

        episode_return = 0.0
        episode_success = False
        success_step: int | None = None
        last_reward = 0.0
        saw_success_key = False
        step = 0

        for step in range(1, max_steps + 1):
            action = policy_predict_fn(obs)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated) or bool(truncated)
            else:  # pragma: no cover - legacy gym compatibility
                obs, reward, done, info = step_out  # type: ignore[misc]

            episode_return += float(reward)
            last_reward = float(reward)

            if isinstance(info, dict) and success_info_key in info:
                saw_success_key = True
                if bool(info[success_info_key]) and not episode_success:
                    episode_success = True
                    success_step = step

            if done:
                break

        # Fallback when the env never reports an explicit success flag.
        if not saw_success_key and last_reward > 0 and step > 0:
            episode_success = True
            success_step = step

        successes.append(episode_success)
        returns.append(episode_return)
        if episode_success and success_step is not None:
            steps_to_success.append(success_step)

    success_rate = float(np.mean(successes)) if successes else 0.0
    mean_return = float(np.mean(returns)) if returns else 0.0
    mean_steps = (
        float(np.mean(steps_to_success)) if steps_to_success else None
    )

    return EpisodeStats(
        success_rate=success_rate,
        mean_return=mean_return,
        mean_steps_to_goal=mean_steps,
        n_episodes=n_episodes,
    )


def compute_retention_matrix(results: dict[str, Any]) -> np.ndarray:
    """Build the N x N retention matrix R[i, j] from a Runner results dict.

    R[i, j] is the ``success_rate`` on task j evaluated right after training
    phase i. Cells where the task hadn't been seen yet (or no evaluation was
    recorded) are NaN.
    """
    task_ids: list[str] = list(results["task_ids"])
    n = len(task_ids)
    task_index = {tid: idx for idx, tid in enumerate(task_ids)}

    matrix = np.full((n, n), np.nan, dtype=float)
    for entry in results.get("evaluations", []):
        phase = int(entry["phase"])
        if not (0 <= phase < n):
            continue
        per_task = entry.get("per_task", {}) or {}
        for tid, stats in per_task.items():
            if stats is None or tid not in task_index:
                continue
            j = task_index[tid]
            matrix[phase, j] = float(stats["success_rate"])
    return matrix


def compute_forgetting(retention: np.ndarray) -> np.ndarray:
    """Per-task forgetting: max retention seen so far minus the final value.

    For each task j, ``forgetting[j] = max_k R[k, j] - R[last, j]``, where the
    max is over phases where R[k, j] is observed (not NaN), and ``last`` is the
    last phase (final row of R). NaN final entries propagate as NaN.
    """
    if retention.ndim != 2 or retention.shape[0] != retention.shape[1]:
        raise ValueError("retention must be a square 2D matrix")

    n = retention.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)

    last_row = retention[-1, :]
    forgetting = np.full(n, np.nan, dtype=float)
    for j in range(n):
        col = retention[:, j]
        observed = col[~np.isnan(col)]
        if observed.size == 0 or np.isnan(last_row[j]):
            continue
        forgetting[j] = float(observed.max() - last_row[j])
    return forgetting


def compute_avg_retention(retention: np.ndarray) -> float:
    """Mean success_rate of the last evaluation phase (NaN-safe)."""
    if retention.size == 0:
        return float("nan")
    last_row = retention[-1, :]
    if np.all(np.isnan(last_row)):
        return float("nan")
    return float(np.nanmean(last_row))


def load_results(path: Path) -> dict[str, Any]:
    """Load a Runner results JSON file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_retention_matrices(
    results_list: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack retention matrices across seeds; return (mean, std).

    Each input dict is the parsed ``results.json`` from one seed. The matrices
    must all have identical shape and ``task_ids``. Raises ``ValueError`` if
    not.

    Uses ``np.nanmean`` / ``np.nanstd`` (ddof=0). Cells where every seed is NaN
    propagate as NaN in the aggregated mean and std (we silence the all-NaN
    warning so callers don't see RuntimeWarnings for the upper-triangular
    region).
    """
    import warnings

    if not results_list:
        raise ValueError("results_list must contain at least one results dict")

    first = results_list[0]
    ref_task_ids = list(first["task_ids"])
    ref_matrix = compute_retention_matrix(first)
    ref_shape = ref_matrix.shape

    stacked: list[np.ndarray] = [ref_matrix]
    for idx, res in enumerate(results_list[1:], start=1):
        task_ids = list(res["task_ids"])
        if task_ids != ref_task_ids:
            raise ValueError(
                f"task_ids mismatch at seed index {idx}: "
                f"{task_ids} vs {ref_task_ids} (shape would differ)"
            )
        mat = compute_retention_matrix(res)
        if mat.shape != ref_shape:
            raise ValueError(
                f"retention matrix shape mismatch at seed index {idx}: "
                f"{mat.shape} vs {ref_shape}"
            )
        stacked.append(mat)

    arr = np.stack(stacked, axis=0)  # (n_seeds, N, N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0, ddof=0)
    return mean, std


def collect_seed_results(method_dir: Path) -> list[dict]:
    """Load all ``seed_*/results.json`` under a method directory, in seed order.

    The seed number is parsed from the directory name (``seed_<N>``). Entries
    without a parseable seed are skipped.
    """
    method_dir = Path(method_dir)
    if not method_dir.is_dir():
        return []

    entries: list[tuple[int, Path]] = []
    for child in method_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("seed_"):
            continue
        try:
            seed_n = int(name[len("seed_"):])
        except ValueError:
            continue
        results_path = child / "results.json"
        if results_path.exists():
            entries.append((seed_n, results_path))

    entries.sort(key=lambda kv: kv[0])
    return [load_results(p) for _, p in entries]
