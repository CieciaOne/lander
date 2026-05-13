"""Evaluation utilities for the rover continual-learning project."""

from rover_cl.eval.metrics import (
    EpisodeStats,
    EpisodeTrajectory,
    aggregate_retention_matrices,
    collect_seed_results,
    compute_avg_retention,
    compute_forgetting,
    compute_retention_matrix,
    evaluate_policy,
    evaluate_with_trajectories,
    load_results,
    rollout_with_trajectory,
)

__all__ = [
    "EpisodeStats",
    "EpisodeTrajectory",
    "aggregate_retention_matrices",
    "collect_seed_results",
    "compute_avg_retention",
    "compute_forgetting",
    "compute_retention_matrix",
    "evaluate_policy",
    "evaluate_with_trajectories",
    "load_results",
    "rollout_with_trajectory",
]
