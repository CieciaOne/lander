"""Load a Mission from a YAML config file.

Schema::

    scenario: scenario_01_sequential_terrains  # name from SCENARIO_REGISTRY
    cl_method: ewc                              # name from cl._REGISTRY
    seed: 0                                     # single-seed run
    # OR (multi-seed sweep; mutually exclusive with `seed`):
    # seeds: [0, 1, 2]
    scenario_kwargs:                            # passed to the scenario factory
      train_timesteps: 30000
      eval_episodes: 10
      max_steps: 600
    cl_kwargs:                                  # passed to make_cl()
      lam: 1000.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rover_cl.cl import _REGISTRY as _CL_REGISTRY
from rover_cl.missions.base import Mission
from rover_cl.missions.scenarios import SCENARIO_REGISTRY, get_scenario


_REQUIRED = ("scenario", "cl_method")


def _build_mission(
    scenario_name: str,
    cl_method: str,
    seed: int,
    scenario_kwargs: dict[str, Any],
    cl_kwargs: dict[str, Any],
) -> Mission:
    # Validate CL method early — matches make_cl()'s ValueError contract so users
    # don't get a confusing error deep inside Runner.run().
    if cl_method.lower() not in _CL_REGISTRY:
        raise ValueError(
            f"Unknown CL method '{cl_method}'. Known: {sorted(_CL_REGISTRY)}"
        )
    # get_scenario raises KeyError listing known scenarios for an unknown name.
    mission = get_scenario(
        scenario_name,
        cl_method=cl_method,
        seed=seed,
        **scenario_kwargs,
    )
    mission.cl_kwargs = cl_kwargs
    return mission


def _parse_config(path: Path | str) -> tuple[str, str, list[int], dict, dict]:
    cfg_path = Path(path)
    with cfg_path.open("r") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config {cfg_path} must be a mapping, got {type(raw).__name__}")

    missing = [k for k in _REQUIRED if k not in raw or raw[k] is None]
    if missing:
        raise ValueError(
            f"Config {cfg_path} missing required field(s): {missing}"
        )

    scenario_name: str = raw["scenario"]
    cl_method: str = raw["cl_method"]
    scenario_kwargs: dict[str, Any] = dict(raw.get("scenario_kwargs") or {})
    cl_kwargs: dict[str, Any] = dict(raw.get("cl_kwargs") or {})

    seeds_field = raw.get("seeds")
    if seeds_field is not None:
        if not isinstance(seeds_field, (list, tuple)) or not seeds_field:
            raise ValueError(
                f"Config {cfg_path}: 'seeds' must be a non-empty list of ints"
            )
        seeds = [int(s) for s in seeds_field]
    else:
        seeds = [int(raw.get("seed", 0) or 0)]

    return scenario_name, cl_method, seeds, scenario_kwargs, cl_kwargs


def load_mission_config(path: Path | str) -> Mission:
    """Load a single Mission from YAML.

    If the YAML specifies a ``seeds`` list, only the FIRST seed is used here.
    Callers that want every seed should use :func:`load_missions_config`.
    """
    scenario_name, cl_method, seeds, scenario_kwargs, cl_kwargs = _parse_config(path)
    return _build_mission(
        scenario_name, cl_method, seeds[0], scenario_kwargs, cl_kwargs
    )


def load_missions_config(path: Path | str) -> list[Mission]:
    """Load one or more Missions from a YAML config.

    Returns one Mission per seed when ``seeds`` is set, otherwise a one-element
    list using the ``seed`` field (default 0).
    """
    scenario_name, cl_method, seeds, scenario_kwargs, cl_kwargs = _parse_config(path)
    return [
        _build_mission(scenario_name, cl_method, s, scenario_kwargs, cl_kwargs)
        for s in seeds
    ]


__all__ = [
    "load_mission_config",
    "load_missions_config",
    "SCENARIO_REGISTRY",
]
