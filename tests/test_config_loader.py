"""Tests for the YAML config loader (no PPO training, all <1s)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rover_cl.configs import load_mission_config
from rover_cl.missions import Mission, Task

pytestmark = pytest.mark.fast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"

STARTER_CONFIGS = [
    ("scenario_01_naive.yaml", "naive"),
    ("scenario_01_replay.yaml", "replay"),
    ("scenario_01_ewc.yaml", "ewc"),
]


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(body)
    return p


@pytest.mark.parametrize("filename,expected_method", STARTER_CONFIGS)
def test_loader_builds_mission_from_each_starter_config(
    filename: str, expected_method: str
) -> None:
    mission = load_mission_config(CONFIGS_DIR / filename)
    assert isinstance(mission, Mission)
    assert mission.cl_method == expected_method
    # mission.name == "{registry_key}_{cl_method}" — canonicalized in
    # get_scenario so paths under results/<scenario>/<method>/seed_<N>/ match
    # the name passed to --compare.
    assert mission.name == f"scenario_01_sequential_terrains_{expected_method}"
    assert len(mission.tasks) == 2
    for t in mission.tasks:
        assert isinstance(t, Task)


def test_cl_kwargs_propagate(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path,
        "scenario: scenario_01_sequential_terrains\n"
        "cl_method: ewc\n"
        "cl_kwargs:\n"
        "  lam: 42.0\n",
    )
    mission = load_mission_config(cfg)
    assert mission.cl_kwargs == {"lam": 42.0}


def test_missing_required_field_raises(tmp_path: Path) -> None:
    cfg_no_scenario = tmp_path / "no_scenario.yaml"
    cfg_no_scenario.write_text("cl_method: naive\n")
    with pytest.raises(ValueError, match="scenario"):
        load_mission_config(cfg_no_scenario)

    cfg_no_method = tmp_path / "no_method.yaml"
    cfg_no_method.write_text("scenario: scenario_01_sequential_terrains\n")
    with pytest.raises(ValueError, match="cl_method"):
        load_mission_config(cfg_no_method)


def test_unknown_scenario_raises(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path,
        "scenario: not_a_real_scenario\ncl_method: naive\n",
    )
    with pytest.raises(KeyError) as exc:
        load_mission_config(cfg)
    # KeyError str() is repr-wrapped; check the message lists a real scenario.
    assert "scenario_01_sequential_terrains" in str(exc.value)


def test_unknown_cl_method_raises(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path,
        "scenario: scenario_01_sequential_terrains\ncl_method: not_a_real_method\n",
    )
    with pytest.raises(ValueError):
        load_mission_config(cfg)


def test_seed_defaults_to_zero(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path,
        "scenario: scenario_01_sequential_terrains\ncl_method: naive\n",
    )
    mission = load_mission_config(cfg)
    assert mission.seed == 0


def test_scenario_kwargs_passed_through(tmp_path: Path) -> None:
    cfg = _write_yaml(
        tmp_path,
        "scenario: scenario_01_sequential_terrains\n"
        "cl_method: naive\n"
        "scenario_kwargs:\n"
        "  train_timesteps: 12345\n",
    )
    mission = load_mission_config(cfg)
    assert mission.tasks[0].train_timesteps == 12345
