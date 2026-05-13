"""Fast tests for the scenario registry.

These don't run any training — they only verify the registry is consistent
and that each scenario factory returns a `Mission` with the expected task
sequence + CL method. Mission integration with the Runner is covered by
`tests/test_scenarios.py` (slow).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rover_cl.missions import SCENARIO_REGISTRY, get_scenario  # noqa: E402
from rover_cl.missions.base import Mission, Task  # noqa: E402

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Concrete (runnable) scenarios
# ---------------------------------------------------------------------------

CONCRETE_SCENARIOS = [
    ("scenario_01_sequential_terrains", ["T1_flat", "T2_corridor"]),
    ("scenario_02_three_terrains", ["T1_flat", "T2_corridor", "T3_obstacle_field"]),
    ("scenario_05_full_terrain_curriculum",
     ["T1_flat", "T2_corridor", "T3_obstacle_field", "T4_dunes"]),
]


@pytest.mark.parametrize("name, expected_task_ids", CONCRETE_SCENARIOS)
def test_scenario_returns_mission_with_expected_tasks(name: str, expected_task_ids: list[str]) -> None:
    mission = get_scenario(name)
    assert isinstance(mission, Mission)
    assert [t.task_id for t in mission.tasks] == expected_task_ids
    assert all(isinstance(t, Task) for t in mission.tasks)


def test_scenario_03_order_sensitivity_both_directions() -> None:
    easy = get_scenario("scenario_03_order_sensitivity", direction="easy_to_hard")
    hard = get_scenario("scenario_03_order_sensitivity", direction="hard_to_easy")
    assert [t.task_id for t in easy.tasks] == ["T1_flat", "T2_corridor", "T3_obstacle_field"]
    assert [t.task_id for t in hard.tasks] == ["T3_obstacle_field", "T2_corridor", "T1_flat"]
    # default cl_method for scenario_03 is "ewc"
    assert easy.cl_method == "ewc"
    assert hard.cl_method == "ewc"


def test_scenario_03_rejects_bad_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        get_scenario("scenario_03_order_sensitivity", direction="sideways")


def test_scenario_04_replay_sweep_passes_buffer_size() -> None:
    m = get_scenario("scenario_04_replay_sweep", buffer_size=5000)
    assert m.cl_method == "replay"
    assert m.cl_kwargs == {"buffer_size_per_task": 5000}
    # mission.name is canonicalized to "{registry_key}_{cl_method}"; the
    # buffer_size is recorded in cl_kwargs, not in the name.
    assert m.name == "scenario_04_replay_sweep_replay"


def test_scenario_04_with_non_replay_method_clears_cl_kwargs() -> None:
    # Sanity: if user picks a non-replay method, buffer_size shouldn't leak in.
    m = get_scenario("scenario_04_replay_sweep", buffer_size=999, cl_method="naive")
    assert m.cl_method == "naive"
    assert m.cl_kwargs == {}


def test_cl_method_propagates_to_mission_name_and_field() -> None:
    for cl in ("naive", "replay", "ewc"):
        m = get_scenario("scenario_01_sequential_terrains", cl_method=cl)
        assert m.cl_method == cl
        assert cl in m.name


# ---------------------------------------------------------------------------
# Stubs (NotImplementedError on call) — registered but not yet built.
# ---------------------------------------------------------------------------

STUB_SCENARIOS = ["scenario_02_threat_classes", "scenario_06_fusion"]


@pytest.mark.parametrize("name", STUB_SCENARIOS)
def test_stub_scenario_raises_with_helpful_message(name: str) -> None:
    with pytest.raises(NotImplementedError, match=name):
        get_scenario(name)


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------

def test_unknown_scenario_lists_known_options() -> None:
    with pytest.raises(KeyError, match="scenario_01"):
        get_scenario("not_a_scenario")


def test_every_registry_entry_is_callable() -> None:
    for name, factory in SCENARIO_REGISTRY.items():
        assert callable(factory), f"{name} is not callable"


EXPECTED_REGISTRY_KEYS = {
    "scenario_01_sequential_terrains",
    "scenario_02_three_terrains",
    "scenario_03_order_sensitivity",
    "scenario_04_replay_sweep",
    "scenario_05_full_terrain_curriculum",
    "scenario_07_blocked_arc",
    "scenario_08_blocked_arc_hills",
    "scenario_09_curriculum_arc",
    "scenario_10_robust_curriculum",
    "scenario_02_threat_classes",
    "scenario_06_fusion",
}


def test_registry_contains_all_expected_scenarios() -> None:
    missing = EXPECTED_REGISTRY_KEYS - set(SCENARIO_REGISTRY)
    assert not missing, f"missing from SCENARIO_REGISTRY: {sorted(missing)}"
