from pathlib import Path

import pytest

from arx_sim.config import available_task_layouts, load_sim_config
from arx_sim.inprocess_app import _apply_task_initial_joint_positions


class _FakeRobot:
    dof_names = ["joint4", "right_joint21"]

    def __init__(self) -> None:
        self.positions = None
        self.targets = None

    def set_dof_positions(self, values, *, dof_indices) -> None:
        self.positions = (values, dof_indices)

    def set_dof_position_targets(self, values, *, dof_indices) -> None:
        self.targets = (values, dof_indices)


def test_grasp_task_is_selectable_and_resolved(tmp_path: Path) -> None:
    config = load_sim_config(task="grasp", run_dir=tmp_path / "run")

    assert "grasp" in available_task_layouts()
    assert config["task_layout"]["name"] == "grasp"
    assert config["simulation"]["robot_pose"]["position"] == [
        4.3,
        2.85,
        0.01,
    ]
    assert config["task_layout"]["initial_joint_positions"] == {"joint4": 0.339}
    assert Path(config["simulation"]["task_overlay"]).is_file()
    assert Path(config["simulation"]["task_overlay"]).name == "task.usda"
    assert config["objects"]["red_cube"]["position"] == [4.12, 1.93, 0.715]
    assert config["objects"]["plate"]["position"] == [4.25, 2.03, 0.686]


def test_mapping_task_is_selectable_and_uses_scene_overrides(tmp_path: Path) -> None:
    config = load_sim_config(task="mapping", run_dir=tmp_path / "run")

    assert "mapping" in available_task_layouts()
    assert config["task_layout"]["name"] == "mapping"
    assert Path(config["simulation"]["environment"]).name == "environment_mapping.usda"
    assert Path(config["simulation"]["environment"]).is_file()
    assert Path(config["simulation"]["task_overlay"]).is_file()
    assert config["task_layout"]["scene_policy"]["disabled_interior_doors"] == [
        "door_bexenl_0",
        "door_bexenl_1",
    ]
    assert len(config["objects"]) == 10
    assert config["objects"]["coffee_table_mug"]["model"] == "ehnmxj"


def test_omitting_task_preserves_base_scene(tmp_path: Path) -> None:
    config = load_sim_config(run_dir=tmp_path / "run")

    assert "task_layout" not in config
    assert "task_overlay" not in config["simulation"]
    assert config["simulation"].get("robot_pose", {}).get("position", [0.0, 0.0, 0.0]) == [
        0.0,
        0.0,
        0.0,
    ]


def test_unknown_task_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown task layout"):
        load_sim_config(task="missing", run_dir=tmp_path / "run")


def test_task_initial_joint_positions_are_applied() -> None:
    robot = _FakeRobot()

    _apply_task_initial_joint_positions(
        robot,
        {"task_layout": {"initial_joint_positions": {"joint4": 0.25}}},
    )

    assert robot.positions == ([0.25], [0])
    assert robot.targets == ([0.25], [0])
