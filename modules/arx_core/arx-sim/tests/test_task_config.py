from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from arx_sim.config import load_sim_config


class TaskConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
    def load(self, task: str | None = None) -> dict:
        return load_sim_config(
            task=task,
            env={"ARX_SIM_ASSET_ROOT": str(self.root)},
            run_dir=self.root / "run",
        )

    def test_grasp_task_loads_overlay_and_world_pose(self) -> None:
        config = self.load("grasp")
        self.assertEqual(config["task_layout"]["name"], "grasp")
        self.assertEqual(Path(config["simulation"]["task_overlay"]).name, "task.usda")
        self.assertEqual(
            config["simulation"]["robot_pose"]["position"],
            [4.3, 2.85, 0.01],
        )
        self.assertEqual(config["objects"]["plate"]["position"], [4.25, 2.03, 0.686])

    def test_default_run_does_not_load_task(self) -> None:
        config = self.load()
        self.assertNotIn("task", config)
        self.assertNotIn("task_overlay", config["simulation"])

    def test_unlocked_task_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown task layout"):
            self.load("other_task")


if __name__ == "__main__":
    unittest.main()
