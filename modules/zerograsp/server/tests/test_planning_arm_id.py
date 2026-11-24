from __future__ import annotations

import unittest
from unittest.mock import patch

from arx_shared.schemas import GraspPlanRequest
from coolas_zerograsp import planning


PREDICTION = {
    "score": 0.9,
    "width_m": 0.04,
    "depth_m": 0.0,
    "rotation_matrix_camera": [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    "translation_m_camera": [0.02, 0.0, 0.0],
}


class PlanningArmIdTest(unittest.TestCase):
    def test_left_arm_uses_left_calibration_keys(self):
        calibration = {
            "right_ee_camera": {
                "matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            },
            "left_ee_camera": {
                "matrix": [
                    [1.0, 0.0, 0.0, 0.10],
                    [0.0, 1.0, 0.0, 0.20],
                    [0.0, 0.0, 1.0, 0.30],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            },
            "right_tcp_gripper": {"xyz_m": [0.0, 0.0, 0.0]},
            "left_tcp_gripper": {"xyz_m": [0.0, 0.0, 0.0]},
        }

        def fake_load_yaml(name):
            if name == "arx.yaml":
                return {"motion": {"x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0], "z_range": [-1.0, 1.0]}}
            if name == "zerograsp.yaml":
                return {"task": {"fixed_open_gripper": True, "fixed_open_gripper_value": -3.2}}
            return {}

        with patch.object(planning, "load_calibration", return_value=calibration), patch.object(planning, "load_yaml", side_effect=fake_load_yaml):
            plan = planning.plan_grasp(GraspPlanRequest(
                arm_id="left",
                prediction=PREDICTION,
                current_end_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ))

        self.assertTrue(plan.safety_passed)
        self.assertEqual(plan.metadata["arm_id"], "left")
        self.assertEqual(plan.metadata["calibration_keys"]["ee_camera"], "left_ee_camera")
        self.assertEqual(plan.metadata["calibration_keys"]["tcp_gripper"], "left_tcp_gripper")
        self.assertEqual(plan.metadata["transforms"]["T_ee_camera"], calibration["left_ee_camera"]["matrix"])


if __name__ == "__main__":
    unittest.main()
