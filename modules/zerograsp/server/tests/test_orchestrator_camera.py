from pathlib import Path
import unittest
from unittest.mock import patch

from arx_shared.schemas import GraspPlan, GraspPrediction, RunGraspRequest, ZeroGraspInputResult
from coolas_zerograsp import orchestrator
from coolas_zerograsp.orchestrator import _camera_id_for_arm, _capture_rgbd


def test_camera_id_for_arm_uses_wrist_cameras() -> None:
    assert _camera_id_for_arm("right") == "right_hand"
    assert _camera_id_for_arm("left") == "left_hand"
    assert _camera_id_for_arm("") == "right_hand"


def test_capture_explicitly_sends_wrist_camera_id(tmp_path: Path) -> None:
    response = {
        "color_path": str(tmp_path / "color.png"),
        "depth_path": str(tmp_path / "depth.png"),
        "params_path": str(tmp_path / "params.json"),
    }
    with patch(
        "coolas_zerograsp.orchestrator._post_json", return_value=response
    ) as post_json:
        _capture_rgbd(
            {"arx_client_url": "http://127.0.0.1:8001"},
            tmp_path,
            camera_id="right_hand",
        )

    post_json.assert_called_once_with(
        "http://127.0.0.1:8001/camera/capture-rgbd",
        {"output_dir": str(tmp_path), "camera_id": "right_hand"},
        timeout=120.0,
    )


class IntermediateWaypointTest(unittest.TestCase):
    def test_intermediate_waypoint_opens_gripper(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            reuse_dir = tmp_path / "reuse_camera"
            reuse_dir.mkdir()
            mask_path = reuse_dir / "mask.png"
            mask_path.write_bytes(b"mask")
            post_calls = []

            def fake_load_yaml(name):
                if name == "services.yaml":
                    return {"arx_client_url": "http://arx", "langsam_url": "http://langsam"}
                if name == "zerograsp.yaml":
                    return {
                        "task": {
                            "depth_fusion_enabled": False,
                            "intermediate_waypoint_enabled": True,
                            "intermediate_waypoint_height_offset_m": 0.10,
                            "intermediate_waypoint_target_ratio": 0.5,
                            "intermediate_waypoint_timeout_sec": 20.0,
                            "final_speed": 1.3,
                            "close_delay": 0.5,
                            "motion_timeout_sec": 120.0,
                        }
                    }
                return {}

            def fake_post_json(url, payload, timeout=300.0):
                post_calls.append((url, payload, timeout))
                return {"ok": True, "message": "ok", "command": [], "data": {}}

            with (
                patch.object(orchestrator, "load_yaml", side_effect=fake_load_yaml),
                patch.object(orchestrator, "_get_json", return_value={"end_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}),
                patch.object(
                    orchestrator,
                    "prepare_zerograsp_input",
                    return_value=ZeroGraspInputResult(
                        input_dir=str(tmp_path / "zg_input"),
                        rgb_path=str(reuse_dir / "color.png"),
                        depth_path=str(reuse_dir / "depth.png"),
                        mask_path=str(mask_path),
                        camera_info_path=str(reuse_dir / "params.json"),
                    ),
                ),
                patch.object(orchestrator, "predict_grasp", return_value=GraspPrediction(grasp_path="grasp.json", output_dir=str(tmp_path / "zg_output"))),
                patch.object(orchestrator, "_apply_target_x_filter", return_value={"score": 1.0}),
                patch.object(
                    orchestrator,
                    "plan_grasp",
                    return_value=GraspPlan(
                        target_pose=[0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                        target_gripper=-3.2,
                        safety_passed=True,
                    ),
                ),
                patch.object(orchestrator, "_post_json", side_effect=fake_post_json),
            ):
                result = orchestrator.run_grasp_task(RunGraspRequest(
                    prompt="red cube.",
                    execute=True,
                    skip_projection=True,
                    output_dir=str(tmp_path / "run"),
                    reuse_camera_dir=str(reuse_dir),
                    reuse_mask_path=str(mask_path),
                ))

            self.assertFalse(result.intermediate_waypoint["hold_gripper"])
            self.assertEqual(result.intermediate_waypoint["target_gripper"], -3.2)
            self.assertEqual(result.intermediate_waypoint["timeout_sec"], 20.0)
            self.assertEqual(len(post_calls), 2)
            waypoint_url, waypoint_payload, waypoint_http_timeout = post_calls[0]
            self.assertEqual(waypoint_url, "http://arx/arm/move-end-pose")
            self.assertFalse(waypoint_payload["hold_gripper"])
            self.assertEqual(waypoint_payload["target_gripper"], -3.2)
            self.assertEqual(waypoint_http_timeout, 30.0)
            self.assertEqual(post_calls[1][1]["target_gripper"], -3.2)
