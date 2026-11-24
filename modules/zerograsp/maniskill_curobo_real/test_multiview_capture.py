"""Tests for the reusable multi-view RGB-D capture geometry."""

from __future__ import annotations

import unittest

import numpy as np

from maniskill_curobo_real.capture_multiview_rgbd import (
    default_multiview_specs,
    default_three_view_specs,
)


class MultiviewCaptureTest(unittest.TestCase):
    def test_default_three_views_keep_existing_camera_and_close_aux_views(self) -> None:
        main_eye = (-0.20, 0.0, 0.27)
        target = (0.05, 0.0, 0.08)
        views = default_three_view_specs(
            main_eye=main_eye,
            main_target=target,
            side_yaw_deg=30.0,
            side_distance=0.25,
            side_height_above_target=0.19,
        )

        self.assertEqual(
            [view.name for view in views],
            [
                "view_0_existing_rgbd",
                "view_1_close_left",
                "view_2_close_right",
            ],
        )
        self.assertEqual(views[0].eye, main_eye)
        self.assertEqual(views[0].target, target)
        self.assertEqual(views[0].role, "existing_baseline_camera")

        target_xy = np.asarray(target[:2], dtype=np.float64)
        for view in views[1:]:
            eye = np.asarray(view.eye, dtype=np.float64)
            self.assertAlmostEqual(float(np.linalg.norm(eye[:2] - target_xy)), 0.25)
            self.assertAlmostEqual(float(eye[2]), target[2] + 0.19)
            self.assertEqual(view.target, target)
        self.assertAlmostEqual(views[1].eye[0], views[2].eye[0])
        self.assertAlmostEqual(views[1].eye[1], -views[2].eye[1])

    def test_four_views_add_close_opposite_camera(self) -> None:
        target = (0.05, 0.0, 0.08)
        views = default_multiview_specs(
            main_eye=(-0.20, 0.0, 0.27),
            main_target=target,
            view_count=4,
        )

        self.assertEqual(len(views), 4)
        self.assertEqual(views[-1].name, "view_3_close_opposite")
        self.assertEqual(views[-1].role, "close_auxiliary_opposite")
        self.assertGreater(views[-1].eye[0], target[0])
        self.assertAlmostEqual(views[-1].eye[1], target[1])

    def test_five_views_keep_four_views_and_add_high_front_camera(self) -> None:
        target = (0.05, 0.0, 0.08)
        views = default_multiview_specs(
            main_eye=(-0.20, 0.0, 0.27),
            main_target=target,
            view_count=5,
            side_yaw_deg=32.0,
            side_distance=0.25,
        )

        self.assertEqual(len(views), 5)
        self.assertEqual(views[3].name, "view_3_close_opposite")
        self.assertEqual(views[4].name, "view_4_close_high_front")
        self.assertEqual(views[4].role, "close_auxiliary_high_front")
        self.assertGreater(views[4].eye[2], views[0].eye[2])
        self.assertEqual(
            views[:4],
            default_multiview_specs(
                main_eye=(-0.20, 0.0, 0.27),
                main_target=target,
                view_count=4,
                side_yaw_deg=32.0,
                side_distance=0.25,
            ),
        )

    def test_rejects_unsupported_view_count(self) -> None:
        with self.assertRaises(ValueError):
            default_multiview_specs(
                main_eye=(-0.20, 0.0, 0.27),
                main_target=(0.05, 0.0, 0.08),
                view_count=6,
            )


if __name__ == "__main__":
    unittest.main()
