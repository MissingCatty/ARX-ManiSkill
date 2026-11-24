"""Check that diagnostic observations cannot credit pose or collision failures."""
import types
import unittest

import torch

from probe import capture


class Costs:
    def __init__(self, feasible):
        self.feasible = feasible
        self.constraints = types.SimpleNamespace(
            names=["cspace", "scene_collision"],
            values=[torch.zeros(32, 1), (~feasible).float().reshape(32, 1)])

    def get_feasible(self, **kwargs):
        return self.feasible


class NativeGateTests(unittest.TestCase):
    def fixture(self):
        position = torch.full((32, 1), .001, dtype=torch.float32)
        rotation = torch.full((32, 1), .01, dtype=torch.float32)
        feasible = torch.ones(32, dtype=torch.bool)
        metrics = types.SimpleNamespace(convergence=types.SimpleNamespace(
            names=["tool_pose_position_tolerance", "tool_pose_orientation_tolerance"],
            values=[position, rotation]), costs_and_constraints=Costs(feasible))
        config = types.SimpleNamespace(position_tolerance=.002, orientation_tolerance=.03,
                                       success_requires_convergence=True)
        result = types.SimpleNamespace(debug_info={"seed_idx": torch.tensor([[0, 1, 2, 3]])},
                                       success=torch.ones(1, 4, dtype=torch.bool))
        return torch.zeros(32, 6, dtype=torch.float32), metrics, result, config

    def test_reports_all_seeds_not_only_returned_four(self):
        output = capture(*self.fixture())
        self.assertEqual(len(output["pose_converged"]), 32)
        self.assertEqual(len(output["returned_success"]), 4)

    def test_pose_failure_is_not_success_when_collision_free(self):
        opt, metrics, result, config = self.fixture()
        metrics.convergence.values[0][0, 0] = .02
        result.success[0, 0] = False
        output = capture(opt, metrics, result, config)
        self.assertFalse(output["native_full_success"][0])
        self.assertTrue(output["feasible"][0])

    def test_scene_collision_is_not_success_when_pose_converges(self):
        opt, metrics, result, config = self.fixture()
        metrics.costs_and_constraints.feasible[0] = False
        metrics.costs_and_constraints.constraints.values[1][0, 0] = 1.
        result.success[0, 0] = False
        output = capture(opt, metrics, result, config)
        self.assertTrue(output["pose_converged"][0])
        self.assertFalse(output["native_full_success"][0])

    def test_inconsistent_native_result_stops_the_observer(self):
        opt, metrics, result, config = self.fixture()
        metrics.convergence.values[0][0, 0] = .02
        with self.assertRaisesRegex(AssertionError, "Native gate mismatch"):
            capture(opt, metrics, result, config)


if __name__ == "__main__":
    unittest.main()
