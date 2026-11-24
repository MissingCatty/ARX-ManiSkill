import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from arx_sim.runtime.simulation_loop import run_simulation_loop


class RealtimeLoopTests(unittest.TestCase):
    def run_clock(self, frame_cost):
        clock = [0.0]
        world = SimpleNamespace(current_time=0.0)

        def step(**kwargs):
            self.assertEqual(kwargs, {'render': True})
            clock[0] += frame_cost
            world.current_time += 0.2

        world.step = step
        stop = Mock()
        stop.is_set.return_value = False
        stop.wait.side_effect = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
        queue = Mock()
        queue.get_nowait.return_value = None
        context = SimpleNamespace(
            config={'simulation': {'realtime': True, 'physics_dt': 1/120}},
            world=world, simulation_app=Mock(), command_queue=queue,
            controller=Mock(), cameras={}, ros2_publishers=None,
        )
        with patch('arx_sim.runtime.simulation_loop.time.monotonic', side_effect=lambda: clock[0]):
            self.assertEqual(run_simulation_loop(context, stop, max_steps=50), 50)
        return clock[0], world.current_time, stop

    def test_fast_renderer_is_paced_without_changing_physics_time(self):
        wall, sim, stop = self.run_clock(0.05)
        self.assertAlmostEqual(sim, 10.0)
        self.assertLess(abs(sim-wall), 0.2)
        self.assertGreater(stop.wait.call_count, 0)

    def test_overloaded_renderer_does_not_fake_simulation_time(self):
        wall, sim, stop = self.run_clock(0.3)
        self.assertAlmostEqual(sim, 10.0)
        self.assertAlmostEqual(wall, 15.0)
        stop.wait.assert_not_called()
