import unittest
from types import SimpleNamespace

from modules.curobo.dual_arm.profiling import SolverTrace


class TimingTests(unittest.TestCase):
    def test_values_and_single_call_are_preserved(self):
        obj = SimpleNamespace(solve=lambda value: value+1)
        trace = SolverTrace(lambda: None)
        trace.wrap(obj, 'solve', 'ik')
        trace.wrap(obj, 'solve', 'ik')
        self.assertEqual(obj.solve(3), 4)
        self.assertEqual(trace.report()['components']['ik']['calls'], 1)

    def test_exception_propagates_without_fake_result(self):
        def fail():
            raise RuntimeError('CUDA failure')
        obj = SimpleNamespace(solve=fail)
        trace = SolverTrace(lambda: None)
        trace.wrap(obj, 'solve', 'ik')
        with self.assertRaises(RuntimeError):
            obj.solve()
        self.assertTrue(trace.records[0]['exception'])

    def test_reset_keeps_wrapper_and_clears_measurements(self):
        obj = SimpleNamespace(solve=lambda: 7)
        trace = SolverTrace(lambda: None)
        trace.wrap(obj, 'solve', 'ik')
        obj.solve(); trace.reset()
        self.assertEqual(trace.records, [])
        self.assertEqual(obj.solve(), 7)
        self.assertEqual(len(trace.records), 1)
