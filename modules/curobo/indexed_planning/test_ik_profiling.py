"""Check diagnostic selection and kernel accounting without CUDA admission."""
import cProfile
from types import SimpleNamespace
import unittest

from .ik_profiling import IKTrace, cpu_rows, device_rows


class DiagnosticTests(unittest.TestCase):
    def test_device_events_do_not_double_count_cpu_parents(self):
        def event(kind, name, elapsed, annotation=False):
            return SimpleNamespace(device_type=SimpleNamespace(name=kind), name=name,
                                   is_user_annotation=annotation,
                                   time_range=SimpleNamespace(elapsed_us=lambda: elapsed))
        rows = device_rows([event('CPU', 'launch', 100), event('CUDA', 'kernel', 4),
                            event('CUDA', 'kernel', 6), event('CUDA', 'copy', 2),
                            event('CUDA', 'enclosing range', 100, True)])
        self.assertEqual(rows, [{'name': 'kernel', 'calls': 2, 'device_microseconds': 10.},
                                {'name': 'copy', 'calls': 1, 'device_microseconds': 2.}])

    def test_disabled_stage_calls_original_without_importing_torch(self):
        trace = IKTrace();obj = SimpleNamespace(call=lambda value: value+1)
        trace.wrap(obj, 'call', 'unused')
        self.assertEqual(obj.call(4), 5)
        self.assertFalse(trace.events)

    def test_original_exception_propagates(self):
        trace = IKTrace()
        def fail(): raise ValueError('source fault')
        obj = SimpleNamespace(call=fail);trace.wrap(obj, 'call', 'unused')
        with self.assertRaisesRegex(ValueError, 'source fault'): obj.call()

    def test_overlapping_requests_are_rejected_and_finish_resets(self):
        trace = IKTrace();trace.begin(None, True)
        with self.assertRaises(RuntimeError): trace.begin(None, True)
        report = trace.finish();self.assertFalse(trace.enabled)
        self.assertFalse(report['numerical_configuration_changed'])
        trace.begin(None, False);self.assertEqual(trace.finish()['samples'], [])

    def test_real_cpu_profile_has_named_call_counts(self):
        def sample(): return sum(range(5))
        profile = cProfile.Profile();profile.runcall(sample)
        row = next(r for r in cpu_rows(profile) if r['function'] == 'sample')
        self.assertEqual(row['calls'], 1)
        self.assertGreaterEqual(row['cumulative_seconds'], row['self_seconds'])


if __name__ == '__main__': unittest.main()
