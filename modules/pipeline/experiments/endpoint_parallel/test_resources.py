import unittest
from .resources import idle_devices, foreign_processes, parse_processes


class ResourceTests(unittest.TestCase):
    def test_unavailable_pid_only_blocks_its_own_gpu(self):
        processes = parse_processes('GPU-a, [N/A]\nGPU-b, 12\n')
        cards = [{'uuid': n, 'free_mib': 24000, 'utilization_percent': 0} for n in ['GPU-a', 'GPU-b', 'GPU-c']]
        self.assertEqual([g['uuid'] for g in idle_devices(cards, processes)], ['GPU-c'])
        self.assertEqual(foreign_processes('GPU-a', 42, processes), [None])

    def test_idle_context_is_still_excluded_from_timing_device(self):
        cards = [{'uuid': 'a', 'free_mib': 23500, 'utilization_percent': 0}]
        self.assertEqual(idle_devices(cards, [{'uuid': 'a', 'pid': 1}]), [])

    def test_unknown_or_busy_utilization_is_not_admitted(self):
        for util in [None, 1, 100]:
            self.assertEqual(idle_devices([{'uuid': 'a', 'free_mib': 24000, 'utilization_percent': util}], []), [])

    def test_known_failed_device_is_skipped(self):
        cards = [{'uuid': n, 'free_mib': 24000, 'utilization_percent': 0} for n in ['a', 'b']]
        self.assertEqual([g['uuid'] for g in idle_devices(cards, [], ['a'])], ['b'])

    def test_during_execution_only_owned_worker_is_allowed(self):
        processes = [{'uuid': 'a', 'pid': 1}, {'uuid': 'a', 'pid': 2}, {'uuid': 'b', 'pid': 3}]
        self.assertEqual(foreign_processes('a', 1, processes), [2])


if __name__ == '__main__': unittest.main()
