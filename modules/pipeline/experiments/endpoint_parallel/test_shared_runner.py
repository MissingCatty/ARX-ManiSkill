import unittest
from .shared_runner import summarize_load


class LoadTests(unittest.TestCase):
    def test_empty_record_has_no_invented_measurement(self):
        report = summarize_load([])
        self.assertEqual(report['samples'], 0)
        self.assertNotIn('planning', report)

    def test_planning_is_separate_from_warmup(self):
        rows = [{'phase': phase, 'free_mib': free, 'GPU_utilization_percent': util, 'foreign_compute_contexts': foreign}
                for phase, free, util, foreign in [('registered', 15000, 90, 2), ('planning', 12000, 40, 3)]]
        report = summarize_load(rows)
        self.assertEqual(report['planning_samples'], 1)
        self.assertEqual(report['planning']['median_GPU_utilization_percent'], 40.)
        self.assertEqual(report['planning']['maximum_foreign_compute_contexts'], 3)

    def test_failed_query_is_not_zero_load(self):
        report = summarize_load([{'phase': 'planning', 'query_error': 'timeout'}])
        self.assertEqual(report['query_errors'], 1)
        self.assertNotIn('planning', report)


if __name__ == '__main__': unittest.main()
