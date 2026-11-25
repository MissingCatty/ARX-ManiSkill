"""Resource isolation and output comparison faults for the offline benchmark."""
from pathlib import Path
import os
import tempfile
import unittest

import numpy as np

from arx_project.layout import write_json
from modules.rayst3r.optimized.benchmark import dependency_ready
from modules.rayst3r.optimized.worker import compare_points, verify_records


class IsolationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def terminal(self):
        write_json(self.root/'progress.json', {'phase': 'complete'})
        write_json(self.root/'services_stopped.json', {'all_owned_exited': True})
        write_json(self.root/'publication_complete.json', {'audit_passed': True})

    def test_active_campaign_wins_even_with_stale_exit_and_publication_receipts(self):
        self.terminal()
        write_json(self.root/'progress.json', {'phase': 'running'})
        self.assertFalse(dependency_ready(self.root)[0])

    def test_all_components_and_final_publication_must_be_finished(self):
        write_json(self.root/'progress.json', {'phase': 'complete'})
        self.assertFalse(dependency_ready(self.root)[0])
        write_json(self.root/'services_stopped.json', {'all_owned_exited': False})
        self.assertFalse(dependency_ready(self.root)[0])
        write_json(self.root/'services_stopped.json', {'all_owned_exited': True})
        self.assertFalse(dependency_ready(self.root)[0])
        write_json(self.root/'publication_complete.json', {'audit_passed': True})
        self.assertTrue(dependency_ready(self.root)[0])

    def test_live_worker_pid_and_failed_integrity_audit_block_execution(self):
        self.terminal()
        write_json(self.root/'resident/rayst3r/rayst3r_ready.json', {'pid': os.getpid()})
        self.assertFalse(dependency_ready(self.root)[0])
        write_json(self.root/'publication_complete.json', {'audit_passed': False})
        with self.assertRaisesRegex(RuntimeError, 'record-integrity'):
            dependency_ready(self.root)

    def test_changed_point_count_nonfinite_and_coordinate_errors_fail(self):
        points = np.zeros((3, 3), dtype=np.float32)
        self.assertTrue(compare_points(points, points.copy(), 1e-6)['passed'])
        self.assertFalse(compare_points(points, points[:2], 1e-6)['passed'])
        changed = points.copy(); changed[0, 0] = .001
        self.assertFalse(compare_points(points, changed, 1e-6)['passed'])
        changed[0, 0] = np.nan
        self.assertFalse(compare_points(points, changed, 1e-6)['passed'])

    def test_independent_record_check_detects_deleted_and_corrupt_outputs(self):
        reg = {'cases': [{'id': 'C001'}], 'settings': {'variants': ['baseline', 'combined'],
               'repetitions': 1, 'coordinate_atol_m': 1e-6}}
        directory = self.root/'trials/C001'
        for name in reg['settings']['variants']:
            path = directory/name; path.mkdir(parents=True)
            np.save(path/'points.npy', np.zeros((2, 3), dtype=np.float32))
        write_json(directory/'result.json', {'comparisons': [{'passed': True}]*2,
            'traces': [{'variant': 'baseline'}, {'variant': 'combined'}],
            'timings_seconds': {'baseline': [1.], 'combined': [.5]}})
        self.assertTrue(verify_records(self.root, reg)['passed'])
        np.save(directory/'combined/points.npy', np.ones((2, 3), dtype=np.float32))
        self.assertFalse(verify_records(self.root, reg)['passed'])
        (directory/'combined/points.npy').unlink()
        self.assertFalse(verify_records(self.root, reg)['passed'])


if __name__ == '__main__':
    unittest.main()
