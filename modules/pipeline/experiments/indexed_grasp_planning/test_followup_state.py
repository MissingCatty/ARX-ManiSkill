"""Exercise the last-request/exit race without launching a controller."""
import unittest

from .followup_state import completed_controller


class FollowupCompletionTests(unittest.TestCase):
    fields = {'MainPID': '0', 'ActiveState': 'inactive', 'ExecMainStatus': '0'}
    complete = {'phase': 'complete', 'completed_requests': 138, 'total_requests': 138}

    def test_stale_137_snapshot_cannot_override_fresh_138_receipt(self):
        stale = {'phase': 'planning', 'completed_requests': 137, 'total_requests': 138}
        reads = []
        def latest():
            reads.append(True)
            return dict(self.complete)
        result = completed_controller(self.fields, latest, 138)
        self.assertEqual(stale['completed_requests'], 137)
        self.assertEqual(result, self.complete)
        self.assertEqual(reads, [True])

    def test_running_controller_is_not_finished(self):
        for state in ['active', 'activating', 'reloading', 'deactivating']:
            fields = {**self.fields, 'ActiveState': state}
            self.assertIsNone(completed_controller(fields, lambda: self.fail('Do not finalize a live controller'), 138))

    def test_nonzero_exit_cannot_pass_with_complete_progress(self):
        with self.assertRaises(RuntimeError):
            completed_controller({**self.fields, 'ExecMainStatus': '1'}, lambda: self.complete, 138)

    def test_incomplete_progress_stops(self):
        with self.assertRaises(RuntimeError):
            completed_controller(self.fields, lambda: {**self.complete, 'phase': 'planning'}, 138)

    def test_missing_last_request_stops(self):
        with self.assertRaises(RuntimeError):
            completed_controller(self.fields, lambda: {**self.complete, 'completed_requests': 137}, 138)

    def test_wrong_quota_or_noninteger_count_stops(self):
        for progress in [{**self.complete, 'total_requests': 137},
                         {**self.complete, 'completed_requests': 138.0}]:
            with self.assertRaises(RuntimeError): completed_controller(self.fields, lambda: progress, 138)

    def test_failed_service_is_not_a_normal_exit(self):
        with self.assertRaises(RuntimeError):
            completed_controller({**self.fields, 'ActiveState': 'failed'}, lambda: self.complete, 138)

    def test_incomplete_status_stops(self):
        with self.assertRaises(RuntimeError): completed_controller({}, lambda: self.complete, 138)


if __name__ == '__main__': unittest.main()
