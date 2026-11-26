"""Prevent duplicate execution and predeclare cross-restart timing treatment."""
import unittest

from .serialization_resume import schedule, timing_rows


class ResumeTests(unittest.TestCase):
    cases = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
    variants = ['control_1', 'fast_binding_1']

    def test_only_unexecuted_slots_are_scheduled(self):
        claims = {'a_control_1': {'state': 'complete'}, 'a_fast_binding_1': {'state': 'complete'},
                  'b_control_1': {'state': 'complete'}, 'b_fast_binding_1': {'state': 'claimed'}}
        old, new = schedule(self.cases, self.variants, claims, ['b_fast_binding_1'])
        self.assertEqual(len(old), 3)
        self.assertEqual([e['key'] for e in new], ['b_fast_binding_1', 'c_control_1', 'c_fast_binding_1'])
        self.assertFalse({e['key'] for e in old} & {e['key'] for e in new})

    def test_claimed_request_requires_explicit_authorization(self):
        with self.assertRaises(ValueError):
            schedule(self.cases, self.variants, {'a_control_1': {'state': 'claimed'}}, [])

    def test_completed_request_cannot_be_replayed(self):
        with self.assertRaises(ValueError):
            schedule(self.cases, self.variants, {'a_control_1': {'state': 'complete'}}, ['a_control_1'])

    def test_unknown_or_duplicate_cohort_claims_stop(self):
        with self.assertRaises(ValueError):
            schedule(self.cases, self.variants, {'other_control_1': {'state': 'complete'}}, [])
        with self.assertRaises(ValueError): schedule([{'id': 'a'}, {'id': 'a'}], self.variants, {}, [])

    def test_unused_authorization_is_rejected(self):
        with self.assertRaises(ValueError): schedule(self.cases, self.variants, {}, ['a_control_1'])

    def test_timing_exclusion_does_not_remove_planning_results(self):
        rows = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
        reg = {'continuation': {'latency_excluded_case_ids': ['b']}}
        self.assertEqual(timing_rows(rows, reg), [{'id': 'a'}, {'id': 'c'}])
        self.assertEqual(len(rows), 3)
        self.assertEqual(timing_rows(rows, {}), rows)


if __name__ == '__main__': unittest.main()
