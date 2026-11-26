"""Reject lost validation records and distinguish identity from success."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from arx_project.layout import sha256
from .validation_launch import monitor_policy
from .validation_replay import compare_pair, summarize, request_order, latency_gates
from .validation_review import check_validation_receipt


class ValidationRecordTests(unittest.TestCase):
    def test_exact_pilot_and_full_schedules(self):
        pilot = ['control_1', 'vectorized_1', 'vectorized_2', 'control_2']
        self.assertEqual(request_order(pilot, 0), pilot)
        self.assertEqual(request_order(pilot, 1), ['vectorized_1', 'control_1', 'control_2', 'vectorized_2'])
        full = ['control_1', 'vectorized_1']
        self.assertEqual(request_order(full, 0), full)
        self.assertEqual(request_order(full, 1), list(reversed(full)))
        self.assertEqual(sum(len(request_order(full, i)) for i in range(69)), 138)

    def test_partial_or_unknown_schedule_stops(self):
        for variants in [['control_1'], ['control_1', 'other'], ['control_1', 'vectorized_1', 'control_2']]:
            with self.assertRaises(ValueError): request_order(variants, 0)

    def test_named_optimization_keeps_balanced_orders(self):
        variants = ['control_1', 'fast_binding_1', 'fast_binding_2', 'control_2']
        self.assertEqual(request_order(variants, 0), variants)
        self.assertEqual(request_order(variants, 1), ['fast_binding_1', 'control_1', 'control_2', 'fast_binding_2'])

    def test_rpc_metric_is_selected_explicitly(self):
        modes = {'control_1': 'control', 'fast_binding_1': 'fast_binding'}
        variants = {name: {'success': True, 'planning_request_seconds': 10., 'planning_rpc_seconds': 12.,
                          'candidate_checks': 1, 'stage_summary': {}} for name in modes}
        result = summarize([{'variants': variants}], modes, 'planning_rpc_seconds')
        self.assertEqual(result['fast_binding']['median_seconds'], 12.)

    def test_total_saving_does_not_override_median_or_p90_gate(self):
        summary = {'control': {'total_seconds': 100., 'median_seconds': 10., 'p90_seconds': 20.},
                   'fast_binding': {'total_seconds': 90., 'median_seconds': 11., 'p90_seconds': 22.}}
        reg = {'minimum_total_time_reduction_for_full_comparison': .05,
               'require_median_reduction': True, 'maximum_p90_increase_fraction': .05}
        saving, gates = latency_gates(summary, 'fast_binding', reg)
        self.assertTrue(gates['total_time_reduction']);self.assertFalse(all(gates.values()))
        summary['fast_binding'].update(median_seconds=9., p90_seconds=20.)
        self.assertTrue(all(latency_gates(summary, 'fast_binding', reg)[1].values()))

    def records(self, root):
        names = {'metrics_sha256': 'metrics.json', 'plan_sha256': 'plan_result.json',
            'order_sha256': 'candidate_order.json', 'timing_sha256': 'stage_timings.json',
            'validation_metrics_sha256': 'validation_metrics.json'}
        for name in names.values(): (root/name).write_text('{}')
        (root/'candidate_snapshot_commit.json').write_text('{"persisted":true}')
        arms = {}
        for side in ['left', 'right']:
            path = root/'arms'/side/'plan_result.json';path.parent.mkdir(parents=True);path.write_text('{}')
            arms[side] = sha256(path)
        return {'state': 'complete', **{k: sha256(root/v) for k, v in names.items()}, 'arm_results_sha256': arms}

    def test_complete_receipt_and_lost_timing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);claim = self.records(root)
            check_validation_receipt(root, claim)
            (root/'stage_timings.json').unlink()
            with self.assertRaises((ValueError, FileNotFoundError)): check_validation_receipt(root, claim)

    def test_lost_optimization_sidecar_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);claim = self.records(root)
            path = root/'binding_serialization.json';path.write_text('{}')
            claim['extra_receipts_sha256'] = {path.name: sha256(path)}
            check_validation_receipt(root, claim);path.unlink()
            with self.assertRaises(FileNotFoundError): check_validation_receipt(root, claim)

    def test_lost_other_arm_or_changed_arm_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);claim = self.records(root)
            missing = deepcopy(claim);del missing['arm_results_sha256']['left']
            with self.assertRaises(ValueError): check_validation_receipt(root, missing)
            (root/'arms/left/plan_result.json').write_text('{"success":false}')
            with self.assertRaises(ValueError): check_validation_receipt(root, claim)

    def test_incomplete_claim_is_not_a_grasp_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);claim = self.records(root);claim['state'] = 'claimed'
            with self.assertRaises(ValueError): check_validation_receipt(root, claim)

    def test_success_alone_does_not_pass_identity(self):
        base = {'success': True, 'selected_arm': 'right', 'selected_original_source_index': 1,
            'candidate_checks': 4, 'joint_endpoints': {'q_grasp': [0]*6}, 'event_decisions': {'right': []}}
        pair = {'control': base, 'vectorized': deepcopy(base)}
        self.assertTrue(all(compare_pair(pair).values()))
        pair['vectorized']['selected_original_source_index'] = 2
        self.assertTrue(compare_pair(pair)['control_success_retained'])
        self.assertFalse(compare_pair(pair)['same_original_selection'])
        pair['vectorized']['joint_endpoints']['q_grasp'][0] = .01
        self.assertFalse(compare_pair(pair)['same_joint_endpoints'])

    def test_resource_policy_keeps_original_thresholds_and_no_retry(self):
        from arx_project.layout import ROOT
        policy = monitor_policy(ROOT/'tmp/indexed_grasp_planning/example')
        self.assertEqual(policy['admission_plans_mib'], [{'planner': 10000}])
        self.assertEqual(policy['cuda_probe_before_dispatch']['minimum_free_mib'], 10000)
        self.assertEqual(policy['dispatch_marker_files'], ['dispatch_claim.json'])
        self.assertFalse(policy['automatic_experiment_retries'])

    def test_summaries_keep_repeated_requests_and_nested_times(self):
        modes = {'control_1': 'control', 'control_2': 'control', 'vectorized_1': 'vectorized', 'vectorized_2': 'vectorized'}
        variants = {name: {'success': True, 'planning_request_seconds': 2 if group == 'control' else 1,
            'candidate_checks': 3, 'stage_summary': {'bind': {'calls': 1, 'wall_seconds': .5, 'exclusive_seconds': .3}}}
                    for name, group in modes.items()}
        result = summarize([{'variants': variants}], modes)
        self.assertEqual(result['control']['requests'], 2)
        self.assertEqual(result['vectorized']['total_seconds'], 2)
        self.assertEqual(result['control']['stage_totals']['bind']['exclusive_seconds'], .6)


if __name__ == '__main__': unittest.main()
