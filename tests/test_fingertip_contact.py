"""Fault tests for the current fingertip-only table contact contract."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from modules.curobo.fingertip_contact import ContactProfile
from modules.curobo.fingertip_contact.physical_contact import classify_contact, ReleaseEnvelope
from modules.curobo.fingertip_contact.planning import contact_pool
from modules.maniskill.planning_common.contact_execution import execution_class, verify_trial


class ContactContractTests(unittest.TestCase):
    def setUp(self):
        self.profile = ContactProfile()
        self.eef = np.eye(4)
        self.eef[:3, :3] = np.diag([1., -1., -1.])
        self.eef[2, 3] = .044
        self.args = {'finger': 'right_link27', 'is_table': True, 'stage': 'close',
            'points': [{'world': [0., 0., -.001], 'normal': [0., 0., 1.],
                        'separation_m': -.001, 'impulse_ns': .01}],
            'eef': self.eef, 'table': {'center': [0., 0., -.25], 'dims': [.2, .2, .5]}, 'tip_plane': .045}

    def check(self, **changes):
        return classify_contact(self.profile, **{**deepcopy(self.args), **changes})

    def test_one_mm_top_contact_is_scoped_and_zero_profile_rejects(self):
        self.assertTrue(self.check()['allowed'])
        self.assertFalse(classify_contact(replace(self.profile, maximum_penetration_m=0.), **self.args)['allowed'])
        self.assertFalse(classify_contact(replace(self.profile, enabled=False), **self.args)['allowed'])

    def test_other_links_objects_and_stages_remain_strict(self):
        for change in [{'finger': 'right_link26'}, {'finger': 'left_link27'}, {'is_table': False},
                       *[{'stage': s} for s in ['observe', 'pregrasp', 'hold', 'unknown']]]:
            with self.subTest(change=change):
                self.assertFalse(self.check(**change)['allowed'])

    def test_body_side_and_excess_depth_are_rejected(self):
        eef = self.eef.copy(); eef[2, 3] = .030
        self.assertFalse(self.check(eef=eef)['allowed'])
        for key, value in [('normal', [1., 0., 0.]), ('separation_m', -.00201),
                           ('world', [.1001, 0., -.001]), ('normal', [0., 0., 100.])]:
            points = deepcopy(self.args['points']); points[0][key] = value
            self.assertFalse(self.check(points=points)['allowed'])

    def test_nonfinite_missing_and_invalid_transforms_cannot_pass(self):
        self.assertFalse(self.check(points=[])['allowed'])
        for key in ['impulse_ns', 'separation_m']:
            points = deepcopy(self.args['points']); points[0][key] = float('nan')
            self.assertFalse(self.check(points=points)['allowed'])
        eef = self.eef.copy(); eef[0, 0] = 2.
        self.assertFalse(self.check(eef=eef)['allowed'])
        eef = self.eef.copy(); eef[3, 0] = 1.
        self.assertFalse(self.check(eef=eef)['allowed'])

    def test_release_cannot_create_contact_or_accumulate_noise(self):
        release = ReleaseEnvelope(self.profile)
        self.assertFalse(release.accept(.001))
        release.anchor(.001)
        self.assertTrue(release.accept(.0008))
        self.assertFalse(release.accept(.00101))
        self.assertTrue(release.accept(.00085))
        self.assertEqual(release.ceiling, .0008)
        self.assertFalse(release.accept(.00091))
        self.assertTrue(release.accept(0.))
        self.assertFalse(release.accept(.00011))

    def test_profile_rejects_widened_geometry_and_solver_budgets(self):
        for args in [{'maximum_penetration_m': .003}, {'distal_length_m': .011},
                     {'release_depth_noise_m': .001}, {'maximum_penetration_m': float('nan')},
                     {'allowed_stages': ['reach', 'closure']}, {'ik_fallback_seeds': 128},
                     {'collision_profile_id': 'old_model'}, {'enabled': 'yes'}]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                replace(self.profile, **args)

    def test_release_requires_valid_fresh_measured_closure(self):
        class Base:
            pass
        cls = contact_pool(Base, self.profile)
        obj = object.__new__(cls)
        state = {'qpos': [0.]*6}
        with self.assertRaises(ValueError):obj.release_anchor(state)
        state['fingertip_table_contact'] = {'profile_id': self.profile.profile_id,
            'stage': 'close', 'valid': True, 'qpos': [0.]*6, 'depth_m': .001}
        self.assertEqual(obj.release_anchor(state), .001)
        for key, value in [('valid', False), ('stage', 'approach'), ('qpos', [1.]*6), ('depth_m', .003)]:
            wrong = deepcopy(state); wrong['fingertip_table_contact'][key] = value
            with self.assertRaises(ValueError):obj.release_anchor(wrong)

    def test_start_override_never_applies_to_initial_reach_or_observation(self):
        class Base:
            def validate_start_state(self, *args):return False
        cls = contact_pool(Base, self.profile)
        obj = object.__new__(cls)
        obj.contact_stage = 'reach'
        obj.contact_view = lambda *args: self.fail('Local contact override used in initial reach')
        self.assertFalse(obj.validate_start_state(None, {}, np.zeros(6)))
        obj.contact_stage = 'final_approach'
        self.assertFalse(obj.validate_start_state(None, {}, np.zeros(6), include_target=True))
        work = tempfile.TemporaryDirectory(); self.addCleanup(work.cleanup)
        context = SimpleNamespace(out=Path(work.name))
        obj.contact_view = lambda *args: SimpleNamespace(validate=lambda *a, **k: np.array([True]), receipts=[{'states': 1}])
        self.assertTrue(obj.validate_start_state(context, {}, np.zeros(6)))

    def test_non_table_checker_and_bounds_cannot_be_overridden(self):
        import torch
        from unittest.mock import Mock
        from modules.curobo.fingertip_contact.validator import LocalContactValidator
        obj = object.__new__(LocalContactValidator)
        obj.context = SimpleNamespace(limits=torch.tensor([[-1., 1.]]*6))
        obj.context.checker = SimpleNamespace(validate=lambda q: torch.zeros(len(q), dtype=torch.bool))
        obj.other_checker = SimpleNamespace(validate=lambda q: torch.zeros(len(q), dtype=torch.bool))
        obj.policy = SimpleNamespace(evaluate_one=Mock(return_value={'valid': True}))
        obj.receipts = []
        self.assertFalse(obj.validate(np.zeros((1, 6)), stage='closure').any())
        obj.policy.evaluate_one.assert_not_called()
        obj.other_checker.validate = lambda q: torch.ones(len(q), dtype=torch.bool)
        self.assertFalse(obj.validate(np.full((1, 6), 2.), stage='closure').any())
        obj.policy.evaluate_one.assert_not_called()
        with self.assertRaises(ValueError):obj.validate(np.full((1, 6), np.nan), stage='closure')
        self.assertTrue(obj.validate(np.zeros((1, 6)), stage='closure').all())
        self.assertEqual(obj.policy.evaluate_one.call_count, 1)

    def test_table_contacts_never_supply_bilateral_grasp_evidence(self):
        work = tempfile.TemporaryDirectory(); self.addCleanup(work.cleanup)
        root = Path(work.name)
        def write(name, data): (root/name).write_text(json.dumps(data))
        write('result.json', {'success': False, 'stages': {}})
        write('execution_origin.json', {'object_initial_z_m': 0.})
        write('contact_entities.json', {'profile_id': self.profile.profile_id,
            'target': 'target', 'table': 'table', 'fingers': {'finger': 'right_link27'},
            'tip_planes': {'right_link27': .045}, 'table_geometry': self.args['table']})
        event = {'a': 'finger', 'b': 'table', 'allowed': True, 'original_allowed': False,
                 'points': self.args['points']}
        row = {'stage': 'close', 'eef_world': self.eef.tolist(), 'contacts': [event], 'object_pose': [0., 0., 0.]}
        (root/'execution.jsonl').write_text(json.dumps(row)+'\n')
        verify_trial(root, self.profile)
        evaluation = json.loads((root/'independent_evaluation.json').read_text())
        self.assertEqual(evaluation['bilateral'], [])
        self.assertEqual(evaluation['allowed_tip_table_events'], 1)
        self.assertFalse(evaluation['success_v2'])
        row['contacts'][0]['points'][0]['separation_m'] = -.003
        (root/'execution.jsonl').write_text(json.dumps(row)+'\n')
        with self.assertRaises(RuntimeError):verify_trial(root, self.profile)


if __name__ == '__main__':
    unittest.main()
