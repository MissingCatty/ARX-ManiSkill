"""Offline checks against the archived ARX robot, never a substitute robot."""
import copy
from pathlib import Path
import time
import unittest
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT, read, digest
from .advisor38_kinematics import CPUArm, ScreenRejected
from .advisor38_prefilter import require_seven
from .advisor36_contract import boundary_poses


class ARXPrefilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = ROOT / 'runs/nearfield_advisor37_20261112_01/seed_preflight_02'
        cls.registration = read(root / 'registration.json')
        cls.request = next(read(p) for p in sorted((root / 'workers/planner/requests').glob('*.done'))
                           if 'cheez_it_center' in str(read(p).get('scene', '')) and read(p).get('label') == 'pregrasp')
        cls.candidate = read(cls.registration['canonical_candidate_file'])[0]['selected']

    def arm(self):
        return CPUArm(Path(self.registration['model_reference']) / 'model/arx.urdf', self.request['snapshot'])

    def test_CPU_chain_agrees_with_recorded_ARX_pose(self):
        arm = self.arm()
        np.testing.assert_allclose(arm.fk(arm.start), self.request['snapshot']['link_world']['right_eef_link'], atol=2e-6)

    def test_actual_ARX_pose_has_in_limit_CPU_solution(self):
        arm = self.arm()
        goal = arm.fk(arm.start)
        q = arm.solve(goal, arm.start + .002, time.monotonic() + 10.)
        self.assertTrue(np.all(q >= arm.limits[:, 0]) and np.all(q <= arm.limits[:, 1]))
        self.assertLess(np.linalg.norm(arm.fk(q)[:3, 3] - goal[:3, 3]), .0005)

    def test_expired_CPU_budget_cannot_pass(self):
        arm = self.arm()
        with self.assertRaises(ScreenRejected):
            arm.solve(arm.fk(arm.start), arm.start, time.monotonic() - 1.)

    def certificate(self):
        return {'pass': True, 'physical_commands': 0, 'candidate_sha256': digest(self.candidate),
                'cases': [{'case': c['case'], 'pass': True,
                           'insertion': {'dense_samples': 601, 'invalid_samples': 0},
                           'lift': {'dense_samples': 1201, 'invalid_samples': 0}}
                          for c in boundary_poses(self.request['goal_world'], 'coupled_xyz')]}

    def test_complete_seven_path_certificate_accepts(self):
        require_seven(self.certificate(), self.candidate)

    def test_missing_lift_evidence_rejects(self):
        cert = self.certificate(); del cert['cases'][2]['lift']
        with self.assertRaises(ValueError):
            require_seven(cert, self.candidate)

    def test_collision_sample_rejects(self):
        cert = self.certificate(); cert['cases'][3]['insertion']['invalid_samples'] = 1
        with self.assertRaises(ValueError):
            require_seven(cert, self.candidate)

    def test_recorded_camera_mount_matches_authored_ARX_calibration(self):
        import yaml
        from .advisor38_static_capture import authored_mount
        root = ROOT / 'runs/nearfield_advisor38_20261112_01/static_objects/cheez_it'
        fault = read(root / 'groups/pilot/FAULT.json')['static_diagnostics']
        config = yaml.safe_load((ROOT / 'arx-sim/config/controllers.yaml').read_text())['controllers']['camera']
        mount = authored_mount(Path(self.registration['model_reference']) / 'model/arx.urdf',
                               fault['after'], config['definitions']['right_wrist']['xform_rows'],
                               config['roles']['right_wrist'].rsplit('/', 1)[0])
        self.assertFalse(np.allclose(fault['actual_mount'], fault['frozen_mount'], atol=1e-5, rtol=0))
        np.testing.assert_allclose(fault['actual_mount'], mount, atol=1e-5, rtol=0)
        corrupted = mount.copy(); corrupted[0, 3] += 2e-5
        self.assertFalse(np.allclose(fault['actual_mount'], corrupted, atol=1e-5, rtol=0))

    def test_prior_fifteen_candidates_exhaust_budget(self):
        from .advisor38_orchestrator import select_candidates
        top = ROOT / 'runs/nearfield_advisor38_20261112_01'
        spec = read(top / 'prefilter_registration.json')
        obj = next(iter(spec['objects']))
        history = [{'object_id': obj, 'processed': 7} for _ in range(15)]
        chosen, exhausted, blocked = select_candidates(top, {'objects': {}, 'candidate_rounds': history},
                {'objects': {obj: spec['objects'][obj]}}, [], False)
        self.assertEqual(chosen, [])
        self.assertEqual(exhausted, [obj])
        self.assertEqual(blocked, [obj])

    def test_unresolved_physical_case_stops_next_candidate(self):
        from .advisor38_orchestrator import select_candidates
        top = ROOT / 'runs/nearfield_advisor38_20261112_01'
        spec = read(top / 'prefilter_registration.json')
        obj = next(iter(spec['objects']))
        with self.assertRaises(RuntimeError):
            select_candidates(top, {'objects': {}, 'candidate_rounds': [{'object_id': obj, 'processed': 6}]},
                              spec, [], False)

    def test_released_object_cannot_be_retested(self):
        from .advisor38_orchestrator import select_candidates
        top = ROOT / 'runs/nearfield_advisor38_20261112_01'
        spec = read(top / 'prefilter_registration.json')
        obj = next(iter(spec['objects']))
        chosen, exhausted, blocked = select_candidates(top, {'objects': {obj: {}}, 'candidate_rounds': []},
                {'objects': {obj: spec['objects'][obj]}}, [{'object_id': obj, 'pass': True}], False)
        self.assertEqual(chosen, [])
        self.assertEqual(exhausted, [])
        self.assertEqual(blocked, [obj])

    def test_missing_boundary_rejects_physical_registration(self):
        cert = self.certificate(); cert['cases'].pop()
        with self.assertRaises(ValueError):
            require_seven(cert, self.candidate)

    def test_duplicate_boundary_rejects_physical_registration(self):
        cert = self.certificate(); cert['cases'][-1] = cert['cases'][0]
        with self.assertRaises(ValueError):
            require_seven(cert, self.candidate)

    def test_changed_candidate_cannot_reuse_screen(self):
        cert = self.certificate(); candidate = copy.deepcopy(self.candidate)
        candidate['Gcanonical_object'][0][3] += .005
        with self.assertRaises(ValueError):
            require_seven(cert, candidate)

    def test_failed_path_cannot_hide_under_top_level_pass(self):
        cert = self.certificate(); cert['cases'][4]['pass'] = False
        with self.assertRaises(ValueError):
            require_seven(cert, self.candidate)


if __name__ == '__main__':
    unittest.main()
