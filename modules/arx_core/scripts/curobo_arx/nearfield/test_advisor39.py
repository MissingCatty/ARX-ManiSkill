"""Offline fault cases and real ARX object surface verification."""
import copy
import unittest
import tempfile
from pathlib import Path
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, digest
from scripts.curobo_arx.sdg_stage1.sampler import composed_mesh
from .advisor36_geometry import Surface
from .advisor39_geometry import triggered, dense_candidates
from .advisor38_geometry import distinct


class ReplenishmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        top = ROOT / 'runs/nearfield_advisor38_20261112_01'
        cls.obj = 'mustard_bottle'
        spec = read(top / 'prefilter_registration.json')['objects'][cls.obj]
        cls.pool = read(spec['candidate_pool'])['candidates']
        cls.item = next(r for r in read(top / 'source_inventory.json') if r['object_id'] == cls.obj)
        cls.original = read(spec['geometry'])

    def records(self):
        return [{'candidate_index': c['index'], 'pass': False} for c in self.pool]

    def test_sixty_failures_trigger(self):
        self.assertTrue(triggered(self.records(), self.pool, 5, False))

    def test_fifty_nine_terminal_do_not_trigger(self):
        self.assertFalse(triggered(self.records()[:-1], self.pool, 5, False))

    def test_existing_qualified_candidate_prevents_trigger(self):
        rows = self.records(); rows[17]['pass'] = True
        self.assertFalse(triggered(rows, self.pool, 5, False))

    def test_physical_cap_and_released_object_prevent_trigger(self):
        self.assertFalse(triggered(self.records(), self.pool, 15, False))
        self.assertFalse(triggered(self.records(), self.pool, 5, True))

    def test_duplicate_receipt_cannot_complete_sixty(self):
        rows = self.records(); rows[-1] = rows[-2]
        with self.assertRaises(ValueError):
            triggered(rows, self.pool, 5, False)

    def test_wrong_candidate_does_not_complete_registered_pool(self):
        rows = self.records(); rows[-1]['candidate_index'] = -1
        self.assertFalse(triggered(rows, self.pool, 5, False))

    def test_dense_real_mustard_contacts_and_old_pose_exclusion(self):
        mesh, _ = composed_mesh(self.item['asset']['usd_path'])
        prior = [np.array(c['Gcanonical_object']) for c in self.pool]
        rows = list(dense_candidates(self.item, self.original, mesh, prior, 1, cap=4))
        self.assertEqual(len(rows), 4)
        surface = Surface(mesh)
        for row in rows:
            goal = np.array(row['Gcanonical_object'])
            actual = surface.pair(np.array(row['pad_center_object']), goal[:3, 1])
            np.testing.assert_allclose(actual['angles'], row['contact_normal_errors_degrees'], atol=1e-8)
            self.assertLessEqual(max(actual['angles']), row['normal_limit_degrees'])
            self.assertTrue(all(distinct(goal, old) for old in prior))
            self.assertLess(row['section_grid_m'], .003)

    def test_expired_geometry_budget_produces_no_candidate(self):
        mesh, _ = composed_mesh(self.item['asset']['usd_path'])
        self.assertEqual(list(dense_candidates(self.item, self.original, mesh, [], 1, deadline=0)), [])

    def supplemental_fixture(self, top, missing_case=False, candidates=5):
        old = ROOT / 'runs/nearfield_advisor38_20261112_01'
        obj = 'household_hot3d_potato_masher'
        item = read(old / 'prefilter_registration.json')['objects'][obj]
        candidate = next(c for c in read(item['candidate_pool'])['candidates'] if c['index'] == 380153)
        proof = read(old / 'prefilter' / obj / '380153/result.json')
        if missing_case:
            proof['cases'].pop()
        write(top / 'candidate.json', candidate); write(top / 'result.json', proof)
        write(top / 'scope_registration.json', {'source_campaign': str(old)})
        history = [r for r in read(old / 'carry_forward_registry.json')['candidate_rounds'] if r['object_id'] == obj]
        if candidates == 15:
            history = history * 3
        record = {'object_id': obj, 'pass': True, 'candidate_sha256': digest(candidate),
                  'candidate_file': str(top / 'candidate.json'), 'candidate_file_sha256': sha(top / 'candidate.json'),
                  'result_file': str(top / 'result.json')}
        return obj, {'objects': {}, 'candidate_rounds': history}, {'objects': {obj: item}}, [record]

    def test_complete_ARX_model_certificate_can_enter_queue(self):
        from .advisor39_orchestrator import selection
        with tempfile.TemporaryDirectory() as directory:
            top = Path(directory); obj, data, spec, rows = self.supplemental_fixture(top)
            selected, exhausted, _ = selection(top, data, spec, [], rows, True, {}, False)
            self.assertEqual([r['object_id'] for r in selected], [obj])
            self.assertEqual(exhausted, [])

    def test_supplement_cannot_bypass_missing_boundary(self):
        from .advisor39_orchestrator import selection
        with tempfile.TemporaryDirectory() as directory:
            top = Path(directory); _, data, spec, rows = self.supplemental_fixture(top, missing_case=True)
            with self.assertRaises(ValueError):
                selection(top, data, spec, [], rows, True, {}, False)

    def test_supplement_cannot_reset_fifteen_candidate_budget(self):
        from .advisor39_orchestrator import selection
        with tempfile.TemporaryDirectory() as directory:
            top = Path(directory); obj, data, spec, rows = self.supplemental_fixture(top, candidates=15)
            selected, exhausted, _ = selection(top, data, spec, [], rows, True, {}, False)
            self.assertEqual(selected, [])
            self.assertEqual(exhausted, [obj])


if __name__ == '__main__':
    unittest.main()
