"""Offline sampling/record faults using the registered actual ARX canonical."""
import copy
from collections import Counter
import unittest
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT, read
from .advisor36_samples import audit, grouped_split
from .advisor37_samples import (generate_wave, envelope, PLANNED, MINIMUM,
                                assign_replacement_splits, quota_status, select_wave)


class ARXTaperedSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        references = read(ROOT / 'runs/nearfield_advisor35_rulefree_20261112_01/canonical_references.json')
        cls.G = next(r['Gcanonical_object'] for r in references if r['object_id'] == 'household_hope_milk_carton')
        cls.rows = generate_wave('household_hope_milk_carton', cls.G, 0)
        grouped_split(cls.rows)

    def records(self, rows, admitted=True):
        return [{'sample_id': r['sample_id'], 'pair_id': r['pair_id'], 'candidate': r,
                 'split': r['split'], 'admitted': admitted, 'capture_admitted': admitted,
                 'pair_check': {'pass': admitted}} for r in rows]

    def test_near_envelope_monotone_and_far_unchanged(self):
        previous = np.zeros(4)
        for z in np.linspace(0, .09, 181):
            xy, rot = envelope(z)
            now = np.r_[xy, rot]
            self.assertTrue((now >= previous - 1e-12).all())
            previous = now
            if z >= .035:
                np.testing.assert_allclose(now, [.020, 10, 10, 15])

    def test_actual_ARX_candidates_keep_core_and_exact_pair_symmetry(self):
        self.assertTrue(audit(self.rows)['pass'])
        self.assertEqual(Counter(r['category'] for r in self.rows),
                         {'funnel': 512, 'boundary_position': 128, 'boundary_rotation': 128, 'grasp': 256})
        for row in self.rows:
            p, rot = np.array(row['desired_translation_m']), np.array(row['desired_rotation_degrees'])
            self.assertEqual(row['category'] == 'grasp', np.linalg.norm(p) <= .004 and np.linalg.norm(rot) <= 3)
            if row['category'] == 'funnel':
                xy, angles = envelope(p[2])
                self.assertTrue((abs(p[:2]) <= xy).all())
                self.assertTrue((abs(rot) <= angles).all())

    def test_one_missing_member_cannot_satisfy_quota(self):
        with self.assertRaises(ValueError):
            quota_status(self.records(self.rows[:1]))
        records = self.records(self.rows[:2])
        records[1]['admitted'] = False
        with self.assertRaises(ValueError):
            quota_status(records)

    def test_capture_only_is_not_independent_admission(self):
        records = self.records(self.rows, False)
        for r in records:
            r['capture_admitted'] = True
        result = quota_status(records)
        self.assertFalse(result['minimum_met'])
        self.assertEqual(result['admitted_observations'], 0)

    def test_quota_requires_every_layer_and_boundary_subtype(self):
        records = self.records(self.rows)
        self.assertTrue(quota_status(records)['planned_counts_met'])
        kept = [r for r in records if r['candidate']['axial_layer_mm'] != '05_15']
        self.assertFalse(quota_status(kept)['minimum_met'])
        self.assertEqual(MINIMUM['funnel/05_15'], 48)
        self.assertEqual(MINIMUM['funnel/50_65'], 72)
        self.assertEqual(MINIMUM['grasp/core'], 192)

    def test_started_pair_never_replayed_and_pool_finite(self):
        selected = select_wave(self.rows, 0, [], [self.rows[0]['sample_id']])
        self.assertEqual(len(selected), 1022)
        self.assertNotIn(self.rows[1]['sample_id'], selected)
        self.assertEqual(select_wave(self.rows, 0, self.records(self.rows)), [])
        with self.assertRaises(ValueError):
            generate_wave('household_hope_milk_carton', self.G, 4)

    def test_duplicate_admission_is_record_damage(self):
        records = self.records(self.rows[:2])
        with self.assertRaises(ValueError):
            quota_status(records + records)

    def test_replacement_cannot_bridge_frozen_splits(self):
        # Inject a split-conflict into actual ARX pose records; no fake robot.
        previous = copy.deepcopy(self.rows[:2])
        previous[0]['split'], previous[1]['split'] = 'train', 'validation'
        fresh = copy.deepcopy(self.rows[:2])
        for r in fresh:
            r['sample_id'] += '_fresh'
            r['pair_id'] += '_fresh'
        result = assign_replacement_splits(fresh, previous)
        self.assertEqual(result['proposal_excluded'], 2)
        self.assertTrue(all(r['split'] == 'excluded' for r in fresh))
        self.assertEqual([r['split'] for r in previous], ['train', 'validation'])

    def test_fresh_wave_ids_are_disjoint_and_BBQ_Spam_isolated(self):
        new = generate_wave('household_hope_milk_carton', self.G, 1)
        self.assertFalse({r['sample_id'] for r in self.rows} & {r['sample_id'] for r in new})
        for obj in ('household_hot3d_bbq_sauce_bottle', 'household_ycb_spam_can'):
            rows = generate_wave(obj, self.G, 0)
            grouped_split(rows, evaluation_only=True)
            more = generate_wave(obj, self.G, 1)[:8]
            assign_replacement_splits(more, rows, True)
            self.assertEqual({r['split'] for r in more}, {'unseen_evaluation'})


if __name__ == '__main__':
    unittest.main()
