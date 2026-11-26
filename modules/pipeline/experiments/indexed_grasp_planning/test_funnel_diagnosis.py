"""Keep candidate outcomes distinct from per-branch collision reasons."""
import unittest

from .funnel_diagnosis import endpoint_category


class EndpointCategoryTests(unittest.TestCase):
    def test_absent_solution_is_not_called_collision(self):
        self.assertEqual(endpoint_category({'kind':'no_local_grasp_endpoint','kinematic_solutions':0,'reasons':[]}),
            'no_kinematic_solution_returned')

    def test_multiple_table_branches_still_make_one_candidate(self):
        self.assertEqual(endpoint_category({'kind':'no_local_grasp_endpoint','kinematic_solutions':2,
            'reasons':['nonfinger_table_collision','real_finger_depth_exceeds_tolerance']}),'solutions_rejected_table')

    def test_mixed_table_and_other_object_branches_have_their_own_bucket(self):
        self.assertEqual(endpoint_category({'kind':'no_local_grasp_endpoint','kinematic_solutions':2,
            'reasons':['finger_body_contact','ground_or_other_object_collision']}),'solutions_rejected_mixed')

    def test_native_and_local_passes_are_both_endpoint_passes(self):
        for kind in ['native_full_world_ik','bounded_ik_with_local_table_check']:
            self.assertEqual(endpoint_category({'kind':kind}),'endpoint_passed')

    def test_missing_branch_reason_is_rejected(self):
        with self.assertRaises(ValueError):endpoint_category({'kind':'no_local_grasp_endpoint','kinematic_solutions':2,'reasons':[]})

    def test_unknown_reason_cannot_silently_be_called_table_collision(self):
        with self.assertRaises(ValueError):endpoint_category({'kind':'no_local_grasp_endpoint','kinematic_solutions':1,'reasons':['unknown']})


if __name__=='__main__':unittest.main()
