import copy
import unittest
from .paired_runtime import choose_pool


class ComparisonInputTests(unittest.TestCase):
    def fixture(self):
        return {'candidates':[{'goal_id':'old'},{'goal_id':'new'}],
            'proposal_extension':{'original_prefix_ids':['old']},'point_cloud':{'depth_sha256':'original'}}

    def test_control_cannot_execute_an_added_candidate(self):
        raw=self.fixture();saved=copy.deepcopy(raw);control=choose_pool(raw,'baseline')
        self.assertEqual(control['candidates'],[{'goal_id':'old'}]);self.assertEqual(raw,saved)
        self.assertEqual(control['point_cloud'],raw['point_cloud'])
        self.assertEqual(choose_pool(raw,'variant')['candidates'],saved['candidates'])

    def test_corrupt_prefix_stops_before_control_execution(self):
        raw=self.fixture();raw['candidates'].reverse()
        with self.assertRaises(RuntimeError):choose_pool(raw,'baseline')
        with self.assertRaises(RuntimeError):choose_pool(raw,'variant')
        with self.assertRaises(RuntimeError):choose_pool({'candidates':[{'goal_id':'new'}]},'baseline')
        self.assertEqual(choose_pool({'candidates':[]},'baseline')['candidates'],[])


if __name__=='__main__':unittest.main()
