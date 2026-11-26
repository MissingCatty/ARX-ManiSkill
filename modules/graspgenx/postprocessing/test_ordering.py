"""Exercise source identity, phase priority and the retained fallback pool."""
import unittest
from unittest.mock import patch

import numpy as np

from .ordering import make_queue,selection_source,validate_order


class QueueTests(unittest.TestCase):
    def payload(self):
        candidates=[]
        for i in range(4):
            pose=np.eye(4);pose[0,3]=i*.02
            candidates.append({'pose':pose.tolist(),'score':1-i*.1,'width':.088,'metadata':{'source':i}})
        return {'candidates':candidates,'native_score_order':True}

    def test_control_preserves_original_rows_and_order(self):
        source=self.payload();queue,manifest=make_queue(source,'control')
        self.assertEqual(queue['candidates'],source['candidates'])
        self.assertEqual(manifest['priority_to_original_source'],list(range(4)))

    @patch('modules.graspgenx.postprocessing.ordering.native_nms')
    def test_priority_changes_only_order_not_any_grasp(self,nms):
        nms.return_value={'representatives':[0,2],'deferred':[1,3]}
        source=self.payload();queue,manifest=make_queue(source,'nms15')
        self.assertEqual([c['metadata']['source'] for c in queue['candidates']],[0,2,1,3])
        for rank,index in enumerate(manifest['priority_to_original_source']):
            self.assertIs(queue['candidates'][rank],source['candidates'][index])
        self.assertEqual([c['metadata']['source'] for c in source['candidates']],list(range(4)))

    def test_deferred_selection_maps_to_its_original_source(self):
        manifest={'source_count':4,'representatives':[0,2],'deferred':[1,3],
            'priority_to_original_source':[0,2,1,3],'representative_count':2}
        self.assertEqual(selection_source(manifest,{'index':2}),
            {'original_source_index':1,'search_priority_rank':2,'priority_phase':'deferred'})

    def test_representatives_win_before_higher_score_deferred_rows(self):
        order=validate_order(4,[0,2],[1,3])
        executable={1,2};first=next(i for i in order if i in executable)
        self.assertEqual(first,2)

    def test_fallback_recovers_when_all_representatives_fail(self):
        order=validate_order(4,[0,2],[1,3]);executable={1}
        self.assertEqual(next(i for i in order if i in executable),1)

    def test_missing_original_candidate_is_rejected(self):
        with self.assertRaises(ValueError):validate_order(4,[0,2],[1])

    def test_duplicate_or_cross_phase_candidate_is_rejected(self):
        with self.assertRaises(ValueError):validate_order(4,[0,2],[2,3])

    def test_within_phase_score_order_cannot_be_reversed(self):
        with self.assertRaises(ValueError):validate_order(4,[2,0],[1,3])

    def test_changed_mapping_is_rejected(self):
        manifest={'source_count':3,'representatives':[0,2],'deferred':[1],
            'priority_to_original_source':[0,1,2],'representative_count':2}
        with self.assertRaises(ValueError):selection_source(manifest,{'index':0})

    def test_out_of_range_selection_is_rejected(self):
        manifest={'source_count':1,'representatives':[0],'deferred':[],
            'priority_to_original_source':[0],'representative_count':1}
        with self.assertRaises(ValueError):selection_source(manifest,{'index':1})


if __name__=='__main__':unittest.main()
