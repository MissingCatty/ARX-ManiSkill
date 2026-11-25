"""Offline tests for admission, one-read evidence and failure accounting."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from arx_project.artifacts import atomic_json
from .campaign import method_resource_choices,placement_still_fits,occupied_devices
from .contracts import verify_io
from .metrics import failure_category,slot_counts

FLOORS={'rayst3r':23000,'zerograsp_single_device':18000,'graspgenx_execution_device':16000}


class Contracts(unittest.TestCase):
    def test_admitted_devices_cannot_be_reused_during_loading(self):
        free={'ray':24000,'execution':21000,'third':19000}
        roles=method_resource_choices(free,FLOORS,'graspgenx')[0][1]
        assignment={'graspgenx':{'gpu_roles':roles}}
        reserved=occupied_devices(assignment,['graspgenx'])
        choices=method_resource_choices(free,FLOORS,'zerograsp',unavailable=reserved)
        self.assertEqual(set(choices[0][1].values()),set(free)-reserved)
        self.assertFalse(set(choices[0][1].values())&reserved)
        self.assertEqual(occupied_devices(assignment,[]),set())

    def test_running_case_is_not_counted_as_interrupted(self):
        self.assertEqual(slot_counts({'done','old_interrupted','now'}, {'done'},
            current='now',active=True,total=4),{'interrupted':1,'in_progress':1,'not_started':1})
        self.assertEqual(slot_counts({'done','old_interrupted','now'}, {'done'},
            current='now',active=False,total=4),{'interrupted':2,'in_progress':0,'not_started':1})
        with self.assertRaises(RuntimeError):slot_counts({'claimed'},{'missing'})

    def test_one_ready_card_can_start_zero_without_waiting_for_ray(self):
        self.assertEqual(method_resource_choices({'a':19352},FLOORS,'graspgenx'),[])
        choices=method_resource_choices({'a':19352},FLOORS,'zerograsp')
        self.assertEqual(set(choices[0][1].values()),{'a'})
        self.assertEqual(method_resource_choices({'a':17999},FLOORS,'zerograsp'),[])

    def test_graspgenx_uses_two_devices_and_preserves_ray_floor(self):
        self.assertEqual(method_resource_choices({'a':22999,'b':20000},FLOORS,'graspgenx'),[])
        choices=method_resource_choices({'a':23000,'b':16000},FLOORS,'graspgenx')
        roles=choices[0][1]
        self.assertNotEqual(roles['rayst3r'],roles['simulation'])
        self.assertEqual(roles['graspgenx'],roles['simulation'])
        self.assertFalse(placement_still_fits({'a':22999,'b':16000},FLOORS,'graspgenx',roles))

    def test_cuda_failed_device_does_not_hide_a_healthy_ready_card(self):
        choices=method_resource_choices(dict(a=19000,d=24000),FLOORS,'zerograsp',unavailable={'d'})
        self.assertTrue(choices)
        self.assertTrue(all('d' not in row[1].values() for row in choices))
        self.assertEqual(method_resource_choices(dict(a=17000,c=24000),FLOORS,'zerograsp',unavailable={'c'}),[])

    def test_persisted_file_does_not_satisfy_the_shared_memory_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            atomic_json(root/'candidate_snapshot_commit.json',{'persisted':True})
            proof={'source':'single_snapshot_read','decode_count':1,'full_arm_pool_copies':0,
                   'candidate_order_poses_and_scores_unchanged':True,'load_seconds':.01}
            atomic_json(root/'candidate_io.json',proof)
            with self.assertRaises(RuntimeError):verify_io(root,require_decode=True)
            proof['source']='shared_memory';atomic_json(root/'candidate_io.json',proof)
            self.assertEqual(verify_io(root,require_decode=True)['decode_count'],1)
            proof['full_arm_pool_copies']=1;atomic_json(root/'candidate_io.json',proof)
            with self.assertRaises(RuntimeError):verify_io(root,require_decode=True)

    def test_empty_candidates_still_require_a_storage_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaises(OSError):verify_io(root,require_decode=False)
            atomic_json(root/'candidate_snapshot_commit.json',{'persisted':True})
            self.assertIsNone(verify_io(root,require_decode=False))

    def test_failure_distribution_distinguishes_stages_and_does_not_label_success_failure(self):
        labels={reason:failure_category({'reason':reason,'success':False}) for reason in [
            'unable_to_reach_high_workspace','zerograsp_no_postprocessed_candidate','no_arm_has_executable_candidate',
            'measured_pregrasp_approach_unreachable','no_bilateral_target_contact','insufficient_hold_height']}
        self.assertEqual(len(set(labels.values())),6)
        self.assertIsNone(failure_category({'reason':'success','success':True}))
        self.assertEqual(failure_category({'reason':'approach_not_converged','success':False}),'approach')


if __name__=='__main__':unittest.main()
