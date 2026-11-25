"""Priority, ground-truth isolation and stale-plan fault tests."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from arx_project.layout import ROOT, read_json, write_json, sha256
from modules.mobile_manipulation.placement import pose_matrix
from modules.mobile_manipulation.workspace_priority.search import HighWorkspaceSearch
from modules.mobile_manipulation.workspace_priority.handoff import validate_prepared


class Model:
    def frames(self, state):
        base = np.asarray(state['links']['base_link'])
        local = np.eye(4)
        local[:3, 3] = [.094118, .0010067, .27+state['qpos'][0]]
        return {'base_link': base, 'body4': base@local}


class Checker:
    blocked = False
    def __init__(self, *args):
        pass
    def check(self, *args):
        return {'passed': not self.blocked, 'obstacle': 'table' if self.blocked else None}
    def path(self, proposal, settings):
        return {'passed': True, 'samples': 2}, [{'time_s': 0}, {'time_s': 1}]


class Backend:
    def __init__(self, succeed_after=0):
        self.calls, self.succeed_after = [], succeed_after
    def plan(self, directory, arm=None):
        gate = read_json(directory/'workspace_gate.json')
        self.assert_high = gate['passed']
        self.calls.append((gate['scope'], arm, read_json(directory/'candidates/candidate_pool.json')))
        return {'success': len(self.calls) > self.succeed_after,
                'selected': {'arm': arm or 'right', 'index': 0}}


class HighWorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = Model()
        self.state = {'joint_names': ['joint4'], 'qpos': [.4], 'joint_limits': [[0., .46]],
            'links': {'base_link': pose_matrix(-1.05, .25, 0, -.5).tolist()},
            'object_center': [999., 999., 999.]}
        self.state['links'] = {k:v.tolist() for k,v in self.model.frames(self.state).items()}
        self.policy = read_json(ROOT/'configs/experiments/high_workspace_placement.json')
        self.policy['require_head_view'] = False
        self.settings = read_json(ROOT/'configs/experiments/sam3_graspgenx_v4.json')['placement']
        self.workspace = {'regions': {
            'shared': {'lower_body4_m': [.2,-.2,-.16], 'upper_body4_m': [.5,.2,.1]},
            'right': {'lower_body4_m': [.2,-.56,-.34], 'upper_body4_m': [.46,.06,.24]},
            'left': {'lower_body4_m': [.2,-.06,-.34], 'upper_body4_m': [.46,.56,.24]}}}
        self.data = {'cells':np.array([[8,-10,-1],[8,0,-1],[8,-5,-1],[8,5,-1]]),
            'sample_count':np.array([2000,2000,2000,20]), 'orientation_count':np.ones(4)*100,
            'down_orientation_count':np.ones(4)*20, 'resolution_m':np.array(.05)}
        self.target = np.array([-.4, .15, .03])
        self.pool = {'candidates':[{'index':7, 'pose':np.eye(4).tolist(), 'score':.3}]}

    def tearDown(self):
        self.temp.cleanup()

    def search(self, backend=None, checker=Checker):
        return HighWorkspaceSearch(self.model,self.settings,backend or Backend(),
            self.workspace,self.policy,checker_factory=checker,density_data=self.data)

    def test_shared_high_space_precedes_any_single_arm_or_motion_cost(self):
        backend = Backend()
        search = self.search(backend)
        chosen, report = search.select(self.state,self.pool,self.target,self.root/'shared')
        self.assertEqual(report['selected_scope'],'shared')
        self.assertTrue(chosen['grasp_planning_verified'])
        self.assertEqual(len(backend.calls),1)
        self.assertEqual(backend.calls[0][2],self.pool)
        self.assertFalse(report['outside_workspace_fallback'])

    def test_single_arm_fallback_keeps_matching_arm_and_original_pool(self):
        backend = Backend(succeed_after=2)
        chosen,report=self.search(backend).select(self.state,self.pool,self.target,self.root/'fallback')
        self.assertEqual(backend.calls[0][0],'shared')
        self.assertEqual({s for s,_,_ in backend.calls[1:]},{'right','left'})
        self.assertEqual(chosen['proposal']['arm'],backend.calls[-1][0])
        self.assertEqual(backend.calls[-1][1],backend.calls[-1][0])
        self.assertTrue(all(pool==self.pool for _,_,pool in backend.calls))
        self.assertEqual(report['full_planning_calls'],3)

    def test_no_full_plan_is_called_for_blocked_high_workspaces(self):
        class Blocked(Checker):
            blocked=True
        backend=Backend()
        chosen,report=self.search(backend,Blocked).select(self.state,self.pool,self.target,self.root/'blocked')
        self.assertIsNone(chosen)
        self.assertEqual(report['reason'],'unable_to_plan')
        self.assertEqual(report['failed_stage'],'high_workspace_or_body_transit')
        self.assertFalse(backend.calls)

    def test_all_native_planning_failures_report_unable_to_plan_without_low_fallback(self):
        backend=Backend(succeed_after=99)
        chosen,report=self.search(backend).select(self.state,self.pool,self.target,self.root/'fail')
        self.assertIsNone(chosen)
        self.assertEqual(report['reason'],'unable_to_plan')
        self.assertEqual(report['failed_stage'],'native_grasp_planning')
        self.assertEqual(len(backend.calls),3)

    def test_measured_target_drives_placement_and_every_goal_is_in_true_dense_cells(self):
        search=self.search()
        bad=deepcopy(self.state);bad['object_center']=[-999.,-999.,-999.]
        a,b=search.proposals(self.state,self.target,'shared'),search.proposals(bad,self.target,'shared')
        self.assertEqual(a,b)
        self.assertTrue(a)
        for goal in a:
            _,gate=search.check_goal(self.state,self.target,goal)
            self.assertTrue(gate['passed'])
            self.assertGreaterEqual(gate['right_sample_count'],search.threshold)
            self.assertGreaterEqual(gate['left_sample_count'],search.threshold)

    def test_geometry_only_does_not_claim_or_call_grasp_planning(self):
        backend=Backend()
        chosen,report=self.search(backend).select(self.state,self.pool,self.target,self.root/'geometry',
                                                geometry_only=True,scopes=['left'])
        self.assertEqual(chosen['proposal']['workspace_scope'],'left')
        self.assertFalse(chosen['grasp_planning_verified'])
        self.assertFalse(backend.calls)
        self.assertEqual(report['full_planning_calls'],0)

    def test_policy_cannot_lower_registered_workspace_thresholds(self):
        workspace = deepcopy(self.workspace)
        workspace['thresholds_per_arm_per_2cm_cell'] = {'sample_count': 29,
            'orientation_count': 26, 'down_orientation_count': 4}
        policy = {**self.policy, 'minimum_sample_count': 28}
        with self.assertRaisesRegex(ValueError, 'differs from the registered'):
            HighWorkspaceSearch(self.model, self.settings, Backend(), workspace, policy,
                                checker_factory=Checker, density_data=self.data)

    def handoff(self):
        saved=deepcopy(self.state)
        self.root.joinpath('camera').mkdir();self.root.joinpath('candidates').mkdir()
        write_json(self.root/'state.json',saved)
        write_json(self.root/'plan_result.json',{'success':True,'selected':{'arm':'right'}})
        for name in ['camera/color.png','camera/depth_m.npy','camera/mask.png','candidates/candidate_pool.json']:
            (self.root/name).write_bytes(b'fixture')
        names=['state.json','plan_result.json','camera/color.png','camera/depth_m.npy',
               'camera/mask.png','candidates/candidate_pool.json']
        target=np.asarray(saved['links']['body4'])@np.array([.425,.025,-.025,1.])
        preparation={'success':True,'prepared_plan':True,'preferred_arm':'right',
            'prepared_target_world':target[:3].tolist(),'workspace_resolution_m':.05,
            'verified_high_cell':[8,0,-1],
            'prepared_artifacts':{n:sha256(self.root/n) for n in names}}
        return saved,preparation

    def test_prepared_plan_rejects_stale_robot_state_and_changed_artifacts(self):
        saved,preparation=self.handoff()
        validate_prepared(self.root,preparation,saved)
        stale=deepcopy(saved);stale['qpos'][0]+=.01
        with self.assertRaisesRegex(RuntimeError,'stale_robot'):
            validate_prepared(self.root,preparation,stale)
        (self.root/'camera/mask.png').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'artifact changed'):
            validate_prepared(self.root,preparation,saved)

    def test_prepared_plan_rejects_wrong_arm_even_if_artifact_hash_is_updated(self):
        saved,preparation=self.handoff()
        write_json(self.root/'plan_result.json',{'success':True,'selected':{'arm':'left'}})
        preparation['prepared_artifacts']['plan_result.json']=sha256(self.root/'plan_result.json')
        with self.assertRaisesRegex(RuntimeError,'outside its high workspace'):
            validate_prepared(self.root,preparation,saved)


if __name__=='__main__':
    unittest.main()
