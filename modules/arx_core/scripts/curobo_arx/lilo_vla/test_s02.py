"""S02 fault injection on recorded ARX data; no CUDA allocation or robot command."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts/curobo_arx'), str(ROOT/'servers/lilo_vla')]
from lilo_vla import trace
from lilo_vla.grasp_handoff import candidates, S02_RULE, RULE, validate_rule
from lilo_vla.interaction_contract import load, RULES
from lilo_vla.observations import policy_observation

SOURCE = ROOT/'runs/lilo_P2_comparison_20261028_02/goal_physical/A1'


class S02Evidence(unittest.TestCase):
    def setUp(self):
        self.obs = json.loads((SOURCE/'lilo_observations/0000/observation.json').read_text())
        self.goal = json.loads((SOURCE/'selected_grasp.json').read_text())['grasp_world']
        self.image, _, self.state = policy_observation(self.obs, 'grasp_goal_world_error_v1', self.goal)
        buf = io.BytesIO(); Image.fromarray(self.image).save(buf, format='PNG'); self.png = buf.getvalue()
        raw = Path(self.obs['rgb_path']).read_bytes()
        with Image.open(io.BytesIO(raw)) as im:pixels = trace.rgb_identity(im)
        self.obs['trace'] = dict(version=trace.VERSION, observation_id=trace.identity(),
            png_sha256=trace.sha(raw), rgb=pixels, snapshot_sha256=trace.sha(trace.canonical(self.obs['snapshot'])),
            capture_finished=trace.stamp(), state_sim_time=self.obs['capture_sim_time'])
        self.args = (self.png, self.state.tolist(), 'pick up the object', 'grasp_goal_world_error_v1')

    def test_actual_saved_arx_image_request_round_trip(self):
        value = trace.make_request(self.obs, *self.args)
        trace.verify_request(value,self.png,Image.open(io.BytesIO(self.png)),*self.args[1:])
        self.assertNotEqual(value['request_id'],value['observation_id'])

    def test_modified_frame_state_instruction_and_ids_rejected(self):
        original = trace.make_request(self.obs,*self.args)
        for key in ('request_png_sha256','request_sha256','observation_id'):
            changed=copy.deepcopy(original);changed[key]='incorrect'
            with self.subTest(key=key),self.assertRaises(ValueError):
                trace.verify_request(changed,self.png,Image.open(io.BytesIO(self.png)),*self.args[1:])
        for state,instruction in (([1.]*8,self.args[2]),(self.args[1],'another object')):
            with self.assertRaises(ValueError):trace.verify_request(original,self.png,Image.open(io.BytesIO(self.png)),state,instruction,self.args[3])
        changed=copy.deepcopy(self.obs);changed['snapshot']['joint_positions'][0]+=.001
        with self.assertRaisesRegex(ValueError,'state changed'):trace.make_request(changed,*self.args)

    def test_legacy_capture_or_missing_velocities_does_not_pass(self):
        changed=copy.deepcopy(self.obs);changed.pop('trace')
        with self.assertRaises(ValueError):trace.make_request(changed,*self.args)
        changed=copy.deepcopy(self.obs);changed['snapshot'].pop('joint_velocities',None)
        changed['trace']['snapshot_sha256']=trace.sha(trace.canonical(changed['snapshot']))
        with self.assertRaisesRegex(ValueError,'velocities'):trace.make_request(changed,*self.args)

    def test_clock_domains_are_not_silently_subtracted(self):
        a=trace.stamp();b=trace.stamp()
        self.assertGreaterEqual(trace.elapsed_seconds(a,b),0)
        with self.assertRaises(ValueError):trace.elapsed_seconds(b,a)
        self.assertIsNone(trace.elapsed_seconds(a,{**b,'clock_domain':'different_host'}))

    def test_pure_40_has_one_tool_axis_candidate_and_frozen_checks(self):
        g=np.array(self.goal);result=candidates(g,S02_RULE)
        self.assertEqual(len(result),1)
        np.testing.assert_allclose(np.array(result[0]['world'])[:3,3],g[:3,3]-.04*g[:3,2],atol=1e-12)
        self.assertEqual(len(candidates(g,RULE)),3)
        config=load(ROOT/'configs/lilo_vla/handoff_10mm.json');config['handoff_rule']=S02_RULE
        self.assertEqual(load(config)['handoff_rule'],RULES[S02_RULE['version']])
        for key,value in [('ordered_backoff_m',[.04,.06]),('image_margin_pixels',0),('reject_pregrasp_target_contact',False)]:
            bad={**S02_RULE,key:value}
            with self.assertRaises(ValueError):validate_rule(bad)
            with self.assertRaises(ValueError):load({**config,'handoff_rule':bad})

    def test_backend_hashes_actual_bf16_forward_tensor_without_changing_actions(self):
        import torch
        from backend import Policy
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor
        processor=PrismaticImageProcessor.from_pretrained(ROOT/'servers/lilo_vla/tests/fixtures',local_files_only=True)
        policy=Policy.__new__(Policy)
        policy.torch=torch;policy.device=torch.device('cpu');policy.lock=threading.Lock()
        policy.key='arx';policy.stats={'arx':{'proprio':{'q01':[-1.]*8,'q99':[1.]*8}}}
        policy.metadata={};policy.projector=None;policy.head=None
        policy.processor=lambda prompt,image:processor(image,return_tensors='pt')
        expected=np.array(json.loads((SOURCE/'policy_0000_prediction.json').read_text())['actions'])
        observed=[]
        class RecordedActionStub:
            def predict_action(self,**kwargs):
                observed.append(trace.tensor_identity(kwargs['pixel_values']))
                return expected.copy(),torch.zeros((1,2))
        policy.model=RecordedActionStub()
        request=trace.make_request(self.obs,*self.args)
        with patch.object(torch.cuda,'synchronize'):
            baseline=policy.predict(Image.fromarray(self.image),*self.args[1:3])
            result=policy.predict(Image.fromarray(self.image),*self.args[1:3],trace={**request,'server_received':trace.stamp()})
        trace.verify_response(request,result['trace'])
        self.assertEqual(baseline['actions'],result['actions'])
        self.assertEqual(observed[0],observed[1])
        self.assertEqual(result['trace']['model_inputs']['pixel_values'],observed[1])
        self.assertEqual(observed[1]['dtype'],'torch.bfloat16')
        wrong=copy.deepcopy(result['trace']);wrong['request_id']='wrong'
        with self.assertRaises(ValueError):trace.verify_response(request,wrong)

    def test_runtime_physics_and_entire_initial_scene_are_compared(self):
        from s02_protocol import check_runtime,check_setup
        original=json.loads((SOURCE/'setup.json').read_text())
        self.assertTrue(check_setup(original,original)['pass_check'])
        modified=copy.deepcopy(original)
        modified['fingerprint']['invariant']['physics_dt']*=2
        self.assertFalse(check_setup(original,modified)['pass_check'])
        modified=copy.deepcopy(original)
        modified['fingerprint']['invariant']['runtime_robot'].pop('get_dof_gains')
        self.assertFalse(check_setup(original,modified)['pass_check'])
        modified=copy.deepcopy(original);modified['initial_conditions']['joint_positions'][0]+=.001
        self.assertFalse(check_setup(original,modified)['pass_check'])
        modified=copy.deepcopy(original)
        key=next(iter(modified['initial_conditions']['objects']))
        modified['initial_conditions']['objects'][key]['position'][0]+=.001
        self.assertFalse(check_setup(original,modified)['pass_check'])

    def test_service_refuses_corrupted_pixels_before_prediction(self):
        import base64
        from types import SimpleNamespace
        from unittest.mock import Mock
        import app as service
        from fastapi import HTTPException
        mocked=Mock()
        policy=SimpleNamespace(metadata={'state_convention':self.args[3]},predict=mocked)
        value=trace.make_request(self.obs,*self.args);value['request_png_sha256']='wrong'
        with patch.object(service.app.state,'policy',policy,create=True):
            with self.assertRaises(HTTPException) as error:
                service.predict(service.PredictRequest(image_png_base64=base64.b64encode(self.png).decode(),
                    state=self.args[1],instruction=self.args[2],state_convention=self.args[3],trace=value))
        self.assertEqual(error.exception.status_code,422);mocked.assert_not_called()


if __name__=='__main__':unittest.main()
