"""Contract regressions on saved ARX data only; no physical grasp claims."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import interaction_contract as contract
from grasp_handoff import candidates, RULE, NEAR_RULES
from observations import encode_state, policy_observation, action_target, teacher_action
from build_dataset import build

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
FIXTURE = ROOT/'servers/lilo_vla/tests/fixtures'
DATA = json.loads((FIXTURE/'arx_fixture.json').read_text())


class InteractionContract(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.config = contract.load(ROOT/'configs/lilo_vla/handoff_10mm.json')
        self.goal = DATA['observation']['snapshot']['link_world']['right_eef_link']
        self.selected = {'grasp_world': self.goal, 'handoff_rule': self.config['handoff_rule']['version']}

    def tearDown(self):
        self.temporary.cleanup()

    def episode(self, role='teacher'):
        value = contract.resolve(self.config, 'mustard_bottle', role)
        for name, data in {'scenario.json': {'asset':'mustard_bottle'},
                           'setup.json': {'fixture':'recorded ARX protocol fixture only'},
                           'setup_verification.json': {'pass':True},
                           'active_asset.json': {'name':'mustard_bottle'},
                           'viewport_initial.json': {'source':'ARX fixture'}}.items():
            contract.write(self.directory/name, data)
        env = contract.prepare_environment({}, self.directory, value)
        frozen = contract.freeze_goal(self.directory, value, self.selected)
        contract.write(self.directory/'selected_grasp.json', self.selected)
        return frozen, env

    def test_teacher_policy_common_semantics_and_hash_match_for_both_rules(self):
        for name in ('handoff_10mm','handoff_40_60_80mm'):
            config = ROOT/'configs/lilo_vla'/f'{name}.json'
            a = contract.resolve(config, 'rubiks_cube', 'teacher')
            b = contract.resolve(config, 'rubiks_cube', 'policy')
            self.assertEqual(a['contract_sha256'], b['contract_sha256'])
            self.assertEqual(a['contract'], b['contract'])
            self.assertEqual(a['instruction'], "pick up the rubik's cube")
            self.assertNotEqual(a['role'], b['role'])
            self.assertIsNone(a['grasp_world'])
            self.assertEqual(a['goal_source']['status'], 'pending_selection')

    def test_registered_rules_equal_existing_geometry_rules(self):
        self.assertEqual(self.config['handoff_rule'], NEAR_RULES['ten_mm'])
        self.assertEqual(contract.load(ROOT/'configs/lilo_vla/handoff_40_60_80mm.json')['handoff_rule'], RULE)

    def test_real_arx_goal_backoff_uses_tool_axis_and_left_world_rotation(self):
        G = np.asarray(self.goal)
        self.assertGreater(np.linalg.norm(G[:3,2]-[0,0,1]), .1)
        option = candidates(G, self.config['handoff_rule'])[0]
        np.testing.assert_allclose(np.asarray(option['world'])[:3,3], G[:3,3]-.01*G[:3,2], atol=1e-12)
        delta = [.005,-.005,.005,.04,-.04,.04]
        from scipy.spatial.transform import Rotation
        moved = np.asarray(candidates(G,self.config['handoff_rule'],delta)[0]['world'])
        np.testing.assert_allclose(moved[:3,3],G[:3,3]-.01*G[:3,2]+delta[:3],atol=1e-12)
        np.testing.assert_allclose(moved[:3,:3],Rotation.from_rotvec(delta[3:]).as_matrix()@G[:3,:3],atol=1e-12)

    def test_unknown_version_or_changed_physics_gate_is_rejected(self):
        for key,value in [('version','unknown'),('execute_steps',8),('camera_role','head'),
                          ('preprocessing_version','different_crop'),('state_convention','object_relative_v1')]:
            changed = copy.deepcopy(self.config);changed[key]=value
            with self.assertRaises(ValueError):contract.load(changed)
        changed=copy.deepcopy(self.config);changed['handoff_rule']['image_margin_pixels']=0
        with self.assertRaises(ValueError):contract.load(changed)

    def test_nonfinite_and_excessive_perturbations_rejected(self):
        for delta in ([float('nan')]*6,[.011,0,0,0,0,0],[0,0,0,.16,0,0],[0,0,0,True,0,0]):
            with self.assertRaises(ValueError):contract.resolve(self.config,'mustard_bottle','teacher',delta=delta)

    def test_grasp_must_be_rigid_not_reflected_or_nonfinite(self):
        for field,value in [((0,0),2.),((0,0),float('inf')),((3,3),0.)]:
            changed=copy.deepcopy(self.goal);changed[field[0]][field[1]]=value
            with self.assertRaises(ValueError):contract.validate_goal(changed)
        changed=np.asarray(self.goal).copy();changed[:3,0]*=-1
        with self.assertRaises(ValueError):contract.validate_goal(changed.tolist())

    def test_external_environment_overrides_are_rejected(self):
        for key in contract.CONTROL_KEYS:
            with self.assertRaisesRegex(ValueError,key):contract.reject_overrides({key:'stale'})

    def test_runtime_checks_generated_values_and_file_contents(self):
        value, env=self.episode()
        self.assertEqual(contract.runtime(env)['episode_sha256'],value['episode_sha256'])
        wrong=dict(env,ARX_LILO_INSTRUCTION='pick up another object')
        with self.assertRaisesRegex(ValueError,'INSTRUCTION'):contract.runtime(wrong)
        contract.write(Path(env['ARX_LILO_HANDOFF_RULE_FILE']), RULE)
        with self.assertRaisesRegex(ValueError,'rule changed'):contract.runtime(env)

    def test_teacher_and_policy_controller_roles_cannot_be_mixed(self):
        _,env=self.episode()
        with self.assertRaisesRegex(ValueError,'Teacher interaction'):
            contract.runtime(dict(env,ARX_LILO_POLICY_URL='http://invalid'))

    def test_unselected_goal_cannot_be_admitted(self):
        value=contract.resolve(self.config,'mustard_bottle','teacher')
        with self.assertRaisesRegex(ValueError,'not been frozen'):contract.validate_resolution(value,require_goal=True)

    def test_fixed_goal_requires_same_object_and_stable_source(self):
        contract.write(self.directory/'scenario.json', {'asset':'mustard_bottle'})
        contract.write(self.directory/'selected_grasp.json',self.selected)
        p=self.directory/'selected_grasp.json'
        with self.assertRaisesRegex(ValueError,'another object'):
            contract.resolve(self.config,'apple_01','teacher',goal_source=p)
        value=contract.resolve(self.config,'mustard_bottle','teacher',goal_source=p)
        changed=copy.deepcopy(self.selected);changed['grasp_world'][0][3]+=.001
        contract.write(p,changed)
        with self.assertRaisesRegex(ValueError,'source changed'):contract.validate_resolution(value)

    def test_selected_goal_and_references_freeze_before_motion(self):
        value,_=self.episode()
        self.assertEqual(contract.validate_episode(self.directory,self.config)['episode_sha256'], value['episode_sha256'])
        contract.write(self.directory/'setup.json',{'fixture':'changed'})
        with self.assertRaisesRegex(ValueError,'reference changed'):contract.validate_episode(self.directory,self.config)

    def test_episode_cannot_change_target_after_freezing(self):
        self.episode()
        changed=copy.deepcopy(self.selected);changed['grasp_world'][0][3]+=.01
        contract.write(self.directory/'selected_grasp.json',changed)
        with self.assertRaisesRegex(ValueError,'metadata changed'):contract.validate_episode(self.directory,self.config)

    def test_old_recordings_cannot_be_relabeled(self):
        with self.assertRaisesRegex(ValueError,'no recorded interaction'):
            contract.validate_episode(self.directory,self.config)
        with self.assertRaisesRegex(ValueError,'cannot be mixed or relabeled'):
            build([self.directory],self.directory/'output',interaction_contract=self.config)
        self.assertFalse((self.directory/'output').exists())

    def test_dataset_cannot_override_recorded_state_convention(self):
        self.episode()
        with self.assertRaisesRegex(ValueError,'State convention conflicts'):
            build([self.directory],self.directory/'output',state_convention='object_relative_v1')

    def test_dataset_instruction_override_rejected_before_writing(self):
        self.episode()
        with self.assertRaisesRegex(ValueError,'instruction conflicts'):
            build([self.directory],self.directory/'output',instruction='pick up another object')
        self.assertFalse((self.directory/'output').exists())

    def test_executor_rejects_bad_interaction_before_gpu_import(self):
        _,env=self.episode()
        env={**os.environ,**env,'ARX_LILO_EXECUTE_STEPS':'8'}
        result=subprocess.run(['python3','-S',str(ROOT/'scripts/curobo_arx/grasp_execute.py')],
            env=env,text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Runtime interaction override: ARX_LILO_EXECUTE_STEPS',result.stderr)
        self.assertNotIn('No module named',result.stderr)

    def test_model_and_service_semantics_are_checked(self):
        value=contract.resolve(self.config,'mustard_bottle','policy')
        health=dict(interaction_semantics=contract.model_semantics(),state_convention=contract.STATE,action_shape=[8,7])
        contract.check_health(value,health)
        for key in ('interaction_semantics','state_convention','action_shape'):
            wrong=copy.deepcopy(health);wrong[key]='wrong'
            with self.assertRaises(ValueError):contract.check_health(value,wrong)

    def test_legacy_model_compatibility_requires_exact_artifact_identity(self):
        for name in ('action_head.pt','dataset_statistics.json'):(self.directory/name).write_text('not a real model')
        contract.write(self.directory/'adapter_manifest.json',dict(state_convention=contract.STATE,
            base_checkpoint_revision=contract.BASE_REVISION,action_alignment_version='command_interval_v2'))
        with self.assertRaisesRegex(ValueError,'no automatic defaults'):
            contract.adapter_compatibility(self.directory,self.config)

    def test_training_and_rollout_use_same_saved_arx_observation_codec(self):
        sys.path.insert(0,str(ROOT/'scripts/curobo_arx'))
        from lilo_vla import rollout
        import build_dataset
        self.assertEqual(Path(build_dataset.policy_observation.__code__.co_filename).resolve(),
                         Path(rollout.policy_observation.__code__.co_filename).resolve())
        observation=copy.deepcopy(DATA['observation']);observation['rgb_path']=str(FIXTURE/'arx_wrist_pregrasp.png')
        a=build_dataset.policy_observation(observation,contract.STATE,self.goal)
        b=rollout.policy_observation(observation,contract.STATE,self.goal)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[2],b[2]);self.assertEqual(a[1],b[1])
        np.testing.assert_allclose(a[2][:6],0,atol=1e-10)

    def test_real_measured_action_round_trip_unchanged(self):
        a,b=DATA['measured_pose_pair']
        row=lambda pose:dict(snapshot={'link_world':{'right_eef_link':pose}},teacher_command={'gripper_target':.1})
        target,gripper=action_target(a,teacher_action(row(a),row(b)))
        np.testing.assert_allclose(target,b,atol=1e-10,rtol=0);self.assertEqual(gripper,.1)

    def test_resolve_only_both_entrypoints_work_without_gpu_rpc_or_site_packages(self):
        env={k:v for k,v in os.environ.items() if k not in contract.CONTROL_KEYS}
        for entry in ('collect_shared_teacher.py','probe_transfer.py'):
            code='''import sys,runpy
sys.path.insert(0,sys.argv[1])
def audit(event,args):
    if event in ('subprocess.Popen','socket.__new__'):raise AssertionError('Unexpected GPU/RPC process or socket')
    if event=='import' and args[0].split('.')[0] in ('torch','numpy','httpx','tensorflow','curobo','client'):
        raise AssertionError('Unexpected numerical/GPU/RPC import: '+args[0])
sys.addaudithook(audit)
sys.argv=sys.argv[2:]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
            command=['python3','-S','-c',code,str(HERE),str(HERE/entry),'--resolve-only',
                '--interaction-contract',str(ROOT/'configs/lilo_vla/handoff_10mm.json'),'--object-name','mustard_bottle']
            result=subprocess.run(command,env=env,text=True,capture_output=True,check=True)
            value=json.loads(result.stdout)
            self.assertEqual(value['contract_sha256'],contract.digest(self.config))
            self.assertIsNone(value['grasp_world'])


if __name__=='__main__':unittest.main()
