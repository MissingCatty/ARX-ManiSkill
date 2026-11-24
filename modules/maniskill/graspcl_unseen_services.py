"""Resident services for the frozen, category-disjoint memory ablation."""
from pathlib import Path
import json
import shutil
import sys
import time

import numpy as np
import graspcl_pipeline as common

CODE = Path(__file__).resolve().parent


def install():
    import short_lift_services as services
    import native_resident_services as native
    from postprocess_backend import NativePostNeural
    import importlib.util
    spec = importlib.util.spec_from_file_location('unseen_physical_table', CODE/'tmp/physical_table_pair_20261118_01.py')
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    kind = sys.argv[sys.argv.index('--kind')+1]
    if kind == 'curobo': policy.install_collision_policy()

    original_guard = services.InputGuard
    class Guard(original_guard):
        def check(self,directory,kind,**flags):
            receipt = super().check(directory,kind,**flags)
            if not receipt.get('bootstrap_only'):
                mapping = common.read(self.reg['semantic_map'])
                categories = {mapping[m]['category'] for m in receipt['scene_models']}
                assert not categories & set(self.reg['forbidden_categories'])
            return receipt
    services.InputGuard = Guard

    # The older worker assumed every bank has ARX feedback. This ablation must
    # also load the genuinely empty online memory, with zero recall templates.
    def initialize(self,root,bank):
        common.seed()
        self.root = root
        self.ids = common.read(bank/'bank_complete.json')['source_ids']
        started = time.perf_counter()
        runtime = root/'graspcl_model'
        shutil.copytree(bank/'model',runtime/'model')
        self.module = common.load_model(runtime)
        scorer = self.module.grasp_scoring_module_cl
        assert len(self.ids) == len(scorer.X_on)
        self.memory = (scorer.X_on,scorer.y_on,scorer.w_on,scorer.tree_on,
            list(self.module.grasp_recall_module_object._db),list(self.module.grasp_recall_module_patch._db))
        if not self.ids:
            assert not self.memory[4] and not self.memory[5]
        self.load_seconds = time.perf_counter()-started
    native.GraspCLWorker.__init__ = initialize

    original_init = NativePostNeural.__init__
    def initialize_zero(self,root,warmup_trial):
        self.root = Path(root)
        original_init(self,root,warmup_trial)
    NativePostNeural.__init__ = initialize_zero
    original_request = NativePostNeural.request
    def shared_request(self,directory,**kwargs):
        directory = Path(directory)
        # Native warmup deliberately has no state file and is never a test.
        if directory == self.root/'neural_warmup':
            return original_request(self,directory,**kwargs)
        reg = common.read(self.root.parent/'registration.json')
        if not reg.get('shared_input_role'):
            return original_request(self,directory,**kwargs)
        canonical = Path(reg['shared_input_cache'])/directory.name
        if reg['shared_input_role'] == 'producer':
            result = original_request(self,directory,**kwargs)
            canonical.mkdir(parents=True,exist_ok=False)
            for name in ['camera','network/raw_outputs']:
                shutil.copytree(directory/name,canonical/name)
            for name in ['raw_grasps.npy','postprocessed_grasps.npy','state.json','postprocessing_manifest.json']:
                shutil.copy2(directory/name,canonical/name)
            common.save(canonical/'manifest.json', {'source_trial':str(directory),
                'sha256':{str(p.relative_to(canonical)):common.sha(p) for p in canonical.rglob('*') if p.is_file()},
                'feedback_used':False,'zero_native_post_candidates':result['native_postprocessed_count']})
        else:
            manifest = common.read(canonical/'manifest.json')
            for name,digest in manifest['sha256'].items():
                assert common.sha(canonical/name)==digest,'Shared input cache changed'
            actual = common.read(directory/'state.json')
            source = common.read(canonical/'state.json')
            # Cached RGB-D uses its original measured optical frame. Joint and
            # collision states remain the current execution's measurements.
            q_error = float(np.max(np.abs(np.asarray(actual['qpos'])-source['qpos'])))
            camera_error = float(np.linalg.norm(np.asarray(actual['camera_world_optical'])[:3,3]-np.asarray(source['camera_world_optical'])[:3,3]))
            from scipy.spatial.transform import Rotation
            ca, cb = np.asarray(actual['camera_world_optical']), np.asarray(source['camera_world_optical'])
            camera_rotation_error = float(Rotation.from_matrix(ca[:3,:3]@cb[:3,:3].T).magnitude())
            object_errors = {}
            for key, obj in actual['objects'].items():
                assert obj['model_id']==source['objects'][key]['model_id']
                # Snapshot uses pose matrices; compare the complete transform.
                a, b = np.asarray(obj['pose']), np.asarray(source['objects'][key]['pose'])
                object_errors[key] = float(np.max(np.abs(a-b)))
            if q_error>.005 or camera_error>.002 or camera_rotation_error>.005 or max(object_errors.values(),default=0)>.002:
                raise RuntimeError('Paired physical reset diverged from canonical observation: '+str((q_error,camera_error,object_errors)))
            backup = directory/'live_capture'
            backup.mkdir(exist_ok=False)
            shutil.move(directory/'camera',backup/'camera')
            shutil.copy2(directory/'state.json',backup/'state.json')
            for name in ['camera','network/raw_outputs']:
                shutil.copytree(canonical/name,directory/name)
            for name in ['raw_grasps.npy','postprocessed_grasps.npy','postprocessing_manifest.json']:
                shutil.copy2(canonical/name,directory/name)
            actual['camera_world_optical'] = source['camera_world_optical']
            actual['shared_perception_source'] = str(canonical)
            common.save(directory/'state.json',actual)
            common.save(directory/'shared_physical_reset_check.json',{'passed':True,'joint_max_error_rad':q_error,
                'camera_position_error_m':camera_error,'camera_rotation_error_rad':camera_rotation_error,'object_pose_max_abs_errors':object_errors,
                'planning_uses_actual_q_and_objects':True,'perception_uses_cached_frame_extrinsics':True})
            result = {'success':True,'count':len(np.load(directory/'raw_grasps.npy')),
                'native_postprocessed_count':manifest['zero_native_post_candidates'],'shared_input_reused':True}
        shared_hashes = {name:common.sha(directory/name) for name in manifest_names(directory)}
        common.save(directory/'shared_input_receipt.json',{'role':reg['shared_input_role'],
            'canonical':str(canonical),'sha256':shared_hashes,'test_feedback_added':0})
        return result
    NativePostNeural.request = shared_request
    services.main()


def manifest_names(directory):
    return ['camera/'+n for n in ['color.png','depth.png','mask.png','params.json']] + [
        'raw_grasps.npy','postprocessed_grasps.npy']


if __name__ == '__main__': install()
