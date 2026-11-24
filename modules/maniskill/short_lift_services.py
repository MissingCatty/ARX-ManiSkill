"""Resident TCP43 services with enforced held-out inputs and frozen feedback."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import resident_backend as service
import graspcl_pipeline as common


class InputGuard:
    def __init__(self,root):
        self.root=root.resolve();self.campaign=self.root.parent
        self.reg=common.read(self.campaign/'registration.json')
        self.allowed=set(self.reg['test_object_ids']);self.excluded=set(self.reg['excluded_old_object_ids'])
        self.entries={e['id']:e for e in self.reg['schedule']}
        self.arms={name for name,_,_ in self.reg['arms']}
        assert not self.allowed&self.excluded

    def check(self,directory,kind,**flags):
        directory=Path(directory).resolve();warmup=Path(self.reg['warmup_trial']).name
        bootstrap=(directory==self.root/'neural_warmup' or directory==self.root/'graspcl_warmup'/warmup
            or (directory.name==warmup and directory.parent.parent==self.campaign/'startup_validation'))
        if bootstrap:
            if (self.campaign/'test_started.json').exists():raise RuntimeError('Training-only warmup cannot run after test claims begin')
            return {'bootstrap_only':True,'new_test_feedback_used':False}
        if directory.parent.name!='trials' or directory.parent.parent.name not in self.arms or directory.parent.parent.parent!=self.campaign:
            raise RuntimeError('Unregistered evaluation input directory')
        entry=self.entries.get(directory.name)
        if entry is None:raise RuntimeError('Unknown held-out case')
        if (directory/'result.json').exists():raise RuntimeError('Completed evaluation result cannot enter an inference request')
        state_name='closed_state.json' if flags.get('lift') else 'pregrasp_state.json' if flags.get('approach') else 'state.json'
        state=common.read(directory/state_name)
        models={a['model_id'] for a in state['objects'].values()}
        if not models<=self.allowed or models&self.excluded:raise RuntimeError('Training/old benchmark object appeared in held-out scene')
        if state['objects']['target']['model_id']!=entry['model_id']:raise RuntimeError('Held-out target identity changed')
        if directory.name in self.reg['source_feedback_ids_in_memory_order']:raise RuntimeError('Evaluation case is a feedback source')
        receipt={'checked_at':time.time(),'component':kind,'flags':flags,'id':directory.name,
            'scene_models':sorted(models),'object_overlap':0,'feedback_id_overlap':0,
            'current_episode_result_present':False,'new_test_feedback_used':False}
        with (directory/'independence_service_checks.jsonl').open('a') as stream:stream.write(json.dumps(receipt)+'\n')
        return receipt


def memory_hash(worker):
    h=hashlib.sha256()
    for a in worker.memory[:3]:h.update(np.asarray(a).tobytes())
    for db in worker.memory[4:]:
        for template in db:
            for grasp in template['grasps']:
                h.update(grasp['metadata']['source_trial_id'].encode())
                h.update(np.asarray(grasp['pose'],dtype=np.float64).tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['zerograsp','graspcl','curobo'],required=True)
    for name in ['socket','root','warmup-trial','bank','policy-registration']:
        parser.add_argument('--'+name,type=Path,required=name in ['socket','root','warmup-trial'])
    args=parser.parse_args();guard=InputGuard(args.root)
    if args.kind=='graspcl':
        import native_resident_services as native
        from table_truncation_services import rescore
        original=native.GraspCLWorker
        class FrozenWorker(original):
            def __init__(self,root,bank):
                super().__init__(root,bank)
                def deny(*args,**kwargs):raise RuntimeError('Held-out evaluation may not add feedback or demonstrations')
                self.module.grasp_scoring_module_cl.add_datapoint=deny
                self.module.grasp_scoring_module_cl.add_demonstration=deny
                self.module.grasp_recall_module_object.add_demonstration=deny
                self.module.grasp_recall_module_patch.add_demonstration=deny
                self.frozen_memory_hash=memory_hash(self)
                common.save(root/'independent_graspcl_guard_ready.json',{'updates_disabled':True,
                    'feedback_points':len(self.ids),'memory_hash':self.frozen_memory_hash,
                    'test_object_overlap':0,'source_ids':self.ids,'test_started':False})
            def request(self,directory,policy):
                checked=guard.check(directory,'graspcl',rescore=bool(policy.get('rescore_only')))
                assert memory_hash(self)==self.frozen_memory_hash
                result=rescore(self,Path(directory)) if policy.get('rescore_only') else super().request(directory,policy)
                assert memory_hash(self)==self.frozen_memory_hash,'Frozen training memory changed during evaluation'
                if not checked.get('bootstrap_only'):
                    exclusion=result['self_case_exclusion']
                    assert not exclusion['excluded_feedback_ids']
                    assert set(exclusion['retained_feedback_ids'])==set(self.ids)
                    common.save(Path(directory)/('frozen_memory_rescore.json' if policy.get('rescore_only') else 'frozen_memory_inference.json'),{
                        'before_sha256':self.frozen_memory_hash,'after_sha256':memory_hash(self),
                        'points':len(self.ids),'test_feedback_added':0,'all_training_sources_disjoint':True})
                return result
        native.GraspCLWorker=FrozenWorker;native.serve_graspcl(args)
    else:
        if args.kind=='zerograsp':
            from postprocess_backend import NativePostNeural
            class Zero(NativePostNeural):
                def request(self,directory,**kwargs):
                    guard.check(directory,'zerograsp');return super().request(directory,**kwargs)
            service.NeuralWorker=Zero
        else:
            import resident_planner
            from fast_resident_planner import FastPlannerPool
            from checked_native_services import checked_pool
            from table_truncation_model import make_pool
            from short_lift_planner import ShortLiftPlanner
            base=make_pool(checked_pool(FastPlannerPool))
            class Pool(base):
                def request(self,directory,**kwargs):
                    guard.check(directory,'curobo',**kwargs);return super().request(directory,**kwargs)
            resident_planner.GraspCLPlanner=ShortLiftPlanner;resident_planner.PlannerPool=Pool
        service.serve(args)


if __name__=='__main__':main()
