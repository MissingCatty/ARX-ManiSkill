"""Owned resident services for the TCP43 table-blocked ablations."""
import argparse
from pathlib import Path
import time

import resident_backend as service


def rescore(worker,directory):
    import numpy as np
    import torch
    import graspcl_pipeline as common
    from graspcl_batch import exclude_self
    directory = Path(directory)
    worker.restore()
    common.seed()
    started = time.perf_counter()
    exclusion = exclude_self(worker.module,worker.ids,directory.name)
    try:
        pool_path = directory / 'graspcl/candidate_pool.json'
        pool = common.read(pool_path)
        scene,_,camera = common.clouds(directory)
        candidates = pool['candidates']
        encoded = []
        for offset in range(0,len(candidates),64):
            part = candidates[offset:offset+64]
            for c in part:
                for key in ['pose','point1','point2']:c[key] = np.asarray(c[key])
            with torch.no_grad():encoded.extend(worker.module.encoder.predict(scene,part))
        if encoded:
            worker.module.grasp_scoring_module_cl.score_grasps(encoded,estimation_mode='posterior')
        for c in encoded:c.pop('score_thompson',None)
        pool['candidates'] = sorted(encoded,key=lambda c:-c['score'])
        pool['refined_candidates_rescored'] = True
        pool['rescore_self_case_exclusion'] = exclusion
        common.save(pool_path,pool)
        write_display(directory,pool,camera,'GraspCL score')
        receipt = {'success':True,'rescored_candidates':len(encoded),
            'encoder_load_count':1,'bank_load_count':1,'bank_updates':0,
            'self_case_exclusion':exclusion,'wall_seconds':time.perf_counter()-started}
        common.save(directory / 'refined_rescore.json',receipt)
        return receipt
    finally:worker.restore()


def write_display(directory,pool,camera=None,label='Network score'):
    import numpy as np
    import graspcl_pipeline as common
    destination = directory / 'graspcl'
    if camera is None:camera = np.asarray(common.read(directory / 'state.json')['camera_world_optical'])
    rows = np.stack([common.display_row(c,camera) for c in pool['candidates']]) if pool['candidates'] else np.empty((0,17))
    np.save(destination / 'raw_grasps.npy',rows,allow_pickle=False)
    common.save(destination / 'network_candidates.json',{'model_name':'ZeroGrasp | ARX TCP43 insertion truncation',
        'count':len(rows),'candidate_stage':'Observed ARX closing sweep and gripper/table truncation; full arm and trajectories still require cuRobo validation',
        'score_label':label,'physical_attempt':False,'action_tcp_eef_z_m':.043,
        'orientation_mode':pool['refinement']['mode']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['zerograsp','graspcl','curobo'],required=True)
    for name in ['socket','root','warmup-trial','bank','policy-registration']:
        parser.add_argument('--'+name,type=Path,required=name in ['socket','root','warmup-trial'])
    args = parser.parse_args()
    if args.kind == 'graspcl':
        import native_resident_services as native
        original = native.GraspCLWorker
        class Worker(original):
            def request(self,directory,policy):
                if policy.get('rescore_only'):return rescore(self,Path(directory))
                return super().request(directory,policy)
        native.GraspCLWorker = Worker
        native.serve_graspcl(args)
    else:
        if args.kind == 'zerograsp':
            from postprocess_backend import NativePostNeural
            service.NeuralWorker = NativePostNeural
        else:
            import resident_planner
            from fast_resident_planner import FastPlannerPool
            from checked_native_services import checked_pool
            from table_truncation_model import TruncationPlanner,make_pool
            resident_planner.GraspCLPlanner = TruncationPlanner
            resident_planner.PlannerPool = make_pool(checked_pool(FastPlannerPool))
        service.serve(args)


if __name__ == '__main__':main()
