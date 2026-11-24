"""Bounded fresh-wrist Graspness service for baseline B, with replayable evidence."""
import argparse
import os
from pathlib import Path
import time
import traceback
import numpy as np
from scripts.curobo_arx.data_factory.common import read, write, sha, verify_files
from scripts.curobo_arx.data_factory.watchdog import exclusive_stop
from scripts.curobo_arx.graspness_teacher.backend import Graspness, NoGrasps, sample_cloud
from training.nearfield_arx.projection import Projector
from .main48_spec import GEOMETRY, point_cloud, select_candidate


class WristGeometry:
    def __init__(self, reg, directory):
        if reg['geometry_baseline']['contract'] != GEOMETRY: raise ValueError('Geometry contract changed')
        self.reg = reg; self.model = Graspness(reg,directory)
        self.projector = Projector(reg['learned_policy']['robot_usd'])

    def infer(self, request, directory):
        from graspnetAPI import GraspGroup
        directory = Path(directory); directory.mkdir(parents=True,exist_ok=False)
        verify_files(request['files']); began = time.monotonic()
        row = read(request['observation']); depth = np.load(request['depth'],allow_pickle=False)
        if abs(row['frame_time'] - row['capture_sim_time']) > 1e-4:
            raise ValueError('Stale wrist observation')
        masks, projection, _ = self.projector.project(request['asset'],row,depth)
        np.savez_compressed(directory/'masks.npz',**masks)
        points, target = point_cloud(depth,row['intrinsics'],masks)
        np.save(directory/'context_points.npy',points); np.save(directory/'target_points.npy',target)
        rejected = None; raw = np.empty((0,17),dtype=np.float32); filtered = raw.copy()
        colliding = np.empty(0,dtype=bool); network_seconds = 0.
        if not projection['target_visibility_pass']: rejected = 'target_visibility_rejected'
        elif not len(points): rejected = 'empty_wrist_cloud'
        else:
            m = self.model; sampled = sample_cloud(points,m.config['num_points'],m.config['seed'])
            np.save(directory/'input_points.npy',sampled)
            batch = m.collate([{'point_clouds':sampled,'coors':sampled/m.config['voxel_size_m'],
                               'feats':np.ones_like(sampled,dtype=np.float32)}])
            batch = {k:v.to('cuda:0') for k,v in batch.items()}
            m.torch.cuda.synchronize(); start = time.monotonic()
            try:
                with m.torch.no_grad(): raw = m.decode(m.net(batch))[0].detach().cpu().numpy()
            except NoGrasps: rejected = 'no_graspable_points'
            m.torch.cuda.synchronize(); network_seconds = time.monotonic()-start
            if len(raw):
                group = GraspGroup(raw)
                detector = m.collision(points,voxel_size=m.config['collision_voxel_size_m'])
                colliding = detector.detect(group,approach_dist=.05,collision_thresh=m.config['collision_threshold'])
                group = group[~colliding]
                if len(group): group = group.nms().sort_by_score()
                filtered = group.grasp_group_array
                if not len(filtered): rejected = 'point_cloud_collision_rejected_all'
        np.save(directory/'raw_predictions.npy',raw); np.save(directory/'collision_mask.npy',colliding)
        np.save(directory/'filtered_predictions.npy',filtered)
        selected = select_candidate(filtered,row['camera_world_optical'],request['G0_world'],target)
        write(directory/'candidate_selection.json',selected)
        verify_files(request['files'])
        return {**selected,'projection':projection,'rejected':rejected or (None if selected['selected'] else 'no_target_candidate_in_G0_neighborhood'),
            'network_seconds':network_seconds,'total_seconds':time.monotonic()-began,
            'context_points':len(points),'target_points':len(target),
            'checkpoint_sha256':self.model.config['checkpoint_sha256'],
            'files':{str(p):sha(p) for p in directory.iterdir() if p.is_file()}}


def serve(campaign):
    campaign=Path(campaign).resolve(); reg=read(campaign/'registration.json'); directory=campaign/'workers/geometry'
    try:
        verify_files({**reg['execution_source_files'],**reg['dependencies']})
        model=WristGeometry(reg,directory)
        warm=[]
        for i,request in enumerate(reg['geometry_baseline']['warmups']):
            warm.append(model.infer(request,directory/f'warmup_{i}'))
        write(directory/'warmup.json',{'calls':warm,'physical_trials':0})
        write(directory/'ready.json',{'registration_sha256':sha(campaign/'registration.json'),
            'gpu_uuid':reg['geometry_baseline']['gpu_uuid'],'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'checkpoint_sha256':reg['graspness']['checkpoint_sha256'],'warmup_calls':len(warm),'wall':time.time()})
        while not (directory/'STOP').exists() and not (campaign/'STOP').exists() and time.time()<reg['deadline_unix']:
            for queued in sorted((directory/'queue').glob('*.json')):
                done=directory/'done'/queued.name
                if done.exists(): continue
                request_path=Path(read(queued)['request']); request=read(request_path)
                if request['registration_sha256']!=sha(campaign/'registration.json') or request['method']!='B':
                    raise ValueError('Foreign geometry request')
                spec=next(t for t in reg['seed_trials'] if t['trial_id']==request['trial_id'])
                if spec['method']!='B' or spec['G0_world']!=request['G0_world']: raise ValueError('Geometry prior changed')
                ident=request['request_id']; reply_path=request_path.with_name(ident+'_reply.json')
                if reply_path.exists(): raise RuntimeError('No geometry overwrite/retry')
                result=model.infer(request,request_path.with_name(ident+'_evidence'))
                reply={**result,'ok':True,'method':'B','request_id':ident,'request_sha256':sha(request_path),
                       'registration_sha256':request['registration_sha256']}
                write(reply_path,reply); write(done,{'request':str(request_path),'reply_sha256':sha(reply_path)})
            time.sleep(.05)
        verify_files({**reg['execution_source_files'],**reg['dependencies']})
        write(directory/'exit.json',{'pass':True,'wall':time.time()})
    except BaseException:
        error=traceback.format_exc();write(directory/'FAULT.json',{'error':error,'wall':time.time()})
        exclusive_stop(campaign/'STOP',{'reason':'geometry_service_fault','error':error});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('campaign');serve(parser.parse_args().campaign)
