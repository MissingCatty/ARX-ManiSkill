"""Pinned official create_detector API, using actual ARX camera depth in metres."""
from argparse import Namespace
import os
from pathlib import Path
import sys
import numpy as np
from scripts.curobo_arx.data_factory.common import read,sha,write
from scripts.curobo_arx.data_factory.geometry import grasp_pool,handoff
from scripts.curobo_arx.data_factory.neural_input import cloud_input


def convert_candidates(rows, row, asset, config):
    rows=np.asarray(rows,dtype=float)
    if rows.ndim!=2 or rows.shape[1]!=17:raise ValueError('Expected official GraspNet [N,17] output')
    valid=np.isfinite(rows).all(axis=1)&(rows[:,1]>0)&(rows[:,1]<=config['max_gripper_width_m'])
    rejected_width=int((~valid).sum());rows=rows[valid]
    rows=rows[np.argsort(-rows[:,0],kind='stable')][:max(32,config['candidate_limit']*4)]
    pool=grasp_pool(rows,row['camera_world_optical'],row['snapshot']['objects']['test_object']['world'],
        asset['size_m'],maximum=len(rows))
    pool['candidates']=sorted(pool['candidates'],key=lambda g:-g['score'])[:config['candidate_limit']]
    for goal in pool['candidates']:
        goal['candidate_source']={'name':'Official AnyGrasp','sdk_revision':config['sdk_revision'],
            'checkpoint_sha256':config.get('checkpoint_sha256'),'input':'Actual ARX head linear-Z depth and calibrated optical frame'}
        goal['pregrasp_world']=handoff(goal['grasp_world'],config['pregrasp_distance_mm'],0,0)['world']
        goal['pregrasp_distance_mm']=config['pregrasp_distance_mm']
        goal['frame_mapping']='Existing frozen ARX GraspNet convention: approach X to EEF Z, closing Y retained, original 40 mm tool offset'
    pool.update(candidate_source='Official AnyGrasp',ranking='NMS then decreasing neural score; bounded top candidates',
        invalid_or_width_rejections=rejected_width,pregrasp_distance_mm=config['pregrasp_distance_mm'])
    return pool


class AnyGrasp:
    def __init__(self, registration, directory):
        self.config=registration['anygrasp'];self.directory=Path(directory)
        if self.config['api']!='create_detector_20260704':raise ValueError('Unsupported explicit AnyGrasp API')
        if sha(self.config['checkpoint'])!=self.config['checkpoint_sha256']:raise RuntimeError('AnyGrasp checkpoint changed')
        sdk=Path(self.config['sdk_root'])/'grasp_detection/gsnet_versions'
        sys.path.insert(0,str(sdk))
        runtime=self.directory/'runtime';runtime.mkdir()
        (runtime/'license').symlink_to(self.config['license_directory'],target_is_directory=True)
        os.chdir(runtime)
        import gsnet
        self.detector=gsnet.create_detector(Namespace(checkpoint_path=self.config['checkpoint'],
            max_gripper_width=self.config['max_gripper_width_m'],gripper_height=self.config['detector_gripper_height_m']))
        if self.detector is None:raise RuntimeError('Official AnyGrasp detector initialization/license validation failed')

    def handle(self, request):
        if request['action']!='infer':raise ValueError('Unsupported AnyGrasp request')
        directory=Path(request['directory']);row=read(directory/'head_observation.json');asset=request['asset']
        if row['object_id']!=asset['object_id'] or row['camera_role']!='head':raise ValueError('AnyGrasp input identity/camera mismatch')
        if abs(row['frame_time']-row['capture_sim_time'])>1e-4:raise ValueError('Stale AnyGrasp camera frame')
        try:points,region=cloud_input(np.load(directory/'head_depth.npy',allow_pickle=False),row,asset)
        except ValueError as error:return {'candidates':[],'rejected':str(error),'candidate_source':'Official AnyGrasp'}
        np.save(directory/'anygrasp_points.npy',points);np.save(directory/'anygrasp_region.npy',region)
        grasps=self.detector.get_grasp(points,{'dense_grasp':False,'collision_detection':True,
            'region_steering':region,'approach_steering':None,'approach_thresh':float(np.pi)})
        if grasps is None:return {'candidates':[],'rejected':'Official AnyGrasp returned no grasp','candidate_source':'Official AnyGrasp'}
        grasps=grasps.nms().sort_by_score()
        rows=np.asarray(grasps.grasp_group_array,dtype=float)
        np.save(directory/'anygrasp_raw_grasps.npy',rows)
        pool=convert_candidates(rows,row,asset,self.config)
        write(directory/'pool.json',pool)
        return pool
