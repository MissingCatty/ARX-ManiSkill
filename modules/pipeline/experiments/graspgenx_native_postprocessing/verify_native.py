"""Compare the original detector on real, saved FP32-origin candidate frames."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from arx_project.layout import ROOT,read_json,write_json,now,sha256
from modules.curobo.dual_arm.model import ArmModel
from modules.graspgenx.native_postprocessing.kernels import native_detect
from modules.graspgenx.native_postprocessing.processor import load_raw
from modules.graspgenx.postprocessing.geometry import GripperGeometry,RobotPointFilter
from modules.graspgenx.postprocessing.filters import CandidatePostprocessor


def run():
    sys.path.insert(0,str(ROOT/'modules/zerograsp/server/vendor/zerograsp_mainline_minimal'))
    from zerograsp.utils.collision_detector import ModelFreeCollisionDetector
    root=ROOT/'tmp/graspgenx_native_postprocessing/offline69_20261127_01'
    reg=json.loads((root/'registration.json').read_text());model=ArmModel()
    cloud=CandidatePostprocessor({s:GripperGeometry(s) for s in ['right','left']},RobotPointFilter(),[])
    axis=np.array([[0,0,1],[0,1,0],[-1,0,0.]])
    torch.set_num_threads(4);results=[]
    ids=['M0001_S1_open_apple','M0020_S1_open_donut','M0074_S4_clutter_scissors']
    for case in [c for c in reg['cases'] if c['id'] in ids]:
        p=Path(case['source_directory']);state=read_json(p/'state.json')
        payload=json.loads((p/'candidates/candidate_pool.json').read_text());poses,_,_=load_raw(p,payload);poses=poses[:24]
        observed,target,_=cloud.observations(state,np.load(p/'camera/depth_m.npy'),
            np.asarray(Image.open(p/'camera/mask.png'))>0,np.load(p/'completed_points_world.npy'),model.frames(state))
        points=np.concatenate([target,observed]);rows=np.zeros((len(poses),17));rows[:,1]=.088;rows[:,2]=.02;rows[:,3]=.04
        rows[:,4:13]=(poses[:,:3,:3]@axis.T).reshape(-1,9);rows[:,13:16]=poses[:,:3,3]-.04*poses[:,:3,2]
        detector=ModelFreeCollisionDetector(torch.tensor(target,dtype=torch.float64),
            torch.zeros((len(target),3),dtype=torch.float64),torch.tensor(observed,dtype=torch.float64));detector.batch_size=4
        bad,dw,depth=detector.detect(SimpleNamespace(grasp_group_array=rows));actual=native_detect(poses,points)
        np.testing.assert_array_equal(bad,actual['bad'])
        np.testing.assert_allclose(.088+dw,actual['widths'],atol=1e-7,rtol=0)
        np.testing.assert_allclose(depth,actual['depths'],atol=1e-7,rtol=0)
        result={'case':case['id'],'queries':len(poses),'observed':len(observed),'target':len(target),'passed':True,
            'max_width_error_m':float(np.max(np.abs(.088+dw-actual['widths']))),
            'max_depth_error_m':float(np.max(np.abs(depth-actual['depths'])))}
        results.append(result);print(result,flush=True)
    write_json(root/'frozen_detector_parity.json',{'at':now(),'passed':True,'queries':72,
        'collision_masks_exactly_equal':True,'size_tolerance_m':1e-7,'saved_source_rotations_dtype':'float32',
        'cases':results,'verification_source_sha256':sha256(__file__)})


if __name__=='__main__':run()
