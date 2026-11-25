"""Versioned model-frame and contact settings for the repaired 200-scene run."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

PROJECT=Path(__file__).resolve().parents[2]
MS=PROJECT.parent/'maniskill'
sys.path.insert(0,str(PROJECT.parent/'graspgenx'))
from arx_frames import VERSION, ORIGIN_BACK_M, CONTACT_OFFSET_M, TCP_M, R_EEF_GENERATOR, T_EEF_GENERATOR, generator_to_eef, generator_to_contact


def descriptor():
    sys.path.insert(0,str(MS))
    from graspgenx_partial_geometry import gripper_descriptor
    physical=gripper_descriptor()
    result=deepcopy(physical)
    result['physical_eef_parameters']=deepcopy(physical['parameters'])
    result['physical_eef_generator_transform']=physical['T_eef_generator']
    result['T_eef_generator']=T_EEF_GENERATOR.copy()
    for name in ['offset_open','offset_mid']:
        result['parameters'][name]=np.asarray(result['parameters'][name]).copy()
        result['parameters'][name][2]+=ORIGIN_BACK_M
    result['parameters']['fingertip_depth']+=ORIGIN_BACK_M
    result.update(repair_contract=VERSION,model_origin_behind_eef_m=ORIGIN_BACK_M,
        tcp_m=TCP_M,contact_offset_m=CONTACT_OFFSET_M,
        robot_or_physics_modified=True,robot_geometry_modified=False,
        physics_change='environment contact_offset .02m -> .001m; original geometry, mass and drives retained',
        conditioner='Same measured ARX finger cavity expressed at a virtual origin 100mm behind the EEF')
    return result


def simulation_config(config):
    result=deepcopy(config)
    result.scene_config.contact_offset=CONTACT_OFFSET_M
    return result


def make_candidates(directory, *, background=False):
    from arx_project.artifacts import publish_candidate_pool
    from common import paths,read,save
    paths()
    import graspcl_pipeline as display
    directory=Path(directory);raw=directory/'graspgenx_raw'
    inference=read(raw/'inference.json')
    assert inference['repair_contract']==VERSION
    assert np.allclose(inference['T_eef_generator'],T_EEF_GENERATOR,atol=0,rtol=0)
    poses=np.load(raw/'grasps_generator_world.npy',allow_pickle=False)
    scores=np.load(raw/'scores.npy',allow_pickle=False)
    tags=inference['branch_tags']
    assert poses.shape==(len(scores),4,4) and len(tags)==len(scores)
    assert np.isfinite(poses).all() and np.isfinite(scores).all()
    candidates=[]
    for i in np.argsort(-scores,kind='stable'):
        eef=generator_to_eef(poses[i]);contact=generator_to_contact(poses[i])
        candidates.append({'pose':contact,'width':.088,'score':float(scores[i]),
            'point1':contact[:3,3]-.044*contact[:3,1],
            'point2':contact[:3,3]+.044*contact[:3,1],
            'metadata':{'origin':'graspgenx','source_generator_index':int(i),
                'source_branch':tags[i],'source_model_score':float(scores[i]),
                'source_generator_pose_world':poses[i],'source_eef_pose_world':eef,
                'T_eef_generator':T_EEF_GENERATOR,'repair_contract':VERSION,
                'model_origin_behind_eef_m':ORIGIN_BACK_M,'storage_tcp_m':TCP_M,
                'pose_refinement_applied':False,'source_network_depth_m':0.}})
    dest=directory/'candidates';dest.mkdir()
    persist_pool = (lambda payload: publish_candidate_pool(directory, payload)) if background else (lambda payload: save(dest/'candidate_pool.json', payload))
    (directory/'graspcl').symlink_to(dest,target_is_directory=True)
    persist_pool({'candidates':candidates,'protocol':VERSION,
        'raw_count':len(poses),'retained_count':len(candidates),'graspcl_invoked':False,
        'native_score_order':True,'topdown_filter':False,'orientation_changes':0,
        'posthoc_grasp_position_changes':0,'model_reference_frame_conversion':True})
    camera=np.asarray(read(directory/'state.json')['camera_world_optical'])
    rows=np.asarray([display.display_row(c,camera) for c in candidates]).reshape(-1,17)
    np.save(directory/'raw_grasps.npy',rows,allow_pickle=False)
    np.save(dest/'raw_grasps.npy',rows,allow_pickle=False)
    save(dest/'network_candidates.json',{'model_name':'RaySt3R + GraspGenX',
        'score_label':'GraspGenX native score','count':len(candidates),
        'repair_contract':VERSION,'graspcl_invoked':False})
    return len(candidates)
