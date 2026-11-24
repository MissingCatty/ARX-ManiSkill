"""Advisor36 pure visual policy contract. No target geometry or G0 input."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .advisor32_rules import rigid,parse_prediction,goal_from_delta

VERSION='arx_rulefree_full6d_explicit_grasp_v1'
TRANSLATION_ANOMALY_M=.120
ROTATION_ANOMALY_DEG=45.


def interpret(current,text):
    parts=parse_prediction('D',text)
    if parts is None:return {'kind':'close','closure_origin':'explicit_model_GRASP','goal_world':None}
    dp,dr=parts
    # Reject numeric anomalies; never clip, synthesize an axis, or close instead.
    if np.linalg.norm(dp)>TRANSLATION_ANOMALY_M+1e-12 or np.linalg.norm(dr)>np.deg2rad(ROTATION_ANOMALY_DEG)+1e-12:
        return {'kind':'reject','reason':'model_numeric_anomaly','goal_world':None}
    return {'kind':'move','goal_world':goal_from_delta(current,dp,dr).tolist(),
        'raw_translation_m':dp.tolist(),'raw_rotation_rad':dr.tolist(),
        'all_six_values_preserved':True,'needs_original_IK_and_collision_checks':True}


def instruction(scale):
    return ('<image><image>\nGrasp the orange-outlined object using the ARX wrist views. '
        'Image1 is wrist RGB. Image2 is fixed metric viridis optical-Z depth from '
        f'{scale["near_m"]*1000:g} to {scale["far_m"]*1000:g} mm; outside values saturate and invalid depth is black. '
        'Orange marks the target; cyan and purple mark the two fingers. Background is dimmed. '
        'REFINE is the full remaining translation and rotation from the CURRENT TCP to a grasp pose, '
        'in CURRENT TCP axes. Rebuild from this observed pose; never accumulate a previous target. '
        'GRASP requests closure at the current pose. '
        'Return one REFINE command with three signed mm values to3 decimals and three rotation-vector degree values to4 decimals, or GRASP.\n'
        'Local state: {"stopped":true}')


def example(images,scale,output='GRASP'):
    if len(images)!=2:raise ValueError('Exactly two synchronized wrist views required')
    parse_prediction('D',output)
    return {'instruction':instruction(scale),'input':'','output':output,'images':list(map(str,images))}


def residual(current,goal):
    d=np.linalg.inv(rigid(current))@rigid(goal)
    return d[:3,3],Rotation.from_matrix(d[:3,:3]).as_rotvec()


def token(current,goal,grasp=False):
    dp,dr=residual(current,goal)
    if grasp:
        if np.linalg.norm(dp)>.004+1e-8 or np.linalg.norm(dr)>np.deg2rad(3)+1e-8:
            raise ValueError('GRASP label outside registered core')
        return 'GRASP'
    return 'REFINE '+' '.join([*(f'{v*1000:.3f}' for v in dp),*(f'{v:.4f}' for v in np.rad2deg(dr))])


def boundary_poses(goal,mode='coupled_xyz'):
    g=rigid(goal);rows=[{'case':'center','goal_world':g.tolist(),'translation_m':[0.,0.,0.],'rotation_degrees':[0.,0.,0.]}]
    if mode not in ('coupled_xyz','lateral_xy_yaw'):raise ValueError('Unregistered six-point boundary')
    for axis in range(3):
        for sign in (-1,1):
            d=np.eye(4);dp=np.zeros(3);dr=np.zeros(3)
            if mode=='coupled_xyz' or axis<2:dp[axis]=sign*.004
            if mode=='coupled_xyz' or axis==2:dr[axis]=sign*3.
            d[:3,3]=dp;d[:3,:3]=Rotation.from_rotvec(np.deg2rad(dr)).as_matrix()
            rows.append({'case':'XYZ'[axis]+('_minus' if sign<0 else '_plus'),'goal_world':(g@d).tolist(),
                'translation_m':dp.tolist(),'rotation_degrees':dr.tolist()})
    return rows


def canonical_release(results):
    if len(results)!=7 or len({r['boundary_case'] for r in results})!=7:return False
    return all(r.get('completed') is True and r.get('success') is True
               and r.get('record_integrity_pass') is True and r.get('seed_pose_verified') is True for r in results)
