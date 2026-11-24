"""ARX object-centric wrist data and Cartesian action conventions."""
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .interaction_contract import SEMANTICS, FIXED
except ImportError:
    from interaction_contract import SEMANTICS, FIXED

POSITION_SCALE_M = SEMANTICS['position_scale_m']
ROTATION_SCALE_RAD = SEMANTICS['rotation_scale_rad']
ACTION_PERIOD_S = FIXED['sample_period_seconds']
STATE_CONVENTIONS=('object_relative_v1','grasp_goal_world_error_v1')


def smooth_segment(start, goal):
    """Quintic interpolation under the existing ARX playback derivative bounds."""
    delta=np.asarray(goal,dtype=np.float64)-np.asarray(start,dtype=np.float64)
    maximum=float(np.abs(delta).max())
    duration=max(.2,1.875*maximum/1.5,(5.774*maximum/5)**.5,(60*maximum/100)**(1/3))
    duration=np.ceil(duration/.025)*.025
    times=np.linspace(0,duration,max(3,int(np.ceil(duration/.025))+1))
    u=times/duration;s=10*u**3-15*u**4+6*u**5
    return np.asarray(start)[None]+s[:,None]*delta[None],times


def relative_state(snapshot):
    ee=np.asarray(snapshot['link_world']['right_eef_link'],dtype=float)
    obj=np.asarray(snapshot['objects']['test_object']['world'],dtype=float)
    positions=dict(zip(snapshot['joint_names'],snapshot['joint_positions']))
    # Match the released LiLo implementation, including world-position difference.
    return np.r_[ee[:3,3]-obj[:3,3],Rotation.from_matrix(obj[:3,:3].T@ee[:3,:3]).as_rotvec(),
                 positions['right_joint27'],positions['right_joint28']]


def encode_state(snapshot,convention='object_relative_v1',grasp_world=None):
    """Versioned 8D policy input shared by dataset construction and rollout.

    Goal version: EEF minus planned grasp position in world axes, and the
    left/world rotation deviation R_eef @ R_grasp.T. Corrective world actions
    have the opposite sign near the goal. No future measured grasp is used.
    """
    if convention=='object_relative_v1':return relative_state(snapshot)
    if convention!='grasp_goal_world_error_v1':raise ValueError('Unknown ARX policy state convention')
    if grasp_world is None:raise ValueError('Goal-conditioned state requires the selected planned grasp')
    goal=np.asarray(grasp_world,dtype=float)
    if goal.shape!=(4,4) or not np.isfinite(goal).all() or not np.allclose(goal[3],[0,0,0,1]):
        raise ValueError('Expected a finite 4x4 planned grasp transform')
    if not np.allclose(goal[:3,:3].T@goal[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(goal[:3,:3])<0:
        raise ValueError('Planned grasp rotation is not rigid')
    ee=np.asarray(snapshot['link_world']['right_eef_link'],dtype=float)
    q=dict(zip(snapshot['joint_names'],snapshot['joint_positions']))
    state=np.r_[ee[:3,3]-goal[:3,3],Rotation.from_matrix(ee[:3,:3]@goal[:3,:3].T).as_rotvec(),
        q['right_joint27'],q['right_joint28']]
    if not np.isfinite(state).all():raise ValueError('Nonfinite goal-conditioned ARX state')
    return state


def action_target(ee, action):
    """Robosuite-style world delta pose; returns an ARX EEF goal, not joint targets."""
    ee=np.asarray(ee,dtype=float);action=np.asarray(action,dtype=float)
    if ee.shape!=(4,4) or action.shape!=(7,) or not np.isfinite(action).all():
        raise ValueError('Expected one finite seven-dimensional Cartesian/gripper action')
    if (np.abs(action[:6])>1+1e-6).any():
        raise ValueError('Policy action exceeds the declared normalized action space')
    target=ee.copy();target[:3,3]+=POSITION_SCALE_M*action[:3]
    target[:3,:3]=Rotation.from_rotvec(ROTATION_SCALE_RAD*action[3:6]).as_matrix()@ee[:3,:3]
    # Existing ARX API: -3.2 opens, +0.1 closes. Dataset convention: 1=open.
    gripper=-3.2 if action[6]>=.5 else .1
    return target,gripper


def teacher_action(first, second):
    a=np.asarray(first['snapshot']['link_world']['right_eef_link'])
    b=np.asarray(second['snapshot']['link_world']['right_eef_link'])
    command=first.get('teacher_command') or {}
    grip=1. if command.get('gripper_target',-3.2)<-1 else 0.
    return np.r_[(b[:3,3]-a[:3,3])/POSITION_SCALE_M,
        Rotation.from_matrix(b[:3,:3]@a[:3,:3].T).as_rotvec()/ROTATION_SCALE_RAD,grip]


def policy_observation(observation, convention, grasp_world=None):
    """One codec for recorded training frames and live ARX policy frames."""
    from PIL import Image
    with Image.open(observation['rgb_path']) as original:
        image, boxes = mask_distractors(original, observation)
    return image, boxes, encode_state(observation['snapshot'], convention, grasp_world)


def target_projection(observation, size):
    world=np.asarray(observation['snapshot']['objects']['test_object']['world'])
    cam=np.asarray(observation['camera_world_optical']);K=np.asarray(observation['intrinsics'])
    center=np.linalg.inv(cam)@world[:,3]
    if center[2]<=0:return None
    return (K@center[:3])[:2]/center[2]


def mask_distractors(image, observation):
    """Mask projected non-target object rectangles; retain target/background/robot."""
    image=np.asarray(image,dtype=np.uint8).copy();height,width=image.shape[:2]
    world_to_camera=np.linalg.inv(np.asarray(observation['camera_world_optical']))
    K=np.asarray(observation['intrinsics'])
    sizes={'red_cube':[.04]*3,'blue_cylinder':[.06,.06,.07],'green_sphere':[.07]*3}
    boxes=[]
    for name,size in sizes.items():
        info=observation['snapshot']['objects'].get(name)
        if info is None:continue
        half=np.asarray(size)/2
        corners=np.array([[x,y,z,1] for x in (-half[0],half[0]) for y in (-half[1],half[1]) for z in (-half[2],half[2])])
        points=(world_to_camera@np.asarray(info['world'])@corners.T).T[:,:3]
        if (points[:,2]<=.01).any():continue
        uv=points@K.T;uv=uv[:,:2]/uv[:,2,None]
        low=np.maximum(np.floor(uv.min(0)).astype(int),0)
        high=np.minimum(np.ceil(uv.max(0)).astype(int),[width,height])
        if (high<=low).any():continue
        image[low[1]:high[1],low[0]:high[0]]=0
        boxes.append({'object':name,'xyxy':[*low.tolist(),*high.tolist()]})
    return image,boxes
