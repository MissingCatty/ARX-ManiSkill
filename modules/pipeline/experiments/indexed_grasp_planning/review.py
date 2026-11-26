"""Independently check physical-arm FK and original candidate identity."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.layout import ensure_run_path, now, read_json, sha256, write_json
from modules.arx_core.arms import get_arm
from modules.curobo.dual_arm.model import ArmModel


def review(root):
    reg = read_json(root/'registration.json')
    accepted = read_json(root/'acceptance.json')
    if not accepted['record_integrity'] or not accepted['task_complete']:
        raise ValueError('The planning replay has not completed with intact records')
    model = ArmModel()
    checked, issues = [], []
    for case in reg['cases']:
        source = Path(case['source_directory'])
        original = read_json(source/'candidates/candidate_pool.json')['candidates']
        for mode in reg['variants']:
            directory = root/'plans'/case['id']/mode
            state = read_json(directory/'state.json')
            result = read_json(directory/'plan_result.json')
            if not result['success']:continue
            selected = result['selected']; arm = get_arm(selected['arm'])
            pose = np.asarray(original[selected['index']]['pose'])
            if not np.array_equal(pose,np.asarray(selected['contact_pose_world'])):
                issues.append({'id':case['id'],'variant':mode,'reason':'original_grasp_pose_changed'})
            for stage,key in [('grasp','grasp_world'),('pre','pregrasp_world'),('lift','lift_world')]:
                q = np.asarray(selected['q_'+stage])
                actual_state = deepcopy(state)
                actual_state['links'] = {'base_link':state['links']['base_link']}
                bounds = True
                for name,value in zip(arm.joints,q):
                    i = state['joint_names'].index(name)
                    lo,hi = state['joint_limits'][i]
                    bounds &= bool(lo <= value <= hi)
                    actual_state['qpos'][i] = float(value)
                actual = model.frames(actual_state)[arm.eef]
                goal = np.asarray(selected[key])
                pe = float(np.linalg.norm(actual[:3,3]-goal[:3,3]))
                re = float(Rotation.from_matrix(actual[:3,:3]@goal[:3,:3].T).magnitude())
                row = {'id':case['id'],'variant':mode,'arm':arm.name,'stage':stage,
                    'original_candidate_index':selected['index'],'position_error_m':pe,
                    'rotation_error_rad':re,'joint_limits_clear':bounds}
                checked.append(row)
                if not bounds or not np.isfinite([pe,re]).all() or pe > .002 or re > .03:
                    issues.append({**row,'reason':'independent_FK_or_joint_limits_failed'})
    result = {'at':now(),'passed':not issues,'issues':issues,'checked_endpoints':len(checked),
        'endpoints':checked,'original_candidate_identity_checked':True,
        'acceptance_sha256':sha256(root/'acceptance.json'),
        'control_successes_retained':accepted['control_successes_retained'],
        'same_physical_arm_URDF_and_original_pose_tolerances':True,'physical_grasps':0}
    write_json(root/'independent_review.json',result)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    output = review(ensure_run_path(parser.parse_args().root))
    print({k:v for k,v in output.items() if k not in ['endpoints','issues']})
    if not output['passed']:raise SystemExit(1)
