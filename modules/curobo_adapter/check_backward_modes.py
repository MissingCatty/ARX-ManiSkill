"""Separate position/quaternion FK gradients with Jacobian output on and off."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from urdf_reference import URDFChain, NAMES, HERE


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['old','new'],required=True);args=parser.parse_args()
    if args.backend=='old':sys.path.insert(0,str(HERE.parent/'curobo/source'))
    import curobo
    import torch
    from curobo.kinematics import Kinematics,KinematicsCfg
    from curobo.types import JointState
    root=HERE/'research/20261117_01';out=root/('backward_modes_'+args.backend);out.mkdir(exist_ok=False)
    trial=HERE.parent/'maniskill/outputs/arx_native_postprocessed_retry11_20261117_01/trials/B0001_S1_open_003_cracker_box'
    cfg=KinematicsCfg.from_robot_yaml_file(str(trial/'plan/arx.yml'))
    model=URDFChain(HERE.parent/'arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf')
    q_array=np.load(root/'jacobian_new/arrays.npz')['q']
    p_weights=np.array([.3,-.2,.7]);q_weights=np.array([.2,.4,-.3,.1])
    position_expected=np.stack([model.fk(row,jacobian=True)[1][:3].T@p_weights for row in q_array])
    def scalar(row):
        t,_=model.fk(row);xyzw=Rotation.from_matrix(t[:3,:3]).as_quat();return float(np.r_[xyzw[3],xyzw[:3]]@q_weights)**2
    rotation_expected=np.stack([[(scalar(row.astype(float)+np.eye(6)[i]*1e-6)-scalar(row.astype(float)-np.eye(6)[i]*1e-6))/2e-6 for i in range(6)] for row in q_array])
    results=[];arrays={'q':q_array,'position_expected':position_expected,'quaternion_expected':rotation_expected}
    for enabled in [False,True]:
        robot=Kinematics(cfg,compute_jacobian=enabled)
        for part in ['position','quaternion','combined']:
            q=torch.tensor(q_array,dtype=torch.float32,device='cuda',requires_grad=True)
            kin=robot.compute_kinematics(JointState.from_position(q,joint_names=NAMES))
            p=(kin.tool_poses.position.reshape(len(q_array),3)*torch.tensor(p_weights,dtype=torch.float32,device='cuda')).sum()
            r=(kin.tool_poses.quaternion.reshape(len(q_array),4)*torch.tensor(q_weights,dtype=torch.float32,device='cuda')).sum(-1).square().sum()
            loss=p if part=='position' else r if part=='quaternion' else p+r
            loss.backward();actual=q.grad.detach().cpu().numpy().copy()
            expected=position_expected if part=='position' else rotation_expected if part=='quaternion' else position_expected+rotation_expected
            error=float(np.max(np.abs(actual-expected)));arrays[f'{enabled}_{part}']=actual
            results.append({'compute_jacobian':enabled,'loss_part':part,'max_absolute_error':error,'passed':error<1e-5})
    np.savez(out/'arrays.npz',**arrays)
    result={'backend':args.backend,'source':curobo.__file__,'configurations':len(q_array),'checks':results,
            'all_passed':all(r['passed'] for r in results),'native_ik_calls':0,'physical_trials':0}
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
