"""Cross-check ARX CUDA Jacobians and FK gradients against independent URDF math."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from urdf_reference import URDFChain, NAMES, HERE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["old", "new"], required=True)
    args = parser.parse_args()
    if args.backend == "old":
        sys.path.insert(0, str(HERE.parent / "curobo/source"))
    import curobo
    import torch
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState
    root = HERE / "research/20261117_01"
    output = root / ("jacobian_" + args.backend)
    output.mkdir(exist_ok=False)
    trial = HERE.parent / "maniskill/outputs/arx_native_postprocessed_retry11_20261117_01/trials/B0001_S1_open_003_cracker_box"
    cfg = KinematicsCfg.from_robot_yaml_file(str(trial / "plan/arx.yml"))
    robot = Kinematics(cfg, compute_jacobian=True)
    model = URDFChain(HERE.parent / "arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf")
    known = json.loads((root / "known_goals.json").read_text())
    qs = [item["q_witness"] for item in known["B0001_S1_open_003_cracker_box"]]
    qs += [r["roots"][0]["q"] for r in json.loads((root / "cpu_roots/results.json").read_text())]
    q = torch.tensor(qs,dtype=torch.float32,device="cuda",requires_grad=True)
    state = robot.compute_kinematics(JointState.from_position(q,joint_names=NAMES))
    native = state.tool_jacobians.detach().cpu().numpy().reshape(len(qs),6,6)
    cpu = np.stack([model.fk(row,jacobian=True)[1] for row in q.detach().cpu().numpy()])
    jacobian_error = float(np.max(np.abs(native-cpu)))
    position_weight = torch.tensor([.3,-.2,.7],dtype=torch.float32,device="cuda")
    quaternion_weight = torch.tensor([.2,.4,-.3,.1],dtype=torch.float32,device="cuda")
    position = state.tool_poses.position.reshape(len(qs),3)
    quaternion = state.tool_poses.quaternion.reshape(len(qs),4)
    loss = ((position*position_weight).sum(-1)+(quaternion*quaternion_weight).sum(-1).square()).sum()
    loss.backward()
    native_gradient = q.grad.detach().cpu().numpy()
    def scalar(row: np.ndarray) -> float:
        t,_=model.fk(row)
        xyzw=Rotation.from_matrix(t[:3,:3]).as_quat();wxyz=np.r_[xyzw[3],xyzw[:3]]
        return float(t[:3,3]@np.array([.3,-.2,.7])+(wxyz@np.array([.2,.4,-.3,.1]))**2)
    cpu_gradient=[]
    for row in q.detach().cpu().numpy().astype(float):
        cpu_gradient.append([(scalar(row+np.eye(6)[i]*1e-6)-scalar(row-np.eye(6)[i]*1e-6))/2e-6 for i in range(6)])
    gradient_error=float(np.max(np.abs(native_gradient-np.asarray(cpu_gradient))))
    np.savez(output/'arrays.npz',q=q.detach().cpu().numpy(),native_jacobian=native,
             cpu_jacobian=cpu,native_gradient=native_gradient,cpu_gradient=cpu_gradient)
    result={'backend':args.backend,'source':curobo.__file__,'configurations':len(qs),
            'max_jacobian_absolute_error':jacobian_error,'max_fk_gradient_absolute_error':gradient_error,
            'passed':jacobian_error<1e-5 and gradient_error<1e-5,'native_ik_calls':0,'physical_trials':0}
    (output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    assert result['passed'], 'ARX kinematic derivative mismatch; inspect saved arrays'


if __name__=='__main__':
    main()
