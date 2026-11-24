"""Compare the actual LM pose objective gradient with finite differences."""
from __future__ import annotations

import argparse,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from urdf_reference import URDFChain,NAMES,HERE


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['old','new'],required=True);args=parser.parse_args()
    if args.backend=='old':sys.path.insert(0,str(HERE.parent/'curobo/source'))
    import torch,curobo
    from curobo._src.solver.seed_ik.seed_ik_solver_cfg import SeedIKSolverCfg
    from curobo._src.solver.seed_ik.seed_ik_solver import SeedIKSolver
    from curobo.types import GoalToolPose
    root=HERE/'research/20261117_01';out=root/('ik_cost_gradient_'+args.backend);out.mkdir(exist_ok=False)
    trial=HERE.parent/'maniskill/outputs/arx_native_postprocessed_retry11_20261117_01/trials/B0001_S1_open_003_cracker_box'
    state=json.loads((trial/'state.json').read_text());bounds=np.array([state['joint_limits'][state['joint_names'].index(n)] for n in NAMES])
    solver=SeedIKSolver(SeedIKSolverCfg.create(robot=str(trial/'plan/arx.yml'),num_seeds=1,use_cuda_graph=False,use_backward=True))
    calc=solver.error_calculator;calc.setup_batch_tensors(1,1)
    model=URDFChain(HERE.parent/'arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf')
    q0=bounds.mean(axis=1);qtarget=q0+np.array([.12,-.14,.10,.18,-.10,.16]);target,_=model.fk(qtarget)
    xyzw=Rotation.from_matrix(target[:3,:3]).as_quat()
    goal=GoalToolPose(tool_frames=['right_eef_link'],position=torch.tensor(target[:3,3],dtype=torch.float32,device='cuda').reshape(1,1,1,1,3),quaternion=torch.tensor([xyzw[3],*xyzw[:3]],dtype=torch.float32,device='cuda').reshape(1,1,1,1,4))
    indices=torch.zeros((1,1),dtype=torch.int32,device='cuda')
    def evaluate(q):
        tensor=torch.tensor(q,dtype=torch.float32,device='cuda').reshape(1,6)
        gradient,_,_,_,cost=calc._compute_pose_errors(tensor,goal,indices,1)
        return gradient.detach().cpu().numpy().reshape(6).copy(),float(cost.detach().cpu().reshape(-1)[0])
    gradient,value=evaluate(q0);checks=[]
    for epsilon in [.01,.001,.0003]:
        numeric=np.array([(evaluate(q0+np.eye(6)[i]*epsilon)[1]-evaluate(q0-np.eye(6)[i]*epsilon)[1])/(2*epsilon) for i in range(6)])
        checks.append({'epsilon':epsilon,'finite_difference':numeric.tolist(),'max_absolute_error':float(np.max(np.abs(gradient-numeric)))})
    result={'backend':args.backend,'source':curobo.__file__,'cost':value,'native_gradient':gradient.tolist(),'checks':checks,'primary_epsilon':.001,'primary_passed':checks[1]['max_absolute_error']<.0005,'native_ik_calls':0,'physical_trials':0}
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
