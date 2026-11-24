"""Original ARX planner plus a non-motion validation for explicit GRASP."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.seed_augmentation.teacher import Teacher as SeedTeacher
from scripts.curobo_arx.data_factory.teacher import Rejection,bounded_planning_start
from scripts.curobo_arx.data_factory.common import read,write,digest,verify_files


class Teacher(SeedTeacher):
    def handle(self,request):
        if request['action']!='closure_validate':return super().handle(request)
        binding=request['binding'];snapshot=request['snapshot'];out=Path(request['output'])
        verify_files(binding['files'])
        if binding['snapshot_sha256']!=digest(snapshot):raise RuntimeError('Closure snapshot binding mismatch')
        self.initialize(read(request['scene']),snapshot)
        names=self.context['names'];torch=self.torch
        if names!=snapshot['right_arm'] or names!=binding['hold_joint_names']:
            raise RuntimeError('Closure joint identity mismatch')
        planner,checker,updater,models=self.context['open']
        measured=dict(zip(snapshot['joint_names'],snapshot['joint_positions']))
        updater.apply(models,{n:float(measured[n]) for n in updater.names})
        inverse=np.linalg.inv(snapshot['link_world']['body4'])
        def fk(q):
            state=planner.compute_kinematics(self.js(q)).tool_poses
            local=np.eye(4);local[:3,3]=state.position.detach().cpu().numpy().reshape(3)
            quat=state.quaternion.detach().cpu().numpy().reshape(4)
            local[:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            return np.asarray(snapshot['link_world']['body4'])@local
        try:
            limits=[snapshot['joint_limits'][snapshot['joint_names'].index(n)] for n in names]
            qactual=torch.tensor(bounded_planning_start([measured[n] for n in names],limits),device='cuda',dtype=torch.float32)
            qhold=torch.tensor(binding['hold_q'],device='cuda',dtype=torch.float32)
            from training.nearfield_arx.contract import pose_error
            match=pose_error(fk(qactual),snapshot['link_world']['right_eef_link'])
            if match['translation_m']>=.002 or match['rotation_rad']>=.01:
                raise RuntimeError('Live ARX/planner FK mismatch at closure')
            if not checker.validate(torch.stack([qactual,qhold])[:,None]).all():
                raise Rejection('closure_open_state_collision')
            held_world=fk(qhold)
            certificate=self.nominal_closed(held_world,qactual,inverse,q_grasp_override=qhold)
            # Current measured configuration must also be nominally clear;
            # original hold target and +12cm lift remain the actual commands.
            if not self.context['closed'][1].validate(qactual[None,None]).all():
                raise Rejection('closure_measured_nominal_closed_collision')
            result={'pass':True,'binding_sha256':digest(binding),'hold_q':binding['hold_q'],
                'hold_world':held_world.tolist(),'nominal_closed':certificate,'live_fk':match,
                'scope':'Non-motion original nominal closed grasp/lift validity at actual held command',
                'physical_execution_certified':False,'motion_commands':0}
        except Rejection as error:
            result={'pass':False,'rejected':str(error),'binding_sha256':digest(binding),'motion_commands':0}
        write(out/'closure_model_validation.json',result)
        return result
