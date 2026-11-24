"""Frozen ARX IK for static observations; no trajectory commands are emitted."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import read,write
from scripts.curobo_arx.data_factory.teacher import Rejection,bounded_planning_start
from .advisor33_teacher import Teacher as Original
from training.nearfield_arx.contract import pose_error


class Teacher(Original):
    def handle(self,request):
        if request['action']=='static_surface_check':
            from .advisor36_surfaces import SurfaceChecker
            from scripts.curobo_arx.data_factory.common import sha
            path=request['geometry'];key=(path,sha(path))
            if getattr(self,'surface_key',None)!=key:
                self.surface_checker=SurfaceChecker(read(path));self.surface_key=key
            result=self.surface_checker.check(request['snapshot'])
            write(Path(request['output'])/'static_surface_receipt.json',result)
            return result
        if request['action']!='static_ik': raise ValueError('Static worker cannot execute movement requests')
        snapshot=request['snapshot'];scene=read(request['scene']);out=Path(request['output'])
        self.current_output=out;self.current_label='static'
        # Preserve the original grasp-target handling. The additional target
        # penetration gate uses actual enabled collider surfaces after FK;
        # conservative planning spheres must not erase all near-field data.
        self.initialize(scene,snapshot,include_target=False)
        if not np.allclose(scene['base_world'],snapshot['link_world']['body4'],atol=1e-6,rtol=0):
            raise RuntimeError('Static scene and current ARX base frame disagree')
        names=self.context['names'];measured=dict(zip(snapshot['joint_names'],snapshot['joint_positions']))
        planner,checker,updater,models=self.context['open']
        updater.apply(models,{n:float(measured[n]) for n in updater.names})
        limits=[snapshot['joint_limits'][snapshot['joint_names'].index(n)] for n in names]
        start=self.torch.tensor(bounded_planning_start([measured[n] for n in names],limits),device='cuda',dtype=self.torch.float32)
        goal=np.asarray(snapshot['objects']['test_object']['world'])@np.asarray(request['desired_eef_object'])
        inverse=np.linalg.inv(snapshot['link_world']['body4'])
        try:
            q=self.ik(goal,start,planner,checker,inverse,seed=request['canonical_arm_seed'])
            state=planner.compute_kinematics(self.js(q)).tool_poses
            fk=np.eye(4);fk[:3,3]=state.position.detach().cpu().numpy().reshape(3)
            quat=state.quaternion.detach().cpu().numpy().reshape(4)
            fk[:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            fk=np.asarray(snapshot['link_world']['body4'])@fk
            error=pose_error(fk,goal)
            if error['translation_m']>.0005 or error['rotation_rad']>np.deg2rad(.2):
                raise Rejection('static_IK_fit_outside_data_quality_limit',error)
            visual=self.visual_goal(request,fk)
            if visual is not None and not visual['pass']: raise Rejection('static_camera_geometry_rejected',visual)
            certificate=None
            if request['category']=='grasp':
                certificate=self.nominal_closed(fk,q,inverse,q_grasp_override=q)
            result={'pass':True,'joint_names':names,'arm_q':q.tolist(),'model_fk_world':fk.tolist(),
                'desired_world':goal.tolist(),'fit':error,'visual_geometry':visual,
                'original_target_excluding_open_check_pass':True,
                'actual_static_target_surface_check_required':True,'nominal_closed':certificate,
                'physical_success_claim':False,'trajectory_commands':0}
        except Rejection as error:
            result={'pass':False,'rejected':str(error),'details':error.details,
                'physical_success_claim':False,'trajectory_commands':0}
        write(out/'static_IK_receipt.json',result)
        return result


if __name__=='__main__':
    from scripts.curobo_arx.seed_augmentation import worker
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--registration',required=True)
    a=p.parse_args();worker.Teacher=Teacher;worker.serve(a.directory,read(a.registration))
