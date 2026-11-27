"""Bind candidate widths to native ARX collision geometry in an optional pool."""
from copy import deepcopy
from pathlib import Path

import numpy as np
from arx_project.layout import write_json
from modules.arx_core.arms import get_arm
from modules.curobo.candidate_io.store import candidate_pool,candidate_window

FINGERS=get_arm('right').finger_joints
COLLISION_MESSAGE='Measured state collides in unchanged ARX model'


def target_state(state,width):
    width=float(width)
    if not np.isfinite(width) or not 0<width<=.088+1e-12:
        raise ValueError('Candidate width is outside the actual ARX stroke')
    result=deepcopy(state)
    for name in FINGERS:result['qpos'][result['joint_names'].index(name)]=width/2
    return result


def binding_proof(validator,width,model):
    locks={name:float(validator.geometry.locked[name]) for name in FINGERS}
    if any(abs(value-width/2)>1e-12 for value in locks.values()):
        raise RuntimeError('Candidate width was not bound to the collision geometry')
    proof={'candidate_width_m':width,'open_finger_m':width/2,
        'actual_collision_finger_locks_m':locks,'robot_model':str(model),
        'same_871_spheres_and_URDF':True}
    if hasattr(validator,'context'):
        cfg=validator.context.checker.kinematics.kinematics_config
        locked=cfg.lock_jointstate
        native=dict(zip(locked.joint_names,locked.position.detach().cpu().reshape(-1).tolist()))
        if any(abs(native[n]-width/2)>2e-7 for n in FINGERS):
            raise RuntimeError('Native GPU collision locks differ from the candidate opening')
        proof['native_GPU_finger_locks_m']={n:native[n] for n in FINGERS}
    return proof


def width_pool(base):
    class WidthPool(base):
        def select_contact(self,context,state,directory,validator):
            directory=Path(directory);candidates=candidate_pool(directory)['candidates']
            if all(abs(float(c['width'])-.088)<1e-12 for c in candidates):
                result=super().select_contact(context,state,directory,validator)
                if result['success']:
                    proof=binding_proof(validator,.088,context.model)
                    result['selected'].update(open_finger_m=.044,width_binding=proof)
                return result
            events=[];rejected={};current_width=.088
            for index,candidate in enumerate(candidates):
                width=float(candidate['width']);measured=target_state(state,width)
                identity=candidate['metadata']['original_native_rank']
                work=directory/'width_targets'/f'{identity:06d}'
                try:
                    if current_width is None or abs(width-current_width)>1e-12:
                        context=self.bind(work,measured,include_target=False)
                        validator=self.contact_view(context,measured);current_width=width
                    proof=binding_proof(validator,width,context.model)
                    with candidate_window(directory,index,index+1):
                        result=super().select_contact(context,measured,directory,validator)
                except AssertionError as error:
                    if str(error)!=COLLISION_MESSAGE:raise
                    result={'success':False,'reason':'no_feasible_native_post_or_recalled_candidate',
                        'candidate_events':[{'candidate_index':0,'kind':'candidate_width_start_in_collision'}],
                        'rejected':{'candidate_width_start_in_collision':1}}
                    current_width=None
                for event in result.get('candidate_events',[]):
                    events.append({**event,'candidate_index':index,'candidate_width_m':width})
                for key,value in result.get('rejected',{}).items():rejected[key]=rejected.get(key,0)+value
                if not result['success']:continue
                # Validate narrowing at the stationary start before permitting
                # a path that assumes this opening. Target remains an obstacle.
                sweep=[];sweep_ok=True
                count=max(2,int(np.ceil(abs(.044-width/2)/.0005))+1)
                if abs(width-.088)>1e-12:
                    for fraction in np.linspace(0,1,count):
                        opening=.044+(width/2-.044)*fraction
                        sample=target_state(state,2*opening)
                        try:self.bind(work/'narrowing'/f'{len(sweep):03d}',sample,include_target=True)
                        except AssertionError as error:
                            if str(error)!=COLLISION_MESSAGE:raise
                            sweep_ok=False;break
                        sweep.append(float(opening))
                    current_width=None
                    if not sweep_ok:
                        rejected['candidate_width_narrowing_collision']=rejected.get('candidate_width_narrowing_collision',0)+1
                        events[-1].update(kind='candidate_width_narrowing_collision')
                        continue
                    context=self.bind(work/'selected',measured,include_target=False)
                    validator=self.contact_view(context,measured);current_width=width
                    proof=binding_proof(validator,width,context.model)
                proof.update(narrowing_samples=sweep,maximum_finger_step_m=.0005,
                    narrowing_target_collision_checked=True)
                result['selected'].update(index=index,open_finger_m=width/2,width_binding=proof)
                result.update(candidate_events=events,rejected=rejected)
                write_json(work/'width_binding.json',proof)
                return result
            return {'success':False,'reason':'no_feasible_native_post_or_recalled_candidate',
                'count':len(candidates),'candidate_events':events,'rejected':rejected}
    return WidthPool


def create(args):
    from modules.curobo.dual_arm.adapter import create as original_create
    from modules.curobo.dual_arm.interleaved_adapter import InterleavedDualArmPlanner,configure
    original=original_create(args,pool_extension=width_pool)

    class WidthPlanner(InterleavedDualArmPlanner):
        def request(self,directory,**flags):
            result=super().request(directory,**flags)
            if result['success'] and not any(flags.get(k) for k in ['approach','lift','observe']):
                selected=result['selected'];proof=selected['width_binding']
                selected['open_finger_m']=proof['open_finger_m']
                write_json(Path(directory)/'plan_result.json',result)
            return result
    return configure(WidthPlanner(original.pool,original.model,original.root))
