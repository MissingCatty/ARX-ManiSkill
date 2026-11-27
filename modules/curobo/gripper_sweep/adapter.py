"""Reuse the existing native sphere batch without changing search scheduling."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time

from arx_project.layout import read_json,write_json
from modules.graspgenx.native_postprocessing.width_planning import width_pool,target_state,WidthSelectionMixin,FINGERS
from modules.curobo.dual_arm.accelerated_adapter import AcceleratedDualArmPlanner,opening_samples
from modules.curobo.indexed_planning.stage_timing import StageTiming,wrap_method


def batch_sweep(pool,work,canonical):
    if getattr(pool,'sphere_reuse_enabled',False) or getattr(pool,'fast_sweep_enabled',False):
        from .cached_sweep import opening_sweep
        return opening_sweep(pool,work,canonical)
    return AcceleratedDualArmPlanner.opening_sweep(SimpleNamespace(pool=pool),work,canonical)


def narrowing_canonical(state,width):
    result=target_state(state,width)
    result['planned_opening']={'target_per_finger_m':float(width/2),
        'observed_solver_finger_positions_m':{n:.044 for n in FINGERS},
        'execution_required_before_arm_motion':True}
    return result


def optimized_pool(base):
    class SweepPool(width_pool(base)):
        def __init__(self,*args,**kwargs):
            self.batch_sweep_enabled=False;self.stage_timing=StageTiming();self._pending_ik=[]
            super().__init__(*args,**kwargs)
            for context in self.contexts.values():
                wrap_method(context,'ik',self.stage_timing,self._ik_label)
                wrap_method(context,'path',self.stage_timing,'path_PRM')
                wrap_method(context,'optimize',self.stage_timing,lambda label,*a,**k:'trajectory_'+label)

        def _ik_label(self,*args,**kwargs):
            if any(f['stage']=='grasp_endpoint' for f in self.stage_timing.stack):return 'grasp_IK'
            return self._pending_ik.pop(0) if self._pending_ik else 'other_IK'

        def bind(self,*args,**kwargs):
            with self.stage_timing.stage('world_and_robot_bind'):return super().bind(*args,**kwargs)

        def contact_view(self,*args,**kwargs):
            with self.stage_timing.stage('contact_validator_update'):return super().contact_view(*args,**kwargs)

        def grasp_endpoint(self,*args,**kwargs):
            self._pending_ik=[]
            with self.stage_timing.stage('grasp_endpoint'):result=super().grasp_endpoint(*args,**kwargs)
            if result[0] is not None:self._pending_ik=['pregrasp_IK','lift_IK']
            return result

        def select_contact(self,*args,**kwargs):
            with self.stage_timing.stage('candidate_search'):return super().select_contact(*args,**kwargs)

        def width_context(self,*args,**kwargs):
            with self.stage_timing.stage('candidate_width_update'):return super().width_context(*args,**kwargs)

        def validate_narrowing(self,work,state,width):
            with self.stage_timing.stage('gripper_narrowing'):
                if not self.batch_sweep_enabled:return super().validate_narrowing(work,state,width)
                canonical=narrowing_canonical(state,width)
                result=batch_sweep(self,work/'narrowing_batch',canonical)
                fractions,target,observed=opening_samples(canonical)
                samples=[float(.044+fraction*(target-.044)) for fraction in fractions]
                first=result.get('first_invalid_sample')
                return {'passed':result['passed'],'samples':samples if result['passed'] else samples[:first],
                    'first_invalid_sample':first,'batched_receipt':str(work/'narrowing_batch/opening_sweep.json')}
    return SweepPool


def create(args):
    from modules.curobo.dual_arm.adapter import create as original_create,DualArmPlanner
    from modules.curobo.dual_arm.interleaved_adapter import InterleavedDualArmPlanner,configure
    original=original_create(args,pool_extension=optimized_pool)
    class SweepPlanner(WidthSelectionMixin,InterleavedDualArmPlanner):
        def opening_sweep(self,work,canonical):
            began=time.perf_counter()
            with self.pool.stage_timing.stage('gripper_opening'):
                if self.pool.batch_sweep_enabled:result=batch_sweep(self.pool,work,canonical)
                else:result=DualArmPlanner.opening_sweep(self,work,canonical)
            result['wall_seconds']=time.perf_counter()-began
            write_json(Path(work)/'opening_sweep.json',result)
            return result
    return configure(SweepPlanner(original.pool,original.model,original.root))
