"""Optional analytic endpoint fast path with original width/V2/path fallback."""
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from arx_project.layout import write_json
from modules.curobo.candidate_io.store import candidate_pool,candidate_window
from .bank import EndpointBank


def analytic_pool(base):
    class AnalyticPool(base):
        def __init__(self,*args,**kwargs):
            self.analytic_enabled=False;self.analytic_bank=None;self.analytic_sessions={}
            self.analytic_statistics=[]
            super().__init__(*args,**kwargs)
            for context in self.contexts.values():
                original=context.ik
                def routed(world,start,_context=context,_original=original):
                    bank=self.analytic_bank
                    if self.analytic_enabled and bank is not None and bank.enabled:
                        for stage in ['pregrasp','lift']:
                            if bank.matches(stage,world):
                                values=bank.branches(stage,start.detach().cpu().numpy())
                                with self.stage_timing.stage('analytic_branch_original_gate'):
                                    accepted=self._checked_branches(_context,None,values,world)
                                if len(accepted):
                                    bank.used=True;bank.events.append({'candidate_index':bank.cursor,'stage':stage,'kind':'analytic_world_endpoint'})
                                    return accepted[0].clone()
                                bank.events.append({'candidate_index':bank.cursor,'stage':stage,'kind':'original_IK_fallback'})
                    return _original(world,start)
                context.ik=routed
            self.analytic_enabled=True

        def prepare_analytic(self,context,state):
            from .session import AnalyticBatchSession
            side=state.get('planning_arm','right')
            if side not in self.analytic_sessions:
                config=yaml.load(Path(context.model).read_text(),Loader=yaml.CSafeLoader)
                self.analytic_sessions[side]=AnalyticBatchSession(config,None,
                    include_self=context.checker.self_collision_cost is not None,
                    native_scene=context.checker.scene_model)
            session=self.analytic_sessions[side]
            session.verify_shared_scene(context.checker.scene_model)
            return session

        def _checked_branches(self,context,validator,values,world):
            import torch
            if not len(values):return torch.empty((0,6),device=context.limits.device)
            geometry=validator.geometry if validator is not None else self.contact_views['open']['validator'].geometry
            frames=geometry.frames(values)['right_eef_link']
            pe=np.linalg.norm(frames[:,:3,3]-world[:3,3],axis=1)
            re=Rotation.from_matrix(frames[:,:3,:3]@world[:3,:3].T).magnitude()
            keep=(pe<=self.contact_profile.position_tolerance_m)&(re<=self.contact_profile.rotation_tolerance_rad)
            q=torch.as_tensor(values[keep],device=context.limits.device,dtype=torch.float32)
            if len(q):
                valid=(validator.validate(q,stage='final_approach') if validator is not None
                       else context.checker.validate(q[:,None]).reshape(-1).detach().cpu().numpy())
                q=q[np.flatnonzero(valid)]
            return q

        def grasp_endpoint(self,context,state,grasp,start,validator):
            bank=self.analytic_bank
            if self.analytic_enabled and bank is not None and bank.enabled:
                if not bank.matches('grasp',grasp):raise RuntimeError('Analytic candidate cursor lost original target')
                values=bank.branches('grasp',start.detach().cpu().numpy())
                with self.stage_timing.stage('analytic_branch_original_gate'):
                    accepted=self._checked_branches(context,validator,values,grasp)
                if len(accepted):
                    bank.used=True
                    return accepted[0].clone(),{'kind':'GPU_analytic_with_original_FK_width_and_V2_gate',
                        'analytic_branches_checked':len(values),'original_IK_fallback':False}
                bank.events.append({'candidate_index':bank.cursor,'stage':'grasp','kind':'original_IK_fallback'})
            return super().grasp_endpoint(context,state,grasp,start,validator)

        def select_contact(self,context,state,directory,validator):
            if not self.analytic_enabled:return super().select_contact(context,state,directory,validator)
            if self.analytic_bank is not None:raise RuntimeError('Analytic candidate windows cannot overlap')
            directory=Path(directory);candidates=candidate_pool(directory)['candidates']
            if not candidates:return super().select_contact(context,state,directory,validator)
            began=time.perf_counter()
            with self.stage_timing.stage('analytic_scene_update'):
                session=self.prepare_analytic(context,state)
            with self.stage_timing.stage('analytic_endpoint_batch'):
                bank=EndpointBank(session,candidates,context.invbase)
            self.analytic_bank=bank;events=[];rejected={}
            try:
                for index in range(len(candidates)):
                    bank.cursor=index;bank.used=False;bank.enabled=True
                    # WidthPool's full-open fast path assumes its incoming
                    # context actually has the original full-open locks.
                    if abs(float(candidates[index]['width'])-.088)<1e-12:
                        locks=validator.geometry.locked
                        if any(abs(float(locks[n])-.044)>1e-12 for n in ['right_joint27','right_joint28']):
                            context,validator=self.width_context(directory/'analytic_full_open_restore'/f'{index:04d}',state)
                    with candidate_window(directory,index,index+1):
                        result=super().select_contact(context,state,directory,validator)
                        if not result['success'] and bank.used:
                            attempt=deepcopy(result);bank.enabled=False
                            bank.events.append({'candidate_index':index,'stage':'complete_candidate','kind':'original_full_candidate_fallback',
                                'analytic_rejected':attempt.get('rejected',{})})
                            write_json(directory/'analytic_attempts'/f'{len(self.analytic_statistics):04d}_{index:04d}.json',attempt)
                            result=super().select_contact(context,state,directory,validator)
                    current=result.get('candidate_events',[])
                    if len(current)!=1 or int(current[0]['candidate_index'])!=0:
                        raise RuntimeError('Width-aware original selector did not finish one candidate')
                    events.append({**current[0],'candidate_index':index})
                    for name,count in result.get('rejected',{}).items():rejected[name]=rejected.get(name,0)+count
                    if result['success']:
                        result['selected']['index']=index
                        result.update(candidate_events=events,rejected=rejected,analytic_endpoints=bank.summary())
                        return result
                return {'success':False,'reason':'no_feasible_native_post_or_recalled_candidate','count':len(candidates),
                    'candidate_events':events,'rejected':rejected,'analytic_endpoints':bank.summary()}
            finally:
                summary=bank.summary();summary['wall_seconds']=time.perf_counter()-began
                self.analytic_statistics.append(summary)
                write_json(directory/'analytic_endpoint_windows'/f'{len(self.analytic_statistics):04d}.json',summary)
                self.analytic_bank=None
    return AnalyticPool
