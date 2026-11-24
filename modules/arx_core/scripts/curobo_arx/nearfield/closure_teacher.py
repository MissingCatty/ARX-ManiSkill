"""Frozen ARX geometry/checkers with a bounded lift motion-goal search."""
import copy
from pathlib import Path
import shutil
import numpy as np
from scripts.curobo_arx.data_factory.common import read, write
from scripts.curobo_arx.data_factory.teacher import Rejection
from .teacher import Teacher as OriginalTeacher
from training.nearfield_arx.closure_recovery import lift_goals, POLICY


class Teacher(OriginalTeacher):
    def request_prior(self, request):
        if hasattr(self,'diagnostic_prior'):
            return np.asarray(self.diagnostic_prior)
        by_id={t['trial_id']:t for t in self.registration['seed_trials']}
        location=Path(request['output']) if request.get('output') else Path(request['scene']).parent
        matches=[by_id[p.name] for p in (location,*location.parents) if p.name in by_id]
        if len(matches)!=1:raise RuntimeError('Lift prior lacks a unique registered trial')
        # This is the fixed coarse prior, not the historical successful Gstar.
        return np.asarray(matches[0]['G0_world'])

    def handle(self, request):
        self.current_prior=self.request_prior(request)
        self.current_output=Path(request['output']) if request.get('output') else Path(request['scene']).parent
        self.current_label=request.get('label',request['action'])
        if request['action']=='plan' and request.get('phase')=='lift':
            return self.plan_lift(request)
        return super().handle(request)

    def nominal_closed(self, grasp, start, inverse, q_grasp_override=None):
        planner,checker,updater,models=self.context['closed']
        updater.apply(models,{n:.022 for n in updater.names})
        opened,open_checker,_,_=self.context['open']
        qg=self.ik(grasp,start,opened,open_checker,inverse) if q_grasp_override is None else q_grasp_override
        if not checker.validate(qg[None,None]).all():raise Rejection('nominal_closed_grasp_collision')
        if not hasattr(self,'current_prior'):raise RuntimeError('No registered coarse lift prior')
        records=[]
        for candidate in lift_goals(grasp,self.current_prior):
            row=copy.deepcopy(candidate);records.append(row)
            try:
                ql=self.ik(np.asarray(candidate['goal_world']),qg,planner,checker,inverse)
                if not checker.validate(ql[None,None]).all():raise Rejection('nominal_closed_lift_collision')
                row.update(pass_=True,q_lift=ql.tolist())
                cert={'pass':True,'nominal_gripper_m':.022,'table_margin_m':.008,
                      'q_grasp':qg.tolist(),'q_lift':ql.tolist(),'geometry_changed':False,
                      'selected_lift_world':candidate['goal_world'],'lift_candidates':records,
                      'lift_fallback_used':candidate['index']>0,
                      'prior_used':'registered fixed G0 orientation',
                      'scope':'Original closed-model validity with bounded8/10/12cm and <=10deg lift goals; not physical success'}
                self.save_nominal(records)
                return cert
            except Rejection as exc:
                row.update(pass_=False,rejected=str(exc),details=exc.details)
                self.save_nominal(records)
        raise Rejection('bounded_lift_goals_exhausted',{'candidates':records})

    def save_nominal(self, records):
        if hasattr(self,'current_output'):
            write(self.current_output/(self.current_label+'_lift_candidates.json'),
                  {'candidates':records,'policy':POLICY,'physical_commands':0})

    def plan_lift(self, request):
        out=Path(request['output']);grasp=np.asarray(request['snapshot']['link_world']['right_eef_link'])
        candidates=[{'goal_world':request['goal_world'],'basis':'accepted preclosure certificate'}]
        candidates += [{**r,'basis':'measured closed EEF bounded fallback'} for r in lift_goals(grasp,self.current_prior)
                       if not np.allclose(r['goal_world'],request['goal_world'],atol=1e-12,rtol=0)]
        records=[]
        for index,candidate in enumerate(candidates):
            folder=out/'lift_goal_search'/f'{index:02d}';folder.mkdir(parents=True,exist_ok=False)
            changed={**request,'goal_world':candidate['goal_world'],'output':str(folder)}
            write(folder/'request.json',changed)
            reply=super().handle(changed);write(folder/'reply.json',reply)
            records.append({'index':index,**candidate,'pass':not reply.get('rejected'),'rejected':reply.get('rejected')})
            write(out/'lift_goal_search.json',{'candidates':records,'physical_commands':0,'maximum_candidates':10})
            if reply.get('rejected'):continue
            # The original executor/auditor reads these canonical certificates.
            # Keep every candidate's original folder as well.
            for file in folder.iterdir():
                if file.is_file() and file.name not in ('request.json','reply.json'):
                    dest=out/file.name
                    if dest.exists():raise RuntimeError('Do not overwrite prior lift evidence')
                    shutil.copy2(file,dest)
            return {**reply,'bounded_lift_search':{'selected_index':index,'candidate_count':len(records),
                    'selected_goal_world':candidate['goal_world'],'all_original_path_checks_retained':True}}
        return {'rejected':'bounded_optimized_lift_search_exhausted','phase':'lift','candidates':records}
