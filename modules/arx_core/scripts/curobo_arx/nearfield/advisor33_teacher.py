"""Original ARX checks with measured-hand/world-Z lift goals, no coarse prior."""
import copy
from pathlib import Path
import shutil
import numpy as np
from scripts.curobo_arx.data_factory.common import write
from scripts.curobo_arx.data_factory.teacher import Rejection
from .teacher import Teacher as Original
from .advisor32_rules import lift_goals


def candidates(grasp):
    grasp = np.asarray(grasp); z = np.array([0., 0., -1.]); y = grasp[:3, 1].copy(); y[2] = 0.
    if np.linalg.norm(y) < 1e-8: y = np.array([0., 1., 0.])
    y /= np.linalg.norm(y); neutral = np.column_stack([np.cross(y, z), y, z])
    return lift_goals(grasp, neutral)


class Teacher(Original):
    def handle(self, request):
        self.current_output = Path(request['output']) if request.get('output') else Path(request['scene']).parent
        self.current_label = request.get('label', request['action'])
        try:
            if request['action']=='plan' and request.get('phase')=='lift':
                return self.plan_lift(request)
            return super().handle(request)
        except ValueError as error:
            from .advisor33_boundary import range_rejection
            result=range_rejection(error,request,getattr(self,'context',{}))
            write(self.current_output/(self.current_label+'_measured_gripper_rejection.json'),result)
            return result

    def nominal_closed(self, grasp, start, inverse, q_grasp_override=None):
        planner, checker, updater, models = self.context['closed']; updater.apply(models, {n:.022 for n in updater.names})
        opened, open_checker, _, _ = self.context['open']
        q = self.ik(grasp, start, opened, open_checker, inverse) if q_grasp_override is None else q_grasp_override
        if not checker.validate(q[None,None]).all(): raise Rejection('nominal_closed_grasp_collision')
        records = []
        for candidate in candidates(grasp):
            row = copy.deepcopy(candidate); records.append(row)
            try:
                ql = self.ik(np.asarray(candidate['goal_world']), q, planner, checker, inverse)
                if not checker.validate(ql[None,None]).all(): raise Rejection('nominal_closed_lift_collision')
                row.update(pass_=True, q_lift=ql.tolist())
                result = {'pass':True, 'nominal_gripper_m':.022, 'table_margin_m':.008,
                    'q_grasp':q.tolist(), 'q_lift':ql.tolist(), 'geometry_changed':False,
                    'selected_lift_world':candidate['goal_world'], 'lift_candidates':records,
                    'prior_used':False, 'scope':'Frozen model qualification, not physical success'}
                write(self.current_output/(self.current_label+'_lift_candidates.json'), {'candidates':records})
                return result
            except Rejection as exc:
                row.update(pass_=False, rejected=str(exc), details=exc.details)
        write(self.current_output/(self.current_label+'_lift_candidates.json'), {'candidates':records})
        raise Rejection('measured_orientation_lift_candidates_exhausted', {'candidates':records})

    def plan_lift(self, request):
        out = Path(request['output']); grasp = np.asarray(request['snapshot']['link_world']['right_eef_link']); records=[]
        for candidate in candidates(grasp):
            folder=out/'lift_goal_search'/f'{candidate["index"]:02d}'; folder.mkdir(parents=True, exist_ok=False)
            changed={**request, 'goal_world':candidate['goal_world'], 'grasp_world':grasp.tolist(), 'output':str(folder)}
            write(folder/'request.json', changed); reply=super().handle(changed); write(folder/'reply.json', reply)
            records.append({**candidate, 'pass':not reply.get('rejected'), 'rejected':reply.get('rejected')})
            write(out/'lift_goal_search.json', {'candidates':records, 'measured_closed_world':grasp.tolist(), 'prior_used':False})
            if reply.get('rejected'): continue
            for p in folder.iterdir():
                if p.is_file() and p.name not in ('request.json','reply.json'):
                    if (out/p.name).exists(): raise RuntimeError('No overwriting lift evidence')
                    shutil.copy2(p,out/p.name)
            return reply
        return {'rejected':'measured_closed_lift_search_exhausted', 'phase':'lift', 'candidates':records}
