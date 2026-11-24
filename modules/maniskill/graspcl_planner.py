"""Select GraspCL canonical poses using the frozen ARX cuRobo implementation."""
import json
from pathlib import Path
import sys

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / 'arx_pipeline'))
import planner


class GraspCLPlanner(planner.Planner):
    def select(self, state, rows, camera):
        # Candidate pool is already stably ordered by GraspCL posterior mean.
        # Do not reapply the baseline GT object-center/height-shift heuristic.
        pool = json.loads((self.out.parent / 'graspcl/candidate_pool.json').read_text())['candidates']
        start = self.measured(state)
        rejected = {key: 0 for key in ['approach_direction', 'grasp_ik', 'closed_collision',
                                      'pre_or_lift_ik', 'path', 'optimizer']}
        for index, candidate in enumerate(pool):
            contact = np.asarray(candidate['pose'])
            if contact[2, 2] > -.35:
                rejected['approach_direction'] += 1
                continue
            grasp = contact.copy()
            grasp[:3, 3] -= .040 * contact[:3, 2]
            pre = grasp.copy()
            pre[:3, 3] -= .08 * grasp[:3, 2]
            lift = grasp.copy()
            lift[2, 3] += .12
            qg = self.ik(grasp, start)
            if qg is None:
                rejected['grasp_ik'] += 1
                continue
            if not self.closed_checker.validate(qg[None, None]).all():
                rejected['closed_collision'] += 1
                continue
            qp, ql = self.ik(pre, qg), self.ik(lift, qg)
            if qp is None or ql is None:
                rejected['pre_or_lift_ik'] += 1
                continue
            reach, approach = self.path(start, qp), self.path(qp, qg)
            if reach is None or approach is None:
                rejected['path'] += 1
                continue
            selected = {'index': index, 'score': candidate['score'],
                'candidate_metadata': candidate['metadata'],
                'contact_pose_world': contact.tolist(),
                'grasp_world': grasp.tolist(), 'pregrasp_world': pre.tolist(), 'lift_world': lift.tolist(),
                'target_contact_world': contact[:3, 3].tolist(), 'table_clearance_raise_m': 0.,
                'q_grasp': qg.tolist(), 'q_pre': qp.tolist(), 'q_lift': ql.tolist()}
            try:
                paths = {'pregrasp': self.optimize('pregrasp', reach, 6),
                         'approach': self.optimize('approach', approach, 5)}
            except RuntimeError:
                rejected['optimizer'] += 1
                raise
            return {'success': True, 'selected': selected, 'paths': paths, 'rejected': rejected}
        return {'success': False, 'reason': 'no_feasible_merged_candidate',
                'rejected': rejected, 'count': len(pool)}


if __name__ == '__main__':
    planner.Planner = GraspCLPlanner
    planner.main()
