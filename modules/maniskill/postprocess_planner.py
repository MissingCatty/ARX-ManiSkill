"""Direct canonical grasp poses into frozen ARX IK/collision/path validation."""
import json
from pathlib import Path
import sys

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / 'arx_pipeline'))
import planner as frozen


class PostprocessPlanner(frozen.Planner):
    def select(self, state, rows, camera):
        pool = json.loads((self.out.parent / 'graspcl/candidate_pool.json').read_text())['candidates']
        start = self.measured(state)
        rejected = {k: 0 for k in ['grasp_ik', 'closed_collision', 'pre_or_lift_ik', 'path', 'optimizer']}
        for index, candidate in enumerate(pool):
            contact = np.asarray(candidate['pose'])
            grasp = contact.copy()
            grasp[:3, 3] -= .040 * contact[:3, 2]
            pre = grasp.copy()
            pre[:3, 3] -= .08 * contact[:3, 2]
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
            selected = {'index': index, 'score': candidate['score'], 'candidate_metadata': candidate['metadata'],
                'candidate_protocol': 'zerograsp_native_post_no_custom_geometry_v1',
                'contact_pose_world': contact.tolist(), 'grasp_world': grasp.tolist(),
                'pregrasp_world': pre.tolist(), 'lift_world': lift.tolist(),
                'target_contact_world': contact[:3, 3].tolist(), 'table_clearance_raise_m': 0.,
                'q_grasp': qg.tolist(), 'q_pre': qp.tolist(), 'q_lift': ql.tolist()}
            try:
                paths = {'pregrasp': self.optimize('pregrasp', reach, 6),
                         'approach': self.optimize('approach', approach, 5)}
            except RuntimeError:
                rejected['optimizer'] += 1
                raise
            return {'success': True, 'selected': selected, 'paths': paths, 'rejected': rejected}
        return {'success': False, 'reason': 'no_feasible_native_post_or_recalled_candidate',
                'rejected': rejected, 'count': len(pool), 'candidate_protocol': 'zerograsp_native_post_no_custom_geometry_v1'}
