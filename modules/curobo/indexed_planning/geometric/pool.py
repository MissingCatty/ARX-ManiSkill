"""Try verified geometric branches, retaining original contact/path gates."""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.layout import write_json
from modules.curobo.candidate_io.store import candidate_pool, candidate_window
from .client import GeometricClient, GeometricSequence


def geometric_pool(base):
    class GeometricPool(base):
        def __init__(self, root, warmup_trial):
            self.geometric_enabled = False
            self.geometric_sequence = None
            super().__init__(root, warmup_trial)
            self.geometric_client = GeometricClient(Path(root)/'geometric_cpu')
            self.geometric_enabled = True
            write_json(Path(root)/'geometric_readiness.json', self.geometric_client.ready)

        def select_contact(self, context, state, directory, validator):
            if not self.geometric_enabled or state.get('kinematic_endpoint_witness') is not None:
                return super().select_contact(context, state, directory, validator)
            if self.geometric_sequence is not None: raise RuntimeError('Geometric endpoint windows cannot overlap')
            sequence = GeometricSequence(candidate_pool(directory)['candidates'],
                context, state, self.geometric_client)
            self.geometric_sequence = sequence
            try:
                events, rejected = [], {k: 0 for k in ['grasp_ik','pre_or_lift_ik','path','local_approach']}
                candidates = candidate_pool(directory)['candidates']
                for index in range(len(candidates)):
                    with candidate_window(directory,index,index+1):
                        result = super().select_contact(context,state,directory,validator)
                        attempt = sequence.events[-1]
                        if not result['success'] and attempt['accepted_geometric_branch']:
                            # An endpoint can be clear while its pre/lift/path
                            # is unusable. Try the complete original candidate
                            # before advancing to any lower-ranked grasp.
                            attempt.update(full_candidate_fallback=True,
                                geometric_path_rejected=dict(result.get('rejected',{})))
                            self.geometric_sequence = None
                            try:
                                result = super().select_contact(context,state,directory,validator)
                            finally:
                                self.geometric_sequence = sequence
                        current = result.get('candidate_events',[])
                        if len(current)!=1 or current[0]['candidate_index']!=0:
                            raise RuntimeError('Original selector did not finish exactly one candidate')
                        events.append({**current[0],'candidate_index':index,
                            'geometric_attempt':dict(attempt)})
                        for name,count in result.get('rejected',{}).items():
                            rejected[name] += count
                        if result['success']:
                            result['selected']['index']=index
                            result.update(rejected=rejected,candidate_events=events,
                                geometric_ik=sequence.summary())
                            return result
                return {'success':False,'reason':'no_feasible_native_post_or_recalled_candidate',
                    'count':len(candidates),'rejected':rejected,'candidate_events':events,
                    'geometric_ik':sequence.summary()}
            finally:
                write_json(context.out/'geometric_ik.json', sequence.summary())
                self.geometric_sequence = None

        def grasp_endpoint(self, context, state, grasp, start, validator):
            import torch
            sequence = self.geometric_sequence
            if sequence is None or state.get('kinematic_endpoint_witness') is not None:
                return super().grasp_endpoint(context, state, grasp, start, validator)
            answer = sequence.next(grasp, start)
            event = {'candidate_index': sequence.cursor-1,
                'geometric_admitted_branches': answer['admitted_branches'],
                'original_IK_fallback': True, 'accepted_geometric_branch': False,
                'roundoff_or_independent_FK_rejected': 0}
            sequence.events.append(event)
            q = np.asarray(answer['q'], dtype=float).reshape(-1, 6)
            if len(q):
                # Test the exact float32 values used by the native checker and
                # executor against authored (double) physical joint limits.
                q = q.astype(np.float32).astype(float)
                names = [f'right_joint{i}' for i in range(21, 27)]
                limits = np.asarray([state['joint_limits'][state['joint_names'].index(n)] for n in names])
                keep = np.isfinite(q).all(1) & (q >= limits[:, 0]).all(1) & (q <= limits[:, 1]).all(1)
                event['roundoff_or_independent_FK_rejected'] += int((~keep).sum())
                q = q[keep]
            if len(q):
                actual = validator.geometry.frames(q)['right_eef_link']
                pe = np.linalg.norm(actual[:, :3, 3]-grasp[:3, 3], axis=1)
                re = Rotation.from_matrix(actual[:, :3, :3]@grasp[:3, :3].T).magnitude()
                keep = (pe <= self.contact_profile.position_tolerance_m) & (re <= self.contact_profile.rotation_tolerance_rad)
                event['roundoff_or_independent_FK_rejected'] += int((~keep).sum())
                q, pe, re = q[keep], pe[keep], re[keep]
            if len(q):
                values = torch.as_tensor(q, device=start.device, dtype=torch.float32)
                valid = np.asarray(validator.validate(values, stage='final_approach')).reshape(-1)
                indices = np.flatnonzero(valid)
                if len(indices):
                    index = int(indices[0]); value = values[index].clone()
                    event.update(original_IK_fallback=False, accepted_geometric_branch=True)
                    return value, {'kind': 'geometric_independent_FK_and_original_contact_gate',
                        'geometric_branches_checked': len(q),
                        'independent_position_error_m': float(pe[index]),
                        'independent_rotation_error_rad': float(re[index]),
                        'native_full_world_clear': bool(context.checker.validate(value[None, None]).all())}
            return super().grasp_endpoint(context, state, grasp, start, validator)

    return GeometricPool
