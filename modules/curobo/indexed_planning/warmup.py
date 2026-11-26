"""Warm failure/optimizer branches and verify padded target-row correspondence."""
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.layout import read_json, write_json


def warm_solver_branches(planner, source_state, directory):
    import torch
    directory = Path(directory)
    state = planner.model.bind(read_json(source_state),'right',directory/'canonical')
    context = planner.pool.bind(directory/'binding',state,include_target=False)
    validator = planner.pool.contact_view(context,state)
    start = context.measured(state)
    measured = np.asarray(state['links']['right_eef_link'])
    unreachable = measured.copy()
    unreachable[:3,3] += context.base[:3,:3]@np.array([10.,0.,0.])
    began = time.perf_counter()
    if context.ik(unreachable,start) is not None:
        raise RuntimeError('The full-world solver accepted a target ten meters away')
    if len(planner.pool.unconstrained_ik(context,unreachable,start)):
        raise RuntimeError('The pure solver accepted a target ten meters away')
    reports = []
    for capacity,engine in planner.pool.batch_engines.items():
        engine.bind(context)
        if any(q is not None for q in engine.native([unreachable]*capacity,start)):
            raise RuntimeError('Batch native warmup accepted unreachable target rows')
        if any(len(q) for q in engine.pure([unreachable]*capacity,start)):
            raise RuntimeError('Batch pure warmup accepted unreachable target rows')
        # Fewer real targets exercise fixed-capacity padding and slicing.
        native = engine.native([unreachable,measured],start)
        pure = engine.pure([unreachable,measured],start)
        if native[0] is not None or native[1] is None or len(pure[0]) or not len(pure[1]):
            raise RuntimeError('Partial-batch target rows lost their individual success results')
        checked = []
        for label,values in [('native',native[1][None]),('pure',pure[1])]:
            actual = validator.geometry.frames(values.detach().cpu().numpy())['right_eef_link']
            pe = np.linalg.norm(actual[:,:3,3]-measured[:3,3],axis=-1)
            re = Rotation.from_matrix(actual[:,:3,:3]@measured[:3,:3].T).magnitude()
            if np.max(pe) > .002 or np.max(re) > .03:
                raise RuntimeError('Independent FK does not match the successful padded target row')
            checked.append({'solver':label,'valid_solutions':len(values),
                'maximum_position_error_m':float(np.max(pe)),'maximum_rotation_error_rad':float(np.max(re))})
        reports.append({'capacity':capacity,'all_unreachable_rows_rejected':True,
            'two_target_padding_slicing_verified':True,'independent_FK':checked,'calls':engine.calls})
        engine.reset_seed()
    for value in planner.pool.contexts.values():value.planner.reset_seed()
    planner.pool.pure_solver.reset_seed()
    torch.cuda.synchronize()
    result = {'passed':True,'seconds':time.perf_counter()-began,'reports':reports,
        'control_negative_targets_warmed':True,'optimizer_failure_branches_warmed':True,
        'physical_motions':0,'counted_as_planning_requests':False}
    write_json(directory/'solver_branch_warmup.json',result)
    return result
