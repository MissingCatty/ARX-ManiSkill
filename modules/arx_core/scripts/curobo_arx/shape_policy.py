"""H008: observed sphere geometry refinement, otherwise unchanged baseline."""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from sphere_refinement import fit_rgbd_sphere, refine


def select_refinement(selected, route, folder, start, ik, path, checker, closed_checker):
    folder = Path(folder)
    original = copy.deepcopy(selected)
    (folder / 'selected_grasp_original.json').write_text(json.dumps(original, indent=2))
    report = {'hypothesis': 'H008', 'scope': 'Observed shape condition; no object label routing',
              'applied': False, 'fallback': 'Unchanged baseline proposal',
              'pad_height_m': .010, 'p95_residual_limit_m': .0015}

    def save(reason):
        report['reason'] = reason
        (folder / 'shape_policy.json').write_text(json.dumps(report, indent=2))

    try:
        fit = fit_rgbd_sphere(folder)
    except ValueError as exc:
        save('Sphere fit rejected: ' + str(exc))
        return selected, route
    report['fit'] = fit
    (folder / 'sphere_fit.json').write_text(json.dumps(fit, indent=2))
    if not fit['accepted']:
        save('Observed surface does not meet the sphere-fit criteria')
        return selected, route
    candidate = refine(original, fit, height_m=.010)
    q0 = torch.tensor(original['q_grasp'], dtype=torch.float32, device=start.device)
    qg = ik(np.asarray(candidate['grasp_world']), q0)
    if qg is None:
        save('Refined grasp IK/collision rejected')
        return selected, route
    qp = ik(np.asarray(candidate['pregrasp_world']), qg)
    ql = ik(np.asarray(candidate['lift_world']), qg)
    if qp is None or ql is None:
        save('Refined pregrasp/lift IK/collision rejected')
        return selected, route
    for key, q in [('q_pre', qp), ('q_grasp', qg), ('q_lift', ql)]:
        candidate[key] = q.tolist()
    open_valid = checker.validate(torch.stack([qp, qg])[:, None]).flatten().tolist()
    closed_valid = closed_checker.validate(torch.stack([qg, ql])[:, None]).flatten().tolist()
    report.update(open_goals_valid=open_valid, nominal_closed_goals_valid=closed_valid)
    if not all(open_valid + closed_valid):
        save('Refined goals rejected under unchanged open/closed world models')
        return selected, route
    refined_route = path(start, qp)
    approach = path(qp, qg)
    if refined_route is None or approach is None:
        save('Refined pregrasp/approach PRM rejected')
        return selected, route
    report['applied'] = True
    report['candidate'] = candidate
    save('Refinement passed existing IK and open/closed path checks')
    return candidate, refined_route
