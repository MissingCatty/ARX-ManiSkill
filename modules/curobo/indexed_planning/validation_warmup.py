"""Compare batched and scalar collision/FK on the actual resident worlds."""
from pathlib import Path

import numpy as np
from arx_project.layout import read_json, write_json
from .branch_validation import validate_rows


def verify_row_validation(planner, source_state, directory):
    import torch
    reports = []
    for side in ['right', 'left']:
        work = Path(directory)/side
        state = planner.model.bind(read_json(source_state), side, work/'canonical')
        context = planner.pool.bind(work/'binding', state, include_target=False)
        validator = planner.pool.contact_view(context, state)
        start = context.measured(state)
        generator = torch.Generator(device=start.device).manual_seed(20261126)
        random = torch.rand((32, 6), generator=generator, device=start.device)
        lower, upper = context.limits.T
        samples = torch.cat([start[None], start[None]+.001,
            lower[None]+random*(upper-lower)[None], (upper+.001)[None]], dim=0)
        scalar, scalar_reasons = [], []
        for row in samples:
            valid = bool(validator.validate(row[None], stage='final_approach').all())
            receipt = validator.receipts[-1]
            scalar.append(valid)
            scalar_reasons.append(None if valid else 'joint_limits' if not receipt['joint_bounds_clear']
                else 'ground_or_other_object_collision' if not receipt['other_world_clear']
                else receipt['narrow_phase'][-1]['reason'] if receipt['narrow_phase'] else 'native_collision')
        batched, reasons = validate_rows(validator, samples)
        if not np.array_equal(scalar, batched) or scalar_reasons != reasons:
            raise RuntimeError('Actual-world scalar and batch collision decisions differ')
        values = samples.cpu().numpy()
        full = validator.geometry.frames(values)['right_eef_link']
        individual = np.concatenate([validator.geometry.frames(q[None])['right_eef_link'] for q in values])
        if not np.allclose(full, individual, rtol=0, atol=1e-12):
            raise RuntimeError('Independent vector FK differs from scalar FK')
        reports.append({'arm': side, 'rows': len(samples), 'passed_rows': int(batched.sum()),
            'failed_rows': int((~batched).sum()), 'same_decisions_and_reasons': True,
            'maximum_FK_element_difference': float(np.max(np.abs(full-individual)))})
    result = {'passed': True, 'reports': reports, 'actual_simulator_connections': 0,
              'formal_planning_requests': 0, 'physical_grasps': 0}
    write_json(Path(directory)/'row_validation.json', result)
    return result
