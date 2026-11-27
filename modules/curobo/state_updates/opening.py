"""Original dense opening check, consuming the bound in-memory model."""
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from arx_project.layout import write_json
from modules.curobo.dual_arm import accelerated_adapter as native
from modules.curobo.gripper_sweep.direct_sweep import native_opening_spheres


def opening_sweep(pool, work, canonical):
    import torch
    work = Path(work)
    began = time.perf_counter()
    fractions, target, observed = native.opening_samples(canonical)
    initial = deepcopy(canonical)
    for name, value in observed.items():
        initial['qpos'][initial['joint_names'].index(name)] = float(value)
    try:
        context = pool.bind(work / 'opening_batch_start', initial, include_target=True)
    except AssertionError as error:
        if str(error) != 'Measured state collides in unchanged ARX model':
            raise
        result = {'passed': False, 'reason': 'gripper_opening_sweep_in_collision', 'fraction': 0.,
                  'samples_checked': 1, 'first_invalid_sample': 0, 'method': 'native_start_gate',
                  'wall_seconds': time.perf_counter() - began}
        write_json(work / 'opening_sweep.json', result)
        return result
    config = pool.memory_bindings.config(context)
    q, checker = context.measured(initial)[None, None], context.checker
    try:
        batch = native_opening_spheres(pool, context, config, q, fractions, target, observed)
        if batch.shape[2] != 871:
            raise ValueError('Opening check must retain the approved 871-sphere model')
        valid = native.validate_sphere_batch(checker, batch, q.expand(len(fractions), -1, -1)).reshape(-1)
        torch.cuda.synchronize()
        mask = valid.detach().cpu().numpy()
        failures = np.flatnonzero(~mask)
        first = int(failures[0]) if len(failures) else None
        result = {'passed': first is None, 'samples_checked': len(fractions), 'first_invalid_sample': first,
            'fraction': None if first is None else float(fractions[first]),
            'reason': 'native_clear' if first is None else 'gripper_opening_sweep_in_collision',
            'maximum_finger_step_m': .0005, 'target_per_finger_m': target,
            'target_and_parked_arm_collision_checked': True, 'method': 'in_memory_model_original_native_FK',
            'sphere_count': 871, 'world_bind_count': 2, 'sampling_fractions': fractions.tolist(),
            'valid_per_sample': mask.tolist(), 'wall_seconds': time.perf_counter() - began}
        write_json(work / 'opening_sphere_batch.json', {'shape': list(batch.shape),
            'finite': bool(torch.isfinite(batch).all()), 'native_affine_lock_refresh': True})
    finally:
        try:
            pool.bind(work / 'opening_batch_restore', canonical, include_target=True)
        except AssertionError as error:
            if str(error) != 'Measured state collides in unchanged ARX model':
                raise
            if 'result' not in locals() or result['passed']:
                raise RuntimeError('Opening batch disagrees with native endpoint gate') from error
    result['wall_seconds'] = time.perf_counter() - began
    write_json(work / 'opening_sweep.json', result)
    return result
