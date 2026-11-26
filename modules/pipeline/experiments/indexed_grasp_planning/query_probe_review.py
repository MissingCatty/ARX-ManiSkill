"""Check captured workload ownership and the component result equivalence."""
import json
from pathlib import Path

import numpy as np

from arx_project.layout import sha256, write_json
from .validation_review import validate


def review(run):
    result = validate(run)
    reg = json.loads((run/'registration.json').read_text());samples = []
    for case in reg['cases']:
        for variant, mode in reg['variant_modes'].items():
            directory = run/'plans'/case['id']/variant
            receipt = json.loads((directory/'query_probe.json').read_text())
            if mode == 'control':
                if receipt['samples']: raise ValueError('Control request unexpectedly ran component probes')
                continue
            if len(receipt['samples']) != 1: raise ValueError('Missing captured actual-query workload')
            sample = receipt['samples'][0];data = directory/'query_components'
            for key, filename in [('query_sha256', 'actual_queries.npz'), ('mesh_sha256', 'actual_meshes.npz'),
                                  ('scene_sha256', 'scene.json')]:
                if sha256(data/filename) != sample[key]: raise ValueError('Actual query/mesh snapshot changed')
            with np.load(data/'actual_queries.npz', allow_pickle=False) as arrays:
                q = arrays['queries'];slots = arrays['mesh_slots'];found = arrays['nearest_found']
                if (len(q) != sample['actual_narrow_phase_queries'] or len(q) != len(slots)
                        or len(q) != len(found) or not np.isfinite(q).all() or not np.isfinite(arrays['spheres']).all()
                        or not ((slots >= 0) & (slots < sample['active_meshes'])).all()
                        or int(found.sum()) != sample['nearest_found']):
                    raise ValueError('Captured query identities or counts are inconsistent')
            with np.load(data/'actual_meshes.npz', allow_pickle=False) as arrays:
                for index in range(sample['active_meshes']):
                    vertices = arrays['vertices_'+str(index)];faces = arrays['faces_'+str(index)]
                    if not np.isfinite(vertices).all() or not ((faces >= 0) & (faces < len(vertices))).all():
                        raise ValueError('Actual GPU mesh snapshot is invalid')
            if not sample['nearest_face_UV_and_ray_sign_bitwise_equal']:
                raise ValueError('Component probes did not match the original combined query')
            samples.append({'case': case['id'], **sample})
    result.update(component_probe_samples=len(samples), actual_workloads_verified=True,
                  original_planning_decisions_retained=True, component_samples=samples)
    write_json(run/'independent_review.json', result)
    return result
