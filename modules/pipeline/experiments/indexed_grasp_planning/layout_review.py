"""Independent planning review plus exact-input collision-layout evidence."""
import json

from arx_project.layout import write_json
from .validation_review import validate


def check_layout(run, result):
    reg = json.loads((run/'registration.json').read_text())
    geometry = json.loads((run/'resident/layout_geometry_preflight/acceptance.json').read_text())
    ready = json.loads((run/'resident/layout_readiness.json').read_text())
    if not all([geometry['passed'], geometry['full_collision_cost_and_gradient_equal'],
                geometry['captured_cases'] == 5, geometry['checked_spheres'] == 5*2048*871,
                ready['passed'], ready['shared_scene_and_robot_parameters'],
                ready['original_and_fast_full_IK_have_separate_CUDA_graphs']]):
        raise ValueError('Captured collision or separate-graph preflight did not pass')
    if not all(geometry['contract'].values()) or geometry['contract'] != ready['contract']:
        raise ValueError('The collision-layout contract changed during warmup')
    for case in reg['cases']:
        for variant in reg['variants']:
            receipt = json.loads((run/'plans'/case['id']/variant/'collision_layout.json').read_text())
            if receipt['variant'] != reg['variant_modes'][variant] or not all(receipt[k] for k in
                    ['geometry_and_thresholds_unchanged', 'separate_CUDA_graphs', 'original_module_unchanged']):
                raise ValueError('Measured collision layout lost its actual evidence')
            if receipt['contract'] != geometry['contract']:
                raise ValueError('Measured collision layout differs from the preflight')
    result.update(actual_collision_spheres_checked=geometry['checked_spheres'],
                  complete_cost_and_gradient_verified=True,
                  original_and_fast_CUDA_graphs_separate=True, profiling_enabled=False)
    write_json(run/'independent_review.json', result)
    return result


def review(run):
    return check_layout(run, validate(run))
