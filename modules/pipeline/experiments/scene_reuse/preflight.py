"""Exact native FK, collision masks, costs and gradients with CPU reuse."""
from pathlib import Path
import numpy as np

from arx_project.layout import ROOT, now, read_json, write_json
from modules.curobo.scene_cache import mesh_stores
from modules.curobo.scene_reuse.prepared_meshes import PreparedMeshCache
from modules.graspgenx.native_postprocessing.width_planning import target_state


def cold_meshes(planner):
    import torch
    torch.cuda.synchronize()
    stores = mesh_stores(planner)
    counts = [len(store.wp_cache) for store in stores]
    for store in stores:
        store.clear(clear_warp_cache=True)
    torch.cuda.synchronize()
    if any(store.wp_cache or store.get_names(0) for store in stores):
        raise RuntimeError('Formal scene geometry cache was not cold')
    return {'stores': len(stores), 'released_mesh_entries': sum(counts),
            'native_scene_tensor_addresses_retained': True}


def compare_state(planner, state, output):
    import torch
    import warp as wp
    from scipy.spatial.transform import Rotation
    pool = planner.pool; reuse = planner.cpu_geometry_reuse; values = {}; traces = {}
    for mode in ['control', 'cached']:
        cold_meshes(planner)
        cache = PreparedMeshCache(ROOT / 'tmp/curobo/mesh_cache' / output.parent.name / output.name) if mode == 'cached' else None
        reuse.begin(cache); pool.memory_updates_enabled = True; pool.memory_bindings.begin()
        try:
            context = pool.bind(output / mode, state, include_target=False)
            validator = pool.contact_view(context, state)
            q = context.measured(state)
            offsets = torch.linspace(-.03, .03, 32, device=q.device)[:, None]
            qs = (q[None] + offsets * torch.tensor([1., -.4, .6, -.3, .2, -.5], device=q.device)[None])
            qs = qs.clamp(context.limits[:, 0]+1e-4, context.limits[:, 1]-1e-4)
            spheres = context.checker.get_kinematics(qs[:, None]).robot_spheres.detach().clone()
            result = {'spheres': spheres.cpu().numpy().copy(),
                      'native': context.checker.validate(qs[:, None]).cpu().numpy().copy(),
                      'local': validator.validate(qs, stage='final_approach').copy()}
            # Surface probes ensure cost/gradient comparisons include real mesh contact.
            scene = read_json(output / mode / 'scene.json')
            name, spec = next(iter(scene['mesh'].items()))
            store = context.checker.scene_model.data.meshes
            key = next(n for n in store.wp_cache if n.startswith(name + '_'))
            point = wp.to_torch(store.wp_cache[key].mesh.points)[0].detach().cpu().numpy()
            pose = spec['pose']; rot = Rotation.from_quat([*pose[4:], pose[3]]).as_matrix()
            center = point @ rot.T + np.asarray(pose[:3])
            probes = spheres.clone(); probes[..., 3] = -1.
            probes[:, 0, 0, :3] = torch.as_tensor(center, device=q.device, dtype=q.dtype)[None]
            probes[:, 0, 0, :3] += offsets * torch.tensor([.1, .2, .3], device=q.device)
            probes[:, 0, 0, 3] = .02
            for label, checker in [('world', context.checker), ('non_table', validator.other_checker)]:
                distance, gradient = checker.get_collision_vector(probes.clone())
                result[label + '_cost'] = distance.cpu().numpy().copy()
                result[label + '_gradient'] = gradient.cpu().numpy().copy()
            values[mode] = result
        finally:
            pool.memory_bindings.finish(); pool.memory_updates_enabled = False
            traces[mode] = reuse.finish()
    exact = {key: np.array_equal(values['control'][key], values['cached'][key]) for key in values['control']}
    if not all(exact.values()):
        raise RuntimeError('Prepared geometry changed native GPU state: ' + str(exact))
    if not np.any(values['control']['world_cost']) or not np.any(values['control']['world_gradient']):
        raise RuntimeError('Surface probes did not exercise mesh contact costs and gradients')
    write_json(output / 'review.json', {'at': now(), 'passed': True, 'states': 32,
        'exact_arrays': exact, 'nonzero_contact_cost_and_gradient': True, 'traces': traces})
    return {'passed': True, 'states': 32, 'exact_arrays': exact}


def run(planner, root, warmup, cases):
    reports = []
    fixtures = [('warmup', read_json(warmup/'state.json'), [.088, .060, .040])]
    fixtures += [(case['id'], read_json(Path(case['source_directory'])/'state.json'), [.088]) for case in cases]
    for cid, state, widths in fixtures:
        for arm in ['right', 'left']:
            canonical = planner.model.bind(state, arm, root / cid / arm / 'physical_arm')
            for width in widths:
                output = root / cid / arm / str(int(width*1000))
                report = compare_state(planner, target_state(canonical, width), output)
                reports.append({'case': cid, 'arm': arm, 'width_m': width, **report})
                write_json(root / 'progress.json', {'at': now(), 'pairs': len(reports), 'total_pairs': 16})
    result = {'at': now(), 'passed': True, 'pairs': len(reports), 'rows': reports,
              'compared_joint_states': sum(r['states'] for r in reports), 'physical_grasps': 0}
    if len(reports) != 16: raise RuntimeError('Wrong component validation cohort')
    write_json(root / 'acceptance.json', result)
    return result
