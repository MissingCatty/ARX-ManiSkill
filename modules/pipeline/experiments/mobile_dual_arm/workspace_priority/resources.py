"""Place resident components on freshly tested GPUs without stopping others."""
from itertools import permutations

from arx_project.layout import read_json, write_json
from modules.pipeline.experiments.dual_arm_grasp.run import allocation_probes, gpu_inventory


def select_resources(*, ray_minimum_mib=18000):
    free = gpu_inventory()
    probes = allocation_probes(free)
    healthy = [gpu for gpu in free if probes[gpu]['passed']]
    choices = [(min(free[r]-ray_minimum_mib, free[s]-12000, free[p]-10000), r, s, p)
               for r, s, p in permutations(healthy, 3)
               if free[r] >= ray_minimum_mib and free[s] >= 12000 and free[p] >= 10000]
    if choices:
        _, ray, sam, plan = max(choices)
        return ({'rayst3r': ray, 'sam3': sam, 'graspgenx': sam,
                 'curobo_dual_arm': plan, 'simulation': plan},
                {ray: ray_minimum_mib, sam: 12000, plan: 10000}, probes)
    # RaySt3R remains separate; the other models and sequential native planner
    # share an initially empty card. Actual post-warmup headroom is still checked
    # by Services.start before any formal scene can be claimed.
    pairs = [(min(free[r]-ray_minimum_mib, free[p]-23000), r, p)
             for r, p in permutations(healthy, 2)
             if free[r] >= ray_minimum_mib and free[p] >= 23000]
    if not pairs:
        raise RuntimeError(f'Need three admitted GPUs or two healthy GPUs with {ray_minimum_mib}/23000 MiB free')
    _, ray, other = max(pairs)
    return ({'rayst3r': ray, 'sam3': other, 'graspgenx': other,
             'curobo_dual_arm': other, 'simulation': other},
            {ray: ray_minimum_mib, other: 23000}, probes)


def verify_ready(root, registration):
    rows = []
    for role in registration['required_components']:
        kind = registration.get('component_modules', {}).get(role, role)
        record = read_json(root/'resident'/role/(kind+'_ready.json'))
        if (record['status'] != 'ready' or record['gpu'] != registration['gpu_roles'][role]
                or not record['warmup_passed'] or record['model_load_count'] != 1):
            raise RuntimeError('Preflight component/device binding differs: '+role)
        rows.append({'role': role, 'kind': kind, 'gpu': record['gpu'],
                     'warmup_passed': True, 'model_load_count': 1})
    free = gpu_inventory()
    if any(free.get(gpu, 0) < 1536 for gpu in set(registration['gpu_roles'].values())):
        raise RuntimeError('Insufficient GPU headroom after full high-workspace preflight')
    write_json(root/'preflight/gpu_assignment_verified.json', {'passed': True,
        'components': rows, 'remaining_mib': free, 'minimum_remaining_mib': 1536,
        'used_gpu_count': len(set(registration['gpu_roles'].values()))})
