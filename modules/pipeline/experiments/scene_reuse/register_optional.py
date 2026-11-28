"""Register only the optional module and refresh the registry's recorded digest."""
from copy import deepcopy

from arx_project.layout import ROOT, now, read_json, sha256, write_json


def register():
    from modules.pipeline.experiments.current_configuration.configuration import load_current
    load_current()
    registry = ROOT/'configs/modules.json'
    active = ROOT/'configs/experiments/active_experiment.json'
    receipt = ROOT/'configs/experiments/history/scene_reuse_optional_20261128_01.json'
    if receipt.exists(): raise ValueError('Optional module registration already has a receipt')
    old_registry, old_active = read_json(registry), read_json(active)
    if 'curobo_scene_reuse' in old_registry['modules']: raise ValueError('Do not overwrite an existing module')
    previous = {str(p): sha256(p) for p in [registry, active]}
    backup = receipt.with_name(receipt.stem+'_previous.json')
    write_json(backup, {'modules': old_registry, 'active_experiment': old_active})
    changed = deepcopy(old_registry)
    changed['modules']['curobo_scene_reuse'] = {
        'source': 'modules/curobo/scene_reuse', 'role': 'motion_planning',
        'environment': 'environments/curobo', 'adapter': 'modules/curobo/scene_reuse/runtime.py',
        'experimental': True, 'default_enabled': False,
        'inputs': ['state.json', 'candidates/candidate_pool.json'],
        'outputs': ['plan_result.json', 'cpu_geometry.json', 'stage_timings.json'],
        'policies': ['configs/experiments/scene_reuse_cpu.json', 'configs/policies/fingertip_table.json'],
        'variant': 'CPU_prepared_arrays_with_original_native_GPU_loader'}
    write_json(registry, changed)
    updated = deepcopy(old_active)
    updated['configuration_hashes'][str(registry)] = sha256(registry)
    write_json(active, updated)
    load_current()
    restored = deepcopy(changed); del restored['modules']['curobo_scene_reuse']
    normalized = deepcopy(updated)
    normalized['configuration_hashes'][str(registry)] = old_active['configuration_hashes'][str(registry)]
    if restored != old_registry or normalized != old_active:
        raise RuntimeError('Optional registration changed existing behavior')
    write_json(receipt, {'at': now(), 'previous_sha256': previous,
        'current_sha256': {str(p): sha256(p) for p in [registry, active]},
        'backup': str(backup.relative_to(ROOT)), 'only_optional_definition_and_registry_digest_changed': True,
        'active_planner_unchanged': True})


if __name__ == '__main__': register()
