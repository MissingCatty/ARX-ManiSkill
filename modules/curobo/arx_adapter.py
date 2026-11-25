"""ARX planning adapter; no perception model imports."""
import importlib.util
import sys
from arx_project.layout import ROOT, module_dir, read_json
def create(args, *, pool_extension=None):
    from modules.curobo.fingertip_contact import load_profile
    profile = load_profile()
    current = read_json(ROOT/'configs/experiments/active_experiment.json')
    if current.get('table_contact_policy_id') != profile.profile_id or current['collision_profile_id'] != profile.collision_profile_id:
        raise ValueError('Unregistered fingertip contact policy')
    expected = profile.maximum_penetration_m if profile.enabled else 0.
    if current['table_penetration_allowance_m'] != expected:
        raise ValueError('Fingertip contact allowance differs from the current registration')
    MS=module_dir("maniskill")
    sys.path[:0]=[str(MS),str(MS/"arx_pipeline")]
    spec = importlib.util.spec_from_file_location('ray_ggx_existing_physics', module_dir('maniskill') / 'compat_tools/physical_table_pair_20261118_01.py')
    policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)
    policy.install_collision_policy()
    import resident_planner
    from fast_resident_planner import FastPlannerPool
    from checked_native_services import checked_pool
    from short_lift_planner import ShortLiftPlanner
    resident_planner.GraspCLPlanner = ShortLiftPlanner
    base = FastPlannerPool
    if profile.enabled:
        from modules.curobo.fingertip_contact.planning import contact_pool
        base = contact_pool(base, profile)
    if pool_extension is not None:
        base = pool_extension(base)
    return checked_pool(base)(args.root / 'planner_warmup', args.warmup_trial)

