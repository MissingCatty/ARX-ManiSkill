"""ARX planning adapter; no perception model imports."""
import importlib.util
import sys
from arx_project.layout import module_dir
def create(args):
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
    return checked_pool(FastPlannerPool)(args.root / 'planner_warmup', args.warmup_trial)

