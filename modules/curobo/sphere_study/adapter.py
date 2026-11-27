"""Explicit diagnostic references inside one isolated planner process."""
import ast
from copy import deepcopy
from functools import partial
import inspect
from pathlib import Path
import textwrap
from types import FunctionType

from arx_project.layout import read_json,write_json
from .profiles import digest


class DiagnosticProfile:
    def __init__(self,profile):
        self.profile=profile

    def require(self,config):
        kin=config.get('robot_cfg',config)['kinematics']
        spheres=kin['collision_spheres']
        if digest(spheres)!=self.profile['sphere_parameter_sha256']:
            raise ValueError('Diagnostic sphere parameters differ from registration')
        if sum(map(len,spheres.values()))!=self.profile['total_spheres']:
            raise ValueError('Diagnostic sphere count differs from registration')
        if set(kin['collision_link_names'])!=set(spheres) or kin.get('collision_sphere_buffer',0.)!=0.:
            raise ValueError('Diagnostic collision links/buffer changed')
        return {'profile_id':self.profile['profile_id'],'total_spheres':self.profile['total_spheres'],
            'sphere_parameter_sha256':self.profile['sphere_parameter_sha256'],
            'counts':{n:len(v) for n,v in spheres.items()},'planning_diagnostic_only':True,
            'source_871_parameter_sha256':self.profile['source_parameter_sha256']}


def opening_with_count(original,count):
    """Only replace the two hardcoded count constants in a private function."""
    class Count(ast.NodeTransformer):
        replacements=0
        def visit_Constant(self,node):
            if type(node.value) is int and node.value==871:
                self.replacements+=1;return ast.copy_location(ast.Constant(count),node)
            return node
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)));change=Count()
    tree=ast.fix_missing_locations(change.visit(tree))
    if change.replacements!=2:raise RuntimeError('Native opening count gate changed; review diagnostic adapter')
    namespace=dict(original.__globals__)
    exec(compile(tree,__file__+':registered_count','exec'),namespace)
    return namespace[original.__name__]


def create(args):
    from modules.curobo.dual_arm.adapter import create as original_create
    from modules.curobo.dual_arm.interleaved_adapter import InterleavedDualArmPlanner,configure
    from modules.graspgenx.native_postprocessing.width_planning import WidthSelectionMixin
    from modules.curobo.gripper_sweep.adapter import optimized_pool
    from modules.curobo.gripper_sweep.codec_adapter import attach_codec
    from modules.curobo.gripper_sweep.direct_sweep import native_opening_spheres
    from modules.curobo.gripper_sweep import cached_sweep
    import modules.graspgenx.native_postprocessing.width_planning as widths
    profile=read_json(args.profile);registered=DiagnosticProfile(profile);original_binding=widths.binding_proof

    def proof(*a,**kw):
        result=original_binding(*a,**kw)
        result.update(same_871_spheres_and_URDF=profile['total_spheres']==871,
            actual_native_sphere_count=profile['total_spheres'],diagnostic_profile_id=profile['profile_id'])
        return result
    widths.binding_proof=proof
    cached_sweep.opening_sweep=opening_with_count(cached_sweep.opening_sweep,profile['total_spheres'])

    def extend(base):
        import resident_planner as resident
        # These module objects exist only in this child. Files and active
        # project registration remain untouched; both guards validate the
        # explicit registered diagnostic profile instead of accepting anything.
        resident.frozen.ARX=Path(args.reference_root)
        resident.frozen.require_current_profile=registered.require
        resident.require_current_profile=registered.require
        return optimized_pool(base)
    original=original_create(args,pool_extension=extend)

    class Planner(WidthSelectionMixin,InterleavedDualArmPlanner):
        def opening_sweep(self,work,canonical):
            with self.pool.stage_timing.stage('gripper_opening'):
                return cached_sweep.opening_sweep(self.pool,work,canonical)

    planner=configure(Planner(original.pool,original.model,original.root))
    attach_codec(planner.pool);pool=planner.pool
    pool.batch_sweep_enabled=True;pool.fast_binding_enabled=True;pool.fast_sweep_enabled=True
    pool.make_opening_spheres=partial(native_opening_spheres,pool)
    pool.diagnostic_sphere_count=profile['total_spheres']
    pool.diagnostic_profile=profile
    return planner
