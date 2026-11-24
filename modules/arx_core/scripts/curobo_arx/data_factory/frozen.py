"""Read-only composed USD/live runtime comparison under clone namespaces."""
import copy
import sys
from .common import ROOT, digest
from .vector_env import ScopedStage


def canonical_physics(invariant, root=None, isolation=None):
    """Normalize only declared clone layout/cross-edges and renderer proxies."""
    current=copy.deepcopy(invariant)
    def local_path(path):
        return '/World'+path[len(root):] if root and (path==root or path.startswith(root+'/')) else path
    usd={}
    for path,value in current['usd'].items():
        if root and path==root:continue
        if root and value['type']=='PhysicsCollisionGroup':
            targets=value['relationships'].get('physics:filteredGroups',[])
            cross=isolation['cross_environment_groups']
            expected=isolation['original_filters'][path]
            if set(targets)!=set(expected+cross):
                raise RuntimeError('Collision filters do not match original plus declared cross-environment exclusions')
            value['relationships']['physics:filteredGroups']=[p for p in targets if p not in cross]
        value['relationships']={k:[local_path(p) for p in paths] for k,paths in value['relationships'].items()}
        if not local_path(path).startswith('/World/COOLASSensors/'):
            usd[local_path(path)]=value
    current['usd']=usd
    current['runtime_objects']={local_path(p):v for p,v in current['runtime_objects'].items()}
    current.pop('configured',None)
    return current


def fingerprint(bridge, historical):
    from arx_sim.diagnostics.research_fingerprint import capture
    context = bridge.context
    stage = context.stage
    try:
        context.stage = ScopedStage(stage,bridge.environment_root,include_globals=True)
        optimization = getattr(bridge,'fingerprint_optimization',None)
        if optimization:
            from .fast_fingerprint import capture_fast
            try:
                raw,timing = capture_fast(bridge,capture,optimization)
            except (RuntimeError,ValueError) as error:
                raise RuntimeError('Frozen physical configuration mismatch: optimized readback failed') from error
        else:
            raw = capture(bridge);timing = None
    finally:
        context.stage = stage
    current=canonical_physics(raw['invariant'],bridge.environment_root,bridge.collision_isolation)
    extra_review=None
    if getattr(bridge,'additional_asset_contracts',None):
        from scripts.curobo_arx.sdg_stage1.asset_contract import check_and_separate
        current,extra_review=check_and_separate(current,bridge.additional_asset_contracts,bridge.root/'additional_asset_runtime_baseline.json')
    reference=canonical_physics(historical['fingerprint']['invariant'])
    sys.path.insert(0,str(ROOT/'scripts/curobo_arx/static_pick'))
    from reproduce_setup import differences
    mismatches = differences(reference,current)
    if extra_review and not extra_review['pass']:mismatches+=extra_review['issues']
    return {'pass':not mismatches,'mismatches':mismatches,'raw':raw,'capture_timing':timing,
            'canonical_sha256':digest(current),'historical_sha256':digest(reference),
            'excluded':'Declared clone parent translation and physics-free renderer proxies; calibrated mounts retained',
            'scope':'Composed collision/static physics plus all available historical live robot/object getters; no setters',
            'additional_assets':extra_review}
