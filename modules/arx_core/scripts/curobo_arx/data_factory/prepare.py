"""Register independent production inputs and freeze a reviewable source copy."""
from arx_project.layout import canonical_path
import argparse
import copy
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import time
from .catalog import build,historical,select_collection,collection_input
from .common import ROOT,VERSION,digest,read,sha,space,write
from .ledger import Ledger


def composed_geometry(catalog):
    """Exact shape grouping ignores name/texture, with explicit near-shape limit."""
    from pxr import Usd,UsdGeom,UsdPhysics
    import numpy as np
    dependencies = {}
    for asset in catalog['objects']:
        if not asset['available']:
            continue
        stage = Usd.Stage.Open(asset['usd_path'])
        default = stage.GetDefaultPrim()
        if not default or abs(UsdGeom.GetStageMetersPerUnit(stage)-1.)>1e-9:
            asset['available']=False
            asset['issues'].append('Missing default prim or unregistered non-metre USD units')
            continue
        geometry = []
        cache = UsdGeom.XformCache()
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                attrs = {a.GetName():str(a.Get()) for a in prim.GetAttributes() if a.GetName() in ('points','faceVertexIndices','faceVertexCounts','radius','height','size','axis') or a.GetName().startswith(('physics:','physxCollision:'))}
                geometry.append({'type':prim.GetTypeName(),'attrs':attrs,'local_transform':np.asarray(cache.GetLocalToWorldTransform(prim)).tolist()})
        if not geometry:
            asset['available']=False
            asset['issues'].append('No composed collision geometry')
            continue
        asset['geometry_group'] = digest(sorted(geometry,key=digest))
        for layer in stage.GetUsedLayers():
            if layer.realPath:
                dependencies[layer.realPath] = sha(layer.realPath)
    known = {}
    from .catalog import KNOWN_ROLES
    for asset in catalog['objects']:
        group = asset.get('geometry_group')
        if group and asset['object_id'] in KNOWN_ROLES:
            if group in known and known[group]!=asset['split']:
                raise ValueError('Identical geometry has conflicting established object roles')
            known[group] = asset['split']
    for asset in catalog['objects']:
        if 'geometry_group' in asset:
            group = asset['geometry_group']
            asset['split'] = known.get(group) or ('validation' if int(digest([catalog['seed'],group])[:8],16)%5==0 else 'train')
    catalog['catalog_sha256'] = digest(catalog['objects'])
    return dependencies


def prepare(out, *, extra=None, max_attempts=None, envs=4, wall_hours=24, block=16, first_object=None,
            candidate_backend='zerograsp', sampling_policy=None, quota_overrides=None, object_policy='common_household',
            placement_profile='active',only_objects=None,distances_mm=None):
    out = Path(out).absolute()
    space(out.parent)
    if out.exists():
        raise FileExistsError('Create a new immutable registration, never overwrite an old campaign')
    if envs<1 or envs>16 or wall_hours<=0 or block<1:
        raise ValueError('Invalid bounded environment/time/block configuration')
    if candidate_backend not in ('zerograsp','sdg','anygrasp','graspness'):
        raise ValueError('Unknown candidate source')
    extra, object_profile = collection_input(extra, object_policy)
    catalog = select_collection(build(extra), object_policy)
    if only_objects is not None:
        from .placements import validation_subset
        catalog=validation_subset(catalog,only_objects)
    if first_object is not None:
        selected=next((a for a in catalog['objects'] if a['object_id']==first_object and a['available'] and a.get('collection_enabled',True)),None)
        if selected is None:raise ValueError('Registered first object must exist and be available')
    dependencies = composed_geometry(catalog)
    if extra is not None:dependencies[str(Path(extra).resolve())] = sha(extra)
    if object_profile is not None:dependencies[str(object_profile)] = sha(object_profile)
    placement_path=None
    if placement_profile=='active':
        selected_profile=read(object_profile).get('placement_profile') if object_profile is not None else None
        if selected_profile:placement_path=ROOT/selected_profile
    elif placement_profile!='historical':
        placement_path=Path(placement_profile).resolve()
    placement_registration=None
    if placement_path is not None:
        from .placements import placement_bank,support_surface
        from scripts.curobo_arx.sdg_stage1.sampler import composed_mesh
        config=read(placement_path)
        environment=historical('RaytracedLighting')['fingerprint']['invariant']['configured']['simulation']['environment']
        if environment.startswith('/workspace/asset/'):environment=str(canonical_path(ROOT/'asset'/environment[len('/workspace/asset/'):]))
        support=support_surface(environment,config['support_prim_suffix'])
        dependencies.update(support['dependencies']);dependencies[str(placement_path)]=sha(placement_path)
        for asset in catalog['objects']:
            if not asset.get('available') or not asset.get('collection_enabled',True):continue
            mesh,_=composed_mesh(asset['usd_path'])
            bank=placement_bank(asset,mesh.vertices,config,support,catalog['seed'])
            asset['collection_placements']=bank['placements']
            asset['collection_placement_rejections']=bank['rejected']
        block=config['block_attempts']
        if not isinstance(block,int) or block<1:raise ValueError('Invalid placement block bound')
        placement_registration={'profile':str(placement_path),'profile_sha256':sha(placement_path),
            'configuration':config,'support':support,'scope':config['scope'],
            'requested_bins_per_object':len(config['xy_offsets_m'])*len(config['yaw_degrees']),
            'registered_supported_bins':sum(len(a.get('collection_placements',[])) for a in catalog['objects']),
            'actual_physical_coverage':0,'historical_benchmark_placements_unchanged':True}
        catalog['catalog_sha256']=digest(catalog['objects'])
    profile={'candidate_reserve':64,'goals_per_object':16,'successes_per_bin':8}
    if quota_overrides:
        if set(quota_overrides)-set(profile):raise ValueError('Unsupported quota override')
        if any(not isinstance(v,int) or v<1 for v in quota_overrides.values()):raise ValueError('Positive integer quotas required')
        profile.update(quota_overrides)
    if profile['goals_per_object']>profile['candidate_reserve']:raise ValueError('Goal quota exceeds reserve')
    target_count = sum(a['available'] and a.get('collection_enabled',True) for a in catalog['objects']) if object_policy!='all' or only_objects is not None else len(catalog['objects'])
    strata=sum(len(a.get('collection_placements',[None])) for a in catalog['objects']
        if a['available'] and a.get('collection_enabled',True)) if placement_registration else target_count
    distances=[10,40,60,80] if distances_mm is None else list(distances_mm)
    if not distances or len(set(distances))!=len(distances) or any(type(d) is not int or not 1<=d<=300 for d in distances):
        raise ValueError('Register unique positive integer entry distances up to 300 mm')
    total = strata*profile['goals_per_object']*len(distances)*profile['successes_per_bin']
    maximum = total*4 if max_attempts is None else max_attempts
    if maximum<1:
        raise ValueError('Positive production attempt bound required')
    quotas = {**profile,'distances_mm':distances,
              'physical_per_bin':32,'planning_rejections_per_bin':64,
              'target_episodes':total,'max_physical_attempts':maximum,
              'max_total_dispatches':maximum+strata*profile['candidate_reserve']*len(distances)*64}
    if placement_registration:quotas['goal_quota_scope']='per registered object placement; all positions and yaw bins required for full coverage'
    if first_object is not None:quotas['first_object']=first_object
    if sampling_policy is not None:
        if sampling_policy!='paired_nominal_perturbed_rounds_v1':raise ValueError('Unknown entry scheduling policy')
        quotas['sampling_policy']=sampling_policy
    render_configs,references = {},{}
    def host_path(value):
        if isinstance(value,dict):return {k:host_path(v) for k,v in value.items()}
        if isinstance(value,list):return [host_path(v) for v in value]
        if isinstance(value,str):
            for prefix in ('/workspace/COOLAS-ARX/asset/','/workspace/asset/'):
                if value.startswith(prefix):return str(canonical_path(ROOT/'asset'/value[len(prefix):]))
        return value
    for renderer in ('RaytracedLighting','PathTracing'):
        ref = historical(renderer)
        references[renderer] = ref
        render_configs[renderer] = host_path(ref['fingerprint']['invariant']['configured'])
    # Source copy, independent from all existing S02 execution directories.
    out.mkdir(parents=True)
    source = out/'source_snapshot/COOLAS-ARX'
    source.mkdir(parents=True)
    files = {}
    for directory in ('shared','arx-ops','arx-sim','scripts/curobo_arx','configs','servers/zerograsp/coolas_zerograsp'):
        src = ROOT/directory
        for p in src.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.yaml','.yml','.json','.usda','.gz','.md'):
                dest = source/p.relative_to(ROOT)
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(p,dest)
                files[str(p)] = sha(p)
    (canonical_path(source/'asset')).symlink_to(canonical_path(ROOT/'asset'),target_is_directory=True)
    vendor=source/'servers/zerograsp/vendor'
    vendor.parent.mkdir(parents=True,exist_ok=True)
    vendor.symlink_to(ROOT/'servers/zerograsp/vendor',target_is_directory=True)
    (source.parent/'curobo').symlink_to(ROOT.parent/'curobo',target_is_directory=True)
    for p in (ROOT/'servers/zerograsp/vendor/zerograsp_mainline_minimal').rglob('*.py'):
        if '__pycache__' not in p.parts:
            files[str(p)] = sha(p)
    for p in (ROOT.parent/'curobo/source/curobo').rglob('*.py'):
        if '__pycache__' not in p.parts:
            files[str(p)] = sha(p)
    model = ROOT/'scripts/curobo_arx/static_pick/reference'
    for d in ('model','model_closed'):
        for p in (model/d).iterdir():
            if p.is_file():dependencies[str(p)] = sha(p)
    for renderer,ref in references.items():
        write(out/'references'/(renderer+'.json'),ref)
        for layer in ref['fingerprint'].get('used_layers',[]):
            resolved=Path(host_path(layer))
            if resolved.is_file():dependencies[str(resolved)]=sha(resolved)
    checkpoint = ROOT/'servers/zerograsp/vendor/zerograsp_mainline_minimal/checkpoints/zerograsp_cvpr2025/zerograsp_synthetic.ckpt'
    if candidate_backend=='zerograsp':dependencies[str(checkpoint)] = sha(checkpoint)
    dependencies[str(ROOT/'configs/lilo_vla/fixtures/tabletop_rt_v1/initial_setup.json')] = sha(ROOT/'configs/lilo_vla/fixtures/tabletop_rt_v1/initial_setup.json')
    storage = ROOT/'runs/research_contract_20261019/storage_policy.json'
    if not storage.exists():storage=ROOT/'configs/lilo_vla/fixtures/tabletop_rt_v1/storage_policy.json'
    shutil.copy2(storage,out/'storage_policy.json')
    registration = {'version':VERSION,'created_unix':time.time(),'seed':catalog['seed'],'catalog':catalog,
        'quotas':quotas,'envs_per_gpu':envs,'object_block_attempts':block,'wall_hours':wall_hours,
        'first_object':first_object,'object_order_scope':'Queue tie order within selected targets; frozen scene assets and their declared initial physics/placements retained',
        'object_policy':object_policy,'placement_diversity':placement_registration,
        'bounded_validation_objects':sorted(only_objects) if only_objects is not None else None,
        'object_profile':str(object_profile) if object_profile is not None else None,
        'diagnostics':{'fatal_signal_trace':True,'periodic_stack':'simulator_main_thread_safe_points',
                       'asynchronous_timed_frame_walk':False,'interval_seconds':120},
        'attempt_seconds':900,'render_configs':render_configs,'reference_files':{k:str(out/'references'/(k+'.json')) for k in references},
        'planning_mesh_capacity':256,'model_reference':str(model),'candidate_backend':candidate_backend,
        'zerograsp_checkpoint':str(checkpoint) if candidate_backend=='zerograsp' else None,
        'zerograsp_checkpoint_sha256':dependencies.get(str(checkpoint)),'source_files':files,'dependencies':dependencies,
        'source_snapshot':str(source),'source_identity':digest(files),'production':True,
        'devices':'resolved and appended in a separate immutable launch registration at start',
        'state_convention':'grasp_goal_world_error_v1','action_chunk':[8,7], 'sample_seconds':.2,
        'frozen':('No robot, camera, scene physics or original evaluator threshold changes; new collection target XY/yaw explicitly registered'
            if placement_registration else 'No robot, scene physics, declared initial state or original evaluator threshold changes'),
        'plan':'research_plan/docs/automated_teacher_pipeline_20261029/README.md'}
    registration['execution_source_files']={str(source/Path(p).relative_to(ROOT)):s for p,s in files.items()
        if Path(p).is_relative_to(ROOT) and (source/Path(p).relative_to(ROOT)).is_file()}
    registration['execution_source_files'].update({p:s for p,s in files.items() if not Path(p).is_relative_to(ROOT)})
    write(out/'catalog.json',catalog)
    write(out/'registration.json',registration)
    ledger = Ledger(out/'ledger.sqlite3')
    ledger.initialize(catalog,quotas)
    ledger.close()
    write(out/'state.json',{'status':'prepared','successes':0,'physical_attempts':0,'quota_complete':False})
    return registration


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--extra-catalog',type=Path)
    p.add_argument('--max-attempts',type=int)
    p.add_argument('--envs-per-gpu',type=int,default=4)
    p.add_argument('--wall-hours',type=float,default=24)
    p.add_argument('--object-block-attempts',type=int,default=16)
    p.add_argument('--first-object',help='Optional first object; remaining full catalog and per-object quotas stay registered')
    p.add_argument('--object-policy',choices=('common_household','all'),default='common_household')
    a=p.parse_args()
    r=prepare(a.out,extra=a.extra_catalog,max_attempts=a.max_attempts,envs=a.envs_per_gpu,wall_hours=a.wall_hours,block=a.object_block_attempts,first_object=a.first_object,object_policy=a.object_policy)
    print(json.dumps({'directory':str(a.out),'objects':len(r['catalog']['objects']),'target':r['quotas']['target_episodes'],'max_physical_attempts':r['quotas']['max_physical_attempts']}))
