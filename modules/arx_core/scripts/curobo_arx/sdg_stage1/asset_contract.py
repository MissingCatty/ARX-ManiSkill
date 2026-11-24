"""Freeze additional objects against their own registered USD, never old history.

The original 67-object historical comparison remains intact. New objects have
authored USD checks plus a pre-motion runtime baseline, explicitly not history.
"""
import hashlib
from pathlib import Path
import sys
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics
from scripts.curobo_arx.data_factory.common import ROOT,digest,read,write


def source_contract(path):
    sys.path.insert(0,str(ROOT/'arx-sim'))
    from arx_sim.diagnostics.research_fingerprint import plain
    stage=Usd.Stage.Open(str(path));root=stage.GetDefaultPrim()
    if not root or not root.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Additional object requires rigid default prim')
    api=UsdPhysics.RigidBodyAPI(root)
    if not api.GetRigidBodyEnabledAttr().Get() or api.GetKinematicEnabledAttr().Get() or root.GetAttribute('physxRigidBody:disableGravity').Get() is True:
        raise ValueError('Additional object must be dynamic with original scene gravity enabled')
    root_transform=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(root)).T
    if not np.allclose(np.linalg.svd(root_transform[:3,:3],compute_uv=False),1.,atol=1e-6):
        raise ValueError('Additional rigid root has non-unit scale; needs a separate declared asset version')
    mass=UsdPhysics.MassAPI(root).GetMassAttr().Get()
    if mass is None or not 0<float(mass)<20:raise ValueError('Additional source needs explicitly positive mass <20kg')
    geometry={'points','faceVertexIndices','faceVertexCounts','extent','size','radius','height','axis','subdivisionScheme'}
    spec={}
    for prim in Usd.PrimRange(root):
        rigid=prim.HasAPI(UsdPhysics.RigidBodyAPI);attrs={}
        for attr in prim.GetAttributes():
            name=attr.GetName()
            if not attr.HasAuthoredValueOpinion() or name in ('physics:velocity','physics:angularVelocity') or name.startswith('state:'):continue
            if not (name.startswith(('physics:','physx','drive:')) or name.startswith('xformOp') and not rigid
                or name in geometry and prim.IsA(UsdGeom.Gprim) and prim.HasAPI(UsdPhysics.CollisionAPI)):continue
            value=attr.Get()
            if value is not None and not isinstance(value,(str,dict)) and hasattr(value,'__len__') and len(value)>64:
                array=np.asarray(value)
                if array.dtype.kind in 'biuf':
                    attrs[name]={'sha256':hashlib.sha256(array.tobytes()).hexdigest(),'shape':list(array.shape),'dtype':str(array.dtype)};continue
            attrs[name]=plain(value)
        rel={r.GetName():[str(p) for p in r.GetTargets()] for r in prim.GetRelationships()
            if r.GetName().startswith(('physics:','material:binding','filtered'))}
        if attrs or rel or rigid:
            spec[str(prim.GetPath())[len(str(root.GetPath())):]]={'type':prim.GetTypeName(),
                'schemas':sorted(prim.GetAppliedSchemas()),'attributes':attrs,'relationships':rel}
    return {'source_root':str(root.GetPath()),'authored':spec,'expected_mass_kg':float(mass),
        'inertia_policy':'Keep original USD auto-inertia; capture positive live inertia at startup and freeze thereafter',
        'historical_runtime_reference_exists':False}


def check_and_separate(current,contracts,baseline_path):
    """Check new assets, then return the unchanged historical subset for comparison."""
    import copy
    historical=copy.deepcopy(current);live={};issues=[]
    for name,contract in contracts.items():
        prefix='/World/Task/assets/asset_'+name
        usd={p:v for p,v in current['usd'].items() if p==prefix or p.startswith(prefix+'/')}
        if not usd:issues.append({'asset':name,'error':'new asset missing from runtime USD'});continue
        for relative,expected in contract['authored'].items():
            path=prefix+relative;actual=usd.get(path)
            if actual is None:issues.append({'path':path,'error':'registered physical prim missing'});continue
            if expected['type']!=actual['type'] or not set(expected['schemas']).issubset(actual['schemas']):
                issues.append({'path':path,'error':'type or required schema changed'})
            for attr,value in expected['attributes'].items():
                if actual['attributes'].get(attr)!=value:issues.append({'path':path,'attribute':attr,'error':'authored physical/geometry value changed'})
            for rel,targets in expected['relationships'].items():
                remap=lambda p:prefix+p[len(contract['source_root']):] if p==contract['source_root'] or p.startswith(contract['source_root']+'/') else p
                if actual['relationships'].get(rel)!=[remap(t) for t in targets]:issues.append({'path':path,'relationship':rel,'error':'registered relationship changed'})
        runtime=current['runtime_objects'].get(prefix)
        if runtime is None:issues.append({'asset':name,'error':'new asset runtime getters missing'});continue
        masses=np.asarray(runtime['get_masses'],dtype=float)
        inertias=np.asarray(runtime['get_inertias'],dtype=float).reshape(-1,3,3)
        if not np.isfinite(masses).all() or not np.allclose(masses,contract['expected_mass_kg'],rtol=1e-5,atol=1e-7):
            issues.append({'asset':name,'error':'runtime mass differs from source'})
        if not np.isfinite(inertias).all() or (np.linalg.eigvalsh(inertias)<=0).any():
            issues.append({'asset':name,'error':'invalid computed inertia'})
        for v in usd.values():v['schemas']=sorted(v['schemas'])
        live[name]={'usd':usd,'runtime':runtime}
        historical['usd']={p:v for p,v in historical['usd'].items() if not(p==prefix or p.startswith(prefix+'/'))}
        historical['runtime_objects'].pop(prefix,None)
    path=Path(baseline_path)
    if issues:return historical,{'pass':False,'issues':issues,'baseline_created':False}
    if path.exists():
        old=read(path)
        if old['live_sha256']!=digest(live):issues.append({'error':'new asset physical properties changed after initial runtime freeze'})
        if old['contract_sha256']!=digest(contracts):issues.append({'error':'new asset source contract changed'})
        created=False
    else:
        write(path,{'live':live,'live_sha256':digest(live),'contract_sha256':digest(contracts),
            'scope':'New object startup evidence before any production command; not historical runtime evidence'})
        created=True
    return historical,{'pass':not issues,'issues':issues,'baseline_created':created,
        'baseline_file':str(path),'asset_count':len(contracts),
        'scope':'Authored USD/source-mass match plus frozen new live physics; historical robot and original67 compared separately'}
