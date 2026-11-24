"""Bounded archived-scene IK diagnosis; never connects to an ARX simulator."""
import argparse
import os
from pathlib import Path
import time
from .common import ROOT, read, sha, write


def diagnose(campaign, block, output, gpu, pool_path=None, mesh_cache=None):
    from .launch import devices, environment
    output=Path(output).absolute()
    output.mkdir(parents=True,exist_ok=False)
    campaign,block=Path(campaign).absolute(),Path(block).absolute()
    inventory=devices()
    device=next(d for d in inventory if str(d['index'])==str(gpu) or d['uuid']==gpu)
    if device['free_mb']<6500:
        raise RuntimeError('Insufficient memory for the two frozen-model planners')
    os.environ.update(environment(device))
    pool_path=Path(pool_path).absolute() if pool_path else block/'perception/pool.json'
    inputs=[campaign/'registration.json',block/'qualification_scene.json',
            block/'perception/head_observation.json',pool_path]
    cached={}
    if mesh_cache:
        # Immutable derived OBJ files only. Exact scene digest remains in the
        # directory key; no robot weights, trajectories or acceptance are reused.
        mesh_cache=Path(mesh_cache).absolute()
        for path in mesh_cache.glob('*/*.obj'):
            dest=output/'scene_versions'/path.relative_to(mesh_cache)
            dest.parent.mkdir(parents=True,exist_ok=True)
            os.link(path,dest)
            cached[str(dest)]=sha(path)
    sources=list((ROOT/'scripts/curobo_arx/data_factory').glob('*.py'))
    write(output/'registration.json',{'wall':time.time(),'gpu':device,
        'kind':'archived_scene_model_only','physical_commands':0,
        'inputs':{str(p):sha(p) for p in inputs},'sources':{str(p):sha(p) for p in sources},
        'derived_mesh_cache':cached})
    from .teacher import Teacher
    teacher=Teacher(read(campaign/'registration.json'),output)
    snapshot=read(block/'perception/head_observation.json')['snapshot']
    pool=read(pool_path)
    started=time.monotonic()
    result=teacher.qualify_pool({'action':'qualify_pool','scene':str(block/'qualification_scene.json'),
        'snapshot':snapshot,'pool':pool})
    write(output/'result.json',{'seconds':time.monotonic()-started,'physical_commands':0,'result':result})
    print('Accepted',len(result['candidates']),'of',len(pool['candidates']),'archived candidates',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign');parser.add_argument('block');parser.add_argument('--out',required=True)
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--pool');parser.add_argument('--mesh-cache')
    args=parser.parse_args()
    diagnose(args.campaign,args.block,args.out,args.gpu,args.pool,args.mesh_cache)
