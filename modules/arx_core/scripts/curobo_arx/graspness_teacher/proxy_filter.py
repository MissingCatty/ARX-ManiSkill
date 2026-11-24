"""Run the unchanged Graspness point-cloud filter in its registered CPU env."""
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scripts.curobo_arx.data_factory.common import read, write, sha


def group(rows):
    rows=np.asarray(rows)
    if rows.ndim!=2 or rows.shape[1]!=17 or not np.isfinite(rows).all():
        raise ValueError('Invalid refined GraspNet rows')
    return SimpleNamespace(translations=rows[:,13:16],rotation_matrices=rows[:,4:13].reshape(-1,3,3),
                           widths=rows[:,1],heights=rows[:,2],depths=rows[:,3])


def run(bundle):
    cfg=read(bundle);source=Path(cfg['collision_source'])
    if sha(source)!=cfg['collision_source_sha256']:raise RuntimeError('Registered source collision filter changed')
    for path,expected in cfg['input_files'].items():
        if sha(path)!=expected:raise RuntimeError('Refinement source point cloud/rows changed')
    spec=importlib.util.spec_from_file_location('registered_collision_filter',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=np.load(cfg['rows'],allow_pickle=False);grasps=group(rows)
    scene=module.ModelFreeCollisionDetector(np.load(cfg['points'],allow_pickle=False),voxel_size=cfg['voxel_size'])
    target=module.ModelFreeCollisionDetector(np.load(cfg['target_points'],allow_pickle=False),voxel_size=cfg['voxel_size'])
    colliding=scene.detect(grasps,approach_dist=.05,collision_thresh=cfg['collision_threshold'])
    _,empty=target.detect(grasps,approach_dist=.05,collision_thresh=cfg['collision_threshold'],
                         return_empty_grasp=True,empty_thresh=cfg['target_empty_threshold'])
    write(cfg['output'],{'colliding':colliding.tolist(),'target_empty':empty.tolist(),
                        'pass':(~colliding&~empty).tolist(),'input_files':cfg['input_files'],
                        'source_sha256':sha(source),'physical_commands':0,'neural_weights_loaded':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('bundle');args=parser.parse_args();run(args.bundle)
