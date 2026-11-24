"""Finite motion-goal search after the unchanged proposal checks find no G.

Tool/camera calibration and robot/scene models are never edited. Every new pose
roundtrips through the original GraspNet-to-ARX converter and original gates.
"""
import copy
import itertools
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, digest
from scripts.curobo_arx.data_factory.geometry import grasp_pool, grasp_equal, handoff, rigid
from .finger_overlap import proposal_variant


def offsets():
    values=[np.eye(3)[axis]*sign*distance for axis in range(3)
            for sign in [-1,1] for distance in [.003,.006,.009,.012]]
    values.extend(np.array(p)*.006 for p in itertools.product([-1,0,1],repeat=3)
                  if sum(v!=0 for v in p)>=2)
    return sorted(values,key=lambda p:(round(float(np.linalg.norm(p)),9),tuple(p)))


def refined_candidate(source, seed_world, delta, observation, asset):
    camera=rigid(observation['camera_world_optical']).copy()
    obj=rigid(observation['snapshot']['objects']['test_object']['world']).copy()
    row=np.asarray(source['network_row'],dtype=float)
    original=grasp_pool(row[None],camera,obj,asset['size_m'],maximum=1)['candidates']
    if len(original)!=1:raise RuntimeError('Cannot reconstruct original neural seed')
    actual=rigid(source['grasp_world']);expected=rigid(original[0]['grasp_world'])
    if not np.allclose(actual[:3,:3],expected[:3,:3],atol=1e-6):
        raise RuntimeError('Seed rotation differs from frozen source conversion')
    # Parent translation is the existing environment-role mapping, not a tool
    # calibration. Verify it through the original converter on every proposal.
    origin=expected[:3,3]-actual[:3,3];camera[:3,3]-=origin;obj[:3,3]-=origin
    canonical=grasp_pool(row[None],camera,obj,asset['size_m'],maximum=1)['candidates'][0]
    if not np.allclose(canonical['grasp_world'],actual,atol=1e-6):raise RuntimeError('Role conversion mismatch')
    source_rotation=camera[:3,:3]@row[4:13].reshape(3,3)
    fixed_axes=source_rotation.T@actual[:3,:3]
    fixed_offset=source_rotation.T@(actual[:3,3]-(camera[:3,:3]@row[13:16]+camera[:3,3]))
    desired=rigid(seed_world).copy();delta=np.asarray(delta,dtype=float)
    if delta.shape!=(3,) or not np.isfinite(delta).all() or np.linalg.norm(delta)>.012000001:
        raise ValueError('Motion-goal search exceeds the registered radius')
    desired[:3,3]+=desired[:3,:3]@delta
    changed=row.copy();rotation=camera[:3,:3].T@desired[:3,:3]@fixed_axes.T
    changed[4:13]=rotation.reshape(-1)
    changed[13:16]=camera[:3,:3].T@(desired[:3,3]-camera[:3,3])-rotation@fixed_offset
    result=grasp_pool(changed[None],camera,obj,asset['size_m'],maximum=1)['candidates']
    if not result:return None  # Original target-association gate, unchanged.
    derived={**copy.deepcopy(source),**result[0]}
    if not np.allclose(derived['grasp_world'],desired,atol=1e-6):raise RuntimeError('Frozen pose conversion roundtrip failed')
    if 'placement_id' in source:
        derived['unscoped_goal_id']=derived['goal_id']
        derived['goal_id']=digest([source['placement_id'],derived['goal_id']])[:20]
    derived.pop('camera_safe_equivalent_flips',None)
    derived['pregrasp_world']=handoff(desired,source['pregrasp_distance_mm'],0,0)['world']
    derived['pose_refinement']={'method':'bounded_local_translation_v1','source_goal_id':source['goal_id'],
        'source_grasp_world':source['grasp_world'],'raw_network_row':source['network_row'],
        'seed_grasp_world':np.asarray(seed_world).tolist(),'local_translation_m':delta.tolist(),
        'radius_limit_m':.012,'calibration_or_robot_model_modified':False}
    return derived


def generate(teacher, request, seeds):
    cfg=teacher.registration;policy=cfg['local_goal_refinement']
    if policy!={'max_seeds':2,'max_per_seed':3,'radius_m':.012,'target_empty_threshold':.01}:
        raise ValueError('Unregistered local-goal search policy')
    root=teacher.directory/'refinement_queries'/str(time.time_ns());root.mkdir(parents=True)
    perception=Path(request['scene']).parent/'perception'
    observation=read(perception/'head_observation.json')
    asset=next(a for a in cfg['catalog']['objects'] if a['object_id']==observation['object_id'])
    if asset['asset_sha256']!=request['visual_context']['asset']['asset_sha256']:
        raise RuntimeError('Refinement target identity differs from the registered catalog')
    from scripts.curobo_arx.data_factory.neural_input import cloud_input
    points,region=cloud_input(np.load(perception/'head_depth.npy',allow_pickle=False),observation,asset)
    expected=request['pool'].get('point_cloud',{}).get('source_depth_sha256')
    if expected and sha(perception/'head_depth.npy')!=expected:raise RuntimeError('Refinement source depth changed')
    candidates=[]
    sources={g['goal_id']:g for g in request['pool']['candidates']}
    for seed in seeds[:policy['max_seeds']]:
        source=sources[seed['goal_id']]
        for delta in offsets():
            candidate=refined_candidate(source,seed['grasp_world'],delta,observation,asset)
            if candidate is not None:candidates.append(candidate)
    if not candidates:return [],{'generated':0,'directory':str(root)}
    np.save(root/'points.npy',points);np.save(root/'target_points.npy',points[region])
    np.save(root/'rows.npy',np.asarray([g['network_row'] for g in candidates]))
    code=Path(cfg['graspness']['source'])/'utils/collision_detector.py'
    bundle={'collision_source':str(code),'collision_source_sha256':sha(code),
        'points':str(root/'points.npy'),'target_points':str(root/'target_points.npy'),'rows':str(root/'rows.npy'),
        'input_files':{str(root/name):sha(root/name) for name in ['points.npy','target_points.npy','rows.npy']},
        'voxel_size':cfg['graspness']['collision_voxel_size_m'],'collision_threshold':cfg['graspness']['collision_threshold'],
        'target_empty_threshold':policy['target_empty_threshold'],'output':str(root/'proxy_result.json')}
    write(root/'bundle.json',bundle);write(root/'generated_candidates.json',candidates)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    env['LD_LIBRARY_PATH']=os.pathsep.join([*cfg['graspness'].get('library_directories',[]),env.get('LD_LIBRARY_PATH','')])
    command=[cfg['graspness']['python'],'-m','scripts.curobo_arx.graspness_teacher.proxy_filter',str(root/'bundle.json')]
    with (root/'proxy.log').open('x') as log:
        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
    proxy=read(root/'proxy_result.json')
    if len(proxy['pass'])!=len(candidates):raise RuntimeError('Source-proxy response lost candidate identity')
    target=teacher.visual_target(request);accepted=[];counts={};geometry=[]
    for candidate,passed in zip(candidates,proxy['pass']):
        if not passed:continue
        ident=candidate['pose_refinement']['source_goal_id']
        if counts.get(ident,0)>=policy['max_per_seed']:continue
        evidence=teacher.finger_overlap.inspect(candidate['grasp_world'],request['snapshot'],target.mesh.vertices)
        geometry.append({'goal_id':candidate['goal_id'],**evidence})
        if evidence['reject_proposal'] or any(grasp_equal(candidate,g) for g in accepted):continue
        accepted.append(candidate);counts[ident]=counts.get(ident,0)+1
    receipt={'generated':len(candidates),'source_proxy_and_target_nonempty':sum(proxy['pass']),
             'authored_geometry_checks':geometry,'retained':len(accepted),'directory':str(root),
             'physical_commands':0,'original_calibration_robot_and_all_original_gates_unchanged':True}
    write(root/'generation_receipt.json',receipt)
    return accepted,receipt


def refining_teacher(base):
    from scripts.curobo_arx.data_factory.teacher import Teacher as OriginalTeacher
    class Teacher(base):
        def qualify_pool(self,request):
            result=super().qualify_pool(request)
            result['model_pool_qualification_calls']=1
            if not proposal_variant(self.registration,request) or result['candidates']:return result
            trace=self.directory/'refinement_queries'/(str(time.time_ns())+'_qualification')
            write(trace/'unrefined_qualification.json',result)
            # Virtual seed selection only: no command is emitted. The additional
            # authored-finger filter has already rejected these original goals.
            seeds=OriginalTeacher.qualify_pool(self,request)
            write(trace/'seed_qualification.json',seeds)
            result['model_pool_qualification_calls']+=1
            result['refinement_trace']=str(trace)
            if not seeds['candidates']:return result
            candidates,receipt=generate(self,request,seeds['candidates'])
            receipt['qualification_trace']=str(trace)
            if not candidates:
                result['goal_refinement']=receipt;return result
            changed=copy.deepcopy(request);changed['pool']['candidates']=candidates
            refined=super().qualify_pool(changed)
            refined['model_pool_qualification_calls']=3
            refined['goal_refinement']={**receipt,'unrefined_qualified':0,
                'refined_qualified':len(refined['candidates'])}
            return refined
    return Teacher
